//! Minimal GGUF reader for BitNet b1.58 2B4T (ggml type I2_S = 36).
//!
//! Verified on-disk facts (see reference/gguf_loader.py and the project memory):
//!  * GGUF v3, little-endian, alignment 32.
//!  * token_embd.weight is F16; the four per-block norms + output_norm are F32.
//!  * q,k,v,o,gate,up,down are I2_S.
//!
//! I2_S tensor layout:
//!   [ceil(ne/4) bytes of 2-bit codes] [one little-endian f32 scale] [pad to 32B]
//! Codes are 128-element block interleaved: for byte j in 0..32 and 2-bit slot
//! s in 0..4 (slot s = bits [2s,2s+1], LSB-first), the element index within the
//! 32-byte / 128-element block is  (3 - s) * 32 + j.  Decoded value = code - 1.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

const GGUF_MAGIC: u32 = 0x4655_4747;
pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_I2_S: u32 = 36;

#[derive(Debug, Clone)]
pub enum Val {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32),
    F32(f32), Bool(bool), Str(String), Arr(Vec<Val>),
    U64(u64), I64(i64), F64(f64),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dims: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
}

pub struct Gguf {
    path: String,
    pub metadata: HashMap<String, Val>,
    pub tensors: HashMap<String, TensorInfo>,
    pub data_offset: u64,
}

impl Gguf {
    pub fn open(path: &str) -> io::Result<Self> {
        let mut f = File::open(path)?;
        if rd_u32(&mut f)? != GGUF_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a GGUF file"));
        }
        let version = rd_u32(&mut f)?;
        if version < 2 || version > 3 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported GGUF version"));
        }
        let n_tensors = rd_u64(&mut f)?;
        let n_meta = rd_u64(&mut f)?;

        let mut metadata = HashMap::new();
        for _ in 0..n_meta {
            let key = rd_str(&mut f)?;
            let t = rd_u32(&mut f)?;
            metadata.insert(key, rd_val(&mut f, t)?);
        }
        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = rd_str(&mut f)?;
            let ndim = rd_u32(&mut f)? as usize;
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                dims.push(rd_u64(&mut f)?);
            }
            let dtype = rd_u32(&mut f)?;
            let offset = rd_u64(&mut f)?;
            tensors.insert(name, TensorInfo { dims, dtype, offset });
        }
        let pos = f.stream_position()?;
        let align = match metadata.get("general.alignment") {
            Some(Val::U32(a)) => *a as u64,
            _ => 32,
        };
        let data_offset = (pos + align - 1) / align * align;
        Ok(Self { path: path.to_string(), metadata, tensors, data_offset })
    }

    // ---- typed metadata getters ----
    pub fn meta_u32(&self, k: &str) -> Option<u32> {
        match self.metadata.get(k)? { Val::U32(v) => Some(*v), Val::I32(v) => Some(*v as u32), _ => None }
    }
    pub fn meta_u64(&self, k: &str) -> Option<u64> {
        match self.metadata.get(k)? {
            Val::U64(v) => Some(*v), Val::U32(v) => Some(*v as u64),
            Val::I32(v) => Some(*v as u64), _ => None,
        }
    }
    pub fn meta_f32(&self, k: &str) -> Option<f32> {
        match self.metadata.get(k)? { Val::F32(v) => Some(*v), Val::F64(v) => Some(*v as f32), _ => None }
    }
    pub fn meta_str(&self, k: &str) -> Option<&str> {
        match self.metadata.get(k)? { Val::Str(s) => Some(s.as_str()), _ => None }
    }

    fn read_at(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Load an F32 tensor as Vec<f32>.
    pub fn load_f32(&self, name: &str) -> io::Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let raw = self.read_at(self.data_offset + t.offset, t.n_elements() * 4)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// Load an F16 tensor as Vec<f32>.
    pub fn load_f16(&self, name: &str) -> io::Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let raw = self.read_at(self.data_offset + t.offset, t.n_elements() * 2)?;
        Ok(raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
    }

    /// Load an I2_S tensor. Returns (ternary codes in {-1,0,1} as [out*in]
    /// row-major, per-tensor scale, in_features, out_features).
    /// ggml dims are [in_features, out_features].
    pub fn load_i2s(&self, name: &str) -> io::Result<(Vec<i8>, f32, usize, usize)> {
        let t = self.tensor(name)?;
        if t.dtype != GGML_I2_S {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "tensor is not I2_S"));
        }
        let ne = t.n_elements();
        let code_bytes = (ne + 3) / 4;
        let raw = self.read_at(self.data_offset + t.offset, code_bytes + 4)?;
        let codes = unpack_i2s(&raw[..code_bytes], ne);
        let s = &raw[code_bytes..code_bytes + 4];
        let scale = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        let in_f = t.dims[0] as usize;
        let out_f = t.dims[1] as usize;
        Ok((codes, scale, in_f, out_f))
    }

    fn tensor(&self, name: &str) -> io::Result<&TensorInfo> {
        self.tensors.get(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("tensor {name}")))
    }
}

/// Expand packed 2-bit I2_S codes (128-element block interleave) to ternary i8.
pub fn unpack_i2s(packed: &[u8], n: usize) -> Vec<i8> {
    let mut out = vec![0i8; n];
    let nb = packed.len() / 32;
    for blk in 0..nb {
        let base = blk * 32;
        let eblk = blk * 128;
        for j in 0..32 {
            let byte = packed[base + j];
            for s in 0..4 {
                let code = ((byte >> (2 * s)) & 0b11) as i8;
                let e = eblk + (3 - s) * 32 + j;
                if e < n {
                    out[e] = code - 1; // 0->-1, 1->0, 2->+1
                }
            }
        }
    }
    out
}

// ---- primitive readers ----
fn rd_u32(f: &mut File) -> io::Result<u32> {
    let mut b = [0u8; 4]; f.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
fn rd_u64(f: &mut File) -> io::Result<u64> {
    let mut b = [0u8; 8]; f.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
fn rd_str(f: &mut File) -> io::Result<String> {
    let len = rd_u64(f)? as usize;
    let mut b = vec![0u8; len]; f.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
fn rd_val(f: &mut File, t: u32) -> io::Result<Val> {
    Ok(match t {
        0 => { let mut b = [0u8; 1]; f.read_exact(&mut b)?; Val::U8(b[0]) }
        1 => { let mut b = [0u8; 1]; f.read_exact(&mut b)?; Val::I8(b[0] as i8) }
        2 => { let mut b = [0u8; 2]; f.read_exact(&mut b)?; Val::U16(u16::from_le_bytes(b)) }
        3 => { let mut b = [0u8; 2]; f.read_exact(&mut b)?; Val::I16(i16::from_le_bytes(b)) }
        4 => Val::U32(rd_u32(f)?),
        5 => { let mut b = [0u8; 4]; f.read_exact(&mut b)?; Val::I32(i32::from_le_bytes(b)) }
        6 => { let mut b = [0u8; 4]; f.read_exact(&mut b)?; Val::F32(f32::from_le_bytes(b)) }
        7 => { let mut b = [0u8; 1]; f.read_exact(&mut b)?; Val::Bool(b[0] != 0) }
        8 => Val::Str(rd_str(f)?),
        9 => {
            let et = rd_u32(f)?;
            let cnt = rd_u64(f)? as usize;
            let mut arr = Vec::with_capacity(cnt);
            for _ in 0..cnt { arr.push(rd_val(f, et)?); }
            Val::Arr(arr)
        }
        10 => Val::U64(rd_u64(f)?),
        11 => { let mut b = [0u8; 8]; f.read_exact(&mut b)?; Val::I64(i64::from_le_bytes(b)) }
        12 => { let mut b = [0u8; 8]; f.read_exact(&mut b)?; Val::F64(f64::from_le_bytes(b)) }
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("bad value type {t}"))),
    })
}

fn f16_to_f32(h: u16) -> f32 {
    let s = (h >> 15) & 1;
    let e = (h >> 10) & 0x1F;
    let m = h & 0x3FF;
    let sign = if s == 1 { -1.0 } else { 1.0 };
    if e == 0 {
        if m == 0 { return sign * 0.0; }
        return sign * (m as f32) * 2f32.powi(-24);
    }
    if e == 0x1F {
        return if m == 0 { sign * f32::INFINITY } else { f32::NAN };
    }
    sign * (1.0 + (m as f32) / 1024.0) * 2f32.powi(e as i32 - 15)
}
