#!/usr/bin/env python3
"""
Save as: check_gguf.py
Run with: python3 check_gguf.py
"""

import struct
from pathlib import Path

GGUF_PATH = "bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf"

def read_gguf_string(data, offset):
    length = struct.unpack_from('<Q', data, offset)[0]
    string = data[offset+8:offset+8+length].decode('utf-8')
    return string, offset + 8 + length

def unpack_ternary_2bit(byte_val):
    """Unpack 4 ternary values from one byte (2 bits each)"""
    vals = []
    for i in range(4):
        bits = (byte_val >> (i * 2)) & 0b11
        # Try different mappings
        vals.append(bits)
    return vals

def main():
    path = Path(GGUF_PATH)
    data = path.read_bytes()
    
    # Quick parse to get tensor info
    n_tensors = struct.unpack_from('<Q', data, 8)[0]
    n_kv = struct.unpack_from('<Q', data, 16)[0]
    
    offset = 24
    for _ in range(n_kv):
        key, offset = read_gguf_string(data, offset)
        vtype = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        if vtype == 8:
            _, offset = read_gguf_string(data, offset)
        elif vtype in [4, 5, 6]:
            offset += 4
        elif vtype == 7:
            offset += 1
        elif vtype == 10:
            offset += 8
        elif vtype == 9:
            arr_type = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            arr_len = struct.unpack_from('<Q', data, offset)[0]
            offset += 8
            if arr_type == 8:
                for _ in range(arr_len):
                    _, offset = read_gguf_string(data, offset)
            elif arr_type in [4, 5, 6]:
                offset += arr_len * 4
            elif arr_type == 10:
                offset += arr_len * 8
    
    # Parse tensors
    tensors = []
    for i in range(n_tensors):
        name, offset = read_gguf_string(data, offset)
        n_dims = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        dims = []
        for _ in range(n_dims):
            dims.append(struct.unpack_from('<Q', data, offset)[0])
            offset += 8
        tensor_type = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        data_offset = struct.unpack_from('<Q', data, offset)[0]
        offset += 8
        n_elements = 1
        for d in dims:
            n_elements *= d
        tensors.append({
            'name': name, 'dims': dims, 'type': tensor_type,
            'offset': data_offset, 'n_elements': n_elements
        })
    
    data_section_start = (offset + 31) // 32 * 32
    
    # Focus on one tensor for deep analysis
    print("="*70)
    print("DEEP ANALYSIS OF blk.0.attn_q.weight")
    print("="*70)
    
    t = next(x for x in tensors if x['name'] == 'blk.0.attn_q.weight')
    abs_offset = data_section_start + t['offset']
    n_elem = t['n_elements']
    expected_bytes = (n_elem * 2 + 7) // 8
    
    print(f"Shape: {t['dims']}")
    print(f"Elements: {n_elem:,}")
    print(f"Expected 2-bit bytes: {expected_bytes:,}")
    print(f"Absolute offset: {abs_offset:,}")
    
    # Read the tensor data
    tensor_data = data[abs_offset:abs_offset + expected_bytes + 64]
    
    print(f"\n--- First 64 bytes ---")
    print(f"Hex: {tensor_data[:32].hex()}")
    print(f"     {tensor_data[32:64].hex()}")
    
    print(f"\n--- Last 64 bytes of expected range ---")
    end_data = data[abs_offset + expected_bytes - 32:abs_offset + expected_bytes + 32]
    print(f"Before end: {end_data[:32].hex()}")
    print(f"After end:  {end_data[32:64].hex()}")
    
    # Check if first bytes look like ternary data
    print(f"\n--- Ternary unpacking test (first 8 bytes) ---")
    for i in range(8):
        byte_val = tensor_data[i]
        trits = unpack_ternary_2bit(byte_val)
        print(f"  Byte {i}: 0x{byte_val:02x} -> {trits}")
    
    # Check distribution of 2-bit values in first 1000 bytes
    print(f"\n--- Distribution of 2-bit patterns (first 10000 bytes) ---")
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for byte_val in tensor_data[:10000]:
        for i in range(4):
            bits = (byte_val >> (i * 2)) & 0b11
            counts[bits] += 1
    total = sum(counts.values())
    for bits, count in sorted(counts.items()):
        print(f"  {bits:02b} ({bits}): {count:,} ({100*count/total:.1f}%)")
    
    # Try different ternary mappings
    print(f"\n--- Possible ternary mappings ---")
    print("  If 0->0, 1->+1, 2->-1, 3->??? :")
    print(f"    +1: {counts[1]:,} ({100*counts[1]/total:.1f}%)")
    print(f"     0: {counts[0]:,} ({100*counts[0]/total:.1f}%)")
    print(f"    -1: {counts[2]:,} ({100*counts[2]/total:.1f}%)")
    print(f"   ???: {counts[3]:,} ({100*counts[3]/total:.1f}%)")
    
    print("  If 0->-1, 1->0, 2->+1, 3->??? :")
    print(f"    +1: {counts[2]:,} ({100*counts[2]/total:.1f}%)")
    print(f"     0: {counts[1]:,} ({100*counts[1]/total:.1f}%)")
    print(f"    -1: {counts[0]:,} ({100*counts[0]/total:.1f}%)")
    print(f"   ???: {counts[3]:,} ({100*counts[3]/total:.1f}%)")

    # Check if value 3 ever appears
    print(f"\n--- Check for value 3 (should not exist in ternary) ---")
    has_3 = counts[3] > 0
    print(f"  Value 3 appears: {has_3} ({counts[3]} times)")
    if has_3:
        print("  WARNING: Value 3 found! This shouldn't happen in pure ternary.")
        print("  The format might use a different bit packing.")
    
    # Try row-wise analysis (check if each row has a scale prefix)
    print(f"\n--- Row-wise analysis ---")
    row_size = t['dims'][1]  # 2560 elements per row
    row_bytes = (row_size * 2 + 7) // 8  # bytes for one row of 2-bit values
    print(f"Row size: {row_size} elements = {row_bytes} bytes (2-bit packed)")
    
    # Check first few rows
    for row_idx in range(3):
        row_start = row_idx * row_bytes
        row_data = tensor_data[row_start:row_start + 16]
        print(f"\n  Row {row_idx} first 16 bytes: {row_data.hex()}")
        # Try interpreting first 2 bytes as f16 scale
        f16 = struct.unpack_from('<e', row_data, 0)[0]
        print(f"    If first 2 bytes are f16 scale: {f16}")
    
    print(f"\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if counts[3] == 0:
        print("""
✓ Only values 0, 1, 2 appear (no 3s) - confirms ternary encoding.
✓ The 32 extra bytes are likely just alignment padding.

The i2_s format appears to be PURE 2-bit packed ternary with no embedded scales.

Your ternary mapping should match your distribution stats above.
Compare with your Rust output to verify the mapping is correct.

LIKELY ISSUE: Since there are no scales, your problem is elsewhere:
  1. SubLN normalization (BitNet uses this, not standard LayerNorm)
  2. Activation quantization (int8 absmax per-token)
  3. RoPE implementation
  4. The tied embeddings output scaling
""")
    else:
        print("""
⚠ Value 3 appears in the data - this is unexpected for ternary.
The bit packing might be different than assumed.
""")

if __name__ == "__main__":
    main()