import numpy as np, sys
from gguf_loader import GGUFLoader
from bitnet_float import rmsnorm, apply_rope, rope_tables
from tokenizers import Tokenizer

tok = Tokenizer.from_file('../bitnet-b1.58-2B-4T/tokenizer.json')
ld = GGUFLoader('../bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf'); cfg = ld.config()
hd=cfg['head_dim'];base=cfg['rope_base'];d=cfg['d_model'];nh=cfg['n_head'];nkv=cfg['n_kv'];eps=cfg['rms_eps'];rep=nh//nkv
embed=ld.load_f16('token_embd.weight').reshape(cfg['vocab'],d); outn=ld.load_f32('output_norm.weight')
PARIS=12366

# Build a permutation perm of length 128: decoded_element[e] = raw_seq[perm[e]]
# raw_seq[4*j+s] = (byte[j] >> 2s) & 3   (j in 0..31, s in 0..3)
def perm_for(scheme):
    P=np.empty(128,dtype=np.int64)
    if scheme=='seq':
        return np.arange(128)
    if scheme=='g32':       # elem = s*32+j  -> seq idx 4j+s
        for j in range(32):
            for s in range(4): P[s*32+j]=4*j+s
        return P
    if scheme=='g32_rev':   # elem = (3-s)*32+j
        for j in range(32):
            for s in range(4): P[(3-s)*32+j]=4*j+s
        return P
    if scheme=='chunk8_stride':  # 4 chunks of 8 bytes -> 32 elems, byte i holds elems i,i+8,i+16,i+24
        for c in range(4):
            for i in range(8):
                for s in range(4): P[c*32 + s*8 + i]=4*(c*8+i)+s
        return P
    if scheme=='chunk8_stride_rev':
        for c in range(4):
            for i in range(8):
                for s in range(4): P[c*32 + (3-s)*8 + i]=4*(c*8+i)+s
        return P
    if scheme=='byte_interleave4':  # elem = j + 32*s already covered; try elem=2-bit planes
        return P
    raise ValueError(scheme)

def make_unpack(perm):
    def unpack(packed,n):
        nb=packed.size//32
        b=packed[:nb*32].reshape(nb,32).astype(np.int16)
        seq=np.empty((nb,128),dtype=np.int16)
        for s in range(4): seq[:, s::4] = (b>>(2*s))&3   # seq[:,4j+s]
        # apply perm: elem[e]=seq[perm[e]]
        el=seq[:,perm]
        return (el.reshape(-1)[:n].astype(np.int8)-1)
    return unpack

Ls=[]
for i in range(cfg['n_layer']):
    p=f'blk.{i}.'; g=lambda nm: ld.load_i2s_packed(p+nm)
    Ls.append(dict(an=ld.load_f32(p+'attn_norm.weight'),fn=ld.load_f32(p+'ffn_norm.weight'),
        asub=ld.load_f32(p+'attn_sub_norm.weight'),fsub=ld.load_f32(p+'ffn_sub_norm.weight'),
        q=g('attn_q.weight'),k=g('attn_k.weight'),v=g('attn_v.weight'),o=g('attn_output.weight'),
        gt=g('ffn_gate.weight'),up=g('ffn_up.weight'),dn=g('ffn_down.weight')))

def fwd(ids, unpack):
    def lin(t,x):
        pk,sc,dm=t; inf,outf=dm; W=unpack(pk,inf*outf).reshape(outf,inf).astype(np.float32)
        s=127.0/np.clip(np.abs(x).max(-1,keepdims=True),1e-5,None); xq=np.clip(np.round(x*s),-128,127)
        return (xq@W.T)*(sc/s)
    T=len(ids); x=embed[np.asarray(ids)].astype(np.float32)
    cosr,sinr=rope_tables(np.arange(T),hd,base); mask=np.triu(np.full((T,T),-np.inf,np.float32),1); scl=1/np.sqrt(hd)
    for L in Ls:
        h=rmsnorm(x,L['an'],eps)
        q=lin(L['q'],h).reshape(T,nh,hd); k=lin(L['k'],h).reshape(T,nkv,hd); v=lin(L['v'],h).reshape(T,nkv,hd)
        q=apply_rope(q,cosr,sinr); k=apply_rope(k,cosr,sinr); kk=np.repeat(k,rep,1); vv=np.repeat(v,rep,1)
        out=np.empty((T,nh,hd),np.float32)
        for hh in range(nh):
            ss=(q[:,hh]@kk[:,hh].T)*scl+mask; ss-=ss.max(-1,keepdims=True); e=np.exp(ss); pp=e/e.sum(-1,keepdims=True); out[:,hh]=pp@vv[:,hh]
        attn=rmsnorm(out.reshape(T,d),L['asub'],eps); x=x+lin(L['o'],attn)
        h2=rmsnorm(x,L['fn'],eps); gg=lin(L['gt'],h2); uu=lin(L['up'],h2)
        act=rmsnorm(np.square(np.maximum(gg,0))*uu,L['fsub'],eps); x=x+lin(L['dn'],act)
    hn=rmsnorm(x,outn,eps); return hn[-1]@embed.T

ids=[cfg['bos']]+tok.encode('The capital of France is',add_special_tokens=False).ids
schemes=sys.argv[1:] or ['seq','g32','g32_rev','chunk8_stride','chunk8_stride_rev']
for sch in schemes:
    perm=perm_for(sch); lg=fwd(ids, make_unpack(perm)); t=int(np.argmax(lg))
    print(f'{sch:18s} top={t:6d} {tok.decode([t])!r:14s} paris_rank={int((lg>lg[PARIS]).sum())}', flush=True)
