/* kernels.c — real hardware microbenchmark for the BitNet integer datapath.
 *
 * Measures, on the actual CPU, the quantities the paper's edge claims rest on:
 *   (1) sustained memory read bandwidth (BitNet inference is weight/KV streaming-
 *       bound, so this sets the per-token latency floor);
 *   (2) ternary x int8 GEMV throughput (the BitLinear core) hot in cache;
 *   (3) attention KV dot-product latency and memory at int8 vs packed 2-bit,
 *       at a 4096-token context (the per-channel-asymmetric 2-bit KV claim).
 *
 * Build:  cc -O3 -march=native -o kernels kernels.c
 * Run:    ./kernels
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9; }

/* ---- (1) memory read bandwidth ---- */
static double bandwidth_gbs(void){
    size_t N = 256ull*1024*1024;            /* 256 MB, >> LLC */
    volatile uint8_t *b = malloc(N);
    for(size_t i=0;i<N;i++) b[i]=(uint8_t)i;
    double best=0;
    for(int rep=0; rep<5; rep++){
        double t=now(); uint64_t s=0;
        for(size_t i=0;i<N;i+=64) s+=b[i];      /* one touch per cache line */
        double dt=now()-t; (void)s;
        double gbs=(N)/dt/1e9;
        if(gbs>best) best=gbs;
    }
    /* full-stream version (read every byte) for a tighter bound */
    double best2=0;
    for(int rep=0; rep<5; rep++){
        double t=now(); uint64_t s=0;
        for(size_t i=0;i<N;i++) s+=b[i];
        double dt=now()-t; (void)s;
        double gbs=(N)/dt/1e9;
        if(gbs>best2) best2=gbs;
    }
    free((void*)b);
    return best2>best?best2:best;
}

/* ---- (2) ternary x int8 GEMV: out[o]=sum_i code[o*I+i]*x[i], code in {-1,0,1} ---- */
static double gemv_ns(int O,int I,int iters){
    int8_t *W = malloc((size_t)O*I); int8_t *x=malloc(I); int32_t *y=malloc((size_t)O*4);
    for(size_t i=0;i<(size_t)O*I;i++) W[i]=(int8_t)((i*2654435761u)%3)-1; /* -1,0,1 */
    for(int i=0;i<I;i++) x[i]=(int8_t)((i%255)-127);
    double t=now();
    for(int it=0; it<iters; it++){
        for(int o=0;o<O;o++){
            const int8_t *w=W+(size_t)o*I; int32_t acc=0;
            for(int i=0;i<I;i++) acc+=w[i]*x[i];
            y[o]=acc;
        }
    }
    double dt=now()-t; volatile int32_t sink=y[O-1]; (void)sink;
    free(W);free(x);free(y);
    return dt/iters*1e9;   /* ns per GEMV */
}

/* ---- (3) KV attention dot over a context, int8 vs packed 2-bit ---- */
static double kv_dot_int8_ns(int T,int D,int iters){
    int8_t *K=malloc((size_t)T*D); int8_t *q=malloc(D); int32_t *s=malloc((size_t)T*4);
    for(size_t i=0;i<(size_t)T*D;i++) K[i]=(int8_t)((i%255)-127);
    for(int i=0;i<D;i++) q[i]=(int8_t)((i%255)-127);
    double t=now();
    for(int it=0;it<iters;it++)
        for(int p=0;p<T;p++){ const int8_t *k=K+(size_t)p*D; int32_t a=0;
            for(int i=0;i<D;i++) a+=k[i]*q[i]; s[p]=a; }
    double dt=now()-t; volatile int32_t sink=s[T-1]; (void)sink;
    free(K);free(q);free(s); return dt/iters*1e9;
}
static double kv_dot_2bit_ns(int T,int D,int iters){
    /* K packed 2-bit (4 per byte); dequant codeword - 1 -> {-1,0,1,2} centered */
    size_t packed=(size_t)T*D/4; uint8_t *K=malloc(packed); int8_t *q=malloc(D); int32_t *s=malloc((size_t)T*4);
    for(size_t i=0;i<packed;i++) K[i]=(uint8_t)(i*2654435761u);
    for(int i=0;i<D;i++) q[i]=(int8_t)((i%255)-127);
    double t=now();
    for(int it=0;it<iters;it++)
        for(int p=0;p<T;p++){ const uint8_t *kp=K+(size_t)p*D/4; int32_t a=0;
            for(int i=0;i<D;i+=4){ uint8_t b=kp[i>>2];
                a+=(((b)&3)-1)*q[i]+(((b>>2)&3)-1)*q[i+1]+(((b>>4)&3)-1)*q[i+2]+(((b>>6)&3)-1)*q[i+3]; }
            s[p]=a; }
    double dt=now()-t; volatile int32_t sink=s[T-1]; (void)sink;
    free(K);free(q);free(s); return dt/iters*1e9;
}

int main(void){
    /* BitNet b1.58 2B4T shapes */
    int d=2560, ff=6912, nkv=5, nh=20, hd=128, L=30, vocab=128256;
    double bw = bandwidth_gbs();
    printf("== measured on this CPU ==\n");
    printf("memory read bandwidth: %.1f GB/s\n\n", bw);

    /* per-token BitLinear GEMV cost (hot-cache compute ceiling) */
    double q_=gemv_ns(d,d,50), kv_=gemv_ns(nkv*hd,d,50), o_=gemv_ns(d,d,50);
    double g_=gemv_ns(ff,d,20), u_=gemv_ns(ff,d,20), dn_=gemv_ns(d,ff,20);
    double layer_ns = q_+2*kv_+o_+g_+u_+dn_;
    double lm_ns = gemv_ns(vocab>4096?4096:vocab, d, 5) * ((double)vocab/(vocab>4096?4096:vocab));
    double tok_ns = layer_ns*L + lm_ns;
    printf("ternary x int8 GEMV, hot cache (compute ceiling):\n");
    printf("  one layer: %.1f us   x%d layers + lm_head: %.2f ms/token  (%.1f tok/s, 1 core)\n\n",
           layer_ns/1e3, L, tok_ns/1e6, 1e9/tok_ns);

    /* weight-streaming-bound latency: 2-bit packed weights ~ params/4 bytes */
    double params = (double)L*(d*d + 2*nkv*hd*d + d*d + 3*ff*d);  /* ternary params */
    double w2bit_GB = params/4.0/1e9;
    printf("weight-stream-bound latency (the real edge floor):\n");
    printf("  ternary params %.2f B -> 2-bit packed %.2f GB ; at %.1f GB/s = %.1f ms/token\n\n",
           params/1e9, w2bit_GB, bw, w2bit_GB/bw*1e3);

    /* KV cache: int8 vs 2-bit at 4096 ctx */
    int T=4096;
    double kvi=kv_dot_int8_ns(T,hd,200), kv2=kv_dot_2bit_ns(T,hd,200);
    double kv_bytes_int8 = (double)L*2*nkv*hd*T;        /* K+V, all layers */
    double kv_bytes_2bit = kv_bytes_int8/4.0;
    printf("KV cache @ 4096 ctx (per token, all heads/layers):\n");
    printf("  footprint: int8 %.1f MB  vs  2-bit %.1f MB  (4x)\n", kv_bytes_int8/1e6, kv_bytes_2bit/1e6);
    printf("  attention dot latency (one head, 4096 keys): int8 %.1f us  vs  2-bit %.1f us\n",
           kvi/1e3, kv2/1e3);
    printf("  KV streaming/token at %.1f GB/s: int8 %.2f ms  vs  2-bit %.2f ms\n",
           bw, kv_bytes_int8/bw/1e9*1e3, kv_bytes_2bit/bw/1e9*1e3);
    return 0;
}
