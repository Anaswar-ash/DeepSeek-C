#pragma once
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum256(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#endif
typedef struct { int fmt; float *qf; int8_t *q8; uint8_t *q4; float *s; int O, I; } QT;
/* ---- Accumulatore int4->float a 512 bit / 512-bit int4->float accumulator ----
 * Stessa matematica lossless di matmul_i4 (nibble->f32, FMA), ma 32 pesi/iter su
 * due catene FMA indipendenti. NON bit-identico al vecchio ordine: la riduzione
 * ad albero accumula MENO errore della somma sequenziale (misurato 2-4x più
 * vicino all'oracolo double sulle forme reali; perplexity invariata, +4-7% sul
 * decode con routing CPU-heavy — vedi docs/experiments/glm52-6x5090-2026-07-12.md).
 * EN: same lossless math as matmul_i4, 32 weights/iter on two independent FMA
 * chains. Not bit-identical to the old order: tree reduction accumulates LESS
 * rounding than sequential summation. I4_ACC512=0 restores the old order (A/B). */
#if defined(__AVX512F__) && defined(__AVX512BW__)
static int g_i4_acc512=1;
static inline float dot_i4f_avx512(const uint8_t *w,const float *x,int I){
    const __m128i m4=_mm_set1_epi8(0x0F); const __m512i b8=_mm512_set1_epi32(8);
    __m512 acc0=_mm512_setzero_ps(),acc1=_mm512_setzero_ps(); int i=0;
    for(;i+32<=I;i+=32){ __m128i by=_mm_loadu_si128((const __m128i*)(w+(i>>1)));
        __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi),n1=_mm_unpackhi_epi8(lo,hi);
        __m512 w0=_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n0),b8));
        __m512 w1=_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n1),b8));
        acc0=_mm512_fmadd_ps(_mm512_loadu_ps(x+i),w0,acc0);
        acc1=_mm512_fmadd_ps(_mm512_loadu_ps(x+i+16),w1,acc1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0,acc1));
}
/* selftest contro il riferimento scalare (I4_ACC512_TEST=1): copre l'ordine dei
 * nibble e ogni multiplo di 32. / selftest vs the scalar reference. */
static int i4_acc512_selftest(void){
    enum { N=224 }; uint8_t w[(N+1)/2]; float x[N];
    for(int i=0;i<N;i++){
        int q=((i*13+5)&15)-8;
        if(!(i&1)) w[i>>1]=(uint8_t)(q+8);
        else w[i>>1]|=(uint8_t)((q+8)<<4);
        x[i]=(float)(((i*29+7)%101)-50)/37.f;
    }
    for(int n=32;n<=N;n+=32){
        float ref=0; for(int i=0;i<n;i++) ref+=x[i]*(float)(((w[i>>1]>>((i&1)*4))&15)-8);
        float got=dot_i4f_avx512(w,x,n),tol=2e-5f*(1.f+fabsf(ref));
        if(fabsf(got-ref)>tol){ fprintf(stderr,"AVX512 i4 selftest n=%d: %.9g != %.9g\n",n,got,ref); return 0; }
    }
    return 1;
}
#endif

/* y[S,O] = x[S,I] @ W^T, W[O,I] f32 */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; for(int i=0;i<I;i++) a+=xs[i]*w[i]; y[(int64_t)s*O+o]=a; } }
}
/* y[S,O] = x[S,I] @ W^T con W quantizzato int8 per-riga + scala[O] (dequant-on-use) */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            __m256 acc=_mm256_setzero_ps();
            for(;i+8<=I;i+=8){ __m256i wi=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w+i)));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i), _mm256_cvtepi32_ps(wi), acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+8<=I;i+=8){ int16x8_t w16=vmovl_s8(vld1_s8(w+i));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),   vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i<I;i++) a+=xs[i]*(float)w[i]; y[(int64_t)s*O+o]=a*sc; } }
}
/* y[S,O] = x[S,I] @ W^T con W int4 impacchettato (2 valori/byte) + scala[O]. */
static void matmul_i4(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
            if(g_i4_acc512){ a=dot_i4f_avx512(w,xs,I); i=I&~31; }
            else {
#endif
#ifdef __AVX2__
            const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));   /* 8 byte=16 nibble */
                __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                __m128i nib=_mm_unpacklo_epi8(lo,hi);                                       /* nibble in ordine */
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m4=vdup_n_u8(0x0F); const int8x8_t b8=vdup_n_s8(8);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint8x8_t by=vld1_u8(w+(i>>1));               /* 8 byte=16 nibble */
                uint8x8x2_t z=vzip_u8(vand_u8(by,m4), vshr_n_u8(by,4));        /* nibble in ordine */
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[0]),b8));
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[1]),b8));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
            }
#endif
            for(;i+1<I;i+=2){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8, hi=(int)(byte>>4)-8;
                a += xs[i]*(float)lo + xs[i+1]*(float)hi; }
            if(i<I){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8; a += xs[i]*(float)lo; }
            y[(int64_t)s*O+o]=a*sc; } }
}
/* Decode hot path for gate+up: same exact q4 dot products as matmul_i4, but one
 * OpenMP dispatch covers both matrices. KTransformers uses persistent pools;
 * this keeps colibri dependency-free while removing one team launch/expert. */
static void matmul_i4_pair(float *yg, float *yu, const float *x,
                           const uint8_t *qg, const float *sg,
                           const uint8_t *qu, const float *su, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int z=0;z<2*O;z++){
        int o=z<O?z:z-O; const uint8_t *w=(z<O?qg:qu)+(int64_t)o*rb;
        float a=0; int i=0;
#ifdef __AVX2__
        const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
        __m256 acc=_mm256_setzero_ps();
        for(;i+16<=I;i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
            __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
            __m128i nib=_mm_unpacklo_epi8(lo,hi);
            __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
            __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
            acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),w0,acc);
            acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),w1,acc); }
        a=hsum256(acc);
#elif defined(__ARM_NEON)
        const uint8x8_t m4=vdup_n_u8(0x0F); const int8x8_t b8=vdup_n_s8(8);
        float32x4_t ac0=vdupq_n_f32(0),ac1=vdupq_n_f32(0);
        for(;i+16<=I;i+=16){ uint8x8_t by=vld1_u8(w+(i>>1));
            uint8x8x2_t n=vzip_u8(vand_u8(by,m4),vshr_n_u8(by,4));
            int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u8(n.val[0]),b8));
            int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u8(n.val[1]),b8));
            ac0=vfmaq_f32(ac0,vld1q_f32(x+i),vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
            ac1=vfmaq_f32(ac1,vld1q_f32(x+i+4),vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
            ac0=vfmaq_f32(ac0,vld1q_f32(x+i+8),vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
            ac1=vfmaq_f32(ac1,vld1q_f32(x+i+12),vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
        a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
        for(;i+1<I;i+=2){ uint8_t b=w[i>>1]; a+=x[i]*(float)((b&15)-8)+x[i+1]*(float)((b>>4)-8); }
        if(i<I) a+=x[i]*(float)((w[i>>1]&15)-8);
        (z<O?yg:yu)[o]=a*(z<O?sg:su)[o];
    }
}

static void matmul_qt(float *y,const float *x,QT *w,int S);
static void expert_gate_up(float *g,float *u,const float *x,QT *wg,QT *wu,int S){
    if(S==1&&wg->fmt==2&&wu->fmt==2&&wg->I==wu->I&&wg->O==wu->O)
        matmul_i4_pair(g,u,x,wg->q4,wg->s,wu->q4,wu->s,wg->I,wg->O);
    else { matmul_qt(g,x,wg,S); matmul_qt(u,x,wu,S); }
}
/* y[S,O] = x[S,I] @ W^T con W int2 impacchettato (4 valori/byte) + scala[O]. nibble 2-bit -> [-2,1]. */
static void matmul_i2(float *y, const float *x, const uint8_t *q2, const float *scale, int S, int I, int O){
    int rb=(I+3)/4;
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const uint8_t *w=q2+(int64_t)o*rb; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            const __m128i m2=_mm_set1_epi8(0x03); const __m256i b2=_mm256_set1_epi32(2);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){ __m128i by=_mm_cvtsi32_si128(*(const int*)(w+(i>>2)));    /* 4 byte=16 valori */
                __m128i p0=_mm_and_si128(by,m2), p1=_mm_and_si128(_mm_srli_epi16(by,2),m2);
                __m128i p2=_mm_and_si128(_mm_srli_epi16(by,4),m2), p3=_mm_and_si128(_mm_srli_epi16(by,6),m2);
                __m128i lo=_mm_unpacklo_epi8(p0,p1), hi=_mm_unpacklo_epi8(p2,p3);
                __m128i nib=_mm_unpacklo_epi16(lo,hi);                                      /* 16 valori in ordine */
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b2));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b2));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m2v=vdup_n_u8(3); const int8x8_t b2v=vdup_n_s8(2);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint32_t wd; memcpy(&wd, w+(i>>2), 4);        /* 4 byte=16 valori */
                uint8x8_t by=vreinterpret_u8_u32(vdup_n_u32(wd));
                uint8x8x2_t z01=vzip_u8(vand_u8(by,m2v),              vand_u8(vshr_n_u8(by,2),m2v));
                uint8x8x2_t z23=vzip_u8(vand_u8(vshr_n_u8(by,4),m2v), vshr_n_u8(by,6));
                uint16x4x2_t zz=vzip_u16(vreinterpret_u16_u8(z01.val[0]), vreinterpret_u16_u8(z23.val[0]));
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[0]),b2v));  /* 16 valori in ordine */
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[1]),b2v));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i<I;i++){ uint8_t byte=w[i>>2]; int sh=(i&3)*2; a += xs[i]*(float)((int)((byte>>sh)&3)-2); }
            y[(int64_t)s*O+o]=a*sc; } }
}
/* ---- KERNEL INTERI (IDOT): attivazioni quantizzate a int8 per riga (absmax/127,
 * stile Q8_0), prodotto scalare INTERO via maddubs/madd AVX2 — niente conversione
 * f32 dei pesi nel ciclo caldo. ~2-3x sui matmul quantizzati; errore aggiunto ~0.3%
 * RMS per matmul (attivazione int8), IDOT=0 torna al percorso f32 esatto. */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
#define IDOT_KERNEL "avx512-vnni"
#elif defined(__AVXVNNI__) && defined(__AVX2__)
#define IDOT_KERNEL "avx-vnni"
#elif defined(__AVX2__)
#define IDOT_KERNEL "avx2"
#elif defined(__ARM_NEON)
#define IDOT_KERNEL "neon"
#elif defined(__VSX__)
#define IDOT_KERNEL "vsx"
#else
#define IDOT_KERNEL "scalar"
#endif
static int g_idot=1;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static int g_i4s=1;   /* SDOT presente: int4 IDOT conviene anche a S=1 (decode). Misurato
                       * su Apple M-series: +14%%, expert-matmul -16%%. EN: with SDOT, int4
                       * IDOT pays even at S=1 (decode); measured on Apple M-series. */
#elif defined(__VSX__)
static int g_i4s=1;   /* POWER8 vec_msum: qui il fallback f32 e' SCALARE, quindi l'IDOT
                       * int4 conviene anche a S=1. Misurato su POWER8 S824 (vedi PR).
                       * EN: on VSX the f32 fallback is plain scalar C, so int4 IDOT
                       * pays even at S=1. Measured on a POWER8 S824 (see PR). */
#else
static int g_i4s=2;   /* senza SDOT / altrove: soglia originale (misura AVX2 dell'autore).
                       * EN: without SDOT / elsewhere: original threshold (author's AVX2). */
#endif
static inline float qrow_i8(const float *x, int8_t *q, int I){
    float amax=0; for(int i=0;i<I;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float s=amax/127.f; if(s<1e-12f) s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv);
    return s;
}
#ifdef __AVX2__
static inline int hsum256_i32(__m256i v){
    __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
    lo=_mm_add_epi32(lo,hi); lo=_mm_hadd_epi32(lo,lo); lo=_mm_hadd_epi32(lo,lo);
    return _mm_cvtsi128_si32(lo);
}
#endif
#if defined(__AVXVNNI__) && defined(__AVX2__)
/* hsum di un __m128i a 4 lane s32 (l'AVX-VNNI 128-bit accumula su 4 lane). */
static inline int hsum128_i32(__m128i v){
    v=_mm_hadd_epi32(v,v); v=_mm_hadd_epi32(v,v); return _mm_cvtsi128_si32(v);
}
#endif
/* dot int8·int8: trucco del segno (|w| unsigned × x·sign(w) signed). Sicuro:
 * coppie <= 128*127*2 = 32512 < 32767, accumulo s32 fino a I=16384. */
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /* VNNI: vpdpbusd u8*s8 -> s32 directly, 64 bytes/iter, no 16-bit intermediate.
     * AVX-512 has no vpsignb: |w| via abs, sign folded into x with a mask-negate
     * (w==0 -> product 0 either way). |x|<=127 (qrow_i8), |w|<=128 as u8: each
     * s32 lane adds <= 4*128*127, safe up to I=16384 like the AVX2 bound. */
    __m512i acc=_mm512_setzero_si512();
    for(;i+64<=I;i+=64){
        __m512i wv=_mm512_loadu_si512((const void*)(w+i));
        __m512i xv=_mm512_loadu_si512((const void*)(x+i));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVXVNNI__) && defined(__AVX2__)
    /* AVX-VNNI 128-bit: vpdpbusd u8*s8 -> s32, 16 byte/iter. Stesso trucco del
     * segno della variante 512-bit: |w| via abs, segno piegato in x con maschera
     * (w==0 -> product 0). __AVX2__ serve per _mm_sign_epi8 / abs. */
    __m128i acc=_mm_setzero_si128();
    for(;i+16<=I;i+=16){
        __m128i wv=_mm_loadu_si128((const __m128i*)(w+i));
        __m128i xv=_mm_loadu_si128((const __m128i*)(x+i));
        __m128i xs=_mm_sign_epi8(xv,wv);              /* x * sign(w); _mm_sign zona __AVX2__ */
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(wv),xs);
    }
    sum=hsum128_i32(acc);
#elif defined(__AVX2__)
    __m256i acc=_mm256_setzero_si256(); const __m256i ones=_mm256_set1_epi16(1);
    for(;i+32<=I;i+=32){
        __m256i wv=_mm256_loadu_si256((const __m256i*)(w+i));
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#elif defined(__ARM_NEON)
    /* ARM: SDOT nativo se disponibile (Apple Silicon: sempre); altrimenti vmull/vpadal.
     * Stesso bound anti-overflow del trucco AVX2: coppie <= 128*127*2 = 32512 < 32767. */
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+16<=I;i+=16){
        int8x16_t wv=vld1q_s8(w+i), xv=vld1q_s8(x+i);
#if defined(__ARM_FEATURE_DOTPROD)
        acc=vdotq_s32(acc,wv,xv);
#else
        int16x8_t p=vmull_s8(vget_low_s8(wv),vget_low_s8(xv));
        p=vmlal_s8(p,vget_high_s8(wv),vget_high_s8(xv));
        acc=vpadalq_s16(acc,p);
#endif
    }
    sum=vaddvq_s32(acc);
#elif defined(__VSX__)
    /* POWER8: vec_msum (s8 x u8 -> s32) somma i prodotti byte DIRETTAMENTE in lane
     * s32, 16 byte/iter: il bound anti-saturazione a 16 bit di maddubs qui non serve.
     * Stesso trucco del segno (|w| u8 per x*sign(w) s8), ma |w| via select+sub MODULO
     * e non vec_abs: -128 deve diventare 128 u8, non saturare a 127.
     * EN: vec_msum accumulates byte products straight into s32 lanes; |w| is built
     * with a modulo subtract select instead of vec_abs so w=-128 wraps to 128 (u8)
     * rather than saturating to 127. |x|<=127 from qrow_i8, so x negation is safe. */
    __vector signed int acc=vec_splats(0);
    const __vector signed char vz=vec_splats((signed char)0);
    for(;i+16<=I;i+=16){
        __vector signed char wv=vec_xl(0,(const signed char*)(w+i));
        __vector signed char xv=vec_xl(0,(const signed char*)(x+i));
        __vector __bool char neg=vec_cmplt(wv,vz);
        __vector signed char xs=vec_sel(xv,vec_sub(vz,xv),neg);
        __vector unsigned char wa=(__vector unsigned char)vec_sel(wv,vec_sub(vz,wv),neg);
        acc=vec_msum(xs,wa,acc);
    }
    sum=vec_extract(acc,0)+vec_extract(acc,1)+vec_extract(acc,2)+vec_extract(acc,3);
#endif
    for(;i<I;i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}
/* dot int4(packed)·int8: nibble -> int8 [-8,7] al volo, poi stesso trucco */
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /* 32 bytes = 64 nibbles -> int8 in [-8,7], one vpdpbusd per 64 values.
     * 256-bit unpack leaves values in per-128-lane order [0-15][32-47]/[16-31][48-63];
     * dot pairing is order-invariant, so permute x's 128-bit blocks to match
     * instead of re-ordering w (one vpermq per iter, off the critical unpack path). */
    const __m256i m4v=_mm256_set1_epi8(0x0F);
    const __m512i b8v=_mm512_set1_epi8(8);
    const __m512i xidx=_mm512_setr_epi64(0,1,4,5,2,3,6,7);
    __m512i acc=_mm512_setzero_si512();
    for(;i+64<=I;i+=64){
        __m256i by=_mm256_loadu_si256((const __m256i*)(w4+(i>>1)));
        __m256i lo=_mm256_and_si256(by,m4v), hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        __m256i z0=_mm256_unpacklo_epi8(lo,hi), z1=_mm256_unpackhi_epi8(lo,hi);
        __m512i wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(z0),z1,1),b8v);
        __m512i xv=_mm512_permutexvar_epi64(xidx,_mm512_loadu_si512((const void*)(x+i)));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVXVNNI__) && defined(__AVX2__)
    /* AVX-VNNI 128-bit, int4: 16 byte = 32 nibble -> int8 [-8,7] in due half
     * (n0/n1), ciascuno alimentato a un vpdpbusd da 16 byte. Stesso unpack
     * 128-bit del ramo AVX2 sotto; 32 elementi/iter come li. */
    const __m128i m4=_mm_set1_epi8(0x0F); const __m128i b8=_mm_set1_epi8(8);
    __m128i acc=_mm_setzero_si128();
    for(;i+32<=I;i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));   /* 16 byte = 32 nibble */
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);  /* nibble in ordine */
        __m128i w0=_mm_sub_epi8(n0,b8), w1=_mm_sub_epi8(n1,b8);
        __m128i x0=_mm_loadu_si128((const __m128i*)(x+i));
        __m128i x1=_mm_loadu_si128((const __m128i*)(x+i+16));
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(w0),_mm_sign_epi8(x0,w0));
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(w1),_mm_sign_epi8(x1,w1));
    }
    sum=hsum128_i32(acc);
#elif defined(__AVX2__)
    const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi8(8);
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i acc=_mm256_setzero_si256();
    for(;i+32<=I;i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));   /* 16 byte = 32 nibble */
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);   /* in ordine */
        __m256i wv=_mm256_sub_epi8(_mm256_set_m128i(n1,n0),b8);
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#elif defined(__ARM_NEON)
    const uint8x16_t m4q=vdupq_n_u8(0x0F); const int8x16_t b8q=vdupq_n_s8(8);
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+32<=I;i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));                          /* 16 byte = 32 nibble */
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4)); /* nibble in ordine */
        int8x16_t w0=vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q);
        int8x16_t w1=vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q);
        int8x16_t x0=vld1q_s8(x+i), x1=vld1q_s8(x+i+16);
#if defined(__ARM_FEATURE_DOTPROD)
        acc=vdotq_s32(acc,w0,x0); acc=vdotq_s32(acc,w1,x1);
#else
        int16x8_t p=vmull_s8(vget_low_s8(w0),vget_low_s8(x0));      /* |w|<=8: nessun overflow */
        p=vmlal_s8(p,vget_high_s8(w0),vget_high_s8(x0));
        acc=vpadalq_s16(acc,p);
        p=vmull_s8(vget_low_s8(w1),vget_low_s8(x1));
        p=vmlal_s8(p,vget_high_s8(w1),vget_high_s8(x1));
        acc=vpadalq_s16(acc,p);
#endif
    }
    sum=vaddvq_s32(acc);
#elif defined(__VSX__)
    /* 16 byte = 32 nibble. vec_mergeh/vec_mergel su ppc64le (GCC) interallacciano come
     * unpacklo/unpackhi x86 (verificato empiricamente su POWER8): i nibble escono in
     * ordine di memoria. |w|<=8 dopo il -8, quindi stesso trucco del segno di dot_i8i8.
     * EN: vec_mergeh/l on ppc64le interleave like x86 unpacklo/hi (verified on POWER8),
     * so nibbles come out in memory order; then the same sign trick as dot_i8i8. */
    const __vector unsigned char m4v=vec_splats((unsigned char)0x0F);
    const __vector unsigned char sh4=vec_splats((unsigned char)4);
    const __vector signed char b8v=vec_splats((signed char)8);
    const __vector signed char vz=vec_splats((signed char)0);
    __vector signed int acc=vec_splats(0);
    for(;i+32<=I;i+=32){
        __vector unsigned char by=vec_xl(0,w4+(i>>1));               /* 16 byte = 32 nibble */
        __vector unsigned char lo=vec_and(by,m4v), hi=vec_sr(by,sh4);
        __vector signed char w0=vec_sub((__vector signed char)vec_mergeh(lo,hi),b8v);
        __vector signed char w1=vec_sub((__vector signed char)vec_mergel(lo,hi),b8v);
        __vector signed char x0=vec_xl(0,(const signed char*)(x+i));
        __vector signed char x1=vec_xl(0,(const signed char*)(x+i+16));
        __vector __bool char n0=vec_cmplt(w0,vz), n1=vec_cmplt(w1,vz);
        acc=vec_msum(vec_sel(x0,vec_sub(vz,x0),n0),
                     (__vector unsigned char)vec_sel(w0,vec_sub(vz,w0),n0),acc);
        acc=vec_msum(vec_sel(x1,vec_sub(vz,x1),n1),
                     (__vector unsigned char)vec_sel(w1,vec_sub(vz,w1),n1),acc);
    }
    sum=vec_extract(acc,0)+vec_extract(acc,1)+vec_extract(acc,2)+vec_extract(acc,3);
#endif
    for(;i+1<I;i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}
static void matmul_q_idot(float *y, const int8_t *xq, const float *sx, const int8_t *q,
                          const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i8i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}
static void matmul_i4_idot(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                           const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i4i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}

typedef struct { int8_t *xq; size_t xq_cap; float *sx; size_t sx_cap; } QScratch;
static _Thread_local QScratch g_qscratch;
static void quant_scratch(size_t xn, size_t sn, int8_t **xq, float **sx){
    if(xn>g_qscratch.xq_cap){
        int8_t *p=realloc(g_qscratch.xq,xn);
        if(!p){ fprintf(stderr,"OOM quant scratch\n"); exit(1); }
        g_qscratch.xq=p; g_qscratch.xq_cap=xn;
    }
    if(sn>g_qscratch.sx_cap){
        float *p=realloc(g_qscratch.sx,sn*sizeof(float));
        if(!p){ fprintf(stderr,"OOM quant scales\n"); exit(1); }
        g_qscratch.sx=p; g_qscratch.sx_cap=sn;
    }
    *xq=g_qscratch.xq; *sx=g_qscratch.sx;
}

static void matmul_qt(float *y, const float *x, QT *w, int S){


    if(w->fmt==0){ matmul(y,x,w->qf,S,w->I,w->O); return; }
    /* int8 IDOT vince sempre (1.4-2.5x). int4 IDOT: l'autore su AVX2 trovo' che a S=1
     * non ripaga (soglia S>=2); ma su ARM/SDOT il singolo token CONVIENE (vedi g_i4s /
     * PR #9 per il gemello VNNI). Soglia configurabile con I4S.
     * EN: int8 IDOT always wins (1.4-2.5x). int4 IDOT: on AVX2 the author found S=1 didn't
     * pay (S>=2 gate); on ARM/SDOT single-token DOES pay (see g_i4s / PR #9 for the VNNI
     * twin). Threshold configurable via I4S. */
    if(g_idot && (w->fmt==1 || (w->fmt==2 && S>=g_i4s))){
        int I=w->I; int8_t *xq; float *sx;
        if(S<0 || I<0 || (size_t)S>SIZE_MAX/(size_t)(I?I:1)){ fprintf(stderr,"matmul_qt: shape overflow\n"); exit(1); }
        quant_scratch((size_t)S*I,(size_t)S,&xq,&sx);
        for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I, xq+(int64_t)s*I, I);
        if(w->fmt==1) matmul_q_idot(y,xq,sx,w->q8,w->s,S,I,w->O);
        else matmul_i4_idot(y,xq,sx,w->q4,w->s,S,I,w->O);
        return;
    }
    if(w->fmt==1) matmul_q(y,x,w->q8,w->s,S,w->I,w->O);
    else if(w->fmt==3) matmul_i2(y,x,w->q4,w->s,S,w->I,w->O);
    else matmul_i4(y,x,w->q4,w->s,S,w->I,w->O);
}
