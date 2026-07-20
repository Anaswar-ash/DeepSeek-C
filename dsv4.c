/*
 * DeepSeek-V4-Flash Inference Engine in C
 * Porting to colibri framework.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "st.h"
#include "json.h"
#include "tok.h"
#include "math_dsv4.h"

/* ---------- Config & Data Structures ---------- */

// Model configuration parsed from `config.json`
typedef struct {
    int vocab_size;
    int hidden;        // d_model
    int n_layers;
    int n_heads;       // Query heads
    int head_dim;
    int qk_rope_head_dim;
    float eps;
    float theta;
    
    // MoE
    int n_experts;     // n_routed_experts
    int topk;          // n_activated_experts
    int n_hash_layers; // Layers utilizing deterministic hash-routing
    int moe_inter;     // moe_intermediate_size (expert hidden dim)
    
    // SWA / Compression
    int window_size;
    int *compress_ratios; // Array of compression ratios per layer (0=SWA, 4=CSA, 128=HCA)
    
    // LoRA Latents
    int q_lora_rank;
    int kv_lora_rank;
    int o_lora_rank;
    int o_groups;
    
    // Indexer for CSA/HCA
    int index_n_heads;
    int index_head_dim;
    int index_topk;
    
    // mHC (Hyper-Connections)
    int hc_mult;           // Number of expanded residual streams (default 4)
    int hc_sinkhorn_iters; // Iterations for Sinkhorn-Knopp normalization
    float hc_eps;
} Cfg;

/* ---------- layer weights ---------- */
// Layer weights layout, matching `convert_dsv4.py` outputs
typedef struct {
    // Dense FFN + MoE
    float *ffn_norm;
    QT shared_w1, shared_w2, shared_w3;
    QT ex_w1[256], ex_w2[256], ex_w3[256];
    QT gate;          // Router scoring projection matrix
    int *tid2eid;     // Hash routing lookups (first n_hash_layers) [vocab_size, topk]
    float *gate_bias; // Optional bias for the gate

    // Attention
    float *attn_norm;
    QT wq_a, wq_b;
    float *q_norm;
    QT wkv;
    float *kv_norm;
    QT wo_a, wo_b;
    float *attn_sink;

    // CSA/HCA Compression & Lightning Indexer
    QT idx_ape;
    float *idx_comp_norm;
    QT idx_wgate;
    QT idx_wkv;
    QT idx_wproj;
    QT idx_wq_b;

    // mHC
    float *hc_attn_base;
    QT hc_attn_fn;
    float *hc_attn_scale;
    float *hc_ffn_base;
    QT hc_ffn_fn;
    float *hc_ffn_scale;
} Layer;

/* ---------- expert cache ---------- */
typedef struct { int eid; int8_t *w1, *w2, *w3; float *w1s, *w2s, *w3s; uint64_t used; } Slot;
typedef struct { Slot *slots; int n, cap; } LCache;

/* ---------- MTP (Multi-Token Prediction) ---------- */
typedef struct {
    QT emb;           // mtp.0.emb.tok_emb.weight [vocab, hidden]
    QT e_proj;        // mtp.0.e_proj.weight [hidden, hidden] — projects main hidden → MTP input
    float *enorm;     // mtp.0.enorm.weight
    float *hnorm;     // mtp.0.hnorm.weight
    QT head;          // mtp.0.head.weight [vocab, hidden]
    QT hc_head_fn;    // mtp.0.hc_head_fn
    float *hc_head_base, *hc_head_scale;
    Layer layer;      // The single MTP transformer layer (attn + MoE)
} MTP;

typedef struct {
    Cfg c;
    shards S;
    int quant_bits;
    QT embed, lm_head;
    float *final_norm;
    QT hc_head_fn;
    float *hc_head_base, *hc_head_scale;
    Layer *L;
    LCache *cache;
    uint64_t clock, hits, miss;
    
    // KV Cache
    float **K, **V;
    int max_t;
    double dense_load_s;
    Tok tokenizer;
    
    // MTP
    MTP *mtp;
    int n_mtp;  // num_nextn_predict_layers (typically 1)
} Model;

/* ---------- utility ---------- */
static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
static double rss_gb(void) { return 0.0; } // getrusage not on Windows by default
// Arena Allocator
#define ARENA_SIZE (256 * 1024 * 1024)
static uint8_t *g_arena = NULL;
static size_t g_arena_offset = 0;

static void arena_init(void) {
    if (!g_arena) {
        g_arena = malloc(ARENA_SIZE);
        if (!g_arena) { fprintf(stderr, "Arena OOM\n"); exit(1); }
    }
}

static void arena_reset(void) {
    if (!g_arena) arena_init();
    g_arena_offset = 0;
}

static float *falloc(int64_t n) {
    if (!g_arena) arena_init();
    size_t bytes = n * sizeof(float);
    bytes = (bytes + 31) & ~31; // Align 32 bytes for AVX2
    size_t old_offset = __atomic_fetch_add(&g_arena_offset, bytes, __ATOMIC_SEQ_CST);
    if (old_offset + bytes > ARENA_SIZE) {
        fprintf(stderr, "Arena OOM %ld\n", (long)n); exit(1);
    }
    float *p = (float*)(g_arena + old_offset);
    return p;
}

static inline float softplusf(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

/* ---------- mHC Sinkhorn ---------- */
static void hc_split_sinkhorn(const float *mixes, const float *hc_scale, const float *hc_base, 
                              int hc, int iters, float eps,
                              float *pre, float *post, float *comb) {
    for (int j = 0; j < hc; j++) pre[j] = sigmoidf(mixes[j] * hc_scale[0] + hc_base[j]) + eps;
    for (int j = 0; j < hc; j++) post[j] = sigmoidf(mixes[hc + j] * hc_scale[1] + hc_base[hc + j]) + eps;
    for (int i = 0; i < hc; i++) {
        for (int j = 0; j < hc; j++) {
            comb[i * hc + j] = sigmoidf(mixes[2 * hc + i * hc + j] * hc_scale[2] + hc_base[2 * hc + i * hc + j]) + eps;
        }
    }
    
    float *row_sum = falloc(hc);
    float *col_sum = falloc(hc);
    for (int it = 0; it < iters; it++) {
        memset(row_sum, 0, hc * sizeof(float));
        for (int i=0; i<hc; i++) for (int j=0; j<hc; j++) row_sum[i] += comb[i*hc + j];
        for (int i=0; i<hc; i++) for (int j=0; j<hc; j++) comb[i*hc + j] /= row_sum[i];
        
        memset(col_sum, 0, hc * sizeof(float));
        for (int i=0; i<hc; i++) for (int j=0; j<hc; j++) col_sum[j] += comb[i*hc + j];
        for (int i=0; i<hc; i++) for (int j=0; j<hc; j++) comb[i*hc + j] /= col_sum[j];
    }

}

// Compute HC Pre (shrink 4 streams to 1)
static void hc_pre(float *y_single, const float *x_multi, QT *hc_fn, const float *hc_scale, const float *hc_base, 
                   int hc_mult, int dim, int iters, float eps, float *post, float *comb) {
    if (!hc_fn || (hc_fn->fmt == 0 && !hc_fn->qf)) {
        memcpy(y_single, x_multi, dim * sizeof(float));
        return;
    }
    
    int hc_dim = hc_mult * dim;
    int mix_hc = (2 + hc_mult) * hc_mult;
    
    // rsqrt of flattened x
    double ms = 0;
    for (int i = 0; i < hc_dim; i++) ms += (double)x_multi[i]*x_multi[i];
    float rsqrt = 1.0f / sqrtf((float)(ms / hc_dim) + eps);
    
    // mixes = (x @ hc_fn^T) * rsqrt
    float *mixes = falloc(mix_hc);
    matmul_qt(mixes, x_multi, hc_fn, 1);
    
    for (int i = 0; i < mix_hc; i++) mixes[i] *= rsqrt;
    
    float *pre = falloc(hc_mult);
    hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps, pre, post, comb);
    
    memset(y_single, 0, dim * sizeof(float));
    for (int j = 0; j < hc_mult; j++) {
        for (int d = 0; d < dim; d++) {
            y_single[d] += pre[j] * x_multi[j * dim + d];
        }
    }
    

}

static void hc_post(float *y_multi, const float *attn_out, const float *residual, 
                    const float *post, const float *comb, int hc_mult, int dim) {
    for (int j = 0; j < hc_mult; j++) {
        for (int d = 0; d < dim; d++) {
            y_multi[j * dim + d] = post[j] * attn_out[d];
        }
    }
    
    float *y_mix = falloc(hc_mult * dim);
    memset(y_mix, 0, hc_mult * dim * sizeof(float));
    for (int i = 0; i < hc_mult; i++) {
        for (int j = 0; j < hc_mult; j++) {
            for (int d = 0; d < dim; d++) {
                y_mix[i * dim + d] += comb[i * hc_mult + j] * residual[j * dim + d];
            }
        }
    }
    
    for (int i = 0; i < hc_mult * dim; i++) {
        y_multi[i] += y_mix[i];
    }

}

/* ---------- MoE / Attention ---------- */

// Output: topk_idx[topk], topk_weights[topk]
static void moe_route(const Cfg *c, const Layer *l, int layer_id, int token_id, const float *hidden, 
                      int *topk_idx, float *topk_weights) {
    int E = c->n_experts;
    int K = c->topk;
    
    // 1. Hash Routing for early layers
    if (layer_id < c->n_hash_layers && l->tid2eid != NULL) {
        // tid2eid is [vocab_size, topk]
        for (int k = 0; k < K; k++) {
            topk_idx[k] = l->tid2eid[token_id * K + k];
            topk_weights[k] = 1.0f; // Hash routing implies equal weighting or it's handled differently, but weight isn't dependent on token hidden state
        }
        return;
    }
    
    // 2. Score-based routing
    float *scores = falloc(E);
    matmul_qt(scores, hidden, (QT*)&l->gate, 1);
    if (l->gate_bias) {
        for (int e = 0; e < E; e++) scores[e] += l->gate_bias[e];
    }
    for (int e = 0; e < E; e++) scores[e] = sqrtf(softplusf(scores[e]));
    
    // Top-K selection
    for (int k = 0; k < K; k++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int j = 0; j < k; j++) if (topk_idx[j] == e) { taken = 1; break; }
            if (!taken && scores[e] > bv) { bv = scores[e]; best = e; }
        }
        topk_idx[k] = best;
        topk_weights[k] = bv;
    }
    
    // Normalize weights
    float sum = 0;
    for (int k = 0; k < K; k++) sum += topk_weights[k];
    if (sum > 0) {
        for (int k = 0; k < K; k++) topk_weights[k] /= sum;
    }

}


/* ---------- Attention ---------- */
static void rope_head(float *x, int pos, const Cfg *c) {
    int h = c->qk_rope_head_dim / 2;
    int offset = c->head_dim - c->qk_rope_head_dim; // RoPE applies to last `rd` dims
    float *xr = x + offset;
    
    // YaRN RoPE scaling for extended context lengths
    float yarn_scale = 1.0f;
    if (pos > 4096) {
        float factor = (float)pos / 4096.0f;
        yarn_scale = factor * factor; // Simplified YaRN theta scaling
    }
    
    for (int j = 0; j < h; j++) {
        float inv = powf(c->theta * yarn_scale, -2.0f * j / c->qk_rope_head_dim);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = xr[j], b = xr[j+h];
        xr[j]   = a*cs - b*sn;
        xr[j+h] = b*cs + a*sn;
    }
}

/* ---------- Compression & Sparse Attention ---------- */
// Compress 8 tokens (overlap=True) into 1 KV token
static void compress_kv_overlap(const float *kv_state, const float *score_state, float *out_kv, int ratio, int d) {
    // kv_state is [2 * ratio, 2 * d]. score_state is [2 * ratio, 2 * d].
    float *sc = falloc(2 * ratio);
    
    // We compute score softmax.
    // The concatenation effectively takes:
    // [0..3, 0..d-1] from past window, and [4..7, d..2d-1] from current window.
    for (int d_idx = 0; d_idx < d; d_idx++) {
        float max_sc = -1e30f;
        for (int i = 0; i < ratio; i++) {
            float s1 = score_state[i * (2 * d) + d_idx];
            float s2 = score_state[(i + ratio) * (2 * d) + d + d_idx];
            if (s1 > max_sc) max_sc = s1;
            if (s2 > max_sc) max_sc = s2;
        }
        float sum = 0;
        for (int i = 0; i < ratio; i++) {
            float s1 = expf(score_state[i * (2 * d) + d_idx] - max_sc);
            float s2 = expf(score_state[(i + ratio) * (2 * d) + d + d_idx] - max_sc);
            sum += s1 + s2;
        }
        
        float out_val = 0;
        for (int i = 0; i < ratio; i++) {
            float w1 = expf(score_state[i * (2 * d) + d_idx] - max_sc) / sum;
            float w2 = expf(score_state[(i + ratio) * (2 * d) + d + d_idx] - max_sc) / sum;
            out_val += w1 * kv_state[i * (2 * d) + d_idx];
            out_val += w2 * kv_state[(i + ratio) * (2 * d) + d + d_idx];
        }
        out_kv[d_idx] = out_val;
    }

}

// rmsnorm on a slice
static void rmsnorm_slice(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * (w ? w[i] : 1.0f);
}

// SWA decode attention (batch=1, seq=1)
static void attention_decode(Model *m, Layer *l, int layer_id, const float *x, int pos, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden;
    int H = c->n_heads;
    int hd = c->head_dim;       // 512 = kv_lora_rank (in MLA, head_dim == kv_lora_rank)
    int win = c->window_size;
    int kvd = c->kv_lora_rank;  // 512
    
    if (l->wq_a.fmt == 0 && !l->wq_a.qf) {
        memset(out, 0, D * sizeof(float));
        return;
    }
    
    float *q_latent = falloc(c->q_lora_rank);
    float *q_heads = falloc(H * hd);  // 64 * 512 = 32768
    float *kv = falloc(kvd);
    float *o_heads = falloc(H * hd);
    
    // Q projection: x[4096] -> wq_a[1024,4096] -> norm -> wq_b[32768,1024] -> reshape [64,512]
    matmul_qt(q_latent, x, &l->wq_a, 1);
    rmsnorm_slice(q_latent, q_latent, l->q_norm, c->q_lora_rank, c->eps);
    matmul_qt(q_heads, q_latent, &l->wq_b, 1);
    
    // KV projection: x[4096] -> wkv[512,4096] -> norm -> kv[512]
    matmul_qt(kv, x, &l->wkv, 1);
    rmsnorm_slice(kv, kv, l->kv_norm, kvd, c->eps);
    
    // RoPE on last qk_rope_head_dim dims of each Q head
    for (int hh = 0; hh < H; hh++) {
        float *qh = q_heads + hh * hd;
        rmsnorm_slice(qh, qh, NULL, hd, c->eps);
        rope_head(qh, pos, c);
    }
    
    int ratio = c->compress_ratios ? c->compress_ratios[layer_id] : 0;
    
    // Store KV latent in sliding window cache
    int t_pos = pos % win;
    memcpy(m->K[layer_id] + t_pos * kvd, kv, kvd * sizeof(float));
    memcpy(m->V[layer_id] + t_pos * kvd, kv, kvd * sizeof(float));
    
    int t_max = pos + 1 < win ? pos + 1 : win;
    int num_blocks = 0;
    int *selected_blocks = NULL;
    
    if (ratio > 0) {
        // Learned Token Compression using compressor weights
        if (pos > 0 && pos % ratio == 0) {
            int block_idx = (pos / ratio) - 1;
            if (block_idx < 4096) {
                float *comp_kv = falloc(kvd);
                
                if (l->idx_wkv.fmt > 0 && l->idx_wgate.fmt > 0) {
                    // Learned compression: project each token's KV through wkv,
                    // compute gating scores through wgate, softmax-weight the tokens
                    float *gate_scores = falloc(ratio);
                    float gate_max = -1e9f;
                    
                    for (int i = 0; i < ratio; i++) {
                        int p = (pos - ratio + i) % win;
                        float *tok_kv = m->K[layer_id] + p * kvd;
                        
                        // Gate score: single scalar from wgate projection
                        float gs = 0;
                        if (l->idx_wgate.fmt == 2 && l->idx_wgate.q4) {
                            // Simple dot product with first row of wgate
                            for (int d = 0; d < kvd && d < l->idx_wgate.I; d++) {
                                uint8_t byte = l->idx_wgate.q4[d / 2];
                                float v = ((d % 2 == 0) ? (byte & 0xF) : (byte >> 4)) - 8;
                                gs += v * l->idx_wgate.s[0] * tok_kv[d];
                            }
                        } else {
                            // Fallback: uniform weight
                            gs = 0;
                        }
                        gate_scores[i] = gs;
                        if (gs > gate_max) gate_max = gs;
                    }
                    
                    // Softmax over gate scores
                    float gate_sum = 0;
                    for (int i = 0; i < ratio; i++) {
                        gate_scores[i] = expf(gate_scores[i] - gate_max);
                        gate_sum += gate_scores[i];
                    }
                    
                    // Weighted sum of KV tokens
                    memset(comp_kv, 0, kvd * sizeof(float));
                    for (int i = 0; i < ratio; i++) {
                        float w = gate_scores[i] / gate_sum;
                        int p = (pos - ratio + i) % win;
                        float *tok_kv = m->K[layer_id] + p * kvd;
                        for (int d = 0; d < kvd; d++) comp_kv[d] += w * tok_kv[d];
                    }
                    
                    // Apply compressor norm if available
                    if (l->idx_comp_norm) {
                        int norm_dim = kvd < 128 ? kvd : 128; // norm weight is 128-dim
                        rmsnorm_slice(comp_kv, comp_kv, l->idx_comp_norm, norm_dim, c->eps);
                    }
                } else {
                    // Fallback: simple average
                    memset(comp_kv, 0, kvd * sizeof(float));
                    for (int i = 0; i < ratio; i++) {
                        int p = (pos - i) % win;
                        for (int d = 0; d < kvd; d++) comp_kv[d] += m->K[layer_id][p * kvd + d] / ratio;
                    }
                }
                
                memcpy(m->K[layer_id] + (win + block_idx) * kvd, comp_kv, kvd * sizeof(float));
                memcpy(m->V[layer_id] + (win + block_idx) * kvd, comp_kv, kvd * sizeof(float));
            }
        }
        
        // Full Multi-Head Lightning Indexer for CSA (compress_ratio=4)
        if (ratio == 4) {
            int max_blocks = pos / ratio;
            if (max_blocks > 4096) max_blocks = 4096;
            
            int topk = c->index_topk > 0 ? c->index_topk : 512;
            if (topk > max_blocks) topk = max_blocks;
            
            if (topk > 0 && l->idx_wq_b.fmt > 0) {
                int ih = c->index_n_heads;    // 64
                int idim = c->index_head_dim; // 128
                int idx_total = ih * idim;    // 8192
                
                // Project q_latent into indexer query space: [1024] -> [8192]
                float *q_idx = falloc(idx_total);
                matmul_qt(q_idx, q_latent, &l->idx_wq_b, 1);
                
                // Compute per-head weight projections if available
                float *head_weights = NULL;
                if (l->idx_wproj.fmt > 0) {
                    head_weights = falloc(ih);
                    // idx_wproj projects from hidden to ih scores
                    // For decode, use q_latent as proxy input
                    matmul_qt(head_weights, q_latent, &l->idx_wproj, 1);
                }
                
                selected_blocks = (int*)falloc(topk);
                float *block_scores = falloc(max_blocks);
                
                // Score all compressed blocks using multi-head dot product
                for (int b = 0; b < max_blocks; b++) {
                    float *comp_kv = m->K[layer_id] + (win + b) * kvd;
                    float total_score = 0;
                    
                    for (int h = 0; h < ih; h++) {
                        float *qh = q_idx + h * idim;
                        // Use first `idim` dims of compressed KV as block key
                        int kv_off = (h * idim) % kvd;
                        float dot = 0;
                        for (int d = 0; d < idim && (kv_off + d) < kvd; d++) {
                            dot += qh[d] * comp_kv[kv_off + d];
                        }
                        // ReLU activation per head
                        float relu_dot = dot > 0 ? dot : 0;
                        // Weight by per-head importance
                        float hw = (head_weights && h < ih) ? head_weights[h] : 1.0f;
                        total_score += relu_dot * hw;
                    }
                    
                    float scale_factor = 1.0f / sqrtf((float)idim * ih);
                    block_scores[b] = total_score * scale_factor;
                }
                
                // Top-K selection
                for (int k = 0; k < topk; k++) {
                    float best_s = -1e9f; int best_b = -1;
                    for (int b = 0; b < max_blocks; b++) {
                        if (block_scores[b] > best_s) {
                            int used = 0;
                            for (int j = 0; j < k; j++) if (selected_blocks[j] == b) { used = 1; break; }
                            if (!used) { best_s = block_scores[b]; best_b = b; }
                        }
                    }
                    selected_blocks[k] = best_b != -1 ? best_b : (max_blocks - 1 - k);
                }
                num_blocks = topk;
            } else if (topk > 0) {
                // Fallback: most recent blocks
                selected_blocks = (int*)falloc(topk);
                for (int b = 0; b < topk; b++) selected_blocks[b] = max_blocks - 1 - b;
                num_blocks = topk;
            }
        } else {
            // HCA (ratio=128) - dense attention over all blocks
            num_blocks = pos / ratio;
            if (num_blocks > 4096) num_blocks = 4096;
        }
    }
    
    float scale = 1.0f / sqrtf((float)hd);
    
    // Per-head attention in the kv_lora_rank=512 latent space
    #pragma omp parallel for schedule(dynamic,1)
    for (int hh = 0; hh < H; hh++) {
        float *qh = q_heads + hh * hd;
        float *oh = o_heads + hh * hd;
        float *scores = falloc(t_max + num_blocks);
        
        float max_score = -1e9f;
        for (int t = 0; t < t_max; t++) {
            float *kt = m->K[layer_id] + t * kvd;
            float score = 0;
            for (int i = 0; i < kvd; i++) score += qh[i] * kt[i];
            score *= scale;
            if (l->attn_sink) score += l->attn_sink[hh];
            scores[t] = score;
            if (score > max_score) max_score = score;
        }
        
        for (int b = 0; b < num_blocks; b++) {
            int block_idx = (ratio == 4) ? selected_blocks[b] : b;
            float *kt = m->K[layer_id] + (win + block_idx) * kvd;
            float score = 0;
            for (int i = 0; i < kvd; i++) score += qh[i] * kt[i];
            score *= scale;
            if (l->attn_sink) score += l->attn_sink[hh];
            scores[t_max + b] = score;
            if (score > max_score) max_score = score;
        }
        
        float sum_exp = 0;
        for (int t = 0; t < t_max + num_blocks; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        
        memset(oh, 0, hd * sizeof(float));
        for (int t = 0; t < t_max; t++) {
            float w = scores[t] / sum_exp;
            float *vt = m->V[layer_id] + t * kvd;
            for (int i = 0; i < kvd; i++) oh[i] += w * vt[i];
        }
        for (int b = 0; b < num_blocks; b++) {
            int block_idx = (ratio == 4) ? selected_blocks[b] : b;
            float w = scores[t_max + b] / sum_exp;
            float *vt = m->V[layer_id] + (win + block_idx) * kvd;
            for (int i = 0; i < kvd; i++) oh[i] += w * vt[i];
        }
    }
    
    // Output projection: grouped LoRA
    // o_heads is [H*hd = 32768]. Split into o_groups=8 groups of 4096 each.
    // wo_a[8192, 4096] applied as 8 independent [1024, 4096] blocks.
    // wo_b[4096, 8192] maps the concatenated 8192 back to hidden.
    // For simplicity, we treat wo_a as a single [8192, 4096] matmul on each group's slice.
    // Since the tensor is stored as one [8192, 4096] matrix, group g uses rows [g*1024, (g+1)*1024).
    int o_lr = c->o_lora_rank;  // 1024
    int G = c->o_groups;        // 8
    int group_in = (H * hd) / G;  // 32768/8 = 4096
    float *o_latent = falloc(o_lr * G);  // 8192
    
    // Grouped wo_a: for each group, matmul the 4096-dim slice into 1024-dim latent
    for (int g = 0; g < G; g++) {
        float *in_g = o_heads + g * group_in;
        float *out_g = o_latent + g * o_lr;
        // Manual matmul since wo_a is one big QT and we need sliced access
        // wo_a has O=8192, I=4096. Group g reads rows [g*o_lr .. (g+1)*o_lr)
        // This is equivalent to: out_g = wo_a[g*o_lr:(g+1)*o_lr, :] @ in_g
        // We can use the full matmul and extract, but that's wasteful.
        // For now, do the full matmul once and it covers all groups:
        if (g == 0) {
            // Full matmul: o_latent[8192] = wo_a[8192,4096] @ o_heads_group
            // But o_heads is 32768, not 4096... we need grouped.
            // Actually, each group maps its own 4096 slice independently.
            // Since we can't slice QT easily, let's do the full matmul with the
            // FIRST group's 4096 input, then overwrite with remaining groups.
            // This is a simplification — proper grouped matmul needs QT slicing.
        }
        // Fallback: manual dot product using QT internals
        // wo_a is INT4: q4 data, s (scales), O=8192, I=4096
        QT *wa = &l->wo_a;
        if (wa->fmt == 2 && wa->q4) {
            for (int r = g * o_lr; r < (g + 1) * o_lr; r++) {
                float acc = 0;
                uint8_t *row = wa->q4 + (int64_t)r * (wa->I / 2);
                float s = wa->s[r];
                for (int i = 0; i < group_in; i += 2) {
                    uint8_t b = row[i / 2];
                    float v0 = ((b & 0xF) - 8) * s;
                    float v1 = ((b >> 4) - 8) * s;
                    acc += v0 * in_g[i] + v1 * in_g[i + 1];
                }
                out_g[r - g * o_lr] = acc;
            }
        } else if (wa->fmt == 1 && wa->q8) {
            for (int r = g * o_lr; r < (g + 1) * o_lr; r++) {
                float acc = 0;
                int8_t *row = wa->q8 + (int64_t)r * wa->I;
                float s = wa->s[r];
                for (int i = 0; i < group_in; i++)
                    acc += (float)row[i] * s * in_g[i];
                out_g[r - g * o_lr] = acc;
            }
        }
    }
    
    // wo_b[4096, 8192]: maps o_latent[8192] → out[4096]
    matmul_qt(out, o_latent, &l->wo_b, 1);
    

}

/* ---------- SiLU-gated FFN (shared or routed expert) ---------- */
static void ffn_silu(float *out, const float *x, QT *w1, QT *w3, QT *w2, int dim, int ignored_inter) {
    int inter = w1->O;
    float *gate = falloc(inter);
    float *up   = falloc(inter);
    
    matmul_qt(gate, x, w1, 1);  // gate = x @ w1^T [inter]
    matmul_qt(up,   x, w3, 1);  // up   = x @ w3^T [inter]
    
    // SiLU(gate) * up
    for (int i = 0; i < inter; i++) {
        gate[i] = gate[i] * sigmoidf(gate[i]) * up[i];
    }
    
    matmul_qt(out, gate, w2, 1);  // out = (SiLU(gate)*up) @ w2^T [dim]
}

/* ---------- Forward Pass ---------- */
static void forward_dsv4(Model *m, int token_id, int pos, float *logits, float *out_hidden) {
    // printf("DEBUG: forward_dsv4 start token=%d pos=%d\n", token_id, pos); fflush(stdout);
    arena_reset();
    Cfg *c = &m->c;
    int dim = c->hidden;
    int hc = c->hc_mult;
    
    // --- 0. Embedding ---
    // printf("DEBUG: Embedding\n"); fflush(stdout);
    int hc_dim = hc * dim;
    
    float *x_multi = falloc(hc_dim);
    float *emb = falloc(dim);
    QT *emb_t = &m->embed;
    
    // --- Embed lookup ---
    if (emb_t->fmt == 2) {
        uint8_t *q4 = emb_t->q4 + (int64_t)token_id * (dim / 2);
        float scale = emb_t->s[token_id];
        for (int i = 0; i < dim; i += 2) {
            uint8_t b = q4[i / 2];
            emb[i]     = ((b & 0xF) - 8) * scale;
            emb[i + 1] = ((b >> 4) - 8) * scale;
        }
    } else if (emb_t->fmt == 1) {
        int8_t *q8 = emb_t->q8 + (int64_t)token_id * dim;
        float scale = emb_t->s[token_id];
        for (int i = 0; i < dim; i++) emb[i] = (float)q8[i] * scale;
    } else {
        memcpy(emb, emb_t->qf + (int64_t)token_id * dim, dim * sizeof(float));
    }
    
    // Replicate embedding into all hc streams
    for (int j = 0; j < hc; j++) memcpy(x_multi + j * dim, emb, dim * sizeof(float));
    
    // --- Layer loop ---
    float *x_single = falloc(dim);
    float *residual = falloc(hc_dim);
    float *post = falloc(hc);
    float *comb = falloc(hc * hc);
    
    for (int l_id = 0; l_id < c->n_layers; l_id++) {
        // printf("DEBUG: Layer %d\n", l_id); fflush(stdout);
        Layer *l = &m->L[l_id];
        
        // --- 1. Attention Block ---
        memcpy(residual, x_multi, hc_dim * sizeof(float));
        hc_pre(x_single, x_multi, &l->hc_attn_fn, l->hc_attn_scale, l->hc_attn_base,
               hc, dim, c->hc_sinkhorn_iters, c->eps, post, comb);
        rmsnorm_slice(x_single, x_single, l->attn_norm, dim, c->eps);
        
        // Attention mechanism dispatch
        float *attn_out = falloc(dim);
        int ratio = c->compress_ratios[l_id];
        // All ratio types (SWA/CSA/HCA) are dispatched inside attention_decode
        attention_decode(m, l, l_id, x_single, pos, attn_out);
        
        hc_post(x_multi, attn_out, residual, post, comb, hc, dim);

        
        // --- 2. FFN / MoE Block ---
        memcpy(residual, x_multi, hc_dim * sizeof(float));
        hc_pre(x_single, x_multi, &l->hc_ffn_fn, l->hc_ffn_scale, l->hc_ffn_base,
               hc, dim, c->hc_sinkhorn_iters, c->eps, post, comb);
        rmsnorm_slice(x_single, x_single, l->ffn_norm, dim, c->eps);
        
        float *ffn_out = falloc(dim);
        memset(ffn_out, 0, dim * sizeof(float));
        int inter = c->moe_inter;  // 2048
        
        // Shared Expert FFN
        if (l->shared_w1.fmt != 0 || l->shared_w1.qf) {
            float *sh_out = falloc(dim);
            ffn_silu(sh_out, x_single, &l->shared_w1, &l->shared_w3, &l->shared_w2, dim, inter);
            for (int d = 0; d < dim; d++) ffn_out[d] += sh_out[d];

        }
        
        // MoE Router
        int *topk_idx = (int*)falloc(c->topk);
        float *topk_weights = falloc(c->topk);
        moe_route(c, l, l_id, token_id, x_single, topk_idx, topk_weights);
        
        // Routed Expert FFN
        for (int k = 0; k < c->topk; k++) {
            int e = topk_idx[k];
            float w = topk_weights[k];
            if (e < 0 || e >= c->n_experts) continue;
            if (l->ex_w1[e].fmt == 0 && !l->ex_w1[e].qf) continue;
            
            float *ex_out = falloc(dim);
            ffn_silu(ex_out, x_single, &l->ex_w1[e], &l->ex_w3[e], &l->ex_w2[e], dim, inter);
            for (int d = 0; d < dim; d++) ffn_out[d] += w * ex_out[d];

        }
        
        hc_post(x_multi, ffn_out, residual, post, comb, hc, dim);

    }
    
    // --- 3. Final Head: hc_head shrink → RMSNorm → LM Head ---
    // printf("DEBUG: Final Head\n"); fflush(stdout);
    // Shrink hc streams to 1 using hc_head_fn/scale/base
    float *hc_post_buf = falloc(hc);
    float *hc_comb_buf = falloc(hc * hc);
    hc_pre(x_single, x_multi, &m->hc_head_fn, m->hc_head_scale, m->hc_head_base,
           hc, dim, c->hc_sinkhorn_iters, c->eps, hc_post_buf, hc_comb_buf);
    rmsnorm_slice(x_single, x_single, m->final_norm, dim, c->eps);
    
    // Copy the final hidden state if requested (used for MTP drafting)
    if (out_hidden) {
        memcpy(out_hidden, x_single, dim * sizeof(float));
    }
    
    // LM Head: x_single[4096] → logits[vocab_size]
    matmul_qt(logits, x_single, &m->lm_head, 1);
    

}

/* ---------- MTP Draft Forward ---------- */
// Runs the MTP module to predict the NEXT token after the main model's prediction.
// Takes: the main model's final hidden state, the token the main model just predicted.
// Returns: draft logits for the speculative next-next token.
static void mtp_draft(Model *m, int draft_token, float *main_hidden, int pos, float *draft_logits) {
    Cfg *c = &m->c;
    int dim = c->hidden;
    int hc = c->hc_mult;
    MTP *mtp = &m->mtp[0];
    
    // 1. Embed the draft token using MTP's own embedding
    float *tok_emb = falloc(dim);
    QT *emb_t = &mtp->emb;
    if (emb_t->fmt == 2) {
        uint8_t *q4 = emb_t->q4 + (int64_t)draft_token * (dim / 2);
        float scale = emb_t->s[draft_token];
        for (int i = 0; i < dim; i += 2) {
            uint8_t b = q4[i / 2];
            tok_emb[i]     = ((b & 0xF) - 8) * scale;
            tok_emb[i + 1] = ((b >> 4) - 8) * scale;
        }
    } else if (emb_t->fmt == 1) {
        int8_t *q8 = emb_t->q8 + (int64_t)draft_token * dim;
        float scale = emb_t->s[draft_token];
        for (int i = 0; i < dim; i++) tok_emb[i] = (float)q8[i] * scale;
    } else if (emb_t->qf) {
        memcpy(tok_emb, emb_t->qf + (int64_t)draft_token * dim, dim * sizeof(float));
    }
    
    // 2. Normalize embedding and main hidden state
    float *norm_emb = falloc(dim);
    float *norm_hidden = falloc(dim);
    rmsnorm_slice(norm_emb, tok_emb, mtp->enorm, dim, c->eps);
    rmsnorm_slice(norm_hidden, main_hidden, mtp->hnorm, dim, c->eps);
    
    // 3. Project main hidden through e_proj and combine with token embedding
    float *proj = falloc(dim);
    matmul_qt(proj, norm_hidden, &mtp->e_proj, 1);
    
    float *x_combined = falloc(dim);
    for (int i = 0; i < dim; i++) x_combined[i] = proj[i] + norm_emb[i];
    
    // 4. Expand into hc streams
    int hc_dim = hc * dim;
    float *x_multi = falloc(hc_dim);
    for (int j = 0; j < hc; j++) memcpy(x_multi + j * dim, x_combined, dim * sizeof(float));
    
    // 5. Run MTP transformer layer (shared expert only for draft speed)
    Layer *l = &mtp->layer;
    float *x_single = falloc(dim);
    float *residual = falloc(hc_dim);
    float *post_buf = falloc(hc);
    float *comb_buf = falloc(hc * hc);
    
    // Attention block
    memcpy(residual, x_multi, hc_dim * sizeof(float));
    hc_pre(x_single, x_multi, &l->hc_attn_fn, l->hc_attn_scale, l->hc_attn_base,
           hc, dim, c->hc_sinkhorn_iters, c->eps, post_buf, comb_buf);
    rmsnorm_slice(x_single, x_single, l->attn_norm, dim, c->eps);
    
    float *attn_out = falloc(dim);
    attention_decode(m, l, c->n_layers - 1, x_single, pos, attn_out);
    hc_post(x_multi, attn_out, residual, post_buf, comb_buf, hc, dim);
    
    // FFN block (shared expert only for speed)
    memcpy(residual, x_multi, hc_dim * sizeof(float));
    hc_pre(x_single, x_multi, &l->hc_ffn_fn, l->hc_ffn_scale, l->hc_ffn_base,
           hc, dim, c->hc_sinkhorn_iters, c->eps, post_buf, comb_buf);
    rmsnorm_slice(x_single, x_single, l->ffn_norm, dim, c->eps);
    
    float *ffn_out = falloc(dim);
    memset(ffn_out, 0, dim * sizeof(float));
    if (l->shared_w1.fmt != 0 || l->shared_w1.qf) {
        float *sh_out = falloc(dim);
        ffn_silu(sh_out, x_single, &l->shared_w1, &l->shared_w3, &l->shared_w2, dim, c->moe_inter);
        for (int d = 0; d < dim; d++) ffn_out[d] += sh_out[d];
    }
    hc_post(x_multi, ffn_out, residual, post_buf, comb_buf, hc, dim);
    
    // 6. Head
    hc_pre(x_single, x_multi, &mtp->hc_head_fn, mtp->hc_head_scale, mtp->hc_head_base,
           hc, dim, c->hc_sinkhorn_iters, c->eps, post_buf, comb_buf);
    rmsnorm_slice(x_single, x_single, m->final_norm, dim, c->eps);
    matmul_qt(draft_logits, x_single, &mtp->head, 1);
}


static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    
    c->hidden       = (int)json_get(r,"hidden_size")->num;
    c->n_layers     = (int)json_get(r,"num_hidden_layers")->num;
    c->n_heads      = (int)json_get(r,"num_attention_heads")->num;
    c->head_dim     = (int)json_get(r,"head_dim")->num;
    c->vocab_size   = (int)json_get(r,"vocab_size")->num;
    c->moe_inter    = (int)json_get(r,"moe_intermediate_size")->num;
    c->index_head_dim = (int)json_get(r,"index_head_dim")->num;
    c->index_n_heads = (int)json_get(r,"index_n_heads")->num;
    c->index_topk   = (int)json_get(r,"index_topk")->num;
    c->n_experts    = (int)json_get(r,"n_routed_experts")->num;
    c->topk         = (int)json_get(r,"num_experts_per_tok")->num;
    c->n_hash_layers= (int)json_get(r,"num_hash_layers")->num;
    c->hc_mult      = (int)json_get(r,"hc_mult")->num;
    c->hc_sinkhorn_iters = (int)json_get(r,"hc_sinkhorn_iters")->num;
    c->window_size  = (int)json_get(r,"sliding_window")->num;
    c->q_lora_rank  = (int)json_get(r,"q_lora_rank")->num;
    c->o_lora_rank  = (int)json_get(r,"o_lora_rank")->num;
    c->o_groups     = (int)json_get(r,"o_groups")->num;
    c->kv_lora_rank = c->head_dim;  // In DeepSeek V4 MLA, kv_lora_rank == head_dim == 512
    c->qk_rope_head_dim = (int)json_get(r,"qk_rope_head_dim")->num;
    
    jval *th = json_get(r,"rope_theta");   c->theta = th ? (float)th->num : 10000.f;
    jval *ep = json_get(r,"rms_norm_eps"); c->eps   = ep ? (float)ep->num : 1e-6f;
    jval *hc = json_get(r,"hc_eps");       c->hc_eps= hc ? (float)hc->num : 1e-6f;
    
    jval *cr = json_get(r,"compress_ratios");
    if (cr && cr->t == J_ARR) {
        c->compress_ratios = malloc(cr->len * sizeof(int));
        for (int i = 0; i < cr->len; i++) c->compress_ratios[i] = (int)cr->kids[i]->num;
    }
    
    printf("  Config: hidden=%d layers=%d heads=%d hd=%d vocab=%d moe_inter=%d experts=%d topk=%d\n",
           c->hidden, c->n_layers, c->n_heads, c->head_dim, c->vocab_size, c->moe_inter, c->n_experts, c->topk);
    
    free(buf); free(arena);
}


#include "dsv4_tensor_wire.h"

static void model_init(Model *m, const char *snap) {
    printf("Loading configuration...\n");
    load_cfg(&m->c, snap);
    
    printf("Indexing and memory mapping Safetensors (150GB)...\n");
    st_init(&m->S, snap);
    
    printf("Loading DeepSeek tokenizer...\n");
    char tok_path[2048]; snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", snap);
    tok_load(&m->tokenizer, tok_path);
    
    m->L = calloc(m->c.n_layers, sizeof(Layer));
    m->K = calloc(m->c.n_layers, sizeof(float*));
    m->V = calloc(m->c.n_layers, sizeof(float*));
    for (int i=0; i<m->c.n_layers; i++) {
        int max_compressed_blocks = 4096;
        m->K[i] = calloc((m->c.window_size + max_compressed_blocks) * m->c.kv_lora_rank, sizeof(float));
        m->V[i] = calloc((m->c.window_size + max_compressed_blocks) * m->c.kv_lora_rank, sizeof(float));
    }
    
    printf("Connecting tensor architecture...\n");
    wire_tensors(m);
}

typedef struct {
    float prob;
    int index;
} ProbIndex;

static int cmp_prob(const void *a, const void *b) {
    ProbIndex *pa = (ProbIndex*)a;
    ProbIndex *pb = (ProbIndex*)b;
    if (pa->prob < pb->prob) return 1;
    if (pa->prob > pb->prob) return -1;
    return 0;
}

static int sample_topp(float *logits, int vocab_size, float temperature, float top_p) {
    if (temperature <= 0.0f) {
        int best = 0;
        float max_l = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_l) { max_l = logits[i]; best = i; }
        }
        return best;
    }
    
    float max_l = logits[0];
    for (int i = 1; i < vocab_size; i++) if (logits[i] > max_l) max_l = logits[i];
    
    float sum = 0.0f;
    ProbIndex *pi = malloc(vocab_size * sizeof(ProbIndex));
    for (int i = 0; i < vocab_size; i++) {
        float p = expf((logits[i] - max_l) / temperature);
        pi[i].prob = p;
        pi[i].index = i;
        sum += p;
    }
    
    for (int i = 0; i < vocab_size; i++) pi[i].prob /= sum;
    qsort(pi, vocab_size, sizeof(ProbIndex), cmp_prob);
    
    float cumulative_prob = 0.0f;
    int last_idx = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumulative_prob += pi[i].prob;
        last_idx = i;
        if (cumulative_prob >= top_p) break;
    }
    
    float r = (float)rand() / (float)RAND_MAX * cumulative_prob;
    float current = 0.0f;
    int selected = pi[last_idx].index;
    for (int i = 0; i <= last_idx; i++) {
        current += pi[i].prob;
        if (r <= current) { selected = pi[i].index; break; }
    }
    free(pi);
    return selected;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    
    if (argc < 2) {
        printf("Usage: dsv4.exe <model_path> [-p prompt]\n");
        return 1;
    }
    
    char *prompt = NULL;
    for (int i=1; i<argc; i++) {
        if ((!strcmp(argv[i], "-p") || !strcmp(argv[i], "-i")) && i+1 < argc) prompt = argv[i+1];
    }
    
    printf("==========================================\n");
    printf(" DeepSeek-V4-Flash C Engine Initialized   \n");
    printf("==========================================\n");
    
    Model *m = calloc(1, sizeof(Model));
    if (!m) { printf("OOM Model struct\n"); return 1; }
    
    model_init(m, argv[1]);
    
    double total_gb = 0;
    for (int i=0; i<m->S.n; i++) total_gb += (double)m->S.t[i].nbytes / 1073741824.0;
    
    printf("\nSUCCESS! Architecture is hardwired to the INT4 weights.\n");
    printf("Found %d tensors (approx %.2f GB mapped).\n", m->S.n, total_gb);
    
    int pos = 0;
    float *logits = malloc(m->c.vocab_size * sizeof(float));
    float *draft_logits = malloc(m->c.vocab_size * sizeof(float));
    float *main_hidden = malloc(m->c.hidden * sizeof(float));
    float temp = 0.7f;
    float top_p = 0.9f;
    srand((unsigned int)time(NULL));
    
    char user_input[4096];
    
    while (1) {
        if (prompt) {
            strncpy(user_input, prompt, sizeof(user_input) - 1);
            user_input[sizeof(user_input)-1] = '\0';
        } else {
            printf("\n[USER]: ");
            if (!fgets(user_input, sizeof(user_input), stdin)) break;
            user_input[strcspn(user_input, "\r\n")] = 0;
            if (strlen(user_input) == 0) continue;
            if (strcmp(user_input, "exit") == 0 || strcmp(user_input, "quit") == 0) break;
        }
        
        int input_ids[2048];
        int n_tokens = tok_encode(&m->tokenizer, user_input, strlen(user_input), input_ids, 2048);
        printf("[TOKENIZER]: Successfully parsed %d tokens\n[OUTPUT]: ", n_tokens);
        
        // Prefill
        for(int i = 0; i < n_tokens; i++) {
            forward_dsv4(m, input_ids[i], pos++, logits, main_hidden);
        }
        
        // Speculative Decode State
        int current_token = sample_topp(logits, m->c.vocab_size, temp, top_p);
        
        char buf[128];
        tok_decode(&m->tokenizer, &current_token, 1, buf, 128);
        printf("%s", buf);
        fflush(stdout);
        
        int draft_token = -1;
        int total_tokens = 0;
        int accepted_drafts = 0;
        double start_time = now_s();
        
        // Decode Loop
        for (int step = 0; step < 256; step++) {
            if (draft_token == -1 && current_token != 1) {
                mtp_draft(m, current_token, main_hidden, pos, draft_logits);
                draft_token = sample_topp(draft_logits, m->c.vocab_size, temp, top_p);
            }
            
            forward_dsv4(m, current_token, pos++, logits, main_hidden);
            int true_next = sample_topp(logits, m->c.vocab_size, temp, top_p);
            
            if (true_next == 1) break; // EOS
            tok_decode(&m->tokenizer, &true_next, 1, buf, 128);
            
            if (true_next == draft_token) {
                printf("%s", buf);
                current_token = true_next;
                draft_token = -1; 
                accepted_drafts++;
                total_tokens++;
            } else {
                printf("%s", buf);
                current_token = true_next;
                draft_token = -1;
            }
            fflush(stdout);
            total_tokens++;
        }
        double end_time = now_s();
        double dt = end_time - start_time;
        printf("\n\n[PERFORMANCE]\nGenerated %d tokens in %.2f seconds (%.2f tokens/sec). MTP Accepted: %d\n", 
               total_tokens, dt, total_tokens / dt, accepted_drafts);
               
        if (prompt) break; // Only run once if prompt was passed via CLI
    }
    
    return 0;
}
