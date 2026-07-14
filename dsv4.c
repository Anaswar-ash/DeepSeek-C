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
    
    // SWA / Compression
    int window_size;
    int *compress_ratios; // Array of compression ratios per layer (0=SWA, 4=CSA, 128=HCA)
    
    // LoRA Latents
    int q_lora_rank;
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
    float *shared_w1, *shared_w2, *shared_w3;
    float *gate;      // Router scoring projection matrix
    int *tid2eid;     // Hash routing lookups (first n_hash_layers) [vocab_size, topk]
    float *gate_bias; // Optional bias for the gate

    // Attention
    float *attn_norm;
    float *wq_a, *wq_b, *q_norm; // Query LoRA up/down projections
    float *wkv, *kv_norm;        // MQA style KV projection
    float *wo_a, *wo_b;          // Output grouped projections
    float *attn_sink;            // Attention sink bias

    // CSA/HCA Compression
    float *c_wkv, *c_wgate, *c_ape; // Compressor weights (reduces chunks)
    float *idx_wq_b, *idx_wproj;    // Lightning Indexer weights (scores chunks)

    // mHC
    // _fn is the projection, _scale is the scalar triplet, _base is the bias
    float *hc_attn_base, *hc_attn_fn, *hc_attn_scale;
    float *hc_ffn_base, *hc_ffn_fn, *hc_ffn_scale;
} Layer;

// Global model state
typedef struct {
    Cfg c;
    shards S;
    int quant_bits;
    float *embed, *lm_head, *final_norm;
    float *hc_head_fn, *hc_head_base, *hc_head_scale;
    Layer *L;

    uint64_t clock, hits, miss;
    
    // KV Cache
    float **K, **V;
    int max_t;
    double dense_load_s;
} Model;

/* ---------- utility ---------- */
static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
static double rss_gb(void) { return 0.0; } // getrusage not on Windows by default
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }

static void matmul(float *out, const float *x, const float *w, int m, int n, int k) {
    // dummy matmul to satisfy compiler
    memset(out, 0, m * k * sizeof(float));
}

/* ---------- mHC Sinkhorn ---------- */
static void hc_split_sinkhorn(const float *mixes, const float *hc_scale, const float *hc_base, 
                              int hc, int iters, float eps,
                              float *pre, float *post, float *comb) {
    for (int j = 0; j < hc; j++) pre[j] = sigmoidf(mixes[j] * hc_scale[0] + hc_base[j]) + eps;
    for (int j = 0; j < hc; j++) post[j] = 2.0f * sigmoidf(mixes[j + hc] * hc_scale[1] + hc_base[j + hc]);
    for (int j = 0; j < hc; j++) {
        for (int k = 0; k < hc; k++) {
            int idx = j * hc + k + hc * 2;
            comb[j * hc + k] = mixes[idx] * hc_scale[2] + hc_base[idx];
        }
    }
    
    // Sinkhorn normalization on comb
    float row_max[16], row_sum[16], col_sum[16]; // safely handles hc up to 16
    for (int j = 0; j < hc; j++) {
        row_max[j] = -1e30f;
        for (int k = 0; k < hc; k++) {
            if (comb[j * hc + k] > row_max[j]) row_max[j] = comb[j * hc + k];
        }
    }
    for (int j = 0; j < hc; j++) {
        row_sum[j] = 0.0f;
        for (int k = 0; k < hc; k++) {
            comb[j * hc + k] = expf(comb[j * hc + k] - row_max[j]);
            row_sum[j] += comb[j * hc + k];
        }
        for (int k = 0; k < hc; k++) comb[j * hc + k] = comb[j * hc + k] / row_sum[j] + eps;
    }
    for (int k = 0; k < hc; k++) {
        col_sum[k] = 0.0f;
        for (int j = 0; j < hc; j++) col_sum[k] += comb[j * hc + k];
    }
    for (int j = 0; j < hc; j++) {
        for (int k = 0; k < hc; k++) comb[j * hc + k] = comb[j * hc + k] / (col_sum[k] + eps);
    }
    for (int iter = 0; iter < iters - 1; iter++) {
        for (int j = 0; j < hc; j++) {
            row_sum[j] = 0.0f;
            for (int k = 0; k < hc; k++) row_sum[j] += comb[j * hc + k];
            for (int k = 0; k < hc; k++) comb[j * hc + k] = comb[j * hc + k] / (row_sum[j] + eps);
        }
        for (int k = 0; k < hc; k++) {
            col_sum[k] = 0.0f;
            for (int j = 0; j < hc; j++) col_sum[k] += comb[j * hc + k];
        }
        for (int j = 0; j < hc; j++) {
            for (int k = 0; k < hc; k++) comb[j * hc + k] = comb[j * hc + k] / (col_sum[k] + eps);
        }
    }
}

// Compute HC Pre (shrink 4 streams to 1)
static void hc_pre(float *y_single, const float *x_multi, const float *hc_fn, const float *hc_scale, const float *hc_base, 
                   int hc_mult, int dim, int iters, float eps, float *post, float *comb) {
    int hc_dim = hc_mult * dim;
    int mix_hc = (2 + hc_mult) * hc_mult;
    
    // rsqrt of flattened x
    double ms = 0;
    for (int i = 0; i < hc_dim; i++) ms += (double)x_multi[i]*x_multi[i];
    float rsqrt = 1.0f / sqrtf((float)(ms / hc_dim) + eps);
    
    // mixes = (x @ hc_fn^T) * rsqrt
    float *mixes = falloc(mix_hc);
    matmul(mixes, x_multi, hc_fn, 1, hc_dim, mix_hc);
    for (int i = 0; i < mix_hc; i++) mixes[i] *= rsqrt;
    
    float *pre = falloc(hc_mult);
    hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps, pre, post, comb);
    
    // y = sum(pre * x, dim=hc_mult)
    memset(y_single, 0, dim * sizeof(float));
    for (int j = 0; j < hc_mult; j++) {
        for (int d = 0; d < dim; d++) {
            y_single[d] += pre[j] * x_multi[j * dim + d];
        }
    }
    
    free(mixes); free(pre);
}

// Compute HC Post (expand 1 stream back to 4)
static void hc_post(float *y_multi, const float *x_single, const float *residual_multi, 
                    const float *post, const float *comb, int hc_mult, int dim) {
    for (int j = 0; j < hc_mult; j++) {
        for (int d = 0; d < dim; d++) {
            // post.unsqueeze(-1) * x.unsqueeze(-2)
            float val = post[j] * x_single[d];
            // sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
            for (int k = 0; k < hc_mult; k++) {
                val += comb[j * hc_mult + k] * residual_multi[k * dim + d];
            }
            y_multi[j * dim + d] = val;
        }
    }
}

/* ---------- MoE Routing ---------- */
static inline float softplusf(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

// Compute routing weights for a single token
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
    // scores = hidden @ gate^T (approximated for 1 token)
    for (int e = 0; e < E; e++) {
        float acc = 0;
        const float *w = l->gate + (int64_t)e * c->hidden;
        for (int d = 0; d < c->hidden; d++) acc += hidden[d] * w[d];
        if (l->gate_bias) acc += l->gate_bias[e];
        scores[e] = sqrtf(softplusf(acc));
    }
    
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
    free(scores);
}


/* ---------- Attention ---------- */
static void rope_head(float *x, int pos, const Cfg *c) {
    int h = c->qk_rope_head_dim / 2;
    int offset = c->head_dim - c->qk_rope_head_dim; // RoPE applies to last `rd` dims
    float *xr = x + offset;
    for (int j = 0; j < h; j++) {
        float inv = powf(c->theta, -2.0f * j / c->qk_rope_head_dim);
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
    free(sc);
}

// Lightning Indexer: Selects top-k indices from compressed KV cache
static void indexer_topk(Model *m, Layer *l, int layer_id, float *x, float *qr, int pos, 
                         int *topk_idxs, int offset) {
    Cfg *c = &m->c;
    int ih = c->index_n_heads;
    int idim = c->index_head_dim;
    
    float *q = falloc(ih * idim);
    matmul(q, qr, l->idx_wq_b, 1, c->q_lora_rank, ih * idim);
    
    // RoPE for indexer queries
    for (int hh = 0; hh < ih; hh++) {
        rope_head(q + hh * idim, pos, c);
    }
    
    float *weights = falloc(ih);
    matmul(weights, x, l->idx_wproj, 1, c->hidden, ih);
    
    float scale = (1.0f / sqrtf((float)idim)) * (1.0f / sqrtf((float)ih));
    
    int ratio = c->compress_ratios[layer_id];
    int num_compressed = pos / ratio; // number of compressed tokens in cache
    int topk = c->index_topk;
    if (num_compressed < topk) topk = num_compressed;
    
    float *scores = falloc(num_compressed);
    memset(scores, 0, num_compressed * sizeof(float));
    
    for (int t = 0; t < num_compressed; t++) {
        float *ckv = m->K[layer_id] + (c->window_size + t) * idim; // Assume compressed KV is stored after sliding window
        float token_score = 0;
        for (int hh = 0; hh < ih; hh++) {
            float acc = 0;
            float *qh = q + hh * idim;
            for (int d = 0; d < idim; d++) {
                acc += qh[d] * ckv[d]; // Einsum
            }
            float relu_acc = acc > 0 ? acc : 0;
            token_score += relu_acc * weights[hh];
        }
        scores[t] = token_score * scale;
    }
    
    // Find top-K
    for (int k = 0; k < topk; k++) {
        float max_s = -1e30f; int best = -1;
        for (int t = 0; t < num_compressed; t++) {
            int taken = 0;
            for (int j = 0; j < k; j++) if (topk_idxs[j] == t + offset) { taken = 1; break; }
            if (!taken && scores[t] > max_s) { max_s = scores[t]; best = t; }
        }
        topk_idxs[k] = best + offset; // global index
    }
    
    free(q); free(weights); free(scores);
}

// rmsnorm on a slice
static void rmsnorm_slice(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * (w ? w[i] : 1.0f);
}

// SWA decode attention (batch=1, seq=1)
static void attention_decode(Model *m, Layer *l, int layer_id, float *x, int pos, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden;
    int H = c->n_heads;
    int hd = c->head_dim;
    int win = c->window_size;
    
    float *q_latent = falloc(c->q_lora_rank);
    float *q_heads = falloc(H * hd);
    float *kv = falloc(hd);
    float *o_heads = falloc(H * hd);
    
    // Q projection
    matmul(q_latent, x, l->wq_a, 1, D, c->q_lora_rank);
    rmsnorm_slice(q_latent, q_latent, l->q_norm, c->q_lora_rank, c->eps);
    matmul(q_heads, q_latent, l->wq_b, 1, c->q_lora_rank, H * hd);
    
    // KV projection
    matmul(kv, x, l->wkv, 1, D, hd);
    rmsnorm_slice(kv, kv, l->kv_norm, hd, c->eps);
    
    // RoPE and Q-norm per head
    for (int hh = 0; hh < H; hh++) {
        float *qh = q_heads + hh * hd;
        rmsnorm_slice(qh, qh, NULL, hd, c->eps); // q *= rsqrt(mean(q^2) + eps)
        rope_head(qh, pos, c);
    }
    rope_head(kv, pos, c);
    
    // Store in sliding window cache
    int cache_idx = pos % win;
    float *cache_row = m->K[layer_id] + cache_idx * hd;
    memcpy(cache_row, kv, hd * sizeof(float));
    
    // Attention loop (SWA over window)
    float scale = 1.0f / sqrtf((float)hd);
    int num_keys = (pos + 1 < win) ? (pos + 1) : win;
    
    for (int hh = 0; hh < H; hh++) {
        float *qh = q_heads + hh * hd;
        float *oh = o_heads + hh * hd;
        float *sc = falloc(num_keys);
        
        for (int t = 0; t < num_keys; t++) {
            int hist_pos = pos - t; // from current pos going backwards
            if (hist_pos < 0) break;
            int c_idx = hist_pos % win;
            float *kh = m->K[layer_id] + c_idx * hd; // KV is shared
            
            float acc = 0;
            for (int d = 0; d < hd; d++) acc += qh[d] * kh[d];
            sc[t] = acc * scale;
        }
        
        // softmax
        float max_sc = -1e30f;
        for (int t = 0; t < num_keys; t++) if (sc[t] > max_sc) max_sc = sc[t];
        float sum_sc = 0;
        for (int t = 0; t < num_keys; t++) {
            sc[t] = expf(sc[t] - max_sc);
            sum_sc += sc[t];
        }
        // Attn sink bias would go here if needed
        float sink_bias = l->attn_sink ? expf(l->attn_sink[hh] - max_sc) : 0;
        sum_sc += sink_bias;
        for (int t = 0; t < num_keys; t++) sc[t] /= sum_sc;
        
        // weighted sum
        memset(oh, 0, hd * sizeof(float));
        for (int t = 0; t < num_keys; t++) {
            int c_idx = (pos - t) % win;
            float *vh = m->K[layer_id] + c_idx * hd;
            for (int d = 0; d < hd; d++) oh[d] += sc[t] * vh[d];
        }
        // Attn sink value is typically 0
        free(sc);
    }
    
    // O projection
    int G = c->o_groups;
    int o_lora = c->o_lora_rank;
    int group_dim = (H * hd) / G;
    float *o_latent = falloc(G * o_lora);
    
    for (int g = 0; g < G; g++) {
        matmul(o_latent + g * o_lora, o_heads + g * group_dim, 
               l->wo_a + g * o_lora * group_dim, 1, group_dim, o_lora);
    }
    matmul(out, o_latent, l->wo_b, 1, G * o_lora, D);
    
    free(q_latent); free(q_heads); free(kv); free(o_heads); free(o_latent);
}

/* ---------- Forward Pass ---------- */
static void forward_dsv4(Model *m, int token_id, int pos, float *logits) {
    Cfg *c = &m->c;
    int dim = c->hidden;
    int hc = c->hc_mult;
    int hc_dim = hc * dim;
    
    // We maintain 4 residual streams for mHC (except layer 0 input which is just token embedding copied 4 times?)
    // Actually, model input is 1 stream, and the first HC Block receives 4 identical copies?
    // According to DeepSeek-V4 paper, the embedding is duplicated to 4 streams.
    float *x_multi = falloc(hc_dim);
    float *emb = m->embed + (int64_t)token_id * dim;
    for (int j = 0; j < hc; j++) {
        memcpy(x_multi + j * dim, emb, dim * sizeof(float));
    }
    
    float *x_single = falloc(dim);
    float *residual = falloc(hc_dim);
    float *post = falloc(hc);
    float *comb = falloc(hc * hc);
    
    for (int l_id = 0; l_id < c->n_layers; l_id++) {
        Layer *l = &m->L[l_id];
        
        // --- 1. Attention Block ---
        memcpy(residual, x_multi, hc_dim * sizeof(float));
        
        // Shrink 4 streams -> 1
        hc_pre(x_single, x_multi, l->hc_attn_fn, l->hc_attn_scale, l->hc_attn_base, hc, dim, c->hc_sinkhorn_iters, c->eps, post, comb);
        
        // Attn Norm
        rmsnorm_slice(x_single, x_single, l->attn_norm, dim, c->eps);
        
        // Attention mechanism dispatch
        float *attn_out = falloc(dim);
        int ratio = c->compress_ratios[l_id];
        if (ratio == 0) {
            attention_decode(m, l, l_id, x_single, pos, attn_out);
        } else {
            // CSA / HCA
            // compress_topk_idxs = indexer_topk(...)
            // sparse_attention(...)
            // For prototyping, we fallback to zero output if not implemented
            memset(attn_out, 0, dim * sizeof(float));
        }
        
        // Expand 1 -> 4 streams and mix
        hc_post(x_multi, attn_out, residual, post, comb, hc, dim);
        free(attn_out);
        
        // --- 2. FFN / MoE Block ---
        memcpy(residual, x_multi, hc_dim * sizeof(float));
        
        // Shrink 4 streams -> 1
        hc_pre(x_single, x_multi, l->hc_ffn_fn, l->hc_ffn_scale, l->hc_ffn_base, hc, dim, c->hc_sinkhorn_iters, c->eps, post, comb);
        
        // FFN Norm
        rmsnorm_slice(x_single, x_single, l->ffn_norm, dim, c->eps);
        
        // MoE Router
        int *topk_idx = malloc(c->topk * sizeof(int));
        float *topk_weights = falloc(c->topk);
        moe_route(c, l, l_id, token_id, x_single, topk_idx, topk_weights);
        
        float *ffn_out = falloc(dim);
        memset(ffn_out, 0, dim * sizeof(float));
        
        // Shared Expert
        // (x @ w1) * w3 -> w2
        // ffn_out += shared_expert(x)
        
        // Routed Experts
        for (int k = 0; k < c->topk; k++) {
            int e = topk_idx[k];
            float w = topk_weights[k];
            // float *ex_out = run_expert(e, x_single);
            // for(d) ffn_out[d] += w * ex_out[d];
            (void)e; (void)w; // Suppress unused warnings
        }
        
        // Expand 1 -> 4 streams and mix
        hc_post(x_multi, ffn_out, residual, post, comb, hc, dim);
        
        free(topk_idx); free(topk_weights); free(ffn_out);
    }
    
    // --- 3. Final Head (MTP/LM) ---
    // hc_head(x_multi) -> shrink to 1 using hc_head_fn/scale/base
    // Then norm, then mm(lm_head).
    
    free(x_multi); free(x_single); free(residual); free(post); free(comb);
}

/* ---------- config loader ---------- */
static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    
    c->hidden       = (int)json_get(r,"hidden_size")->num;
    c->n_layers     = (int)json_get(r,"num_hidden_layers")->num;
    c->n_heads      = (int)json_get(r,"num_attention_heads")->num;
    // c->n_kv_heads   = (int)json_get(r,"num_key_value_heads")->num;
    c->head_dim     = (int)json_get(r,"head_dim")->num;
    c->index_head_dim = (int)json_get(r,"index_head_dim")->num;
    c->index_n_heads = (int)json_get(r,"index_n_heads")->num;
    c->index_topk   = (int)json_get(r,"index_topk")->num;
    c->n_experts    = (int)json_get(r,"n_routed_experts")->num;
    c->topk         = (int)json_get(r,"num_experts_per_tok")->num;
    // c->inter        = (int)json_get(r,"moe_intermediate_size")->num;
    // c->vocab        = (int)json_get(r,"vocab_size")->num;
    c->n_hash_layers= (int)json_get(r,"num_hash_layers")->num;
    c->hc_mult      = (int)json_get(r,"hc_mult")->num;
    c->hc_sinkhorn_iters = (int)json_get(r,"hc_sinkhorn_iters")->num;
    c->window_size  = (int)json_get(r,"sliding_window")->num;
    c->q_lora_rank  = (int)json_get(r,"q_lora_rank")->num;
    c->o_lora_rank  = (int)json_get(r,"o_lora_rank")->num;
    c->qk_rope_head_dim = (int)json_get(r,"qk_rope_head_dim")->num;
    
    jval *th = json_get(r,"rope_theta");   c->theta = th ? (float)th->num : 10000.f;
    jval *ep = json_get(r,"rms_norm_eps"); c->eps   = ep ? (float)ep->num : 1e-6f;
    jval *hc = json_get(r,"hc_eps");       c->hc_eps= hc ? (float)hc->num : 1e-6f;
    // jval *nt = json_get(r,"norm_topk_prob"); c->norm_topk = (nt && nt->t==J_BOOL) ? nt->boolean : 0;
    
    jval *cr = json_get(r,"compress_ratios");
    if (cr && cr->t == J_ARR) {
        c->compress_ratios = malloc(cr->len * sizeof(int));
        for (int i = 0; i < cr->len; i++) c->compress_ratios[i] = (int)cr->kids[i]->num;
    }
    
    free(buf); free(arena);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("DeepSeek-V4-Flash C Engine Initialized\n");
    printf("Running mock memory test...\n");
    
    Model m = {0};
    m.c.hidden = 128;
    m.c.n_layers = 1;
    m.c.n_heads = 4;
    m.c.head_dim = 32;
    m.c.n_experts = 4;
    m.c.topk = 2;
    m.c.n_hash_layers = 0;
    m.c.hc_mult = 4;
    m.c.hc_sinkhorn_iters = 3;
    m.c.eps = 1e-6f;
    m.c.window_size = 64;
    m.c.compress_ratios = malloc(1 * sizeof(int));
    m.c.compress_ratios[0] = 0; // SWA
    m.c.o_lora_rank = 32;
    m.c.o_groups = 1;
    m.c.q_lora_rank = 32;
    m.c.qk_rope_head_dim = 32;
    m.c.theta = 10000.f;

    m.embed = falloc(10 * m.c.hidden);
    m.L = malloc(m.c.n_layers * sizeof(Layer));
    m.K = malloc(m.c.n_layers * sizeof(float*));
    for (int i = 0; i < m.c.n_layers; i++) {
        m.K[i] = falloc(m.c.window_size * m.c.head_dim);
        Layer *l = &m.L[i];
        memset(l, 0, sizeof(Layer));
        l->hc_attn_fn = falloc(m.c.hc_mult * m.c.hidden);
        l->hc_attn_scale = falloc(3);
        l->hc_attn_base = falloc(m.c.hc_mult * 2 + m.c.hc_mult * m.c.hc_mult);
        l->attn_norm = falloc(m.c.hidden);
        
        l->hc_ffn_fn = falloc(m.c.hc_mult * m.c.hidden);
        l->hc_ffn_scale = falloc(3);
        l->hc_ffn_base = falloc(m.c.hc_mult * 2 + m.c.hc_mult * m.c.hc_mult);
        l->ffn_norm = falloc(m.c.hidden);
        
        l->gate = falloc(m.c.n_experts * m.c.hidden);
        
        l->wq_a = falloc(m.c.hidden * m.c.q_lora_rank);
        l->q_norm = falloc(m.c.q_lora_rank);
        l->wq_b = falloc(m.c.q_lora_rank * m.c.n_heads * m.c.head_dim);
        l->wkv = falloc(m.c.hidden * m.c.head_dim);
        l->kv_norm = falloc(m.c.head_dim);
        
        l->wo_a = falloc((m.c.n_heads * m.c.head_dim / m.c.o_groups) * m.c.o_lora_rank);
        l->wo_b = falloc(m.c.o_groups * m.c.o_lora_rank * m.c.hidden);
    }
    
    float logits[128] = {0};
    forward_dsv4(&m, 0, 0, logits);
    
    printf("Mock test completed successfully without memory faults!\n");
    return 0;
}
