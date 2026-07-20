#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

static void *qalloc(size_t n){
    void *p = NULL;
#ifdef _WIN32
    p = _aligned_malloc(n, 4096);
#else
    if(posix_memalign(&p, 4096, n)) p = NULL;
#endif
    if(!p){ printf("\n[FATAL] OOM qalloc %zu bytes\n", n); fflush(stdout); exit(1); }
    return p;
}

#include <windows.h>
#include <io.h>

static void *get_mmap_base(int fd) {
    static int mapped_fds[512];
    static void *mapped_bases[512];
    static int num_mapped = 0;
    for (int i = 0; i < num_mapped; i++) {
        if (mapped_fds[i] == fd) return mapped_bases[i];
    }
    HANDLE hFile = (HANDLE)_get_osfhandle(fd);
    HANDLE hMap = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    void *base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!base) { fprintf(stderr, "mmap failed for fd %d\n", fd); exit(1); }
    mapped_fds[num_mapped] = fd;
    mapped_bases[num_mapped] = base;
    num_mapped++;
    return base;
}

static void qt_load(shards *S, QT *t, const char *name, int O_hint, int I_hint) {
    if (!st_has(S, name)) { memset(t, 0, sizeof(QT)); return; }
    
    char buf[256]; snprintf(buf, sizeof(buf), "%s.qs", name);
    if (st_has(S, buf)) { // It's a quantized tensor
        int64_t O = O_hint > 0 ? O_hint : st_numel(S, buf);
        int64_t numel = st_numel(S, name);
        int64_t I = I_hint > 0 ? I_hint : (numel * 2 / O);
        
        if (numel == O * I) t->fmt = 1;         // INT8
        else if (numel * 2 == O * I) t->fmt = 2; // INT4
        else if (numel * 4 == O * I) t->fmt = 3; // INT2
        else t->fmt = 2; // Default fallback
        
        t->O = (int)O;
        t->I = (int)I;
        
        st_tensor *ts = st_find(S, name);
        if (!ts) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
        
        void *base = get_mmap_base(ts->fd);
        if (t->fmt == 1) t->q8 = (int8_t*)((char *)base + ts->off);
        else t->q4 = (uint8_t*)((char *)base + ts->off);
        
        t->s = qalloc(t->O * sizeof(float));
        st_read_f32(S, buf, t->s, 0);
    } else { // FP32
        t->fmt = 0;
        t->O = O_hint; t->I = I_hint;
        
        t->qf = qalloc((int64_t)t->O * t->I * sizeof(float));
        st_read_f32(S, name, t->qf, 0);
    }
}

static float* fp32_load(shards *S, const char *name) {
    if (!st_has(S, name)) return NULL;
    int64_t numel = st_numel(S, name);
    float *p = qalloc(numel * sizeof(float));
    st_read_f32(S, name, p, 0);
    return p;
}

static void wire_tensors(Model *m) {
    Cfg *c = &m->c;
    shards *S = &m->S;
    char buf[256];
    
    printf("  [wire] Base tensors...\n");
    printf("  [wire] embed.weight\n");
    qt_load(S, &m->embed, "embed.weight", c->vocab_size, c->hidden);
    printf("  [wire] head.weight\n");
    qt_load(S, &m->lm_head, "head.weight", c->vocab_size, c->hidden);
    printf("  [wire] norm.weight\n");
    m->final_norm = fp32_load(S, "norm.weight");
    
    printf("  [wire] hc_head_base.weight\n");
    m->hc_head_base = fp32_load(S, "hc_head_base.weight");
    printf("  [wire] hc_head_scale.weight\n");
    m->hc_head_scale= fp32_load(S, "hc_head_scale.weight");
    printf("  [wire] hc_head_fn.weight\n");
    qt_load(S, &m->hc_head_fn, "hc_head_fn.weight", (2 + c->hc_mult) * c->hc_mult, c->hc_mult * c->hidden);
    
    printf("  [wire] Starting %d layers...\n", c->n_layers);
    int dim = c->hidden;
    
    for (int i = 0; i < c->n_layers; i++) {
        printf("  [wire] Layer %d\n", i);
        Layer *l = &m->L[i];
        l->ffn_norm = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.ffn_norm.weight", i) ? buf : buf);
        
        // Experts
        int ex_inter = 2048; // DeepSeek-V4 routed expert intermediate size is 2048
        for (int e = 0; e < m->c.n_experts; e++) {
            qt_load(S, &l->ex_w1[e], snprintf(buf, sizeof(buf), "layers.%d.ffn.experts.%d.w1.weight", i, e) ? buf : buf, ex_inter, dim);
            qt_load(S, &l->ex_w2[e], snprintf(buf, sizeof(buf), "layers.%d.ffn.experts.%d.w2.weight", i, e) ? buf : buf, dim, ex_inter);
            qt_load(S, &l->ex_w3[e], snprintf(buf, sizeof(buf), "layers.%d.ffn.experts.%d.w3.weight", i, e) ? buf : buf, ex_inter, dim);
        }
        
        qt_load(S, &l->shared_w1, snprintf(buf, sizeof(buf), "layers.%d.ffn.shared_experts.w1.weight", i) ? buf : buf, 0, dim);
        qt_load(S, &l->shared_w2, snprintf(buf, sizeof(buf), "layers.%d.ffn.shared_experts.w2.weight", i) ? buf : buf, dim, 0);
        qt_load(S, &l->shared_w3, snprintf(buf, sizeof(buf), "layers.%d.ffn.shared_experts.w3.weight", i) ? buf : buf, 0, dim);
        
        qt_load(S, &l->gate, snprintf(buf, sizeof(buf), "layers.%d.ffn.gate.weight", i) ? buf : buf, m->c.n_experts, dim);
        l->gate_bias = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.ffn.gate.bias", i) ? buf : buf);
        
        l->attn_norm = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.attn_norm.weight", i) ? buf : buf);
        qt_load(S, &l->wq_a, snprintf(buf, sizeof(buf), "layers.%d.attn.wq_a.weight", i) ? buf : buf, m->c.q_lora_rank, dim);
        qt_load(S, &l->wq_b, snprintf(buf, sizeof(buf), "layers.%d.attn.wq_b.weight", i) ? buf : buf, 0, m->c.q_lora_rank);
        l->q_norm = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.attn.q_norm.weight", i) ? buf : buf);
        
        qt_load(S, &l->wkv, snprintf(buf, sizeof(buf), "layers.%d.attn.wkv.weight", i) ? buf : buf, 0, dim);
        l->kv_norm = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.attn.kv_norm.weight", i) ? buf : buf);
        
        qt_load(S, &l->wo_a, snprintf(buf, sizeof(buf), "layers.%d.attn.wo_a.weight", i) ? buf : buf, 0, (m->c.n_heads * m->c.head_dim) / m->c.o_groups);
        qt_load(S, &l->wo_b, snprintf(buf, sizeof(buf), "layers.%d.attn.wo_b.weight", i) ? buf : buf, dim, 0);
        
        l->hc_attn_base = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.hc_attn_base", i) ? buf : buf);
        l->hc_attn_scale = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.hc_attn_scale", i) ? buf : buf);
        qt_load(S, &l->hc_attn_fn, snprintf(buf, sizeof(buf), "layers.%d.hc_attn_fn", i) ? buf : buf, (2 + m->c.hc_mult) * m->c.hc_mult, m->c.hc_mult * dim);
        
        l->hc_ffn_base = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.hc_ffn_base", i) ? buf : buf);
        l->hc_ffn_scale = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.hc_ffn_scale", i) ? buf : buf);
        qt_load(S, &l->hc_ffn_fn, snprintf(buf, sizeof(buf), "layers.%d.hc_ffn_fn", i) ? buf : buf, (2 + m->c.hc_mult) * m->c.hc_mult, m->c.hc_mult * dim);

        // Indexer loading
        qt_load(S, &l->idx_ape, snprintf(buf, sizeof(buf), "layers.%d.attn.indexer.compressor.ape", i) ? buf : buf, 0, 0);
        l->idx_comp_norm = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.attn.indexer.compressor.norm.weight", i) ? buf : buf);
        qt_load(S, &l->idx_wgate, snprintf(buf, sizeof(buf), "layers.%d.attn.indexer.compressor.wgate.weight", i) ? buf : buf, 0, 0);
        qt_load(S, &l->idx_wkv, snprintf(buf, sizeof(buf), "layers.%d.attn.indexer.compressor.wkv.weight", i) ? buf : buf, 0, 0);
        qt_load(S, &l->idx_wproj, snprintf(buf, sizeof(buf), "layers.%d.attn.indexer.weights_proj.weight", i) ? buf : buf, 0, 0);
        qt_load(S, &l->idx_wq_b, snprintf(buf, sizeof(buf), "layers.%d.attn.indexer.wq_b.weight", i) ? buf : buf, 0, 0);
    }
    
    // MTP (Multi-Token Prediction) wiring
    m->n_mtp = 1; // num_nextn_predict_layers
    m->mtp = calloc(m->n_mtp, sizeof(MTP));
    
    for (int mi = 0; mi < m->n_mtp; mi++) {
        MTP *mtp = &m->mtp[mi];
        printf("  [wire] MTP module %d\n", mi);
        
        qt_load(S, &mtp->emb, snprintf(buf, sizeof(buf), "mtp.%d.emb.tok_emb.weight", mi) ? buf : buf, c->vocab_size, c->hidden);
        qt_load(S, &mtp->e_proj, snprintf(buf, sizeof(buf), "mtp.%d.e_proj.weight", mi) ? buf : buf, c->hidden, c->hidden);
        mtp->enorm = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.enorm.weight", mi) ? buf : buf);
        mtp->hnorm = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.hnorm.weight", mi) ? buf : buf);
        qt_load(S, &mtp->head, snprintf(buf, sizeof(buf), "mtp.%d.head.weight", mi) ? buf : buf, c->vocab_size, c->hidden);
        
        qt_load(S, &mtp->hc_head_fn, snprintf(buf, sizeof(buf), "mtp.%d.hc_head_fn", mi) ? buf : buf, 0, 0);
        mtp->hc_head_base = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.hc_head_base", mi) ? buf : buf);
        mtp->hc_head_scale = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.hc_head_scale", mi) ? buf : buf);
        
        // Wire the MTP transformer layer (same structure as a main layer)
        Layer *l = &mtp->layer;
        l->ffn_norm = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.ffn_norm.weight", mi) ? buf : buf);
        qt_load(S, &l->shared_w1, snprintf(buf, sizeof(buf), "mtp.%d.ffn.shared_experts.w1.weight", mi) ? buf : buf, 0, dim);
        qt_load(S, &l->shared_w2, snprintf(buf, sizeof(buf), "mtp.%d.ffn.shared_experts.w2.weight", mi) ? buf : buf, dim, 0);
        qt_load(S, &l->shared_w3, snprintf(buf, sizeof(buf), "mtp.%d.ffn.shared_experts.w3.weight", mi) ? buf : buf, 0, dim);
        qt_load(S, &l->gate, snprintf(buf, sizeof(buf), "mtp.%d.ffn.gate.weight", mi) ? buf : buf, m->c.n_experts, dim);
        l->gate_bias = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.ffn.gate.bias", mi) ? buf : buf);
        
        l->attn_norm = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.attn_norm.weight", mi) ? buf : buf);
        qt_load(S, &l->wq_a, snprintf(buf, sizeof(buf), "mtp.%d.attn.wq_a.weight", mi) ? buf : buf, m->c.q_lora_rank, dim);
        qt_load(S, &l->wq_b, snprintf(buf, sizeof(buf), "mtp.%d.attn.wq_b.weight", mi) ? buf : buf, 0, m->c.q_lora_rank);
        l->q_norm = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.attn.q_norm.weight", mi) ? buf : buf);
        qt_load(S, &l->wkv, snprintf(buf, sizeof(buf), "mtp.%d.attn.wkv.weight", mi) ? buf : buf, 0, dim);
        l->kv_norm = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.attn.kv_norm.weight", mi) ? buf : buf);
        qt_load(S, &l->wo_a, snprintf(buf, sizeof(buf), "mtp.%d.attn.wo_a.weight", mi) ? buf : buf, 0, (m->c.n_heads * m->c.head_dim) / m->c.o_groups);
        qt_load(S, &l->wo_b, snprintf(buf, sizeof(buf), "mtp.%d.attn.wo_b.weight", mi) ? buf : buf, dim, 0);
        l->attn_sink = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.attn.attn_sink", mi) ? buf : buf);
        
        // MTP layer hc weights
        l->hc_attn_base = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.hc_attn_base", mi) ? buf : buf);
        l->hc_attn_scale = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.hc_attn_scale", mi) ? buf : buf);
        qt_load(S, &l->hc_attn_fn, snprintf(buf, sizeof(buf), "mtp.%d.hc_attn_fn", mi) ? buf : buf, (2 + m->c.hc_mult) * m->c.hc_mult, m->c.hc_mult * dim);
        l->hc_ffn_base = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.hc_ffn_base", mi) ? buf : buf);
        l->hc_ffn_scale = fp32_load(S, snprintf(buf, sizeof(buf), "mtp.%d.hc_ffn_scale", mi) ? buf : buf);
        qt_load(S, &l->hc_ffn_fn, snprintf(buf, sizeof(buf), "mtp.%d.hc_ffn_fn", mi) ? buf : buf, (2 + m->c.hc_mult) * m->c.hc_mult, m->c.hc_mult * dim);
    }
}
