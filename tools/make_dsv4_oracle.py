import torch
import torch.nn.functional as F
import math

# --- 1. Sinkhorn C-emulation test ---
def c_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, iters=20, eps=1e-6):
    b, s, _ = mixes.size()
    mixes = mixes.view(-1, (2 + hc_mult) * hc_mult)
    n = mixes.size(0)
    
    pre = torch.zeros(n, hc_mult)
    post = torch.zeros(n, hc_mult)
    comb = torch.zeros(n, hc_mult, hc_mult)
    
    for i in range(n):
        mix = mixes[i]
        # pre
        for j in range(hc_mult):
            pre[i, j] = torch.sigmoid(mix[j] * hc_scale[0] + hc_base[j]) + eps
        
        # post
        for j in range(hc_mult):
            post[i, j] = 2.0 * torch.sigmoid(mix[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult])
            
        # comb init
        for j in range(hc_mult):
            for k in range(hc_mult):
                idx = j * hc_mult + k + hc_mult * 2
                comb[i, j, k] = mix[idx] * hc_scale[2] + hc_base[idx]
        
        # Sinkhorn iter
        c = comb[i]
        # row max
        row_max = c.max(dim=-1, keepdim=True)[0]
        c = torch.exp(c - row_max)
        c = c / c.sum(dim=-1, keepdim=True) + eps
        c = c / (c.sum(dim=-2, keepdim=True) + eps)
        
        for _ in range(iters - 1):
            c = c / (c.sum(dim=-1, keepdim=True) + eps)
            c = c / (c.sum(dim=-2, keepdim=True) + eps)
            
        comb[i] = c
        
    return pre.view(b, s, hc_mult), post.view(b, s, hc_mult), comb.view(b, s, hc_mult, hc_mult)

def torch_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, iters=20, eps=1e-6):
    b, s, _ = mixes.size()
    mixes = mixes.view(-1, (2 + hc_mult) * hc_mult)
    
    # Original PyTorch logic from kernel.py (TileLang kernel translated to torch)
    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(mixes[:, hc_mult:2*hc_mult] * hc_scale[1] + hc_base[hc_mult:2*hc_mult])
    
    comb = mixes[:, 2*hc_mult:] * hc_scale[2] + hc_base[2*hc_mult:]
    comb = comb.view(-1, hc_mult, hc_mult)
    
    # Sinkhorn
    row_max = comb.max(dim=-1, keepdim=True)[0]
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        
    return pre.view(b, s, hc_mult), post.view(b, s, hc_mult), comb.view(b, s, hc_mult, hc_mult)

def test_sinkhorn():
    print("Testing Sinkhorn C-emulation...")
    hc_mult = 4
    b, s = 2, 4
    mixes = torch.randn(b, s, (2 + hc_mult) * hc_mult)
    hc_scale = torch.randn(3)
    hc_base = torch.randn((2 + hc_mult) * hc_mult)
    
    pre_t, post_t, comb_t = torch_hc_split_sinkhorn(mixes, hc_scale, hc_base)
    pre_c, post_c, comb_c = c_hc_split_sinkhorn(mixes, hc_scale, hc_base)
    
    err_pre = (pre_t - pre_c).abs().max()
    err_post = (post_t - post_c).abs().max()
    err_comb = (comb_t - comb_c).abs().max()
    
    print(f"Max diff PRE:  {err_pre:.6e}")
    print(f"Max diff POST: {err_post:.6e}")
    print(f"Max diff COMB: {err_comb:.6e}")
    assert err_pre < 1e-5 and err_post < 1e-5 and err_comb < 1e-5
    print("Sinkhorn TEST PASSED!")

def main():
    test_sinkhorn()

if __name__ == "__main__":
    main()
