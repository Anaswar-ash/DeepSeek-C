# DeepSeek-C: The DeepSeek-V4-Flash Engine

A highly optimized, memory-mapped pure C engine designed exclusively to run the massive **DeepSeek-V4-Flash** (284B parameters) Mixture-of-Experts (MoE) model locally on consumer hardware.

This engine completely bypasses the traditional requirement for hundreds of gigabytes of VRAM or RAM. Instead, it streams 4-bit quantized model weights directly from your NVMe SSD into the CPU via `mmap`.

---

## 🧠 The Model: DeepSeek-V4-Flash

DeepSeek-V4-Flash is an incredibly sophisticated language model with **284 Billion total parameters**. 
Running a model of this size conventionally would require extreme enterprise hardware. However, it utilizes a **Mixture-of-Experts (MoE)** architecture.

In an MoE architecture, not all parameters are used for every word. Out of the 256 available "experts" in the model, only a small subset is routed and activated per token. DeepSeek-V4-Flash features:
- **43 Hidden Layers**
- **64 Attention Heads**
- **256 Routed Experts** (plus shared experts)
- **Multi-Token Prediction (MTP) Heads** for speculative decoding.

By compressing this model from FP8 down to INT4, we reduce its size from ~700 GB down to roughly **160 GB**. Because only a few experts activate at a time, we only need to load a fraction of that 160 GB into active memory during generation.

---

## ⚙️ How It Works: Core Engine Logic

DeepSeek-C is built on three core technical pillars that make running a 284B model possible on a standard desktop:

### 1. Zero-Copy Disk Streaming (`mmap`)
Instead of loading the 160GB model into RAM, DeepSeek-C uses the operating system's Memory-Mapped Files API (`mmap` on POSIX, `MapViewOfFile` on Windows). 
- The engine maps the massive `.safetensors` files directly into virtual memory.
- When the model evaluates a token, the router decides which experts are needed. The CPU then accesses those specific addresses.
- The OS automatically pages those specific expert blocks from the NVMe SSD into fast RAM. 
- The OS Page Cache acts as a dynamic L2 cache: frequently used experts stay resident in RAM, while unused ones are silently evicted. **Your SSD effectively becomes your VRAM.**

### 2. SIMD Vectorized Integer Kernels (AVX2)
The model weights are packed tightly into 4-bit integers (two weights per byte). DeepSeek-C implements custom quantized matrix-multiplication kernels in pure C using **AVX2/AVX-512 Intrinsics**.
- It unpacks the INT4 weights on-the-fly and multiplies them using integer-dot instructions (like `vpmaddubsw`).
- This allows the CPU to process hundreds of billions of operations per second across multiple OpenMP threads without relying on a GPU.

### 3. Asynchronous OpenAI-Compatible Server
The engine includes a built-in HTTP server powered by Mongoose. It spins up a threaded `/v1/chat/completions` endpoint that streams tokens back to the user via Server-Sent Events (SSE), making it instantly compatible with UI frontends like SillyTavern or AnythingLLM.

---

## 🔬 DeepSeek-V4 Specific Features Implemented

DeepSeek-V4 relies on complex architectural quirks that are fully implemented from scratch in this engine:

- **Manifold-Constrained Hyper-Connections (mHC):** Full mathematical implementation of the Sinkhorn-Knopp normalization used to stabilize V4's residual streams.
- **YaRN RoPE Interpolation:** Advanced multi-band Rotary Position Embeddings interpolation, allowing the model to recall information perfectly across massive context windows.
- **Lightning Indexer & Sparse Attention:** Compressed KV cache selection using multi-head indexer projections, allowing the model to route over 4000+ context blocks seamlessly.
- **Multi-Token Prediction (MTP):** DeepSeek-V4 includes a drafting head that predicts multiple tokens into the future. DeepSeek-C utilizes this for **Speculative Decoding**. It drafts tokens forward and verifies them in a single batched pass. *(Note: The MTP head is kept at INT8 precision, as compressing it to INT4 destroys the draft acceptance rate).*

---

## 💻 Hardware Requirements

Because the engine streams weights dynamically, **Storage I/O Speed is the primary bottleneck.**

- **Storage:** NVMe M.2 SSD is **strictly required**. (Speeds of 3000MB/s+ recommended). Traditional HDDs will result in less than 1 token per minute.
- **CPU:** Any modern x86-64 processor with AVX2 support (Intel 4th Gen / AMD Ryzen 1st Gen or newer).
- **RAM:** 16 GB minimum (to hold the KV Cache, OS buffers, and the dense attention layers).
- **GPU:** None required.

---

## 🚀 Quickstart

### 1. Compile the Engine
Ensure you have GCC installed (e.g., MinGW-w64 on Windows).
```bash
make dsv4
```
*This produces a single, dependency-free `dsv4.exe` binary.*

### 2. Download and Quantize the Model
We provide a memory-safe python tool that downloads the FP8 weights straight from HuggingFace, compresses them into INT4, and deletes the massive originals on-the-fly to save space.

*Requires Python 3.9+ and `pip install torch safetensors huggingface_hub numpy`*

```bash
# 1. Download the main 4-bit weights
python tools/convert_dsv4.py --repo deepseek-ai/DeepSeek-V4-Flash --outdir ./v4_int4

# 2. Download the 8-bit MTP speculative heads
python tools/convert_dsv4.py --repo deepseek-ai/DeepSeek-V4-Flash --outdir ./v4_int4 --mtp
```

### 3. Start Generating
Point the compiled executable to your newly converted model directory:
```bash
./dsv4.exe ./v4_int4
```
The server is now live at `http://localhost:8080/v1/chat/completions`!

---

## License
MIT License.
