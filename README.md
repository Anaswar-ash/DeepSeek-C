# DeepSeek-C

A highly optimized, memory-mapped C engine for running the massive **DeepSeek-V4-Flash** Mixture-of-Experts (MoE) model locally on consumer hardware.

This engine bypasses the need for hundreds of gigabytes of RAM by streaming 4-bit quantized model weights directly from disk into the CPU via `mmap`.

## Features
- **Pure C Engine:** No bloated dependencies, no Python runtime required for generation.
- **OpenAI-Compatible HTTP Server:** Built-in lightweight web server (via Mongoose) exposing `/v1/chat/completions` with Server-Sent Events (SSE) streaming support.
- **Native Windows MSVC Support:** Fully compatible with MSVC compilation, bypassing POSIX dependencies while securely memory mapping files via `MapViewOfFile`.
- **Memory Mapping (`mmap`):** Streams the 158B+ parameters directly from an SSD, requiring only a fraction of system RAM.
- **Manifold-Constrained Hyper-Connections (mHC):** Full mathematical implementation of the Sinkhorn-Knopp normalization used in V4's residual streams.
- **Advanced Context Extrapolation:** Fixed and fully implemented multi-band **YaRN RoPE** (Rotary Position Embeddings) interpolation for perfect long-context recall.
- **Multi-Token Prediction (MTP):** Includes speculative decoding drafts with segregated KV cache slot isolation to double generation speeds without cache corruption.
- **Lightning Indexer & Sparse Attention:** Accurate compressed KV cache selection using multi-head indexer projections, allowing the model to quickly route over 4000+ context blocks seamlessly.
- **Dual-Routing MoE:** Supports both hash-based early expert routing and sigmoid-based top-k routing.
- **Fast Multithreading:** AVX2/AVX-512 and OpenMP optimized matrix multiplication.

## Hardware Requirements
Because the engine streams the weights dynamically during text generation, storage speed is the primary bottleneck.
- **Storage:** NVMe M.2 SSD highly recommended (External Thunderbolt/USB-C SSDs like SanDisk G-DRIVE will also work). Traditional HDDs are not supported.
- **CPU:** Any modern x86-64 processor with AVX2 support (Intel 4th Gen / AMD Ryzen 1st Gen or newer).
- **RAM:** 16 GB minimum (to hold the KV Cache and OS buffers).
- **GPU:** None required.

## Quickstart

### 1. Compile the Engine
Make sure you have GCC (MinGW-w64 on Windows) installed.
```bash
make dsv4
```

### 2. Download and Quantize the Model
Use the provided python tool to download the FP8 weights straight from HuggingFace and compress them into INT4.
*Note: You need `safetensors` and `huggingface_hub` installed (`pip install safetensors huggingface_hub`).*

```bash
python tools/convert_fp8_to_int4.py --hf_url deepseek-ai/DeepSeek-V4-Flash --out_dir ./v4_int4
```
*(This will download roughly 150GB of data. Ensure you have adequate storage space!)*

### 3. Run the API Server
Once the conversion is complete, point the executable to your model directory to start the server:
```bash
./dsv4.exe ./v4_int4
```
This will launch the engine and open an OpenAI-compatible API on `http://localhost:8080`. You can now stream generations via `POST /v1/chat/completions`.

## Architecture Notes
Built upon the [Colibri](https://github.com/vllm-project/colibri) minimalist engine architecture, heavily modified to support the DeepSeek-V4 attention mechanisms (CSA/HCA) and custom routing logits.

## License
MIT License.
