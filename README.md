# DeepSeek-C

A highly optimized, memory-mapped C engine for running the massive **DeepSeek-V4-Flash** Mixture-of-Experts (MoE) model locally on consumer hardware.

This engine bypasses the need for hundreds of gigabytes of RAM by streaming 4-bit quantized model weights directly from disk into the CPU via `mmap`.

## Features
- **Pure C Engine:** No bloated dependencies, no Python runtime required for generation.
- **Memory Mapping (`mmap`):** Streams the 158B+ parameters directly from an SSD, requiring only a fraction of system RAM.
- **Manifold-Constrained Hyper-Connections (mHC):** Full mathematical implementation of the Sinkhorn-Knopp normalization used in V4's residual streams.
- **Lightning Indexer & Sparse Attention:** Accurate compressed KV cache selection using multi-head indexer projections, allowing the model to quickly route over 4000+ context blocks seamlessly.
- **Dual-Routing MoE:** Supports both hash-based early expert routing and sigmoid-based top-k routing.
- **Fast Multithreading:** AVX2/AVX-512 and OpenMP optimized matrix multiplication.
- **Built-in Quantization:** Includes a multi-threaded Python downloading script that shrinks official FP8 HuggingFace weights down to 4-bit integer format on the fly.

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

### 3. Run the Model
Once the conversion is complete, point the executable to your model directory and chat:
```bash
./dsv4.exe ./v4_int4 -p "Write a story about a brilliant software engineer."
```

## Architecture Notes
Built upon the [Colibri](https://github.com/vllm-project/colibri) minimalist engine architecture, heavily modified to support the DeepSeek-V4 attention mechanisms (CSA/HCA) and custom routing logits.

## License
MIT License.
