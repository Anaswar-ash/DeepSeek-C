# Changelog

All notable changes to the DeepSeek-C engine will be documented in this file.

## [Unreleased]

## [v0.4.0] - 2026-07-23

### Added
- **OpenAI-Compatible API Server:** Integrated the Mongoose HTTP server framework to replace the CLI interface. Exposes `/v1/chat/completions` with full support for Server-Sent Events (SSE) streaming and real-time generation.
- **Native MSVC Support:** Rewrote OS compatibility layers in `compat.h` to fully support Windows compilation. Handled `off_t` file size truncation, POSIX `O_BINARY` CRT byte translation, and Windows-native memory mapping (`MapViewOfFile`).

### Fixed
- **Multi-Token Prediction (MTP) KV Cache Corruption:** Segregated the drafting phase KV allocations to prevent the MTP layer from overwriting the main transformer's memory space, restoring generation coherence.
- **YaRN RoPE Mathematical Accuracy:** Replaced the legacy linear RoPE interpolation with true DeepSeek multi-band YaRN scaling, correctly resolving high/low frequency boundary distortions in long-context queries.
- **Arena Memory Leaks:** Enforced strict `arena_reset()` boundaries during MTP drafts and JSON request parsing to prevent buffer overflows across sequential generation turns.

## [v0.3.0] - 2026-07-20

### Added
- **True Lightning Indexer Support:** Replaced the sequential mock block selector with the actual mathematical `matmul_qt` dot-product implementation of the DeepSeek-V4 Lightning Indexer. The model now correctly queries the multi-head compressed latent space, eliminating repetition loops and generating highly coherent text.
- **Surgical Weight Extraction:** Updated the python converter to use HTTP Range requests, enabling the extraction of the ~5 GB Lightning Indexer `idx_wq_b` and `idx_wproj` matrices without needing to download the entire 756 GB model repository.

## [v0.2.0] - 2026-07-16

### Added
- **OpenMP Parallelization:** Accelerated matrix multiplications (`matmul` and `matmul_qt`) and multi-expert routing steps using `#pragma omp parallel for`.
- **Memory Arena Allocator:** Replaced slow heap allocations (`falloc`) with a fast, static bump-pointer arena for per-token generation loops, effectively eliminating memory leaks and segmentation faults during decoding.
- **Dynamic KV-Caching:** Overhauled the key-value cache system to support dynamic sliding windows instead of static linear arrays.

## [v0.1.0] - 2026-07-10

### Added
- **DeepSeek-V4-Flash Structural Support:** Added initial structure maps for Manifold-Constrained Hyper-Connections (mHC), Compressed Sparse Attention (CSA), and Heavily Compressed Attention (HCA).
- **INT4 Quantization:** Introduced the foundational Python script to stream `FP8` weights from HuggingFace and compile them into memory-mapped `INT4` safetensors.
