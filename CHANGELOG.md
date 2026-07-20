# Changelog

All notable changes to the DeepSeek-C engine will be documented in this file.

## [Unreleased]

### Added
- **Multi-Token Prediction (MTP) Support** (Upcoming in Phase 4): Framework planned for speculative multi-token decoding to dramatically improve generation speed.

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
