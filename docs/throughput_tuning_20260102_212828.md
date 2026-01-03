# Throughput Tuning Results (2026-01-02 21:28:28)

Run directory:
- `data/runs/throughput_tuning/20260102_212828/`

## Outcome

This run hit an out-of-memory condition immediately (first tested combo):

```
batch_size=8 num_workers=8 pin_memory=True
status=oom
```

Error excerpt:
- `OutOfMemoryError: HIP out of memory. Tried to allocate 50.00 MiB...`
- `GPU 0 ... total capacity 110.00 GiB ... 4.54 MiB is free`
- `allocated by PyTorch: 26.23 GiB; reserved but unallocated: 840.88 MiB`

## Interpretation

This is consistent with the tuner being run in a **large-model configuration** (e.g., ViT-Giant-like) without the same memory savers planned for Phase 5 (mixed precision, FlashAttention/efficient attention, and gradient checkpointing).

Also note: the error indicates most device memory is consumed by something *other than* PyTorch's tracked allocations. This can happen if:
- another process is holding large VRAM/unified memory allocations
- runtime/caching allocations are large
- fragmentation prevents satisfying a contiguous allocation

## Next Step

Re-run the tuner after ensuring:
- no other training processes are holding VRAM
- mixed precision is enabled for the benchmark run
- allocator fragmentation mitigation is enabled if needed (PyTorch suggests `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)
