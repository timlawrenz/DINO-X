# Notes: ROCm/PyTorch allocator fragmentation

If you see errors like:

- `HIP out of memory ... reserved but unallocated memory is large`

Try restarting the Python process and consider enabling allocator behavior that reduces fragmentation:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Despite the name, `PYTORCH_CUDA_ALLOC_CONF` is used by PyTorch on both CUDA and ROCm builds.
