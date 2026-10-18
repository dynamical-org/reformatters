# Zarr V3 chunk and shard layout tool

Use [chunk_shard_size.py](../src/scripts/chunk_shard_size.py) to search layouts or compare chunk sizes, shard sizes, and access costs. Its code defines the heuristics; use `--help` for options.

```bash
uv run src/scripts/chunk_shard_size.py \
  --time 73000:3:hour --latitude 721:0.25:degrees --longitude 1440:0.25:degrees \
  --search
```

Dimensions accept `length:step:units` or just `length`. For a manual layout, replace `--search` with `--chunk_shape` and `--shard_shape`; shape entries follow the script's `VALID_DIMS` order, restricted to supplied dimensions.

Choose layouts for expected access patterns and memory limits. Shards must be integer multiples of chunks; forecast layouts keep one initialization per shard. Treat estimated compression and search recommendations as inputs to validation, not measured storage costs.
