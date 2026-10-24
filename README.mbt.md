# moonhnsw

Pure MoonBit approximate nearest-neighbor search: a local vector index built on the HNSW graph (Malkov & Yashunin, 2016) — the same algorithm inside [hnswlib](https://github.com/nmslib/hnswlib), [usearch](https://github.com/unum-cloud/usearch) and Qdrant.

Feed in `f32` vectors, get back the approximate top-k nearest neighbors. No FFI, no system dependencies — the whole index is MoonBit, so it runs on **native, wasm and js** backends alike.

## Features

- **HNSW graph index** with incremental `add`, deterministic `search(top-k)`, and binary `save` / `load`.
- **Three metrics** with a smaller-is-closer convention: `L2` (squared Euclidean), `InnerProduct` (negated dot product), `Cosine` (vectors normalized once on insertion).
- **Tunable** `m`, `ef_construction`, and per-query `ef` — the standard recall/speed trade-off knobs.
- **Zero-copy friendly storage**: vectors live in one flattened `FixedArray[Float]`; traversal marks visited nodes with generation stamps instead of hash sets.
- **Self-contained serialization**: one `save()` call produces a byte buffer you can write anywhere; `load()` restores an index that answers searches identically and keeps accepting insertions.
- **Typed errors** (`HnswError`) with readable messages.

## Quick start

```mbt nocheck
let index = @moonhnsw.Hnsw::new(8, @moonhnsw.Distance::Cosine)

// Insert vectors (they are copied into the index); ids are 0, 1, 2, ...
let v : FixedArray[Float] = FixedArray::from_array([
  0.1, 0.2, 0.0, 0.4, 0.1, 0.0, 0.9, 0.3,
])
let id = index.add(v)

// Approximate top-10; larger `ef` = higher recall, slower query.
let hits : Array[@moonhnsw.Neighbor] = index.search(query, 10, ef=64)
for hit in hits {
  println("id=\{hit.id} distance=\{hit.distance}")
}

// Persist and restore.
let bytes : FixedArray[Byte] = index.save()
let restored = @moonhnsw.Hnsw::load(bytes)
```

Add the dependency to your `moon.mod`:

```
moon add yiwnagmolu/moonhnsw
```

## API

| Signature | Description |
| --- | --- |
| `Hnsw::new(dim, distance, params~)` | Create an empty index over `dim`-dimensional vectors. |
| `Hnsw::add(vector) -> Int` | Insert one vector, returns its id. Raises on dimension mismatch. |
| `Hnsw::search(query, k, ef~ = 64) -> Array[Neighbor]` | Approximate top-k, sorted by distance ascending. |
| `Hnsw::save() -> FixedArray[Byte]` | Serialize the full index. |
| `Hnsw::load(bytes) -> Hnsw` | Restore an index; rejects malformed payloads with `CorruptData`. |
| `Hnsw::size() -> Int` | Number of stored vectors. |
| `distance(metric, a, b) -> Double` | Metric value between two vectors — handy for brute-force ground truth. |
| `HnswError::message() -> String` | Human-readable error text. |

Distance conventions (smaller = closer):

- `L2` — squared Euclidean distance.
- `InnerProduct` — negated inner product.
- `Cosine` — negated inner product after L2 normalization; the index normalizes stored vectors once and each query on entry. Zero vectors are kept as-is.

## Parameters

| Parameter | Default | Guidance |
| --- | --- | --- |
| `m` | 16 | Links per node on upper layers (layer 0 keeps `2m`). 12–48 is the usual range; higher = better recall, more memory. |
| `ef_construction` | 200 | Beam width while inserting. 100–500; higher builds a better graph. |
| `ef` (per search) | 64 | Beam width at query time. `ef >= k` is enforced; 64–512 covers most workloads. |

## Benchmarks

Measured with the bundled `examples/benchmark` on uniform random data (n=2000, dim=32, m=16, ef_construction=200, 100 queries, k=10), executed on the **wasm-gc** backend via `moonrun` on Windows — a conservative setting; the native backend is expected to be substantially faster.

| Phase | Result |
| --- | --- |
| Build | 942 ms total (~471 us per vector) |
| Search, ef=64 | recall@10 = **1.00**, 5530 queries/s |
| Search, ef=128 | recall@10 = **1.00**, 3426 queries/s |
| Search, ef=256 | recall@10 = **1.00**, 2072 queries/s |

Recall is verified against exact brute-force ground truth in the test suite for all three metrics (recall@10 >= 0.9 on random datasets).

Run it yourself:

```
moon run --target wasm-gc examples/benchmark
```

## Implementation notes

- Multi-layer graph where each node's top layer is drawn from a geometric distribution (SplitMix64 PRNG, seeded); insertion descends greedily from the entry point, then runs a beam search (`ef_construction`) per layer and connects via the diversity heuristic (Algorithm 4 of the paper).
- New nodes keep `m` links per layer; overflowing neighbor lists of existing nodes are re-selected with the same heuristic.
- Beam search uses a binary heap for candidates (min) and results (max), and an O(1) generation-stamp array for the visited set.
- Distance accumulation stays in `f32` (as in hnswlib): inputs are f32 and only comparisons matter, so skipping per-element f64 widening is measurably faster on every backend.
- Storage layout: one flattened `FixedArray[Float]` of `count * dim` values, doubled geometrically; graph edges as `Array[Array[Array[Int]]]`.
- Serialization is little-endian with a magic header and version byte; every length and id is validated on load.

## Limitations & roadmap

- No deletion or vector update (classic HNSW limitation) — planned as tombstone-based deletion.
- Single-threaded; indexes are not to be shared across concurrent writers.
- Ids are internal insertion order; map them to your own keys in application code.

## Development

```
moon test                 # run the test suite
moon test --target wasm-gc
moon info && moon fmt     # refresh generated interfaces, format
moon run --target wasm-gc examples/benchmark
```

> On Windows the native backend additionally requires MSVC Build Tools (`cl.exe` or `clang-cl`); wasm-gc and js targets work out of the box.

## References

- Malkov, Yashunin. *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. TPAMI 2018 (arXiv:1603.09320).
- [hnswlib](https://github.com/nmslib/hnswlib) and [usearch](https://github.com/unum-cloud/usearch) for parameter conventions.

## License

MIT
