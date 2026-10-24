# moonhnsw

Pure MoonBit approximate nearest-neighbor search: a local vector index built on the HNSW graph (Malkov & Yashunin, 2016) — the same algorithm inside [hnswlib](https://github.com/nmslib/hnswlib), [usearch](https://github.com/unum-cloud/usearch) and Qdrant.

Feed in `f32` vectors, get back the approximate top-k nearest neighbors. No FFI, no system dependencies — the whole index is MoonBit, so it runs on **native, wasm and js** backends alike.

## Features

- **HNSW graph index** with incremental `add`, deterministic `search(top-k)`, approximate `search_radius`, and binary `save` / `load`.
- **Tombstone deletion**: `delete(id)` hides vectors from all searches while keeping the graph connected; `compact()` rebuilds a tombstone-free index and hands you an old-id → new-id mapping.
- **Introspection**: `stats()` reports per-layer node counts, average degree and a rough memory estimate; `get_vector(id)` and `contains(id)` expose the stored data.
- **External labels**: `add_labeled(vector, label)` tags each vector with a `UInt64` key (a document or chunk id); labels persist through `save` / `load` and `compact`.
- **Composable**: `absorb()` merges another index (keeping labels), `copy()` deep-clones, and `save_bytes()` / `load_bytes()` wrap the payload as `Bytes` for binary pipelines.
- **Three metrics** with a smaller-is-closer convention: `L2` (squared Euclidean), `InnerProduct` (negated dot product), `Cosine` (vectors normalized once on insertion).
- **Tunable** `m`, `ef_construction`, and per-query `ef` — the standard recall/speed trade-off knobs.
- **Zero-copy friendly storage**: vectors live in one flattened `FixedArray[Float]`; traversal marks visited nodes with generation stamps instead of hash sets; distances accumulate in f32 like hnswlib.
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
| `Hnsw::add_labeled(vector, label) -> Int` | Insert tagged with a `UInt64` label of your choice (need not be unique). |
| `Hnsw::label_of(id) -> UInt64` | Label stored for an id; defaults to the id itself. |
| `Hnsw::search(query, k, ef~ = 64) -> Array[Neighbor]` | Approximate top-k, sorted by distance ascending. |
| `Hnsw::search_radius(query, radius, max_results~ = 64)` | All neighbors within a distance bound, ascending; approximate like top-k. |
| `Hnsw::delete(id)` | Tombstone-delete a vector; it disappears from every search. |
| `Hnsw::contains(id) -> Bool` | True when `id` is stored and not deleted. |
| `Hnsw::get_vector(id) -> FixedArray[Float]` | Copy of the stored vector (normalized for Cosine). |
| `Hnsw::stats() -> HnswStats` | Per-layer node counts and average degree, for tuning. |
| `Hnsw::compact() -> CompactResult` | Tombstone-free rebuild plus old-id → new-id mapping. |
| `Hnsw::absorb(other) -> FixedArray[Int]` | Merge another index's live vectors (labels kept); returns old-id → new-id mapping. |
| `Hnsw::copy() -> Hnsw` | Independent deep clone, tombstones and labels included. |
| `Hnsw::ids() -> Array[Int]` | Ascending ids of all non-deleted vectors. |
| `Hnsw::save() -> FixedArray[Byte]` | Serialize the full index. |
| `Hnsw::load(bytes) -> Hnsw` | Restore an index; rejects malformed payloads with `CorruptData`. |
| `Hnsw::save_bytes() / load_bytes(Bytes)` | Same payload wrapped as immutable `Bytes`. |
| `Hnsw::size() -> Int` | Number of non-deleted (searchable) vectors. |
| `Hnsw::total() -> Int` | Number of vectors ever inserted, including tombstones. |
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

| Metric | n | Build | ef=64 | ef=128 | ef=256 |
| --- | --- | --- | --- | --- | --- |
| L2 | 2 000 | 1014 ms | recall 1.00, 5003 q/s | recall 1.00, 3253 q/s | recall 1.00, 1829 q/s |
| InnerProduct | 2 000 | 1281 ms | recall 1.00, 6224 q/s | recall 1.00, 3781 q/s | recall 1.00, 2250 q/s |
| Cosine | 2 000 | 965 ms | recall 1.00, 5168 q/s | recall 1.00, 3225 q/s | recall 1.00, 1939 q/s |
| L2 (scale) | 10 000 | 9554 ms | recall 0.97, 3003 q/s | recall 1.00, 1328 q/s | recall 1.00, 761 q/s |
| L2 (large) | 100 000 | 184.0 s | recall 0.83, 1992 q/s | recall 0.95, 885 q/s | recall 1.00, 550 q/s |

At 10k vectors the ef trade-off becomes visible: `ef=64` reaches 0.97 recall, `ef=128` already 1.00. At 100k the graph is 4 layers tall (~30 MB estimated); random data needs `ef=256` for exact recall — raise `m` if you want the same recall at lower `ef`.

Recall is verified against exact brute-force ground truth in the test suite for all three metrics (recall@10 >= 0.9 on random datasets).

Run it yourself:

```
moon run --target wasm-gc examples/benchmark
moon run --target wasm-gc examples/benchmark-large
```

## Implementation notes

- The graph is a hierarchy of proximity subgraphs. Every node lives on layer 0; each node's top layer is drawn from a geometric distribution, so upper layers hold roughly `1/m` of the nodes below them:

```
layer 2   e──────────────              (entry, a handful of nodes)
layer 1   a───b─────g───────           (greedy descent)
layer 0   a─b─c─d─e─f─g─h─i─...        (all nodes, beam search)
```

  Insertion draws the node's top layer (SplitMix64 PRNG, seeded), descends greedily from the entry point, then runs a beam search (`ef_construction`) per layer and connects via the diversity heuristic (Algorithm 4 of the paper).
- New nodes keep `m` links per layer; overflowing neighbor lists of existing nodes are re-selected with the same heuristic.
- Beam search uses a binary heap for candidates (min) and results (max), and an O(1) generation-stamp array for the visited set.
- Distance accumulation stays in `f32` (as in hnswlib): inputs are f32 and only comparisons matter, so skipping per-element f64 widening is measurably faster on every backend.
- Storage layout: one flattened `FixedArray[Float]` of `count * dim` values, doubled geometrically; graph edges as `Array[Array[Array[Int]]]`.
- Serialization is little-endian with a magic header and version byte; every length and id is validated on load.

## Limitations & roadmap

- Deletion is tombstone-based: deleted nodes keep occupying memory (ids and graph slots are never reclaimed) and in-place vector update is not supported.
- Radius search shares the graph-approximation of top-k search: a neighbor inside the radius can be missed when it is only reachable through farther nodes.
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
