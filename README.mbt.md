# moonhnsw

[![CI](https://github.com/yiwnagmolu/moonhnsw/actions/workflows/ci.yml/badge.svg)](https://github.com/yiwnagmolu/moonhnsw/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.12.4-blue)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Pure MoonBit approximate nearest-neighbor search: a local vector index built on the HNSW graph (Malkov & Yashunin, 2016) — the same algorithm inside [hnswlib](https://github.com/nmslib/hnswlib), [usearch](https://github.com/unum-cloud/usearch) and Qdrant.

Feed in `f32` vectors, get back the approximate top-k nearest neighbors. No FFI, no system dependencies — the whole index is MoonBit, so it runs on **native, wasm and js** backends alike.

## Features

- **HNSW graph index** with incremental `add`, deterministic `search(top-k)`, approximate `search_radius`, and binary `save` / `load`.
- **Tombstone deletion**: `delete(id)` hides vectors from all searches while keeping the graph connected; `compact()` rebuilds a tombstone-free index and hands you an old-id → new-id mapping.
- **Introspection**: `stats()` reports per-layer node counts, average degree and a rough memory estimate; `get_vector(id)` and `contains(id)` expose the stored data; `export_dot(layer)` renders any layer as Graphviz DOT; `Neighbor` and `HnswStats` derive `Eq` / `Debug` / `ToJson`, so results drop straight into assertions, logs and JSON pipelines.
- **External labels**: `add_labeled(vector, label)` tags each vector with a `UInt64` key (a document or chunk id); labels persist through `save` / `load` and `compact`.
- **Composable**: `absorb()` merges another index (keeping labels), `copy()` deep-clones, and `save_bytes()` / `load_bytes()` wrap the payload as `Bytes` for binary pipelines.
- **SQ8 scalar quantization**: `storage=SQ8` learns per-dimension bounds from the first 256 vectors and then stores one byte per dimension. The vector store is about 4x smaller; retrieval quality depends on the data. Stored vectors added after freezing clamp to the learned range; queries keep their original values when distances are calculated.
- **MMR diverse search**: `search_mmr(query, k, lambda)` re-ranks a wide candidate pool for relevance-vs-redundancy, the de-duplication step RAG contexts need.
- **Empirical ef calibration**: `calibrate_ef` measures recall@k against an exact scan of representative queries and recommends the smallest tested beam width meeting your target.
- **Adaptive filtered retrieval**: `search_filtered_adaptive` widens an unfiltered candidate pool until it contains `k` filter matches or reaches a configured cap; it reports rounds and distance work.
- **Guaranteed exact retrieval**: `search_exact` and `search_radius_exact` scan the live stored vectors with stable distance/id ties. Use them for small collections, strict completeness, and a trustworthy recall baseline; SQ8 results are exact over its quantized store.
- **Integrity-checked persistence**: payloads carry a CRC-32 trailer, so flipped bits and silent truncation are rejected on load instead of surfacing as degraded searches.
- **Three metrics** with a smaller-is-closer convention: `L2` (squared Euclidean), `InnerProduct` (negated dot product), `Cosine` (vectors normalized once on insertion).
- **Tunable** `m`, `ef_construction`, and per-query `ef` — the standard recall/speed trade-off knobs.
- **Allocation-aware storage**: Float32 vectors live in one flattened `FixedArray[Float]`, while frozen SQ8 vectors live in a byte array; traversal marks visited nodes with generation stamps instead of hash sets. Common Float32 distances accumulate in f32, with a Double fallback for extreme finite values.
- **Self-contained serialization**: one `save()` call produces a byte buffer you can write anywhere; `load()` restores an index that answers searches identically and keeps accepting insertions.
- **Typed errors** (`HnswError`) with readable messages.
- **Finite-input checks**: NaN and infinite vectors/queries are rejected before graph mutation; loading also validates stored values after checksum verification.

## Quick start

The module is available on [Mooncakes](https://mooncakes.io/docs/yiwnagmolu/moonhnsw/). To start from an empty project, run `moon new moonhnsw-demo` and `cd moonhnsw-demo`. Then add the published version:

```sh
moon add yiwnagmolu/moonhnsw@0.12.4
```

In the `moon.pkg` of the package using the library (for example, `cmd/main/moon.pkg`), add:

```moon.pkg
import {
  "yiwnagmolu/moonhnsw" @moonhnsw,
}

pkgtype(kind: "executable")
```

Then put this complete example in `cmd/main/main.mbt` and run `moon run --target wasm-gc cmd/main` (or choose another installed target):

```mbt nocheck
///|
fn main raise {
  let index = @moonhnsw.Hnsw::new(2, @moonhnsw.Distance::L2)
  let vector : FixedArray[Float] = FixedArray::from_array([1.0, 2.0])
  let _ = index.add(vector)
  let hits = index.search(vector, 1, ef=64)
  println("nearest id=\{hits[0].id} distance=\{hits[0].distance}")
}
```

This install path was checked in a fresh project against the published Mooncakes package. For larger examples of labels, filtered search and persistence, see the API table below and the repository tests.

## API

| Signature | Description |
| --- | --- |
| `Hnsw::new(dim, distance, params~)` | Create an empty index over `dim`-dimensional vectors. |
| `Hnsw::add(vector) -> Int` | Insert one vector, returns its id. Raises on dimension mismatch. |
| `Hnsw::add_labeled(vector, label) -> Int` | Insert tagged with a `UInt64` label of your choice (need not be unique). |
| `Hnsw::label_of(id) -> UInt64` | Label stored for an id; defaults to the id itself. |
| `Hnsw::upsert_labeled(vector, label) -> Int` | Insert-or-replace by label: the document-update flow. |
| `Hnsw::find_by_label(label) -> Array[Int]` | Ids carrying a label (labels may repeat). |
| `Hnsw::search_batch(queries, k, ef~)` | One call for a list of queries. |
| `Hnsw::search(query, k, ef~ = 64) -> Array[Neighbor]` | Approximate top-k, sorted by distance ascending. |
| `Hnsw::search_exact(query, k, filter~) -> Array[Neighbor]` | Exhaustive top-k over live stored vectors, distance/id tie order. |
| `Hnsw::search_filtered(query, k, ef~, filter)` | Top-k restricted by a filter closure over ids (metadata filtering). |
| `Hnsw::search_filtered_adaptive(query, k, ef_start~, ef_max~, filter)` | Grow a post-filter candidate pool until k matches or the cap; returns neighbors, final ef, rounds, and distance work. |
| `Hnsw::calibrate_ef(queries, k, target_recall, efs)` | Measure recall and mean distance work at each ef; select the smallest ef meeting the target. |
| `Hnsw::search_mmr(query, k, lambda~, ef~)` | Maximal Marginal Relevance; lambda=1 keeps relevance, lambda=0 prioritizes diversity after the first result. |
| `Hnsw::search_radius(query, radius, max_results~, filter~)` | Radius search; composes with the filter closure. |
| `Hnsw::search_radius_exact(query, radius, max_results~, filter~)` | Exhaustive radius search with a completeness guarantee within the result limit. |
| `Hnsw::shrink_to_fit()` | Release store slack without rebuilding the graph. |
| `Hnsw::from_vectors(dim, distance, vectors, params~)` | One-call construction from a list of vectors. |
| `Hnsw::delete(id)` | Tombstone-delete a vector; it disappears from every search. |
| `Hnsw::contains(id) -> Bool` | True when `id` is stored and not deleted. |
| `Hnsw::get_vector(id) -> FixedArray[Float]` | Copy of the stored vector (normalized for Cosine). |
| `Hnsw::stats() -> HnswStats` | Per-layer node counts and average degree, for tuning. |
| `Hnsw::compact() -> CompactResult` | Tombstone-free rebuild plus old-id → new-id mapping. |
| `Hnsw::absorb(other) -> FixedArray[Int]` | Merge another index's live vectors (labels kept); returns old-id → new-id mapping. |
| `Hnsw::copy() -> Hnsw` | Independent deep clone, tombstones and labels included. |
| `Hnsw::ids() -> Array[Int]` | Ascending ids of all non-deleted vectors. |
| `Hnsw::export_dot(layer) -> String` | Graphviz DOT of one graph layer, for visualization. |
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
- `Cosine` — negated inner product after L2 normalization; the index normalizes stored vectors once and each query on entry, and `distance(Cosine, a, b)` normalizes both arguments. Zero vectors are kept as-is.

For `InnerProduct` and `Cosine`, useful radius thresholds can be negative. For
example, `search_radius(query, -0.8)` with Cosine asks for similarity at least
0.8. `L2` radius thresholds must be nonnegative.

## Choosing search width

Given representative queries, measure a recall/cost curve and use the chosen
width for later queries against the same index state:

```mbt nocheck
///|
let calibration = index.calibrate_ef(sample_queries, 10, 0.95, [16, 32, 64, 128])

///|
let hits = index.search(query, 10, ef=calibration.selected_ef)
```

Calibration compares each trial with an exact scan of live stored vectors. It
is an offline tuning step, not a per-query operation; SQ8 calibration uses
the same quantized distance as SQ8 search. The measured recall applies only
to the supplied queries and current index. Check `calibration.met_target`
before treating `selected_ef` as meeting your target.

For a restrictive filter, `search_filtered_adaptive` tries a small nearest
candidate pool first and doubles its width until enough matches are present:

```mbt nocheck
///|
let result = index.search_filtered_adaptive(
  query,
  10,
  ef_start=32,
  ef_max=512,
  filter=fn(id) { id % 2 == 0 },
)

///|
let hits = result.neighbors
```

This applies the filter after graph search. It may return fewer than `k`
neighbors at the cap, and its results are approximate. The result records
`final_ef`, `rounds`, and `distance_computations` for tuning. The existing
`search_filtered` applies the filter during traversal and can explore more
of the graph for selective filters.

When a missing neighbor is unacceptable, use `search_exact` or
`search_radius_exact`. Both scan every live vector that passes the filter;
their cost grows linearly with collection size. `search_radius` and the
adaptive family remain approximate. For SQ8, exact means exact relative to
the quantized values stored by the index.

## Parameters

| Parameter | Default | Guidance |
| --- | --- | --- |
| `m` | 16 | Links per node on upper layers (layer 0 keeps `2m`). 12–48 is the usual range; higher = better recall, more memory. |
| `ef_construction` | 200 | Beam width while inserting. 100–500; higher builds a better graph. |
| `ef` (per search) | 64 | Beam width at query time. `ef >= k` is enforced; 64–512 covers most workloads. |
| `storage` | `Float32` | `SQ8` stores one byte per dimension after learning bounds from the first 256 vectors (~4x smaller vector store, data-dependent recall); later inserted values clamp to those bounds, while queries are not clamped. |

## Benchmarks

The reproducible [real-data benchmark](docs/real-benchmarks.md) uses held-out
queries from four UCI datasets (11–64 dimensions) and the JS release backend,
covering L2, InnerProduct and Cosine. On this Windows Ryzen 7 5800H run of
v0.12.4, L2 Float32 at `ef=64` reached 7,172 QPS on 7,494 pen-digit vectors
(exact scan: 847 QPS), 5,041 QPS on 16,000 letter vectors (exact: 332 QPS),
and 2,596 QPS on 3,823 64-dimensional optical-digit vectors (exact: 472 QPS).
Tie-aware recall@10 was 1.000 on those three Float32 workloads; letter strict
ID overlap was 0.9445 because many vectors tie at the cutoff distance. SQ8
used less serialized space with data-dependent recall: at `ef=64`, its
tie-aware recall was 0.9940 on pen digits and letters, 0.9615 on Wine Quality,
and 0.9955 on optical digits. These are median-of-three short query runs, not
a general throughput guarantee. The 64-dimensional image features are not a
proxy for high-dimensional text embeddings. The report includes all ef widths,
metrics, source hashes and raw logs.
The [acceptance self-audit](docs/acceptance-audit.md) maps the original project
proposal to code, tests, benchmark evidence and remaining measurement limits.

The following table is historical context from the bundled `examples/benchmark` on uniform random data (dim=32, m=16, ef_construction=200, k=10), executed on the **wasm-gc** backend via `moonrun` on Windows. The 2k and 10k rows were rerun for v0.11.0 on 2026-10-28 (100 queries at 2k, 50 at 10k); the 100k row is an earlier v0.10.0 manual run. Its SQ8 row predates the v0.12.4 distance correction and is not a current-version quality or speed claim. Wall-clock numbers vary by run and machine. Speedups use a fair single-pass brute-force top-k in the same process; `dists/query` counts distance computations.

| Metric | n | Build | ef=64 | ef=128 | ef=256 |
| --- | --- | --- | --- | --- | --- |
| L2 | 2 000 | 577 ms | recall 1.00, 8297 q/s (0.49x brute, 980 dists) | recall 1.00, 5049 q/s (0.30x) | recall 1.00, 2903 q/s (0.17x) |
| InnerProduct | 2 000 | 676 ms | recall 1.00, 10809 q/s (0.58x brute, 891 dists) | recall 1.00, 6028 q/s (0.32x) | recall 1.00, 3205 q/s (0.17x) |
| Cosine | 2 000 | 488 ms | recall 1.00, 9099 q/s (0.49x brute, 970 dists) | recall 1.00, 5174 q/s (0.28x) | recall 1.00, 3197 q/s (0.17x) |
| L2 (scale) | 10 000 | 5290 ms | recall 0.97, 5669 q/s (1.50x brute, 1390 dists) | recall 1.00, 3164 q/s (0.84x) | recall 1.00, 1435 q/s (0.38x) |
| L2 (large) | 100 000 | 210 s | recall 0.83, 2041 q/s (**13.9x brute**, 1687 dists) | recall 0.95, 872 q/s (6.0x) | recall 1.00, 481 q/s (3.3x) |
| L2 SQ8 | 2 000 | 713 ms | recall 0.99, 6297 q/s (0.56x brute) | recall 0.99, 4101 q/s (0.36x) | recall 0.99, 2711 q/s (0.24x) |

**Read the speedup column honestly:** on these runs, a tight brute-force loop wins at 2k vectors and ef=64 crosses over around 10k. In the earlier 100k run, the graph touched **1.7% of the dataset** per query (1687 vs 100000 distances) for 0.83 recall and measured 13.9x brute-force throughput. That older wall-clock result should be remeasured before comparing it with v0.11.0.

The benchmark now also reports MMR query throughput, total distance work,
and an ordered-result checksum. In a v0.11.1 L2 run with 2,000 vectors,
`k=10`, and `ef=128`, MMR distance work fell from 6,862.94 to 2,494.94
per query. Its ordered-result checksum stayed the same across L2, inner
product, cosine, 10k L2, and SQ8 runs. These are short, fixed-seed wasm-gc
measurements; rerun them on your target before relying on wall-clock figures.

Since v0.11.2, the same example also times 40,000 calls to the public
`distance(Cosine, a, b)` helper and prints a distance checksum. The helper
normalizes with the same f32 rounding as before while avoiding two temporary
arrays per call. Insertion also avoids redundant vector copies; the bundled
build timings are too variable to assign it a reliable wall-clock speedup.

At 10k vectors the ef trade-off becomes visible: `ef=64` reaches 0.97 recall, `ef=128` reaches 1.00. At 100k the graph was 4 layers tall (~30 MB estimated); that random dataset needed `ef=256` for exact recall. SQ8 shrinks the estimated index from 0.53 MB to 0.34 MB at 2k vectors (the vector store itself drops 4x) at 0.99 recall, and built about 24% slower in this run.

Recall is verified against exact brute-force ground truth in the test suite for all three metrics (recall@10 >= 0.9 on random datasets).

Run it yourself:

```
moon run --target wasm-gc examples/benchmark
moon run --target wasm-gc examples/benchmark-large
```

## When to use / production readiness

**Good fit** — embedded, single-writer semantic search and RAG contexts; dedup and clustering via `search_radius`; disk-backed indexes loaded via `load` (CRC-checked); memory-constrained deployments via SQ8 when measured recall is acceptable. The bundled measurements span 2k–100k vectors, with the largest run on an older release.

**Know the boundaries** — indexes are single-threaded and not concurrency-safe (wrap in your own actor/lock); exact scanning can beat HNSW on small collections but the crossover depends on data and backend (the JS UCI runs above favored HNSW even at 7,494 vectors); deletion is tombstone-based, so `compact` (a full rebuild) is the way to reclaim space; benchmarks cover one Windows machine on JS and wasm-gc — profile on your target backend before committing to SLAs. No incremental persistence: `save` writes the full index.

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
- Float32 storage accumulates distances in `f32`; frozen SQ8 decodes stored byte values and scores them against the original query using `Double`, including queries outside learned bounds.
- Storage layout: Float32 vectors use one flattened `FixedArray[Float]` of `count * dim` values, doubled geometrically; frozen SQ8 uses one byte per component. Graph edges use `Array[Array[Array[Int]]]`.
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
moon check --target wasm-gc --deny-warn
moon info && moon fmt     # refresh generated interfaces, format
moon run --target wasm-gc examples/benchmark
```

> On Windows the native backend additionally requires MSVC Build Tools (`cl.exe` or `clang-cl`); wasm-gc and js targets work out of the box.

## References

- Malkov, Yashunin. *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. TPAMI 2018 (arXiv:1603.09320).
- [hnswlib](https://github.com/nmslib/hnswlib) and [usearch](https://github.com/unum-cloud/usearch) for parameter conventions.

This is an independent MoonBit implementation of the published HNSW algorithm,
not a port or copy of either library. The repository does not bundle their code.

## License

MIT; see [LICENSE](LICENSE). The benchmark downloads UCI datasets at runtime;
the source data is not bundled. Dataset attribution, DOIs and CC BY 4.0 links
are in the [real-data benchmark report](docs/real-benchmarks.md).
