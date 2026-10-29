# Changelog

All notable changes to moonhnsw are documented here. Releases are tagged in
the repository; the binary serialization format stays backward compatible
(newer loaders accept older payloads).

## 0.12.0 — 2026-10-29

### Added

- `search_exact` and `search_radius_exact` scan live stored vectors for
  guaranteed top-k and radius results, with filters and stable distance/id
  tie ordering. SQ8 scans use the index's quantized distance semantics.
- A reproducible JS release benchmark on two UCI datasets (pen digits and
  letter features), including 200 held-out queries, Float32 and SQ8 results,
  exact-scan baselines, tie-aware recall, distance counts, source hashes,
  raw output and hardware details.
- Boundary tests for exact scans, ties, tombstones, selective filters, SQ8,
  empty indexes, invalid inputs and valid-checksum corrupt payloads.

### Changed

- Reject NaN and infinite input vectors and queries before they can corrupt
  graph ordering or SQ8 bounds. `upsert_labeled` validates before deleting
  previous vectors, and loading rejects non-finite stored f32 values and SQ8
  metadata even when the payload checksum is valid.
- `calibrate_ef` now uses the public exact scan for its ground truth. Binary
  serialization format v5 remains unchanged and old payload versions load.
- Mooncakes registry review found an existing brute-force strategy in
  `trkbt10/vcdb`; this API differs by exact-scanning the same HNSW instance,
  preserving its tombstones, filters and SQ8 stored-distance semantics.

## 0.11.2 — 2026-10-29

### Changed

- Reuse the input f32 vector while inserting L2 and inner-product data, and
  reuse the normalized scratch vector for cosine. This removes an unused norm
  pass and a per-insert query allocation/copy; stored vectors remain isolated
  from later caller mutations.
- Compute public `distance(Cosine, a, b)` with the same f32 normalization and
  dot-product rounding without allocating two temporary arrays per call. The
  benchmark now reports its throughput and distance checksum.
- Read labels in a single preflighted little-endian bulk pass, and use bulk
  f32 reads/writes for SQ8 bounds. Bulk read lengths now use overflow-safe
  checks. Serialized format v5 and public APIs are unchanged.

### Fixed

- Keep candidates with `+Infinity` L2 distance in unbounded top-k searches.
  Finite f32 inputs can overflow the f32 squared-distance accumulator; radius
  searches still honor their explicit threshold.

## 0.11.1 — 2026-10-29

### Changed

- Cache each MMR candidate's maximum similarity to selected results, reducing
  reranking pair distances from O(pool × k²) to O(pool × k). With a 128-item
  pool and k=10, pair distances drop from 5,475 to 1,107; the bundled 2k L2
  run measured 6,862.94 to 2,494.94 total distances per query, with the same
  ordered-result checksum.
- Reverse the drained result heap in place, avoiding one array allocation and
  copy per graph search. Bundled search recall and distance counts stayed the
  same across L2, InnerProduct, Cosine, and SQ8.
- Report MMR throughput, total distance work, and an ordered-result checksum
  in the benchmark example. The binary format and public API are unchanged.

## 0.11.0 — 2026-10-28

### Added

- `calibrate_ef`: compare representative queries against an exact scan of
  live stored vectors, report recall@k and mean distance computations for
  each candidate `ef`, and recommend the smallest tested width meeting the
  caller's target. Frozen SQ8 uses its search distance for ground truth.
- `search_filtered_adaptive`: widen a post-filter HNSW candidate pool until
  `k` matches are found or `ef_max` is reached; return the final width,
  round count, neighbors, and total distance work.
- Public tests for both APIs, plus v1–v4 payload fixtures and corruption
  regression cases. The serialized format remains v5.

### Fixed

- Restore standard MMR `lambda` semantics: 1.0 is pure relevance, while 0.0
  selects the nearest first and then favors diversity. Pools now grow to at
  least `k` even when `ef < k`.
- Normalize inputs to the public `distance(Cosine, a, b)` helper; allow
  negative radius thresholds for Cosine and InnerProduct.
- Reject wrong-dimension upserts before deleting old labeled vectors. Batch
  searches now report the sum of their distance computations.
- Check v4/v5 CRC before allocating from untrusted header lengths; reject
  trailing data and invalid graph structure during load. The constructor and
  loader now cap dimensions at 1,000,000 and `m` at 1,073,741,823 to avoid
  invalid allocation and link-budget arithmetic.
- Count equal-distance neighbors as correct in empirical recall, and reject
  invalid SQ8 quantization metadata even when the payload checksum is valid.

### Ecosystem check

- On 2026-10-28, the Mooncakes registry listed 3,371 unique modules.
  `trkbt10/vcdb` already provides HNSW, so HNSW itself is not a unique
  Mooncakes capability. The inspected vector-search packages did not expose
  equivalent adaptive filtered HNSW widening or empirical HNSW `ef`
  calibration APIs. `yiwnagmolu/moonhnsw` was not yet listed in the registry.

## 0.10.2 — 2026-10-28

### Changed

- Clear the reusable candidate heap directly after graph traversal instead of
  popping every unused candidate. This reduces the cleanup work from
  O(r log r) to O(r) for r remaining candidates, with no change to search
  results, distance computations, or the serialized format.
- The bundled wasm-gc benchmark retained the same recall and distance counts
  across all metrics and search widths; its short throughput runs varied too
  much to claim a reliable end-to-end speedup.

## 0.10.1 — 2026-10-26

### Added

- `Show` for `HnswError` (renders `message()`) and `Neighbor` (compact
  `Neighbor(id, distance)`), so errors and results print through string
  interpolation directly. `derive(Show)` is deprecated in this toolchain;
  the impls are hand-written with explicit extend promotions.

### Changed

- Hot-loop field loads (`alive`, `links`, `count`) hoisted to locals in
  `search_layer`; within noise on the benchmark, kept for the invariant it
  documents. README gains version and license badges.

## 0.10.0 — 2026-10-26

### Added

- `Hnsw::distance_computations()`: distance evaluations of the last search,
  the hardware-independent efficiency metric of the graph (reset at each
  search entry point).
- Benchmarks report dists/query and speedup against a fair single-pass
  brute-force baseline, plus a timed brute column in the 100k example.
- README gains a "When to use / production readiness" section stating the
  honest crossover: graph search wins above ~10k vectors (13.9x brute at
  100k, touching 1.7% of the dataset at ef=64) and loses below it.

### Changed

- The brute-force baseline in benchmarks switched from k argmin passes
  (k*n distances) to a single-pass running top-k; previous speedup figures
  were overstated by roughly k times.

## 0.9.2 — 2026-10-26

### Changed

- `hnsw.mbt` (1450 lines) split block-for-block into topic files: insert,
  search, lifecycle, distance, with hnsw.mbt keeping types, the constructor
  and accessors. No behavior change; the full suite, generated interface and
  formatting are identical. AGENTS.md documents the new layout.

## 0.9.1 — 2026-10-25

### Added

- The benchmark now measures persistence round-trips and payload sizes.
  Baseline on wasm-gc: 10k x 32d L2 saves 2.39 MB in 100.5 ms and loads in
  91.4 ms; SQ8 stores 0.30 MB and round-trips in ~12 ms at 2k.

### Changed

- Labels serialize through a bulk u64 pass (one capacity check per run
  instead of eight per label). At 10k labels the effect is below noise -
  80 KB of a 2.39 MB payload - but it removes the last per-element
  serialization loop.

## 0.9.0 — 2026-10-25

### Added

- Filter composition: `search_radius` and `search_mmr` take the same filter
  closure as `search_filtered`, completing the combinator matrix.
- `Hnsw::shrink_to_fit`: reallocates the vector stores to the exact current
  size — memory relief without the graph rebuild `compact` pays.
- `Hnsw::from_vectors`: one-call construction from a list of vectors.

### Fixed

- Two latent SQ8 bugs, surfaced by new SQ8 radius/compact coverage:
  `freeze_and_quantize_all` dropped the 256th vector (off-by-one against
  the counter), and `Hnsw::new` silently reset `params.storage` to
  `Float32`, so `compact`/`absorb` of an SQ8 index produced full-precision
  copies. `compact` now re-quantizes survivors through `get_vector`.

## 0.8.1 — 2026-10-25

### Changed

- L2/IP searches walk the caller's query array directly (only Cosine pays
  for a normalized copy) and the traversal heaps reuse backing arrays
  drained on return; the candidate heap is emptied in place, a subtlety the
  stress test caught on the first cut. Measured: InnerProduct build
  1365 -> 1175 ms on the wasm-gc benchmark.
- The link section of a payload - the dominant part of large graphs -
  serializes and loads through bulk u32 helpers with one capacity check per
  run instead of per integer. Byte layout unchanged.

## 0.8.0 — 2026-10-25

### Added

- `Hnsw::upsert_labeled(vector, label)`: insert-or-replace by business key —
  re-embed a document, then upsert; replaced vectors are tombstoned and stay
  reclaimable via `compact`.
- `Hnsw::find_by_label(label)`: reverse id lookup for a label (labels may
  repeat).
- `Hnsw::search_batch(queries, k, ef)`: one call for a list of queries.
- A 180-step randomized-operation stress test: adds, deletes and checked
  searches with `verify()` and brute-force nearest-neighbor agreement
  throughout.

## 0.7.1 — 2026-10-25

### Changed

- SQ8 search quantizes its query once per operation instead of once per
  visited neighbor (reference-identity caching, invalidated at every public
  entry point): 2959 -> 4306 qps at ef=64 and 1315 -> 1775 qps at ef=256 on
  the wasm-gc benchmark, build 1458 -> 1255 ms.
- Vector payloads move through bulk serialization helpers with one capacity
  check per block instead of per byte.

## 0.7.0 — 2026-10-25

### Added

- SQ8 scalar quantization: `HnswParams::new(storage=SQ8)` learns
  per-dimension bounds from the first 256 inserted vectors, freezes, and
  stores one byte per dimension (~4x smaller vector store). Out-of-range
  values clamp; distances run over the byte store with the full-precision
  query on the search side; `get_vector` returns the dequantized vector.
  Recall 0.99 at n=2000 dim=32 on the bundled benchmark.
- `Hnsw::search_mmr`: Maximal Marginal Relevance re-ranking over an ef-wide
  candidate pool for diversity-aware retrieval.
- CRC-32 integrity trailer: v4+ payloads are checksummed and corrupted
  payloads are rejected with `CorruptData` on load.

### Changed

- Serialization format bumped to v5 (storage section + CRC-32 trailer);
  v1–v4 payloads keep loading unchanged.

## 0.6.0 — 2026-10-25

### Added

- `Hnsw::search_filtered`: approximate top-k restricted by a user filter
  closure over ids (combine with `label_of` for metadata filtering). The
  all-true filter reproduces plain `search` bit-for-bit.

## 0.5.0 — 2026-10-25

### Added

- `Hnsw::export_dot(layer)`: Graphviz DOT export of one graph layer, with
  deleted nodes dashed and user labels shown.
- `ToJson` for `Neighbor` and `HnswStats`.
- Clustered-data recall tests and a post-load construction parity test
  (rng_state persistence made observable).

## 0.4.2 — 2026-10-24

### Added

- AGENTS.md contributor guide and a README layer diagram.

## 0.4.1 — 2026-10-24

### Added

- `Hnsw::absorb(other)`: merge another index's live vectors (labels kept),
  returning an old-id to new-id mapping.
- `Hnsw::copy` (deep clone) and `Hnsw::ids` (ascending alive ids).
- `Hnsw::save_bytes` / `Hnsw::load_bytes` for `Bytes`-based pipelines.

## 0.4.0 — 2026-10-24

### Added

- External labels: `Hnsw::add_labeled` / `Hnsw::label_of`, persisted in
  serialization format v3 (v1/v2 payloads still load).
- `estimated_bytes` in `HnswStats`.
- 100k-vector large-scale benchmark example (`examples/benchmark-large`).

## 0.3.0 — 2026-10-24

### Added

- `Hnsw::contains` / `Hnsw::get_vector` accessors.
- `Hnsw::stats`: per-layer node counts and average degree.
- `Hnsw::compact`: tombstone-free rebuild with an id mapping.

## 0.2.0 — 2026-10-24

### Added

- `Hnsw::search_radius`: approximate radius search.
- Tombstone deletion via `Hnsw::delete` (format v2, v1-compatible).
- GitHub Actions CI: tests on wasm-gc and js, wasm build, format gate,
  benchmark smoke run.

### Changed

- Distance accumulation switched to f32 like hnswlib (~16% faster queries,
  6% faster builds on the wasm-gc benchmark).

## 0.1.0 — 2026-10-24

### Added

- Initial release: HNSW graph index with `add` / `search` / `save` / `load`,
  L2 / InnerProduct / Cosine metrics, typed errors, blackbox and whitebox
  test suites with brute-force recall verification, and a benchmark example.
