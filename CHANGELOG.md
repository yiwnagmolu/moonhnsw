# Changelog

All notable changes to moonhnsw are documented here. Releases are tagged in
the repository; the binary serialization format stays backward compatible
(newer loaders accept older payloads).

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
