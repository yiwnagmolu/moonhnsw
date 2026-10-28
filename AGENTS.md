# Project Agents.md Guide

This is a [MoonBit](https://docs.moonbitlang.com) project: **moonhnsw**, a
pure MoonBit HNSW approximate nearest-neighbor search library (module
`yiwnagmolu/moonhnsw`).

## Project Structure

- The library lives in the root package, split by topic:
  - `hnsw.mbt` — types (`Distance`, `HnswParams`, `HnswError`, `Neighbor`,
    `HnswStats`, the `Hnsw` struct), the constructor and accessors.
  - `insert.mbt` — insertion paths, vector capacity growth, level draws.
  - `search.mbt` — the query family (top-k, filtered, radius, MMR, batch)
    plus beam traversal and neighbor selection.
  - `lifecycle.mbt` — delete, compact, absorb, copy, shrink_to_fit, verify
    and DOT export.
  - `distance.mbt` — metric helpers and the f32 distance kernels.
  - `quant.mbt` — SQ8 scalar quantization (bounds, freeze, byte distances).
  - `serial.mbt` — binary serialization (format v5, CRC-32 trailer).
  - `crc.mbt` — the IEEE CRC-32 kernel.
  - `rng.mbt` — the SplitMix64 level-draw generator.
  - `heap.mbt` — the internal binary heap.
- `hnsw_wbtest.mbt` holds whitebox tests (internal helpers, brute-force
  recall harness, tamper and stress tests); `api_test.mbt` and `audit_test.mbt`
  hold blackbox tests that exercise the package the way an external user would,
  with package-qualified names.
- `examples/benchmark` runs in CI as a smoke test; `examples/benchmark-large`
  (100k vectors) is for occasional manual runs only — never add it to CI.

## Commands

- `moon test --target wasm-gc` — run the suite (also `--target js`, and
  `--target native` where a C toolchain exists; on Windows that means MSVC).
- `moon info && moon fmt` — refresh generated interfaces, then format. Always
  run both before committing; CI gates on `moon fmt --check`.
- `moon run --target wasm-gc examples/benchmark` — build / recall / QPS
  numbers per metric and scale.
- `moon doc` — build the documentation site.

## Conventions

- Code is organized in `///|`-separated blocks; keep internal helpers and
  types `priv`.
- Public result types derive `Eq` / `@debug.Debug` with explicit `pub extend`
  declarations (the implicit method promotion is deprecated).
- Raising functions declare `raise HnswError`; user-input mistakes surface as
  `InvalidParams` / `DimMismatch` / `InvalidId`, damaged payloads as
  `CorruptData`.
- Every feature lands with blackbox or whitebox coverage. Recall claims are
  always checked against brute-force ground truth, and deterministic seeds
  keep the tests reproducible.

## Serialization format (v5)

Little-endian: magic `MHNSW`, u8 version, u32 dim, u8 metric, u32 `m`, u32
`ef_construction`, u64 seed, i32 entry, i32 max_level, u32 count, u64 rng
state, one alive byte per vector, one u64 label per vector, u8 storage code
(and for SQ8: frozen flag + per-dimension min/max/scale), the vector values
(f32, or one byte per dimension when frozen), then per node the per-layer
link lists; v4+ payloads end with a CRC-32 over every preceding byte.
Newer loaders must keep accepting older payload versions.
