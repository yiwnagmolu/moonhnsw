# Project Agents.md Guide

This is a [MoonBit](https://docs.moonbitlang.com) project: **moonhnsw**, a
pure MoonBit HNSW approximate nearest-neighbor search library (module
`yiwnagmolu/moonhnsw`).

## Project Structure

- The library lives in the root package: the HNSW graph and public API
  (`hnsw.mbt`), binary serialization (`serial.mbt`), the deterministic
  SplitMix64 PRNG (`rng.mbt`) and the internal binary heap (`heap.mbt`).
- `hnsw_wbtest.mbt` holds whitebox tests (internal helpers, brute-force
  recall harness, tamper tests); `api_test.mbt` holds blackbox tests that
  exercise the package the way an external user would, with
  package-qualified names.
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

## Serialization format (v3)

Little-endian: magic `MHNSW`, u8 version, u32 dim, u8 metric, u32 `m`, u32
`ef_construction`, u64 seed, i32 entry, i32 max_level, u32 count, u64 rng
state, one alive byte per vector, one u64 label per vector, `count * dim`
f32 vector values, then per node the per-layer link lists. Newer loaders
must keep accepting older payload versions.
