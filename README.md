# moonhnsw

[![CI](https://github.com/yiwnagmolu/moonhnsw/actions/workflows/ci.yml/badge.svg)](https://github.com/yiwnagmolu/moonhnsw/actions/workflows/ci.yml)
[![Mooncakes](https://img.shields.io/badge/Mooncakes-0.12.4-blue)](https://mooncakes.io/docs/yiwnagmolu/moonhnsw/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**moonhnsw** is a MoonBit library for approximate nearest-neighbor search over `f32` vectors. It implements an HNSW graph with L2, inner-product and cosine distances. The core index runs on native, wasm and JS targets. See the [full API and usage guide](README.mbt.md) for parameters, distance conventions, limitations and examples.

## Install and run

From an existing MoonBit project, install the published version:

```sh
moon add yiwnagmolu/moonhnsw@0.12.4
```

If you need a new project first, run `moon new moonhnsw-demo` and `cd moonhnsw-demo`. In the package that will use the index, put this in `cmd/main/moon.pkg` (or adapt the path to your package):

```moon.pkg
import {
  "yiwnagmolu/moonhnsw" @moonhnsw,
}

pkgtype(kind: "executable")
```

Put this complete program in `cmd/main/main.mbt`:

```mbt
///|
fn main raise {
  let index = @moonhnsw.Hnsw::new(2, @moonhnsw.Distance::L2)
  let vector : FixedArray[Float] = FixedArray::from_array([1.0, 2.0])
  let _ = index.add(vector)
  let hits = index.search(vector, 1, ef=64)
  println("nearest id=\{hits[0].id} distance=\{hits[0].distance}")
}
```

Run `moon run --target wasm-gc cmd/main`. The output is `nearest id=0 distance=0`. The package import in `moon.pkg` is required in addition to `moon add`.

## Scope and verification

The library supports incremental insertion, approximate and exact search, filtering, radius search, deletion and compaction, SQ8 quantization, labels, and checked binary save/load. The [full guide](README.mbt.md) documents the API and the limits of approximate retrieval.

The repository includes [tests](api_test.mbt) for core paths and edge cases, a [runnable benchmark example](examples/benchmark/main.mbt), a [real-data benchmark report](docs/real-benchmarks.md) with UCI source hashes and raw logs, and a [nine-point acceptance audit](docs/acceptance-audit.md). To verify locally, run:

```sh
moon fmt --check
moon check --target wasm-gc --deny-warn
moon test --target wasm-gc
moon test --target js
moon build --target wasm
moon run --target wasm-gc examples/benchmark
```

The [CI workflow](.github/workflows/ci.yml) also runs native tests, checks the minimum `moonc` version (`0.10.16`), and builds documentation.

## License and sources

Licensed under [MIT](LICENSE), an [OSI-approved license](https://opensource.org/license/mit). This is an independent MoonBit implementation of the HNSW algorithm described by Malkov and Yashunin; [hnswlib](https://github.com/nmslib/hnswlib) and [usearch](https://github.com/unum-cloud/usearch) informed parameter conventions. Their source code is not bundled. UCI benchmark data is downloaded at runtime; dataset attribution, licenses and DOIs are in the [benchmark report](docs/real-benchmarks.md).
