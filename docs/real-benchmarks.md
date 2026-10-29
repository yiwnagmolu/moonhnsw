# Real-data nearest-neighbor benchmarks (2026-10-30)

This report measures **moonhnsw 0.12.2**, compiled to JavaScript,
on three public UCI datasets. It supplements the uniform-random wasm-gc runs in
the README; the datasets, backend and machine differ, so their timings are not
direct before/after speedup measurements.

## Data and reproduction

- [Pen-Based Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits), E. Alpaydin and F. Alimoglu, DOI [10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K): 7,494 official training vectors; the first 200 of 3,498 official test vectors are queries.
- [Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition), D. Slate, DOI [10.24432/C5ZP40](https://doi.org/10.24432/C5ZP40): first 16,000 vectors for construction and the next 200 for queries, following UCI's suggested 16,000/4,000 split.
- [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis, DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T): white-wine data, first 4,000 rows for construction and the next 200 for queries; the final quality score is excluded.

Pen digits and letters have 16 numerical features; wine quality has 11.
Class labels and quality scores are excluded: this is nearest-neighbor
retrieval, not classification or regression accuracy. Raw feature values
are converted to f32 without scaling beyond the index's Cosine metric.
UCI identifies all three datasets as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The chemistry features have different scales, so L2 distances and raw inner
products are workload measurements, not claims of meaningful wine similarity.
`benchmarks/real/run.cjs` downloads the original CSV files from UCI, verifies
their pinned SHA-256 hashes, and caches them outside the repository. It then
invokes the MoonBit JS release build. No dataset copy is checked in.

Run from the repository root with Node.js 24+ and MoonBit installed:

```text
node benchmarks/real/run.cjs pendigits
node benchmarks/real/run.cjs letters
node benchmarks/real/run.cjs winequality
```

The underlying MoonBit executable is in `benchmarks/real/main.mbt`. The Node
launcher only loads data and starts it. Construction, exact search, HNSW
search, distance counts and timings execute in MoonBit. Source hashes and
complete output from these runs is in [pendigits](benchmarks/pendigits-2026-10-30.txt),
[letters](benchmarks/letters-2026-10-30.txt), and
[winequality](benchmarks/winequality-2026-10-30.txt).

## Method

- Windows x64; AMD Ryzen 7 5800H; Node v24.21.0; `moon 0.1.20261023`,
  `moonc v0.10.17+d971670cd`; JS release build.
- `m=16`, `ef_construction=200`, default fixed seed; `k=10` and
  `ef=64/128/256`. All three distance metrics are measured; SQ8 is measured
  for L2. Every query uses an independent held-out data row.
- Exact top-k uses the new public `search_exact`, which scans every live
  stored vector. The Float32 exact result is the reference for Float32 and
  SQ8. SQ8 recall therefore includes quantization error.
- `id_recall@10` counts overlap with the exact result's ten IDs.
  `tie_recall@10` also accepts a returned vector whose full-precision
  distance is no greater than the exact tenth distance. The Letter dataset
  has many distance ties, so the second measure avoids treating an equally
  near vector as a miss.
- Query throughput is the median of three timed passes over 200 queries,
  after warm-up. Quality checks are outside those timed passes. The checksum
  weights every returned ID by query and rank over all three passes; it
  exposes accidental changes in output ordering. Build time is one run.
- The Letter and Wine Quality splits follow source row order rather than a
  randomized sample. Their measurements describe these fixed subsets and
  should not be treated as a population-level quality estimate.

## Results

Payload is the serialized size. `d/q` counts MoonBit distance evaluations
per query. Exact scan evaluates all training vectors. QPS varies across
short runs; these numbers describe this one machine, build and sample. SQ8
exact-scan QPS was not measured; its recall is against the Float32 truth.

### Pen digits: 7,494 × 16, 200 held-out queries

| Metric / store | Build ms | Payload MiB | Exact QPS | ef | ID recall | Tie recall | HNSW QPS | d/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 / Float32 | 4147 | 1.34 | 819 | 64 | 0.9995 | 1.0000 | 7786 | 350 |
| L2 / Float32 |  |  |  | 128 | 0.9990 | 1.0000 | 3926 | 487 |
| L2 / Float32 |  |  |  | 256 | 0.9990 | 1.0000 | 2494 | 714 |
| L2 / SQ8 | 2887 | 1.00 | — | 64 | 0.9885 | 0.9890 | 11717 | 350 |
| L2 / SQ8 |  |  |  | 128 | 0.9885 | 0.9890 | 6179 | 487 |
| L2 / SQ8 |  |  |  | 256 | 0.9885 | 0.9890 | 3268 | 714 |
| InnerProduct / Float32 | 6423 | 1.14 | 991 | 64 | 0.9930 | 0.9930 | 6615 | 616 |
| InnerProduct / Float32 |  |  |  | 128 | 0.9995 | 0.9995 | 3564 | 882 |
| InnerProduct / Float32 |  |  |  | 256 | 1.0000 | 1.0000 | 2115 | 1264 |
| Cosine / Float32 | 2769 | 1.34 | 1873 | 64 | 1.0000 | 1.0000 | 11455 | 347 |
| Cosine / Float32 |  |  |  | 128 | 1.0000 | 1.0000 | 6648 | 483 |
| Cosine / Float32 |  |  |  | 256 | 1.0000 | 1.0000 | 3490 | 706 |

### Letters: 16,000 × 16, 200 held-out queries

| Metric / store | Build ms | Payload MiB | Exact QPS | ef | ID recall | Tie recall | HNSW QPS | d/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 / Float32 | 13110 | 2.88 | 391 | 64 | 0.9445 | 1.0000 | 4665 | 488 |
| L2 / Float32 |  |  |  | 128 | 0.9510 | 1.0000 | 2503 | 766 |
| L2 / Float32 |  |  |  | 256 | 0.9480 | 1.0000 | 1573 | 1245 |
| L2 / SQ8 | 7055 | 2.14 | — | 64 | 0.9345 | 0.9920 | 9949 | 472 |
| L2 / SQ8 |  |  |  | 128 | 0.9370 | 0.9920 | 5301 | 743 |
| L2 / SQ8 |  |  |  | 256 | 0.9345 | 0.9920 | 2682 | 1212 |
| InnerProduct / Float32 | 20841 | 2.33 | 477 | 64 | 0.9860 | 0.9985 | 3660 | 1112 |
| InnerProduct / Float32 |  |  |  | 128 | 0.9850 | 0.9995 | 2570 | 1700 |
| InnerProduct / Float32 |  |  |  | 256 | 0.9890 | 0.9995 | 1500 | 2489 |
| Cosine / Float32 | 8315 | 2.88 | 771 | 64 | 0.9905 | 1.0000 | 8049 | 472 |
| Cosine / Float32 |  |  |  | 128 | 0.9920 | 1.0000 | 4385 | 748 |
| Cosine / Float32 |  |  |  | 256 | 0.9940 | 1.0000 | 2276 | 1215 |

### Wine Quality (white): 4,000 × 11, 200 held-out queries

This smaller, real-valued dataset tests a different feature distribution and
shows that SQ8 quality varies by workload. The raw log is a separate
2026-10-30 run on the same machine and toolchain.

| Metric / store | Build ms | Payload MiB | Exact QPS | ef | ID recall | Tie recall | HNSW QPS | d/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 / Float32 | 1319 | 0.65 | 3581 | 64 | 0.9910 | 1.0000 | 15754 | 277 |
| L2 / Float32 |  |  |  | 128 | 0.9930 | 1.0000 | 7638 | 389 |
| L2 / Float32 |  |  |  | 256 | 0.9910 | 1.0000 | 4531 | 563 |
| L2 / SQ8 | 1040 | 0.52 | — | 64 | 0.9230 | 0.9285 | 17723 | 276 |
| L2 / SQ8 |  |  |  | 128 | 0.9220 | 0.9285 | 8342 | 388 |
| L2 / SQ8 |  |  |  | 256 | 0.9230 | 0.9285 | 4930 | 562 |
| InnerProduct / Float32 | 944 | 0.50 | 3702 | 64 | 1.0000 | 1.0000 | 15785 | 254 |
| InnerProduct / Float32 |  |  |  | 128 | 1.0000 | 1.0000 | 10376 | 299 |
| InnerProduct / Float32 |  |  |  | 256 | 1.0000 | 1.0000 | 7339 | 392 |
| Cosine / Float32 | 1148 | 0.65 | 3549 | 64 | 0.9835 | 1.0000 | 16220 | 293 |
| Cosine / Float32 |  |  |  | 128 | 0.9825 | 1.0000 | 8969 | 416 |
| Cosine / Float32 |  |  |  | 256 | 0.9855 | 1.0000 | 4218 | 626 |

At ef=64, the JS L2 HNSW call was about 9.5× the Float32 exact-scan QPS on
pen digits and about 12× on letters, while touching 4.7% and 3.0% of the
vectors respectively. These ratios are specific to the two workloads. SQ8
used 25% less serialized space on pen digits and 26% less on letters; it
traded roughly 1–2 percentage points of tie-aware recall for higher QPS.
On wine quality, SQ8 reduced serialized size by about 20% but tie-aware
recall fell to 0.9285; the data-dependent quality cost is material.

The tie-aware recall is 1.0 for Float32 at the tested widths, but the ID
recall on letters is lower because many vectors share the kth distance.
Neither result implies exact graph search on arbitrary future queries. The
new exact API is the appropriate choice when a completeness guarantee is
required.
