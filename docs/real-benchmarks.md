# Real-data nearest-neighbor benchmarks (2026-10-30)

This report measures **moonhnsw 0.12.4**, compiled to JavaScript,
on four public UCI datasets. It supplements the uniform-random wasm-gc runs in
the README; the datasets, backend and machine differ, so their timings are not
direct before/after speedup measurements.

## Data and reproduction

- [Pen-Based Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits), E. Alpaydin and F. Alimoglu, DOI [10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K): 7,494 official training vectors; the first 200 of 3,498 official test vectors are queries.
- [Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition), D. Slate, DOI [10.24432/C5ZP40](https://doi.org/10.24432/C5ZP40): first 16,000 vectors for construction and the next 200 for queries, following UCI's suggested 16,000/4,000 split.
- [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis, DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T): white-wine data, first 4,000 rows for construction and the next 200 for queries; the final quality score is excluded.
- [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), E. Alpaydin and C. Kaynak, DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49): all 3,823 official training vectors; the first 200 of 1,797 official test vectors are queries. The train and test writers differ.

Pen digits and letters have 16 numerical features; wine quality has 11;
optical digits has 64 image features. Optical digits adds dimensional coverage
but is not a substitute for a 384+ dimensional text-embedding workload.
Class labels and quality scores are excluded: this is nearest-neighbor
retrieval, not classification or regression accuracy. Raw feature values
are converted to f32 without scaling beyond the index's Cosine metric.
UCI identifies all four datasets as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The chemistry features have different scales, so L2 distances and raw inner
products are workload measurements, not claims of meaningful wine similarity.
`benchmarks/real/run.cjs` downloads the original delimited data files
from UCI, verifies their pinned SHA-256 hashes, and caches them outside the
repository. It then invokes the MoonBit JS release build. No dataset copy is
checked in.

Run from the repository root with Node.js 24+ and MoonBit installed:

```text
node benchmarks/real/run.cjs pendigits
node benchmarks/real/run.cjs letters
node benchmarks/real/run.cjs winequality
node benchmarks/real/run.cjs optdigits
```

The underlying MoonBit executable is in `benchmarks/real/main.mbt`. The Node
launcher only loads data and starts it. Construction, exact search, HNSW
search, distance counts and timings execute in MoonBit. Source hashes and
complete output from these runs are in [pendigits](benchmarks/pendigits-2026-10-30.txt),
[letters](benchmarks/letters-2026-10-30.txt),
[winequality](benchmarks/winequality-2026-10-30.txt), and
[optdigits](benchmarks/optdigits-2026-10-30.txt).

## Method

- Windows x64; AMD Ryzen 7 5800H; Node v24.21.0; `moon 0.1.20261023`,
  `moonc v0.10.17+d971670cd`; JS release build.
- `m=16`, `ef_construction=200`, default fixed seed; `k=10` and
  `ef=64/128/256`. All three distance metrics are measured; SQ8 is measured
  for L2. Query rows are separate from indexed training rows.
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
- Query sets use the first 200 rows of their held-out source partitions. The
  Letter and Wine Quality train/query partitions also follow source row order
  rather than a randomized sample. These measurements describe fixed subsets
  and should not be treated as population-level quality estimates.
- In 0.12.4, SQ8 scores the original query against dequantized stored values.
  Earlier 0.12.2 SQ8 results used a different query-distance rule, so their
  recall and QPS cannot be interpreted as a single-variable performance gain.

## Results

Payload is the serialized size. `d/q` counts MoonBit distance evaluations
per query. Exact scan evaluates all training vectors. QPS varies across
short runs; these numbers describe this one machine, build and sample. SQ8
exact-scan QPS was not measured; its recall is against the Float32 truth.

### Pen digits: 7,494 × 16, 200 held-out queries

| Metric / store | Build ms | Payload MiB | Exact QPS | ef | ID recall | Tie recall | HNSW QPS | d/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 / Float32 | 4478 | 1.34 | 847 | 64 | 0.9995 | 1.0000 | 7172 | 350 |
| L2 / Float32 |  |  |  | 128 | 0.9990 | 1.0000 | 3977 | 487 |
| L2 / Float32 |  |  |  | 256 | 0.9990 | 1.0000 | 2710 | 714 |
| L2 / SQ8 | 3236 | 1.00 | — | 64 | 0.9930 | 0.9940 | 8584 | 350 |
| L2 / SQ8 |  |  |  | 128 | 0.9930 | 0.9940 | 4628 | 487 |
| L2 / SQ8 |  |  |  | 256 | 0.9930 | 0.9940 | 2824 | 715 |
| InnerProduct / Float32 | 7973 | 1.14 | 934 | 64 | 0.9930 | 0.9930 | 6095 | 616 |
| InnerProduct / Float32 |  |  |  | 128 | 0.9995 | 0.9995 | 3189 | 882 |
| InnerProduct / Float32 |  |  |  | 256 | 1.0000 | 1.0000 | 1953 | 1264 |
| Cosine / Float32 | 3060 | 1.34 | 1785 | 64 | 1.0000 | 1.0000 | 10830 | 347 |
| Cosine / Float32 |  |  |  | 128 | 1.0000 | 1.0000 | 5713 | 483 |
| Cosine / Float32 |  |  |  | 256 | 1.0000 | 1.0000 | 2415 | 706 |

### Letters: 16,000 × 16, 200 held-out queries

| Metric / store | Build ms | Payload MiB | Exact QPS | ef | ID recall | Tie recall | HNSW QPS | d/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 / Float32 | 13564 | 2.88 | 332 | 64 | 0.9445 | 1.0000 | 5041 | 488 |
| L2 / Float32 |  |  |  | 128 | 0.9510 | 1.0000 | 2777 | 766 |
| L2 / Float32 |  |  |  | 256 | 0.9480 | 1.0000 | 1581 | 1245 |
| L2 / SQ8 | 8695 | 2.14 | — | 64 | 0.9390 | 0.9940 | 8438 | 472 |
| L2 / SQ8 |  |  |  | 128 | 0.9385 | 0.9945 | 3722 | 743 |
| L2 / SQ8 |  |  |  | 256 | 0.9370 | 0.9945 | 2130 | 1212 |
| InnerProduct / Float32 | 20856 | 2.33 | 473 | 64 | 0.9860 | 0.9985 | 4448 | 1112 |
| InnerProduct / Float32 |  |  |  | 128 | 0.9850 | 0.9995 | 2553 | 1700 |
| InnerProduct / Float32 |  |  |  | 256 | 0.9890 | 0.9995 | 1495 | 2489 |
| Cosine / Float32 | 8225 | 2.88 | 847 | 64 | 0.9905 | 1.0000 | 7795 | 472 |
| Cosine / Float32 |  |  |  | 128 | 0.9920 | 1.0000 | 3791 | 748 |
| Cosine / Float32 |  |  |  | 256 | 0.9940 | 1.0000 | 1924 | 1215 |

### Wine Quality (white): 4,000 × 11, 200 held-out queries

This smaller, real-valued dataset tests a different feature distribution and
shows that SQ8 quality varies by workload. The raw log is a separate
2026-10-30 run on the same machine and toolchain.

| Metric / store | Build ms | Payload MiB | Exact QPS | ef | ID recall | Tie recall | HNSW QPS | d/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 / Float32 | 1221 | 0.65 | 3803 | 64 | 0.9910 | 1.0000 | 14678 | 277 |
| L2 / Float32 |  |  |  | 128 | 0.9930 | 1.0000 | 8311 | 389 |
| L2 / Float32 |  |  |  | 256 | 0.9910 | 1.0000 | 4351 | 563 |
| L2 / SQ8 | 1336 | 0.52 | — | 64 | 0.9535 | 0.9615 | 15494 | 276 |
| L2 / SQ8 |  |  |  | 128 | 0.9550 | 0.9615 | 8242 | 387 |
| L2 / SQ8 |  |  |  | 256 | 0.9530 | 0.9615 | 4376 | 562 |
| InnerProduct / Float32 | 1054 | 0.50 | 3605 | 64 | 1.0000 | 1.0000 | 15760 | 254 |
| InnerProduct / Float32 |  |  |  | 128 | 1.0000 | 1.0000 | 10632 | 299 |
| InnerProduct / Float32 |  |  |  | 256 | 1.0000 | 1.0000 | 7691 | 392 |
| Cosine / Float32 | 1108 | 0.65 | 4230 | 64 | 0.9835 | 1.0000 | 15378 | 293 |
| Cosine / Float32 |  |  |  | 128 | 0.9825 | 1.0000 | 8956 | 416 |
| Cosine / Float32 |  |  |  | 256 | 0.9855 | 1.0000 | 4579 | 626 |

### Optical digits: 3,823 × 64, 200 held-out queries

The UCI split uses different writers for training and testing. Its 64 image
features are four times the highest dimension of the other datasets here,
but remain far below common text-embedding dimensions.

| Metric / store | Build ms | Payload MiB | Exact QPS | ef | ID recall | Tie recall | HNSW QPS | d/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 / Float32 | 5631 | 1.39 | 472 | 64 | 0.9970 | 1.0000 | 2596 | 439 |
| L2 / Float32 |  |  |  | 128 | 0.9960 | 1.0000 | 1827 | 636 |
| L2 / Float32 |  |  |  | 256 | 0.9970 | 1.0000 | 1049 | 986 |
| L2 / SQ8 | 4120 | 0.69 | — | 64 | 0.9930 | 0.9955 | 4269 | 439 |
| L2 / SQ8 |  |  |  | 128 | 0.9930 | 0.9955 | 2638 | 636 |
| L2 / SQ8 |  |  |  | 256 | 0.9930 | 0.9955 | 1608 | 985 |
| InnerProduct / Float32 | 8320 | 1.32 | 580 | 64 | 0.9980 | 1.0000 | 2713 | 669 |
| InnerProduct / Float32 |  |  |  | 128 | 0.9995 | 1.0000 | 1680 | 981 |
| InnerProduct / Float32 |  |  |  | 256 | 0.9975 | 1.0000 | 1048 | 1460 |
| Cosine / Float32 | 2626 | 1.39 | 1237 | 64 | 1.0000 | 1.0000 | 6300 | 426 |
| Cosine / Float32 |  |  |  | 128 | 1.0000 | 1.0000 | 3885 | 616 |
| Cosine / Float32 |  |  |  | 256 | 1.0000 | 1.0000 | 2183 | 957 |

At ef=64, the JS L2 HNSW call was about 8.5× the Float32 exact-scan QPS on
pen digits and 15× on letters, while touching 4.7% and 3.0% of the vectors
respectively. Optical digits, at 64 dimensions, reached 2,596 QPS versus
472 exact-scan QPS, with 1.0000 tie-aware recall on these 200 queries. These
ratios are workload-specific. SQ8 used 25% less serialized space on pen digits
and 26% less on letters, with lower tie-aware recall than Float32.
On wine quality, SQ8 reduced serialized size by about 20% but tie-aware
recall was 0.9615 at ef=64; the data-dependent quality cost remains material.

The tie-aware recall is 1.0 for Float32 at the tested widths, but the ID
recall on letters is lower because many vectors share the kth distance.
Neither result implies exact graph search on arbitrary future queries. The
new exact API is the appropriate choice when a completeness guarantee is
required.
