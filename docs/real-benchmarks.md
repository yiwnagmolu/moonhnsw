# Real-data nearest-neighbor benchmarks (2026-10-29)

This report measures the **moonhnsw MoonBit library**, compiled to JavaScript,
on two public UCI datasets. It supplements the uniform-random wasm-gc runs in
the README; the datasets, backend and machine differ, so their timings are not
direct before/after speedup measurements.

## Data and reproduction

- [Pen-Based Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits), E. Alpaydin and F. Alimoglu, DOI [10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K): 7,494 official training vectors; the first 200 of 3,498 official test vectors are queries.
- [Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition), D. Slate, DOI [10.24432/C5ZP40](https://doi.org/10.24432/C5ZP40): first 16,000 vectors for construction and the next 200 for queries, following UCI's suggested 16,000/4,000 split.

Both datasets have 16 numerical features. Class labels are excluded; this is
nearest-neighbor retrieval, not classification accuracy. Raw feature values
are converted to f32. The UCI pages identify both datasets as CC BY 4.0.
`benchmarks/real/run.cjs` downloads the original CSV files from UCI, verifies
their pinned SHA-256 hashes, and caches them outside the repository. It then
invokes the MoonBit JS release build. No dataset copy is checked in.

Run from the repository root with Node.js 24+ and MoonBit installed:

```text
node benchmarks/real/run.cjs pendigits
node benchmarks/real/run.cjs letters
```

The underlying MoonBit executable is in `benchmarks/real/main.mbt`. The Node
launcher only loads data and starts it. Construction, exact search, HNSW
search, distance counts and timings execute in MoonBit. Source hashes and
complete output from this run are in [pendigits](benchmarks/pendigits-2026-10-29.txt)
and [letters](benchmarks/letters-2026-10-29.txt).

## Method

- Windows x64; AMD Ryzen 7 5800H; Node v24.21.0; `moon 0.1.20261023`,
  `moonc v0.10.17+d971670cd`; JS release build.
- `m=16`, `ef_construction=200`, default fixed seed; `k=10` and
  `ef=64/128/256`. Every query uses an independent held-out data row.
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

## Results

Payload is the serialized size. `d/q` counts MoonBit distance evaluations
per query. Exact scan evaluates all training vectors. QPS varies across
short runs; these numbers describe this one machine, build and sample. SQ8
exact-scan QPS was not measured; its recall is against the Float32 truth.

### Pen digits: 7,494 × 16, 200 held-out queries

| Metric / store | Build ms | Payload MiB | Exact QPS | ef | ID recall | Tie recall | HNSW QPS | d/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 / Float32 | 3795 | 1.34 | 929 | 64 | 0.9995 | 1.0000 | 8312 | 350 |
| L2 / Float32 |  |  |  | 128 | 0.9990 | 1.0000 | 5280 | 487 |
| L2 / Float32 |  |  |  | 256 | 0.9990 | 1.0000 | 2922 | 714 |
| L2 / SQ8 | 2315 | 1.00 | — | 64 | 0.9885 | 0.9890 | 12707 | 350 |
| L2 / SQ8 |  |  |  | 128 | 0.9885 | 0.9890 | 7049 | 487 |
| L2 / SQ8 |  |  |  | 256 | 0.9885 | 0.9890 | 2995 | 714 |
| Cosine / Float32 | 2731 | 1.34 | 1839 | 64 | 1.0000 | 1.0000 | 11019 | 347 |
| Cosine / Float32 |  |  |  | 128 | 1.0000 | 1.0000 | 5247 | 483 |
| Cosine / Float32 |  |  |  | 256 | 1.0000 | 1.0000 | 2701 | 706 |

### Letters: 16,000 × 16, 200 held-out queries

| Metric / store | Build ms | Payload MiB | Exact QPS | ef | ID recall | Tie recall | HNSW QPS | d/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 / Float32 | 13596 | 2.88 | 362 | 64 | 0.9445 | 1.0000 | 5256 | 488 |
| L2 / Float32 |  |  |  | 128 | 0.9510 | 1.0000 | 2607 | 766 |
| L2 / Float32 |  |  |  | 256 | 0.9480 | 1.0000 | 1317 | 1245 |
| L2 / SQ8 | 7270 | 2.14 | — | 64 | 0.9345 | 0.9920 | 9570 | 472 |
| L2 / SQ8 |  |  |  | 128 | 0.9370 | 0.9920 | 5137 | 743 |
| L2 / SQ8 |  |  |  | 256 | 0.9345 | 0.9920 | 2271 | 1212 |
| Cosine / Float32 | 8592 | 2.88 | 688 | 64 | 0.9905 | 1.0000 | 7462 | 472 |
| Cosine / Float32 |  |  |  | 128 | 0.9920 | 1.0000 | 2635 | 748 |
| Cosine / Float32 |  |  |  | 256 | 0.9940 | 1.0000 | 1944 | 1215 |

At ef=64, the JS L2 HNSW call was about 9× the Float32 exact-scan QPS on
pen digits and about 15× on letters, while touching 4.7% and 3.0% of the
vectors respectively. These ratios are specific to the two workloads. SQ8
used 25% less serialized space on pen digits and 26% less on letters; it
traded roughly 1–2 percentage points of tie-aware recall for higher QPS.

The tie-aware recall is 1.0 for Float32 at the tested widths, but the ID
recall on letters is lower because many vectors share the kth distance.
Neither result implies exact graph search on arbitrary future queries. The
new exact API is the appropriate choice when a completeness guarantee is
required.
