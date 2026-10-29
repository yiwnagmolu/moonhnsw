// Run through `node benchmarks/real/run.cjs pendigits|letters`; the launcher
// downloads and checks the official UCI files, then provides their f32 values
// to this MoonBit executable. Timed sections run in MoonBit only.

///|
extern "js" fn dataset_name() -> String =
  #| () => globalThis.__moonhnsw_dataset.name

///|
extern "js" fn dataset_dim() -> Int32 =
  #| () => globalThis.__moonhnsw_dataset.dim

///|
extern "js" fn train_count() -> Int32 =
  #| () => globalThis.__moonhnsw_dataset.train.length / globalThis.__moonhnsw_dataset.dim

///|
extern "js" fn query_count() -> Int32 =
  #| () => globalThis.__moonhnsw_dataset.test.length / globalThis.__moonhnsw_dataset.dim

///|
extern "js" fn train_value(i : Int32) -> Float =
  #| (i) => globalThis.__moonhnsw_dataset.train[i]

///|
extern "js" fn query_value(i : Int32) -> Float =
  #| (i) => globalThis.__moonhnsw_dataset.test[i]

///|
fn read_rows(n : Int32, dim : Int32, train : Bool) -> Array[FixedArray[Float]] {
  let rows : Array[FixedArray[Float]] = []
  for i in 0..<n {
    let row : FixedArray[Float] = FixedArray::make(dim, 0.0)
    for j in 0..<dim {
      let offset = i * dim + j
      row[j] = if train { train_value(offset) } else { query_value(offset) }
    }
    rows.push(row)
  }
  rows
}

///|
fn fmt2(x : Double) -> String {
  ((x * 100.0 + 0.5).to_int().to_double() / 100.0).to_string()
}

///|
fn fmt4(x : Double) -> String {
  ((x * 10000.0 + 0.5).to_int().to_double() / 10000.0).to_string()
}

///|
fn timed_exact(
  idx : @moonhnsw.Hnsw,
  queries : Array[FixedArray[Float]],
  k : Int32,
) -> (Array[Array[@moonhnsw.Neighbor]], Double, Int64) raise @moonhnsw.HnswError {
  let truths : Array[Array[@moonhnsw.Neighbor]] = []
  let mut checksum = 0L
  let times : Array[Double] = []
  let _ = idx.search_exact(queries[0], k)
  for repeat in 0..<3 {
    let t = @bench.monotonic_clock_start()
    for qi in 0..<queries.length() {
      let hits = idx.search_exact(queries[qi], k)
      if repeat == 0 {
        truths.push(hits)
      }
      for rank in 0..<hits.length() {
        checksum += (qi + 1).to_int64() *
          (rank + 1).to_int64() *
          (hits[rank].id + 1).to_int64()
      }
    }
    times.push(@bench.monotonic_clock_end(t))
  }
  times.sort()
  (truths, times[1], checksum)
}

///|
fn bench_search(
  idx : @moonhnsw.Hnsw,
  train : Array[FixedArray[Float]],
  queries : Array[FixedArray[Float]],
  truths : Array[Array[@moonhnsw.Neighbor]],
  metric : @moonhnsw.Distance,
  k : Int32,
  ef : Int32,
) -> Unit raise @moonhnsw.HnswError {
  let _ = idx.search(queries[0], k, ef~)
  let times : Array[Double] = []
  let mut checksum = 0L
  let mut total_dists = 0L
  let mut matched_ids = 0
  let mut matched_ties = 0
  // Quality accounting is outside the timed section. A tied neighbor at the
  // exact kth distance is valid even if it has a different internal id.
  for qi in 0..<queries.length() {
    let hits = idx.search(queries[qi], k, ef~)
    total_dists += idx.distance_computations()
    let threshold = truths[qi][k - 1].distance
    for hit in hits {
      for truth in truths[qi] {
        if hit.id == truth.id {
          matched_ids += 1
          break
        }
      }
      let full_distance = @moonhnsw.distance(metric, queries[qi], train[hit.id])
      if full_distance <= threshold {
        matched_ties += 1
      }
    }
  }
  for _ in 0..<3 {
    let t = @bench.monotonic_clock_start()
    for qi in 0..<queries.length() {
      let hits = idx.search(queries[qi], k, ef~)
      for rank in 0..<hits.length() {
        checksum += (qi + 1).to_int64() *
          (rank + 1).to_int64() *
          (hits[rank].id + 1).to_int64()
      }
    }
    times.push(@bench.monotonic_clock_end(t))
  }
  times.sort()
  let queries_n = queries.length().to_double()
  println(
    "  ef=\{ef}: id_recall@\{k}=\{fmt4(matched_ids.to_double() / (queries_n * k.to_double()))} tie_recall@\{k}=\{fmt4(matched_ties.to_double() / (queries_n * k.to_double()))} median_qps=\{fmt2(queries_n * 1000000.0 / times[1])} dists/query=\{fmt2(total_dists.to_double() / queries_n)} checksum=\{checksum}",
  )
}

///|
fn run() -> Unit raise @moonhnsw.HnswError {
  let name = dataset_name()
  let dim = dataset_dim()
  let n = train_count()
  let nq = query_count()
  let k = 10
  println("dataset=\{name} train=\{n} queries=\{nq} dim=\{dim} k=\{k}")
  let train = read_rows(n, dim, true)
  let queries = read_rows(nq, dim, false)
  for metric in [@moonhnsw.Distance::L2, @moonhnsw.Distance::Cosine] {
    let metric_name = if metric is @moonhnsw.Distance::L2 {
      "L2"
    } else {
      "Cosine"
    }
    let idx = @moonhnsw.Hnsw::new(dim, metric)
    let start = @bench.monotonic_clock_start()
    for row in train {
      let _ = idx.add(row)
    }
    let build_us = @bench.monotonic_clock_end(start)
    let saved = idx.save()
    println(
      "metric=\{metric_name} storage=Float32 build_ms=\{fmt2(build_us / 1000.0)} payload_MiB=\{fmt2(saved.length().to_double() / 1048576.0)} est_MiB=\{fmt2(idx.stats().estimated_bytes.to_double() / 1048576.0)}",
    )
    let (truths, exact_us, exact_checksum) = timed_exact(idx, queries, k)
    println(
      "  exact: median_qps=\{fmt2(nq.to_double() * 1000000.0 / exact_us)} dists/query=\{n} checksum=\{exact_checksum}",
    )
    for ef in [64, 128, 256] {
      bench_search(idx, train, queries, truths, metric, k, ef)
    }
    if metric is @moonhnsw.Distance::L2 {
      let sq8 = @moonhnsw.Hnsw::new(
        dim,
        metric,
        params=@moonhnsw.HnswParams::new(storage=@moonhnsw.Storage::SQ8),
      )
      let t = @bench.monotonic_clock_start()
      for row in train {
        let _ = sq8.add(row)
      }
      let sq8_us = @bench.monotonic_clock_end(t)
      println(
        "metric=L2 storage=SQ8 build_ms=\{fmt2(sq8_us / 1000.0)} payload_MiB=\{fmt2(sq8.save().length().to_double() / 1048576.0)} est_MiB=\{fmt2(sq8.stats().estimated_bytes.to_double() / 1048576.0)}",
      )
      for ef in [64, 128, 256] {
        bench_search(sq8, train, queries, truths, metric, k, ef)
      }
    }
  }
}

///|
fn main {
  match (Ok(run()) catch { e => Err(e) }) {
    Ok(_) => ()
    Err(e) => println("benchmark error: \{e.message()}")
  }
}
