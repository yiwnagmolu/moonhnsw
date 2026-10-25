// Benchmark: build throughput, recall@10 and query throughput of the public
// API against a brute-force ground truth on uniform random data — for every
// distance metric, plus a larger scale run.
//
// Run with:  moon run --release examples/benchmark

///|
priv struct Rng {
  mut state : UInt64
}

///|
fn Rng::next_f64(self : Rng) -> Double {
  let state = self.state + 0x9E3779B97F4A7C15UL
  self.state = state
  let mut z = state
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL
  ((z ^ (z >> 31)) >> 11).to_double() * (1.0 / 9007199254740992.0)
}

///|
fn gen_dataset(seed : UInt64, n : Int32, dim : Int32) -> FixedArray[Float] {
  let rng = { state: seed, }
  let data : FixedArray[Float] = FixedArray::make(n * dim, 0.0)
  let mut i = 0
  while i < n * dim {
    data[i] = Float::from_double(rng.next_f64())
    i += 1
  }
  data
}

///|
fn row_of(
  data : FixedArray[Float],
  row : Int32,
  dim : Int32,
) -> FixedArray[Float] {
  let out : FixedArray[Float] = FixedArray::make(dim, 0.0)
  FixedArray::unsafe_blit(out, 0, data, row * dim, dim)
  out
}

///|
/// L2-normalizes every row of a flat dataset into a new array.
fn normalize_flat(
  data : FixedArray[Float],
  n : Int32,
  dim : Int32,
) -> FixedArray[Float] {
  let out : FixedArray[Float] = FixedArray::make(n * dim, 0.0)
  let mut i = 0
  while i < n {
    let off = i * dim
    let mut norm_sq = 0.0
    let mut j = 0
    while j < dim {
      let v = data.unsafe_get(off + j).to_double()
      norm_sq += v * v
      j += 1
    }
    j = 0
    if norm_sq > 0.0 {
      let inv = 1.0 / norm_sq.sqrt()
      while j < dim {
        out[off + j] = Float::from_double(
          data.unsafe_get(off + j).to_double() * inv,
        )
        j += 1
      }
    } else {
      while j < dim {
        out[off + j] = data.unsafe_get(off + j)
        j += 1
      }
    }
    i += 1
  }
  out
}

///|
/// Distance between query `q` and flat-dataset row `row`, allocation-free.
/// For `InnerProduct` the value is negated (smaller = more similar), which
/// also covers Cosine when both sides are pre-normalized.
fn dist_row(
  metric : @moonhnsw.Distance,
  data : FixedArray[Float],
  row : Int32,
  dim : Int32,
  q : FixedArray[Float],
) -> Double {
  let off = row * dim
  match metric {
    @moonhnsw.Distance::L2 => {
      let mut s : Float = 0.0
      let mut j = 0
      while j < dim {
        let d = data.unsafe_get(off + j) - q.unsafe_get(j)
        s += d * d
        j += 1
      }
      s.to_double()
    }
    _ => {
      let mut s : Float = 0.0
      let mut j = 0
      while j < dim {
        s += data.unsafe_get(off + j) * q.unsafe_get(j)
        j += 1
      }
      -s.to_double()
    }
  }
}

///|
/// Exact top-k ids via a single pass over all rows, maintaining a running
/// top-k — the fair brute-force baseline (n distances per query).
fn brute_topk(
  metric : @moonhnsw.Distance,
  truth_data : FixedArray[Float],
  n : Int32,
  dim : Int32,
  q : FixedArray[Float],
  k : Int32,
) -> Array[Int32] {
  // Running top-k as parallel id/distance arrays, worst at the end.
  let ids : FixedArray[Int32] = FixedArray::make(k, -1)
  let ds : FixedArray[Double] = FixedArray::make(k, 1.0e308)
  let mut filled = 0
  for i in 0..<n {
    let d = dist_row(metric, truth_data, i, dim, q)
    if filled < k {
      // Insertion into the sorted prefix.
      let mut j = filled
      while j > 0 && ds.unsafe_get(j - 1) > d {
        ids[j] = ids.unsafe_get(j - 1)
        ds[j] = ds.unsafe_get(j - 1)
        j -= 1
      }
      ids[j] = i
      ds[j] = d
      filled += 1
    } else if d < ds.unsafe_get(k - 1) {
      let mut j = k - 1
      while j > 0 && ds.unsafe_get(j - 1) > d {
        ids[j] = ids.unsafe_get(j - 1)
        ds[j] = ds.unsafe_get(j - 1)
        j -= 1
      }
      ids[j] = i
      ds[j] = d
    }
  }
  let out : Array[Int32] = []
  for j in 0..<filled {
    out.push(ids.unsafe_get(j))
  }
  out
}

///|
fn fmt2(x : Double) -> String {
  ((x * 100.0 + 0.5).to_int().to_double() / 100.0).to_string()
}

///|
fn bench_metric(
  name : String,
  metric : @moonhnsw.Distance,
  n : Int32,
  dim : Int32,
  n_queries : Int32,
  k : Int32,
  m : Int32,
  ef_construction : Int32,
  seed : UInt64,
  storage~ : @moonhnsw.Storage = @moonhnsw.Storage::Float32,
) -> Unit raise @moonhnsw.HnswError {
  println(
    "--- \{name}  n=\{n} dim=\{dim} queries=\{n_queries} k=\{k} m=\{m} ef_construction=\{ef_construction} ---",
  )
  let raw = gen_dataset(seed, n, dim)
  let queries = gen_dataset(seed + 1UL, n_queries, dim)
  // Ground truth for Cosine = negated inner product over pre-normalized
  // vectors; the index normalizes internally, the truth side does it here.
  let for_cosine = match metric {
    @moonhnsw.Distance::Cosine => true
    _ => false
  }
  let truth_metric = if for_cosine {
    @moonhnsw.Distance::InnerProduct
  } else {
    metric
  }
  let truth_data = if for_cosine { normalize_flat(raw, n, dim) } else { raw }
  // Build.
  let t0 = @bench.monotonic_clock_start()
  let idx = @moonhnsw.Hnsw::new(
    dim,
    metric,
    params=@moonhnsw.HnswParams::new(m~, ef_construction~, storage~),
  )
  for i in 0..<n {
    let _ = idx.add(row_of(raw, i, dim))
  }
  let build_us = @bench.monotonic_clock_end(t0)
  let s = idx.stats()
  println(
    "memory          : \{fmt2(s.estimated_bytes.to_double() / 1048576.0)} MB (est.)",
  )
  println(
    "build           : \{fmt2(build_us / 1000.0)} ms total, \{fmt2(build_us / n.to_double())} us per vector",
  )
  // Persistence round-trip.
  let ts = @bench.monotonic_clock_start()
  let payload = idx.save()
  let save_us = @bench.monotonic_clock_end(ts)
  let tl = @bench.monotonic_clock_start()
  let reloaded = @moonhnsw.Hnsw::load(payload)
  let load_us = @bench.monotonic_clock_end(tl)
  println(
    "persistence     : save \{fmt2(save_us / 1000.0)} ms, load \{fmt2(load_us / 1000.0)} ms (\{fmt2(payload.length().to_double() / 1048576.0)} MB payload)",
  )
  let _ = reloaded
  // Ground truth (not timed).
  let truths : Array[Array[Int32]] = []
  for qi in 0..<n_queries {
    let q = row_of(queries, qi, dim)
    let qt = if for_cosine { normalize_flat(q, 1, dim) } else { q }
    truths.push(brute_topk(truth_metric, truth_data, n, dim, qt, k))
  }
  // Warm-up search.
  let _ = idx.search(row_of(queries, 0, dim), k, ef=64)
  // Timed brute-force baseline for the speedup column.
  let tb = @bench.monotonic_clock_start()
  for qi in 0..<n_queries {
    let q = row_of(queries, qi, dim)
    let qt = if for_cosine { normalize_flat(q, 1, dim) } else { q }
    let _ = brute_topk(truth_metric, truth_data, n, dim, qt, k)
  }
  let brute_us = @bench.monotonic_clock_end(tb)
  let brute_qps = n_queries.to_double() / (brute_us / 1000000.0)
  // Search at several beam widths.
  for ef in [64, 128, 256] {
    let mut hits = 0
    let mut dists_total = 0L
    let t1 = @bench.monotonic_clock_start()
    for qi in 0..<n_queries {
      let res = idx.search(row_of(queries, qi, dim), k, ef~)
      dists_total += idx.distance_computations()
      let truth = truths[qi]
      for r in res {
        for t in truth {
          if r.id == t {
            hits += 1
            break
          }
        }
      }
    }
    let us = @bench.monotonic_clock_end(t1)
    let recall = hits.to_double() / (n_queries * k).to_double()
    let qps = n_queries.to_double() / (us / 1000000.0)
    let speedup = qps / brute_qps
    let dists_per_query = dists_total.to_double() / n_queries.to_double()
    println(
      "search ef=\{ef}\t: recall@\{k}=\{fmt2(recall)}  qps=\{fmt2(qps)}  \{fmt2(speedup)}x brute  \{fmt2(dists_per_query)} dists/query (brute \{n})",
    )
  }
}

///|
fn run_all() -> Unit raise @moonhnsw.HnswError {
  let dim = 32
  let k = 10
  let m = 16
  let ef_construction = 200
  bench_metric(
    "L2",
    @moonhnsw.Distance::L2,
    2000,
    dim,
    100,
    k,
    m,
    ef_construction,
    20261024UL,
  )
  bench_metric(
    "InnerProduct",
    @moonhnsw.Distance::InnerProduct,
    2000,
    dim,
    100,
    k,
    m,
    ef_construction,
    20261025UL,
  )
  bench_metric(
    "Cosine",
    @moonhnsw.Distance::Cosine,
    2000,
    dim,
    100,
    k,
    m,
    ef_construction,
    20261026UL,
  )
  bench_metric(
    "L2 scale",
    @moonhnsw.Distance::L2,
    10000,
    dim,
    50,
    k,
    m,
    ef_construction,
    20261027UL,
  )
  bench_metric(
    "L2 SQ8",
    @moonhnsw.Distance::L2,
    2000,
    dim,
    100,
    k,
    m,
    ef_construction,
    20261024UL,
    storage=@moonhnsw.Storage::SQ8,
  )
}

///|
fn main {
  match (Ok(run_all()) catch { e => Err(e) }) {
    Ok(_) => ()
    Err(e) => println("error: \{e.message()}")
  }
}
