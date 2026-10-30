// Large-scale benchmark: 100k vectors, L2. Build takes minutes on the wasm
// backends, so this is meant for occasional manual runs rather than CI:
//
//   moon run --target wasm-gc examples/benchmark-large

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
/// Squared L2 distance between query `q` and flat-dataset row `row`.
fn l2_row(
  data : FixedArray[Float],
  row : Int32,
  dim : Int32,
  q : FixedArray[Float],
) -> Double {
  let off = row * dim
  let mut s : Float = 0.0
  let mut j = 0
  while j < dim {
    let d = data.unsafe_get(off + j) - q.unsafe_get(j)
    s += d * d
    j += 1
  }
  s.to_double()
}

///|
/// Exact top-k ids via a single pass over all rows — the fair brute-force
/// baseline (n distances per query).
fn brute_topk(
  data : FixedArray[Float],
  n : Int32,
  dim : Int32,
  q : FixedArray[Float],
  k : Int32,
) -> Array[Int32] {
  let ids : FixedArray[Int32] = FixedArray::make(k, -1)
  let ds : FixedArray[Double] = FixedArray::make(k, 1.0e308)
  let mut filled = 0
  for i in 0..<n {
    let d = l2_row(data, i, dim, q)
    if filled < k {
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
fn fmt_mb(bytes : Int64) -> String {
  fmt2(bytes.to_double() / (1024.0 * 1024.0))
}

///|
fn run_all() -> Unit raise @moonhnsw.HnswError {
  let n = 100000
  let dim = 32
  let n_queries = 20
  let k = 10
  let m = 16
  let ef_construction = 200
  println(
    "--- L2 scale  n=\{n} dim=\{dim} queries=\{n_queries} k=\{k} m=\{m} ef_construction=\{ef_construction} ---",
  )
  let data = gen_dataset(20261028UL, n, dim)
  let queries = gen_dataset(20261029UL, n_queries, dim)
  // Build.
  let t0 = @bench.monotonic_clock_start()
  let idx = @moonhnsw.Hnsw::new(
    dim,
    @moonhnsw.Distance::L2,
    params=@moonhnsw.HnswParams::new(m~, ef_construction~),
  )
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  let build_us = @bench.monotonic_clock_end(t0)
  println(
    "build           : \{fmt2(build_us / 1000.0)} ms total, \{fmt2(build_us / n.to_double())} us per vector",
  )
  let s = idx.stats()
  println(
    "graph           : max_level=\{s.max_level}  est. memory=\{fmt_mb(s.estimated_bytes)} MB",
  )
  // Ground truth (not timed).
  let truths : Array[Array[Int32]] = []
  for qi in 0..<n_queries {
    truths.push(brute_topk(data, n, dim, row_of(queries, qi, dim), k))
  }
  // Warm-up search.
  let _ = idx.search(row_of(queries, 0, dim), k, ef=64)
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
    // A timed brute-force pass on the same queries for the speedup column.
    let tb = @bench.monotonic_clock_start()
    for qi in 0..<n_queries {
      let _ = brute_topk(data, n, dim, row_of(queries, qi, dim), k)
    }
    let brute_us = @bench.monotonic_clock_end(tb)
    let brute_qps = n_queries.to_double() / (brute_us / 1000000.0)
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
fn main raise {
  run_all()
}
