// Whitebox tests: exercise internal helpers (heap, rng, brute force) and the
// public API from the same package. Public-surface behavior is additionally
// covered from an external package's perspective in api_test.mbt.

///|
test "heap min and max ordering" {
  let h = Heap::new(true)
  h.push(3.0, 3)
  h.push(1.0, 1)
  h.push(2.0, 2)
  assert_eq(h.top().0, 1.0)
  assert_eq(h.pop().1, 1)
  assert_eq(h.pop().1, 2)
  assert_eq(h.pop().1, 3)
  assert_eq(h.size(), 0)
  let g = Heap::new(false)
  g.push(1.0, 1)
  g.push(5.0, 5)
  g.push(3.0, 3)
  assert_eq(g.top().1, 5)
  assert_eq(g.pop().1, 5)
  assert_eq(g.pop().1, 3)
  assert_eq(g.pop().1, 1)
}

///|
/// Deterministic SplitMix64 PRNG for test datasets.
priv struct TestRng {
  mut state : UInt64
}

///|
fn TestRng::new(seed : UInt64) -> TestRng {
  { state: seed, }
}

///|
fn TestRng::next_u64(self : TestRng) -> UInt64 {
  let (state, out) = splitmix_next(self.state)
  self.state = state
  out
}

///|
/// Uniform random double in [0, 1).
fn TestRng::next_f64(self : TestRng) -> Double {
  (self.next_u64() >> 11).to_double() * (1.0 / 9007199254740992.0)
}

///|
test "rng is deterministic per seed" {
  let a = TestRng::new(123UL)
  let b = TestRng::new(123UL)
  let mut same = true
  for _ in 0..<8 {
    if a.next_u64() != b.next_u64() {
      same = false
    }
  }
  let c = TestRng::new(123UL)
  let d = TestRng::new(124UL)
  let mut diff = false
  for _ in 0..<8 {
    if c.next_f64() != d.next_f64() {
      diff = true
    }
  }
  assert_true(same)
  assert_true(diff)
}

///|
test "small exact query" {
  let idx = Hnsw::new(2, L2)
  assert_eq(idx.add(FixedArray::from_array([0.0, 0.0])), 0)
  assert_eq(idx.add(FixedArray::from_array([1.0, 0.0])), 1)
  assert_eq(idx.add(FixedArray::from_array([0.0, 1.0])), 2)
  assert_eq(idx.size(), 3)
  let res = idx.search(FixedArray::from_array([0.9, 0.05]), 1)
  assert_eq(res.length(), 1)
  assert_eq(res[0].id, 1)
  assert_true(res[0].distance < 0.05)
}

///|
test "edge cases" {
  let idx = Hnsw::new(4, L2)
  assert_eq(idx.search(FixedArray::make(4, 0.0), 5).length(), 0)
  let _ = idx.add(FixedArray::from_array([1.0, 0.0, 0.0, 0.0]))
  // k larger than the number of stored vectors
  assert_eq(idx.search(FixedArray::make(4, 0.0), 5).length(), 1)
  // k = 0
  assert_eq(idx.search(FixedArray::make(4, 0.0), 0).length(), 0)
  // dimension mismatches raise
  let bad_add : Result[Int32, HnswError] = Ok(idx.add(FixedArray::make(3, 0.0))) catch {
    e => Err(e)
  }
  assert_true(bad_add is Err(_))
  let bad_search : Result[Array[Neighbor], HnswError] = Ok(
    idx.search(FixedArray::make(5, 0.0), 1),
  ) catch {
    e => Err(e)
  }
  assert_true(bad_search is Err(_))
  // invalid constructor arguments raise
  let bad_dim : Result[Hnsw, HnswError] = Ok(Hnsw::new(0, L2)) catch {
    e => Err(e)
  }
  assert_true(bad_dim is Err(_))
}

///|
test "duplicate vectors" {
  let idx = Hnsw::new(2, L2)
  let v : FixedArray[Float] = FixedArray::from_array([1.0, 1.0])
  for _ in 0..<50 {
    let _ = idx.add(v)
  }
  let res = idx.search(v, 5)
  assert_eq(res.length(), 5)
  assert_eq(res[0].distance, 0.0)
}

///|
fn gen_dataset(
  seed : UInt64,
  n : Int32,
  dim : Int32,
  low : Double,
  high : Double,
) -> FixedArray[Float] {
  let rng = TestRng::new(seed)
  let data : FixedArray[Float] = FixedArray::make(n * dim, 0.0)
  let mut i = 0
  while i < n * dim {
    data[i] = Float::from_double(low + rng.next_f64() * (high - low))
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
/// Exact top-k ids by brute force, using the same distance convention as the
/// index. Callers pre-normalize `data` and `q` for Cosine.
fn brute_topk(
  metric : Distance,
  data : FixedArray[Float],
  n : Int32,
  dim : Int32,
  q : FixedArray[Float],
  k : Int32,
) -> Array[Int32] {
  let dists : Array[(Double, Int32)] = []
  for i in 0..<n {
    dists.push((distance_between(metric, q, 0, data, i * dim, dim), i))
  }
  let _ = sort_pairs_ascending(dists)
  let out : Array[Int32] = []
  for i in 0..<k {
    out.push(dists[i].1)
  }
  out
}

///|
fn normalize_dataset(
  data : FixedArray[Float],
  n : Int32,
  dim : Int32,
) -> FixedArray[Float] {
  let out : FixedArray[Float] = FixedArray::make(n * dim, 0.0)
  for i in 0..<n {
    let _ = copy_normalized(data, i * dim, out, i * dim, dim, true)
  }
  out
}

///|
fn recall_of(
  metric : Distance,
  seed : UInt64,
  n : Int32,
  dim : Int32,
  m : Int32,
  ef_construction : Int32,
  ef : Int32,
  k : Int32,
  storage~ : Storage = Float32,
) -> Double raise HnswError {
  // Uniform data in [0, 1)^dim avoids zero vectors for Cosine.
  let data = gen_dataset(seed, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(
    dim,
    metric,
    params=HnswParams::new(m~, ef_construction~, seed=seed + 1UL, storage~),
  )
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  let queries = gen_dataset(seed + 2UL, 20, dim, 0.0, 1.0)
  let truth_data = match metric {
    Cosine => normalize_dataset(data, n, dim)
    _ => data
  }
  let mut hits = 0
  for qi in 0..<20 {
    let raw_q = row_of(queries, qi, dim)
    let q = match metric {
      Cosine => {
        let nq : FixedArray[Float] = FixedArray::make(dim, 0.0)
        let _ = copy_normalized(raw_q, 0, nq, 0, dim, true)
        nq
      }
      _ => raw_q
    }
    let res = idx.search(raw_q, k, ef~)
    let truth = brute_topk(metric, truth_data, n, dim, q, k)
    for r in res {
      for t in truth {
        if r.id == t {
          hits += 1
          break
        }
      }
    }
  }
  hits.to_double() / (20 * k).to_double()
}

///|
test "recall@10 for L2" {
  let recall = recall_of(L2, 42UL, 400, 16, 16, 200, 100, 10)
  assert_true(recall >= 0.9)
}

///|
test "recall@10 for inner product" {
  let recall = recall_of(InnerProduct, 43UL, 400, 16, 16, 200, 150, 10)
  assert_true(recall >= 0.85)
}

///|
test "recall@10 for cosine" {
  let recall = recall_of(Cosine, 44UL, 400, 16, 16, 200, 100, 10)
  assert_true(recall >= 0.9)
}

///|
test "hierarchy emerges for larger indices" {
  let data = gen_dataset(45UL, 400, 8, 0.0, 1.0)
  let idx = Hnsw::new(8, L2)
  for i in 0..<400 {
    let _ = idx.add(row_of(data, i, 8))
  }
  // With m = 16 the chance that no node reaches layer 1 is ~e^-25.
  assert_true(idx.max_level >= 1)
}

///|
test "save/load roundtrip preserves search results" {
  let n = 200
  let dim = 8
  let data = gen_dataset(7UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(
    dim,
    L2,
    params=HnswParams::new(m=12, ef_construction=120, seed=99UL),
  )
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  let idx2 = Hnsw::load(idx.save())
  assert_eq(idx2.size(), n)
  assert_eq(idx2.max_level, idx.max_level)
  assert_eq(idx2.entry, idx.entry)
  idx2.verify()
  let queries = gen_dataset(8UL, 5, dim, 0.0, 1.0)
  for qi in 0..<5 {
    let q = row_of(queries, qi, dim)
    let a = idx.search(q, 10, ef=64)
    let b = idx2.search(q, 10, ef=64)
    assert_eq(a.length(), b.length())
    for j in 0..<a.length() {
      assert_eq(a[j].id, b[j].id)
      assert_eq(a[j].distance, b[j].distance)
    }
  }
  // A loaded index keeps accepting insertions.
  assert_eq(idx2.add(FixedArray::make(dim, 0.5)), n)
  assert_eq(idx2.size(), n + 1)
}

///|
test "save/load empty index and every metric roundtrips" {
  for metric in [L2, InnerProduct, Cosine] {
    let idx = Hnsw::new(3, metric)
    let idx2 = Hnsw::load(idx.save())
    assert_eq(idx2.size(), 0)
    assert_eq(idx2.search(FixedArray::make(3, 1.0), 5).length(), 0)
    let _ = idx2.add(FixedArray::from_array([1.0, 0.0, 0.0]))
    let _ = idx2.add(FixedArray::from_array([0.0, 1.0, 0.0]))
    assert_eq(idx2.size(), 2)
  }
}

///|
test "cosine distances live in [-2, 0] and rank by angle" {
  let idx = Hnsw::new(2, Cosine)
  let _ = idx.add(FixedArray::from_array([1.0, 0.0]))
  let _ = idx.add(FixedArray::from_array([0.0, 1.0]))
  let res = idx.search(FixedArray::from_array([1.0, 0.0]), 2)
  // distance = -cos(angle): identical direction scores -1, orthogonal 0.
  assert_eq(res[0].id, 0)
  assert_true(res[0].distance < -0.99)
  assert_true(res[1].distance > -0.01 && res[1].distance < 0.01)
}

///|
test "corrupt payloads are rejected" {
  let idx = Hnsw::new(4, L2)
  let _ = idx.add(FixedArray::from_array([1.0, 0.0, 0.0, 0.0]))
  let bytes = idx.save()
  // Truncated payload.
  let truncated : FixedArray[Byte] = FixedArray::make(
    bytes.length() - 10,
    b'\x00',
  )
  FixedArray::unsafe_blit(truncated, 0, bytes, 0, truncated.length())
  let r1 : Result[Hnsw, HnswError] = Ok(Hnsw::load(truncated)) catch {
    e => Err(e)
  }
  assert_true(r1 is Err(_))
  // Broken magic.
  let bad_magic : FixedArray[Byte] = FixedArray::make(bytes.length(), b'\x00')
  FixedArray::unsafe_blit(bad_magic, 0, bytes, 0, bytes.length())
  bad_magic[0] = b'X'
  let r2 : Result[Hnsw, HnswError] = Ok(Hnsw::load(bad_magic)) catch {
    e => Err(e)
  }
  assert_true(r2 is Err(_))
}

///|
test "delete excludes vectors from search results" {
  let n = 100
  let dim = 8
  let data = gen_dataset(51UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  // Querying with an inserted vector must return that vector first.
  let q = row_of(data, 7, dim)
  let before = idx.search(q, 1)
  assert_eq(before[0].id, 7)
  let _ = idx.delete(7)
  assert_eq(idx.size(), n - 1)
  assert_eq(idx.total(), n)
  let after = idx.search(q, 1)
  assert_true(after[0].id != 7)
  // Asking for more results than alive vectors returns exactly the alive set.
  let all = idx.search(q, n, ef=200)
  assert_eq(all.length(), n - 1)
  for r in all {
    assert_true(r.id != 7)
  }
}

///|
test "delete all vectors empties the index, re-add still works" {
  let dim = 4
  let idx = Hnsw::new(dim, L2)
  for i in 0..<20 {
    let _ = idx.add(FixedArray::make(dim, Float::from_double(i.to_double())))
  }
  for i in 0..<20 {
    let _ = idx.delete(i)
  }
  assert_eq(idx.size(), 0)
  assert_eq(idx.total(), 20)
  assert_eq(idx.search(FixedArray::make(dim, 1.0), 5).length(), 0)
  let new_id = idx.add(FixedArray::make(dim, 99.0))
  assert_eq(new_id, 20)
  assert_eq(idx.size(), 1)
  let res = idx.search(FixedArray::make(dim, 99.0), 1)
  assert_eq(res[0].id, 20)
}

///|
test "double delete and unknown id raise InvalidId" {
  let idx = Hnsw::new(2, L2)
  let _ = idx.add(FixedArray::from_array([1.0, 0.0]))
  let _ = idx.add(FixedArray::from_array([0.0, 1.0]))
  let _ = idx.delete(0)
  let again : Result[Unit, HnswError] = Ok(idx.delete(0)) catch { e => Err(e) }
  match again {
    Err(InvalidId(id)) => assert_eq(id, 0)
    _ => fail("expected InvalidId")
  }
  let unknown : Result[Unit, HnswError] = Ok(idx.delete(999)) catch {
    e => Err(e)
  }
  assert_true(unknown is Err(_))
  let negative : Result[Unit, HnswError] = Ok(idx.delete(-1)) catch {
    e => Err(e)
  }
  assert_true(negative is Err(_))
}

///|
test "save/load preserves deletions" {
  let n = 60
  let dim = 8
  let data = gen_dataset(52UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  for dead in [3, 17, 42, 59] {
    let _ = idx.delete(dead)
  }
  assert_eq(idx.size(), n - 4)
  let idx2 = Hnsw::load(idx.save())
  assert_eq(idx2.size(), n - 4)
  assert_eq(idx2.total(), n)
  let queries = gen_dataset(53UL, 5, dim, 0.0, 1.0)
  for qi in 0..<5 {
    let q = row_of(queries, qi, dim)
    let a = idx.search(q, 10, ef=64)
    let b = idx2.search(q, 10, ef=64)
    assert_eq(a.length(), b.length())
    for j in 0..<a.length() {
      assert_eq(a[j].id, b[j].id)
      assert_eq(a[j].distance, b[j].distance)
    }
    for r in b {
      assert_true(r.id != 3 && r.id != 17 && r.id != 42 && r.id != 59)
    }
  }
  // A loaded index keeps accepting insertions after deletions.
  let new_id = idx2.add(FixedArray::make(dim, 0.5))
  assert_eq(new_id, n)
  assert_eq(idx2.size(), n - 3)
}

///|
test "radius search matches brute force" {
  let n = 200
  let dim = 4
  let data = gen_dataset(61UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  let queries = gen_dataset(62UL, 10, dim, 0.0, 1.0)
  let mut hits = 0
  let mut truth_total = 0
  for qi in 0..<10 {
    let q = row_of(queries, qi, dim)
    // Radius = distance of the 10th nearest neighbor.
    let top10 = brute_topk(L2, data, n, dim, q, 10)
    let radius = distance_between(L2, q, 0, data, top10[9] * dim, dim)
    let truth : Array[Int32] = []
    for i in 0..<n {
      if distance_between(L2, q, 0, data, i * dim, dim) <= radius {
        truth.push(i)
      }
    }
    truth_total += truth.length()
    let res = idx.search_radius(q, radius)
    for r in res {
      assert_true(r.distance <= radius + 1.0e-9)
      for t in truth {
        if r.id == t {
          hits += 1
          break
        }
      }
    }
  }
  let recall = hits.to_double() / truth_total.to_double()
  assert_true(recall >= 0.85)
}

///|
test "radius zero returns exact matches only" {
  let idx = Hnsw::new(2, L2)
  let v : FixedArray[Float] = FixedArray::from_array([1.0, 1.0])
  for _ in 0..<5 {
    let _ = idx.add(v)
  }
  let _ = idx.add(FixedArray::from_array([0.5, 0.5]))
  // The graph entry point may be the outlier; exact matches must still be
  // found through it.
  let res = idx.search_radius(v, 0.0)
  assert_eq(res.length(), 5)
  for r in res {
    assert_eq(r.distance, 0.0)
  }
}

///|
test "radius search caps results and validates input" {
  let n = 50
  let dim = 4
  let data = gen_dataset(63UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  let q = row_of(data, 0, dim)
  let res = idx.search_radius(q, 1.0e9, max_results=5)
  assert_eq(res.length(), 5)
  for j in 1..<res.length() {
    assert_true(res[j - 1].distance <= res[j].distance)
  }
  // max_results <= 0 yields no results instead of an error.
  assert_eq(idx.search_radius(q, 1.0e9, max_results=0).length(), 0)
  // A negative radius raises.
  let bad : Result[Array[Neighbor], HnswError] = Ok(idx.search_radius(q, -1.0)) catch {
    e => Err(e)
  }
  assert_true(bad is Err(_))
  // A dimension mismatch raises.
  let baddim : Result[Array[Neighbor], HnswError] = Ok(
    idx.search_radius(FixedArray::make(3, 0.0), 1.0),
  ) catch {
    e => Err(e)
  }
  assert_true(baddim is Err(_))
}

///|
test "radius search respects deletions" {
  let dim = 4
  let idx = Hnsw::new(dim, L2)
  let v : FixedArray[Float] = FixedArray::from_array([1.0, 0.0, 0.0, 0.0])
  for _ in 0..<6 {
    let _ = idx.add(v)
  }
  let _ = idx.delete(0)
  let _ = idx.delete(2)
  let res = idx.search_radius(v, 0.0)
  assert_eq(res.length(), 4)
  for r in res {
    assert_true(r.id != 0 && r.id != 2)
  }
}

///|
test "contains and get_vector accessors" {
  let idx = Hnsw::new(3, L2)
  let _ = idx.add(FixedArray::from_array([1.0, 2.0, 3.0]))
  let _ = idx.add(FixedArray::from_array([4.0, 5.0, 6.0]))
  assert_true(idx.contains(0) && idx.contains(1))
  assert_true(!idx.contains(2))
  assert_true(!idx.contains(-1))
  let v = idx.get_vector(1)
  assert_eq(v[0], 4.0)
  assert_eq(v[2], 6.0)
  // Mutating the returned copy does not touch the index.
  v[0] = 99.0
  assert_eq(idx.get_vector(1)[0], 4.0)
  let _ = idx.delete(1)
  assert_true(!idx.contains(1))
  let dead : Result[FixedArray[Float], HnswError] = Ok(idx.get_vector(1)) catch {
    e => Err(e)
  }
  assert_true(dead is Err(_))
}

///|
test "get_vector returns the normalized vector for cosine" {
  let idx = Hnsw::new(2, Cosine)
  let _ = idx.add(FixedArray::from_array([3.0, 4.0]))
  let v = idx.get_vector(0)
  // (3, 4) normalized is (0.6, 0.8), within f32 rounding.
  assert_true(v[0].to_double() > 0.5999 && v[0].to_double() < 0.6001)
  assert_true(v[1].to_double() > 0.7999 && v[1].to_double() < 0.8001)
}

///|
test "stats reports level distribution and degrees" {
  let n = 400
  let dim = 8
  let data = gen_dataset(71UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  let s = idx.stats()
  assert_eq(s.count, n)
  assert_eq(s.alive, n)
  assert_eq(s.max_level, idx.max_level)
  assert_eq(s.nodes_per_level.length(), idx.max_level + 1)
  // Every node participates on layer 0; upper layers shrink geometrically.
  assert_eq(s.nodes_per_level[0], n)
  if s.max_level >= 1 {
    assert_true(s.nodes_per_level[1] > 0 && s.nodes_per_level[1] < n)
  }
  // Degrees respect the link budgets: 2m on layer 0, m above (m = 16).
  for l in 0..<s.nodes_per_level.length() {
    let budget = if l == 0 { 32.0 } else { 16.0 }
    assert_true(s.avg_degree_per_level[l] <= budget)
  }
  assert_true(s.avg_degree_per_level[0] > 0.0)
  // The memory estimate must at least cover the vector storage.
  assert_true(s.estimated_bytes >= (n * dim * 4).to_int64())
}

///|
test "stats on empty and partially deleted indexes" {
  let idx = Hnsw::new(3, L2)
  let s0 = idx.stats()
  assert_eq(s0.max_level, -1)
  assert_eq(s0.nodes_per_level.length(), 0)
  let _ = idx.add(FixedArray::make(3, 1.0))
  let _ = idx.add(FixedArray::make(3, 2.0))
  let _ = idx.delete(0)
  let s = idx.stats()
  assert_eq(s.alive, 1)
  assert_eq(s.count, 2)
  assert_eq(s.nodes_per_level[0], 2)
}

///|
test "compact reclaims deleted space and preserves neighbors" {
  let n = 150
  let dim = 8
  let data = gen_dataset(81UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  for dead in [5, 20, 60, 61, 62, 149] {
    let _ = idx.delete(dead)
  }
  let result = idx.compact()
  let fresh = result.index
  let mapping = result.mapping
  assert_eq(mapping.length(), n)
  // Dead ids map to -1; alive ids map onto consecutive new ids.
  for dead in [5, 20, 60, 61, 62, 149] {
    assert_eq(mapping[dead], -1)
  }
  assert_eq(fresh.size(), n - 6)
  assert_eq(fresh.total(), n - 6)
  // Top-1 neighbors agree between the original and compacted index.
  let queries = gen_dataset(82UL, 5, dim, 0.0, 1.0)
  for qi in 0..<5 {
    let q = row_of(queries, qi, dim)
    let old_top = idx.search(q, 1, ef=128)
    if idx.contains(old_top[0].id) {
      let new_top = fresh.search(q, 1, ef=128)
      assert_eq(new_top[0].id, mapping[old_top[0].id])
    }
  }
}

///|
test "compact of empty and fully deleted indexes" {
  let idx = Hnsw::new(3, L2)
  let empty = idx.compact()
  assert_eq(empty.index.size(), 0)
  assert_eq(empty.mapping.length(), 0)
  let _ = idx.add(FixedArray::make(3, 1.0))
  let _ = idx.add(FixedArray::make(3, 2.0))
  let _ = idx.delete(0)
  let _ = idx.delete(1)
  let all_dead = idx.compact()
  assert_eq(all_dead.index.size(), 0)
  assert_eq(all_dead.mapping[0], -1)
  assert_eq(all_dead.mapping[1], -1)
}

///|
test "labels default to ids and can be customized" {
  let idx = Hnsw::new(2, L2)
  let a = idx.add(FixedArray::from_array([1.0, 0.0]))
  let b = idx.add_labeled(FixedArray::from_array([0.0, 1.0]), 0xDEADBEEFUL)
  assert_eq(a, 0)
  assert_eq(b, 1)
  assert_eq(idx.label_of(0), 0UL)
  assert_eq(idx.label_of(1), 0xDEADBEEFUL)
  // Duplicate labels are allowed; lookup stays keyed by id.
  let c = idx.add_labeled(FixedArray::from_array([0.5, 0.5]), 0xDEADBEEFUL)
  assert_eq(idx.label_of(c), 0xDEADBEEFUL)
  // Unknown ids raise.
  let bad : Result[UInt64, HnswError] = Ok(idx.label_of(3)) catch {
    e => Err(e)
  }
  assert_true(bad is Err(_))
  let _ = idx.delete(0)
  let deleted : Result[UInt64, HnswError] = Ok(idx.label_of(0)) catch {
    e => Err(e)
  }
  assert_true(deleted is Err(_))
}

///|
test "labels survive save/load and compact" {
  let n = 20
  let dim = 4
  let data = gen_dataset(91UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add_labeled(row_of(data, i, dim), (1000 + i).to_uint64())
  }
  let _ = idx.delete(3)
  let _ = idx.delete(11)
  let loaded = Hnsw::load(idx.save())
  for i in 0..<n {
    if idx.contains(i) {
      assert_eq(loaded.label_of(i), (1000 + i).to_uint64())
    }
  }
  // Compaction remaps ids but preserves labels.
  let result = idx.compact()
  let fresh = result.index
  assert_eq(fresh.size(), n - 2)
  for old_id in 0..<n {
    let new_id = result.mapping[old_id]
    if new_id >= 0 {
      assert_eq(fresh.label_of(new_id), (1000 + old_id).to_uint64())
    } else {
      assert_true(old_id == 3 || old_id == 11)
    }
  }
}

///|
test "absorb merges another index with labels" {
  let dim = 4
  let a = Hnsw::new(
    dim,
    L2,
    params=HnswParams::new(m=8, ef_construction=100, seed=1UL),
  )
  let b = Hnsw::new(
    dim,
    L2,
    params=HnswParams::new(m=8, ef_construction=100, seed=2UL),
  )
  let data_a = gen_dataset(95UL, 10, dim, 0.0, 1.0)
  let data_b = gen_dataset(96UL, 7, dim, 0.0, 1.0)
  for i in 0..<10 {
    let _ = a.add_labeled(row_of(data_a, i, dim), (100 + i).to_uint64())
  }
  for i in 0..<7 {
    let _ = b.add_labeled(row_of(data_b, i, dim), (200 + i).to_uint64())
  }
  let _ = b.delete(2)
  let mapping = a.absorb(b)
  assert_eq(a.size(), 16)
  // Only non-deleted vectors transfer: b's tombstone stays behind.
  assert_eq(a.total(), 16)
  assert_eq(mapping.length(), 7)
  assert_eq(mapping[2], -1)
  for i in 0..<7 {
    if i != 2 {
      assert_eq(a.label_of(mapping[i]), (200 + i).to_uint64())
    }
  }
  // Searches reach vectors that came from both sides.
  let qa = row_of(data_a, 0, dim)
  let res_a = a.search(qa, 1, ef=64)
  assert_eq(res_a[0].id, 0)
  let qb = row_of(data_b, 0, dim)
  let res_b = a.search(qb, 1, ef=64)
  assert_eq(a.label_of(res_b[0].id), 200UL)
}

///|
test "absorb validates dimension and metric" {
  let a = Hnsw::new(4, L2)
  let wrong_dim = Hnsw::new(3, L2)
  let wrong_metric = Hnsw::new(4, Cosine)
  let bad_dim : Result[FixedArray[Int32], HnswError] = Ok(a.absorb(wrong_dim)) catch {
    e => Err(e)
  }
  assert_true(bad_dim is Err(_))
  let bad_metric : Result[FixedArray[Int32], HnswError] = Ok(
    a.absorb(wrong_metric),
  ) catch {
    e => Err(e)
  }
  assert_true(bad_metric is Err(_))
}

///|
test "copy is independent and ids lists alive vectors" {
  let n = 10
  let dim = 4
  let data = gen_dataset(97UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add_labeled(row_of(data, i, dim), (50 + i).to_uint64())
  }
  let _ = idx.delete(4)
  let _ = idx.delete(7)
  let cp = idx.copy()
  assert_eq(cp.size(), 8)
  assert_eq(cp.total(), 10)
  // Mutating the copy leaves the original untouched.
  let _ = cp.delete(0)
  let _ = cp.add(FixedArray::make(dim, 9.0))
  assert_eq(idx.size(), 8)
  assert_eq(idx.total(), 10)
  // ids() enumerates exactly the alive vectors, ascending.
  let alive = idx.ids()
  assert_eq(alive.length(), 8)
  for j in 0..<alive.length() {
    assert_true(alive[j] != 4 && alive[j] != 7)
    assert_true(idx.contains(alive[j]))
  }
  assert_true(alive[0] < alive[alive.length() - 1])
}

///|
test "save_bytes and load_bytes roundtrip" {
  let n = 30
  let dim = 4
  let data = gen_dataset(98UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add_labeled(row_of(data, i, dim), (77 + i).to_uint64())
  }
  let _ = idx.delete(5)
  let payload : Bytes = idx.save_bytes()
  let restored = Hnsw::load_bytes(payload)
  assert_eq(restored.size(), n - 1)
  assert_eq(restored.total(), n)
  let queries = gen_dataset(99UL, 3, dim, 0.0, 1.0)
  for qi in 0..<3 {
    let q = row_of(queries, qi, dim)
    let a = idx.search(q, 5, ef=64)
    let b = restored.search(q, 5, ef=64)
    assert_eq(a.length(), b.length())
    for j in 0..<a.length() {
      assert_eq(a[j].id, b[j].id)
      assert_eq(a[j].distance, b[j].distance)
    }
  }
  for i in 0..<n {
    if idx.contains(i) {
      assert_eq(restored.label_of(i), (77 + i).to_uint64())
    }
  }
}

///|
test "search results support equality and debug printing" {
  let dim = 4
  let data = gen_dataset(105UL, 40, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<40 {
    let _ = idx.add(row_of(data, i, dim))
  }
  let q = row_of(data, 3, dim)
  let r1 = idx.search(q, 5, ef=64)
  // A copied index answers identically; Neighbor derives Eq, so whole result
  // lists compare directly instead of field by field.
  assert_eq(r1, idx.copy().search(q, 5, ef=64))
  // Metric values compare too.
  assert_true(r1[0] != { id: r1[0].id, distance: r1[0].distance + 1.0, })
}

///|
test "verify accepts healthy indexes and detects tampering" {
  let n = 60
  let dim = 8
  let data = gen_dataset(107UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  let _ = idx.delete(9)
  idx.verify()
  Hnsw::load(idx.save()).verify()
  idx.compact().index.verify()
  // Corrupt one link id; verify() must reject the graph.
  let mut tampered = false
  for id in 0..<n {
    if !tampered && idx.links[id][0].length() > 0 {
      idx.links[id][0][0] = 999999
      tampered = true
    }
  }
  assert_true(tampered)
  let bad : Result[Unit, HnswError] = Ok(idx.verify()) catch { e => Err(e) }
  assert_true(bad is Err(_))
}

///|
test "Neighbor and HnswStats render to JSON" {
  let n : Neighbor = { id: 7, distance: -1.5, }
  match n.to_json() {
    Object(fields) => {
      assert_eq(fields["id"], Json::number(7.0))
      assert_eq(fields["distance"], Json::number(-1.5))
    }
    _ => abort("expected a JSON object")
  }
  let idx = Hnsw::new(3, L2)
  let _ = idx.add(FixedArray::make(3, 1.0))
  match idx.stats().to_json() {
    Object(fields) => {
      assert_eq(fields["count"], Json::number(1.0))
      assert_true(fields.contains("estimated_bytes"))
      assert_true(fields.contains("nodes_per_level"))
    }
    _ => abort("expected a JSON object")
  }
}

///|
test "export_dot renders nodes, edges and deletion styling" {
  let dim = 4
  let data = gen_dataset(111UL, 12, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<12 {
    let _ = idx.add_labeled(row_of(data, i, dim), (900 + i).to_uint64())
  }
  let _ = idx.delete(3)
  let dot = idx.export_dot(0)
  assert_true(dot.contains("digraph moonhnsw_layer0"))
  // Every node appears with its label; the deleted one is dashed.
  for i in 0..<12 {
    assert_true(dot.contains("n\{i} [label=\"\{i} (label \{900 + i})\""))
  }
  assert_true(dot.contains("style=dashed"))
  assert_true(dot.contains("-> "))
  // The top layer renders too, whatever its height.
  let top = idx.export_dot(idx.max_level)
  assert_true(top.contains("digraph moonhnsw_layer\{idx.max_level}"))
  // Out-of-range layers raise.
  let bad_low : Result[String, HnswError] = Ok(idx.export_dot(-1)) catch {
    e => Err(e)
  }
  assert_true(bad_low is Err(_))
  let bad_high : Result[String, HnswError] = Ok(
    idx.export_dot(idx.max_level + 1),
  ) catch {
    e => Err(e)
  }
  assert_true(bad_high is Err(_))
}

///|
test "recall on clustered data" {
  // Three tight blobs in 8 dimensions: real embeddings are clustered, not
  // uniform, so the graph must stay navigable here too.
  let n = 300
  let dim = 8
  let crng = TestRng::new(120UL)
  let centers : FixedArray[Float] = FixedArray::make(3 * dim, 0.0)
  for i in 0..<(3 * dim) {
    centers[i] = Float::from_double(crng.next_f64() * 10.0)
  }
  let rng = TestRng::new(121UL)
  let data : FixedArray[Float] = FixedArray::make(n * dim, 0.0)
  for i in 0..<n {
    let c = i % 3 * dim
    let off = i * dim
    for j in 0..<dim {
      // Sum of three uniforms approximates a gaussian offset.
      let g = (rng.next_f64() + rng.next_f64() + rng.next_f64() - 1.5) * 0.4
      data[off + j] = Float::from_double(centers[c + j].to_double() + g)
    }
  }
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  let queries = gen_dataset(122UL, 20, dim, 0.0, 10.0)
  let mut hits = 0
  for qi in 0..<20 {
    let q = row_of(queries, qi, dim)
    let res = idx.search(q, 10, ef=100)
    let truth = brute_topk(L2, data, n, dim, q, 10)
    for r in res {
      for t in truth {
        if r.id == t {
          hits += 1
          break
        }
      }
    }
  }
  assert_true(hits.to_double() / 200.0 >= 0.9)
}

///|
test "construction after load matches the original index" {
  // rng_state is persisted, so a loaded index must build the identical graph
  // when fed the same insertions.
  let dim = 8
  let data = gen_dataset(123UL, 40, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2, params=HnswParams::new(seed=777UL))
  for i in 0..<40 {
    let _ = idx.add(row_of(data, i, dim))
  }
  let loaded = Hnsw::load(idx.save())
  let extra = gen_dataset(124UL, 10, dim, 0.0, 1.0)
  for i in 0..<10 {
    let v = row_of(extra, i, dim)
    assert_eq(loaded.add(v), idx.add(v))
  }
  assert_eq(loaded.max_level, idx.max_level)
  assert_eq(loaded.entry, idx.entry)
  // And both answer searches identically.
  let queries = gen_dataset(125UL, 5, dim, 0.0, 1.0)
  for qi in 0..<5 {
    assert_eq(
      idx.search(row_of(queries, qi, dim), 5, ef=64),
      loaded.search(row_of(queries, qi, dim), 5, ef=64),
    )
  }
}

///|
test "filtered search returns only matching vectors" {
  let n = 200
  let dim = 8
  let data = gen_dataset(131UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add_labeled(row_of(data, i, dim), i.to_uint64())
  }
  let q = row_of(data, 3, dim)
  // A filter that keeps everything must agree with plain search exactly.
  assert_eq(
    idx.search_filtered(q, 10, ef=64, filter=fn(_) { true }),
    idx.search(q, 10, ef=64),
  )
  // A filter that rejects everything yields no results.
  assert_eq(
    idx.search_filtered(q, 10, ef=64, filter=fn(_) { false }).length(),
    0,
  )
  // Even-id filtering: results respect the filter and match the exact
  // top-10 among even ids.
  let res = idx.search_filtered(q, 10, ef=200, filter=fn(id) { id % 2 == 0 })
  assert_eq(res.length(), 10)
  let pool : Array[(Double, Int32)] = []
  for i in 0..<n {
    if i % 2 == 0 {
      pool.push((distance_between(L2, q, 0, data, i * dim, dim), i))
    }
  }
  let _ = sort_pairs_ascending(pool)
  for j in 0..<10 {
    assert_eq(res[j].id, pool[j].1)
    assert_eq(res[j].distance, pool[j].0)
  }
  // Label-style filtering — labels equal ids in this index.
  let labeled = idx.search_filtered(q, 5, ef=128, filter=fn(id) { id >= 100 })
  for r in labeled {
    assert_true(idx.label_of(r.id) >= 100UL)
  }
  // Deleted vectors never pass, whatever the filter says.
  let _ = idx.delete(6)
  let res2 = idx.search_filtered(q, 10, ef=200, filter=fn(_) { true })
  for r in res2 {
    assert_true(r.id != 6)
  }
}

///|
test "v4 payload checksum detects bit flips" {
  let n = 50
  let dim = 8
  let data = gen_dataset(141UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  let bytes = idx.save()
  // Flip one byte in the middle of the payload.
  let flipped : FixedArray[Byte] = FixedArray::make(bytes.length(), b'\x00')
  FixedArray::unsafe_blit(flipped, 0, bytes, 0, bytes.length())
  let mid = bytes.length() / 2
  flipped[mid] = (flipped[mid].to_int() ^ 0xFF).to_byte()
  let bad : Result[Hnsw, HnswError] = Ok(Hnsw::load(flipped)) catch {
    e => Err(e)
  }
  match bad {
    // The flip may trip an earlier field check or the trailer; either way
    // the payload must be rejected.
    Err(CorruptData(_)) => ()
    _ => abort("expected the flipped payload to be rejected")
  }
  // Flipping a checksum byte itself must report the checksum specifically.
  let flipped_crc : FixedArray[Byte] = FixedArray::make(bytes.length(), b'\x00')
  FixedArray::unsafe_blit(flipped_crc, 0, bytes, 0, bytes.length())
  let last = bytes.length() - 1
  flipped_crc[last] = (flipped_crc[last].to_int() ^ 0xFF).to_byte()
  let bad_crc : Result[Hnsw, HnswError] = Ok(Hnsw::load(flipped_crc)) catch {
    e => Err(e)
  }
  match bad_crc {
    Err(CorruptData(msg)) => assert_true(msg.contains("checksum"))
    _ => abort("expected a checksum mismatch")
  }
  // The untouched payload still loads and verifies.
  Hnsw::load(bytes).verify()
}

///|
test "recall with SQ8 storage stays high" {
  let l2 = recall_of(L2, 42UL, 400, 16, 16, 200, 150, 10, storage=SQ8)
  assert_true(l2 >= 0.8)
  let cosine = recall_of(Cosine, 44UL, 400, 16, 16, 200, 150, 10, storage=SQ8)
  assert_true(cosine >= 0.75)
}

///|
test "SQ8 store uses far less memory than full floats" {
  let n = 1000
  let dim = 16
  let data = gen_dataset(151UL, n, dim, 0.0, 1.0)
  let full = Hnsw::new(dim, L2)
  let quantized = Hnsw::new(dim, L2, params=HnswParams::new(storage=SQ8))
  for i in 0..<n {
    let v = row_of(data, i, dim)
    let _ = full.add(v)
    let _ = quantized.add(v)
  }
  assert_true(quantized.q_frozen)
  // The byte store is a quarter of the float store for the same capacity;
  // total index size also shrinks (graph edges dominate the remainder).
  let bytes_full = full.vectors.length().to_int64() * 4L
  let bytes_quant = quantized.quants.length().to_int64()
  assert_true(bytes_quant * 2L < bytes_full)
  assert_true(quantized.stats().estimated_bytes < full.stats().estimated_bytes)
}

///|
test "SQ8 roundtrip, dequantization and clamping" {
  let n = 300
  let dim = 8
  let data = gen_dataset(152UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2, params=HnswParams::new(storage=SQ8))
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  assert_true(idx.q_frozen)
  // Dequantized vectors sit within half a quantization step per dimension.
  let approx = idx.get_vector(10)
  let exact = row_of(data, 10, dim)
  for j in 0..<dim {
    assert_true((approx[j].to_double() - exact[j].to_double()).abs() < 0.01)
  }
  // Frozen store round-trips bit-exactly.
  let restored = Hnsw::load(idx.save())
  assert_true(restored.q_frozen)
  let queries = gen_dataset(153UL, 4, dim, 0.0, 1.0)
  for qi in 0..<4 {
    assert_eq(
      idx.search(row_of(queries, qi, dim), 5, ef=64),
      restored.search(row_of(queries, qi, dim), 5, ef=64),
    )
  }
  // An extreme vector clamps into the frozen bounds instead of misbehaving.
  let wild : FixedArray[Float] = FixedArray::make(dim, 1000000.0)
  let wild_id = idx.add(wild)
  assert_eq(idx.search(wild, 1, ef=64)[0].id, wild_id)
  // A pre-freeze payload also round-trips.
  let small = Hnsw::new(dim, L2, params=HnswParams::new(storage=SQ8))
  let _ = small.add(FixedArray::make(dim, 0.5))
  let small2 = Hnsw::load(small.save())
  assert_eq(small2.size(), 1)
  assert_eq(small2.search(FixedArray::make(dim, 0.5), 1)[0].id, 0)
}

///|
test "MMR search trades relevance for diversity" {
  let dim = 8
  // Six near-copies of the query plus plenty of spread-out filler.
  let base = gen_dataset(161UL, 1, dim, 0.0, 1.0)
  let data = gen_dataset(162UL, 120, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for c in 0..<6 {
    let near : FixedArray[Float] = FixedArray::make(dim, 0.0)
    for j in 0..<dim {
      near[j] = base[j] + Float::from_double(c.to_double() * 0.0001)
    }
    let _ = idx.add(near)
  }
  for i in 0..<120 {
    let _ = idx.add(row_of(data, i, dim))
  }
  let q : FixedArray[Float] = FixedArray::make(dim, 0.0)
  FixedArray::unsafe_blit(q, 0, base, 0, dim)
  // Plain search fills the result with the near-copies.
  let plain = idx.search(q, 6, ef=128)
  let mut copies_plain = 0
  for r in plain {
    if r.id < 6 {
      copies_plain += 1
    }
  }
  assert_true(copies_plain >= 5)
  // High-lambda MMR spreads out instead.
  let diverse = idx.search_mmr(q, 6, lambda=1.0, ef=128)
  assert_eq(diverse.length(), 6)
  let mut copies_mmr = 0
  for r in diverse {
    if r.id < 6 {
      copies_mmr += 1
    }
  }
  assert_true(copies_mmr <= 2)
  // Lambda 0 keeps the plain relevance ordering.
  assert_eq(idx.search_mmr(q, 5, lambda=0.0, ef=128), idx.search(q, 5, ef=128))
  // A single result is always the nearest neighbor.
  assert_eq(
    idx.search_mmr(q, 1, lambda=0.3, ef=128)[0].id,
    idx.search(q, 1, ef=128)[0].id,
  )
  // Lambda out of range raises.
  let bad : Result[Array[Neighbor], HnswError] = Ok(
    idx.search_mmr(q, 5, lambda=1.5),
  ) catch {
    e => Err(e)
  }
  assert_true(bad is Err(_))
}

///|
test "find_by_label and upsert_labeled implement the update flow" {
  let dim = 4
  let idx = Hnsw::new(dim, L2)
  let old : FixedArray[Float] = FixedArray::make(dim, 1.0)
  let _ = idx.add_labeled(old, 500UL)
  let _ = idx.add_labeled(FixedArray::make(dim, 2.0), 501UL)
  // Duplicate labels are allowed.
  let _ = idx.add_labeled(FixedArray::make(dim, 3.0), 500UL)
  assert_eq(idx.find_by_label(500UL).length(), 2)
  assert_eq(idx.find_by_label(999UL).length(), 0)
  // Upserting replaces every vector with that label.
  let replacement : FixedArray[Float] = FixedArray::make(dim, 9.0)
  let new_id = idx.upsert_labeled(replacement, 500UL)
  assert_eq(idx.find_by_label(500UL).length(), 1)
  assert_eq(idx.find_by_label(500UL)[0], new_id)
  assert_true(!idx.contains(0) && !idx.contains(2))
  assert_true(idx.contains(1))
  assert_eq(idx.size(), 2)
  // The surviving vector holds the new content.
  let got = idx.get_vector(new_id)
  for j in 0..<dim {
    assert_eq(got[j], 9.0)
  }
  // Search reflects the replacement, not the old copies.
  let res = idx.search(replacement, 2, ef=64)
  assert_eq(res[0].id, new_id)
}

///|
test "search_batch matches per-query searches" {
  let n = 60
  let dim = 4
  let data = gen_dataset(191UL, n, dim, 0.0, 1.0)
  let idx = Hnsw::new(dim, L2)
  for i in 0..<n {
    let _ = idx.add(row_of(data, i, dim))
  }
  let queries = gen_dataset(192UL, 6, dim, 0.0, 1.0)
  let qs : Array[FixedArray[Float]] = []
  for qi in 0..<6 {
    qs.push(row_of(queries, qi, dim))
  }
  let batched = idx.search_batch(qs, 5, ef=64)
  assert_eq(batched.length(), 6)
  for qi in 0..<6 {
    assert_eq(batched[qi], idx.search(qs[qi], 5, ef=64))
  }
}

///|
test "randomized operations keep invariants and match brute force" {
  let dim = 4
  let idx = Hnsw::new(
    dim,
    L2,
    params=HnswParams::new(m=8, ef_construction=64, seed=171UL),
  )
  let pool = gen_dataset(181UL, 200, dim, 0.0, 1.0)
  let alive_mirror : Array[Bool] = Array::make(200, false)
  let mut next_add = 0
  let rng = TestRng::new(180UL)
  // Phase 1: plain adds.
  for _ in 0..<60 {
    let _ = idx.add(row_of(pool, next_add, dim))
    alive_mirror[next_add] = true
    next_add += 1
  }
  // Phase 2: mixed adds, deletes and checked searches.
  for step in 0..<120 {
    let roll = rng.next_f64()
    if roll < 0.5 && next_add < 200 {
      let _ = idx.add(row_of(pool, next_add, dim))
      alive_mirror[next_add] = true
      next_add += 1
    } else if roll < 0.8 {
      let target = (rng.next_f64() * next_add.to_double()).to_int()
      if target < next_add && alive_mirror[target] {
        let _ = idx.delete(target)
        alive_mirror[target] = false
      }
    } else {
      let q = row_of(
        pool,
        (rng.next_f64() * next_add.to_double()).to_int(),
        dim,
      )
      let res = idx.search(q, 5, ef=64)
      for r in res {
        assert_true(r.id >= 0 && r.id < next_add && alive_mirror[r.id])
      }
      // The nearest result must sit inside the exact top-5 over the alive
      // mirror.
      let dists : Array[(Double, Int32)] = []
      for i in 0..<next_add {
        if alive_mirror[i] {
          dists.push((distance_between(L2, q, 0, pool, i * dim, dim), i))
        }
      }
      let _ = sort_pairs_ascending(dists)
      let truth_n = if dists.length() < 5 { dists.length() } else { 5 }
      if truth_n > 0 && res.length() > 0 {
        let mut ok = false
        for t in 0..<truth_n {
          if res[0].id == dists[t].1 {
            ok = true
          }
        }
        assert_true(ok)
      }
    }
    if step % 16 == 0 {
      idx.verify()
    }
  }
  idx.verify()
  // Final bookkeeping agrees with the mirror on every axis.
  let mut alive_total = 0
  for i in 0..<200 {
    if alive_mirror[i] {
      alive_total += 1
    }
  }
  assert_eq(idx.size(), alive_total)
  assert_eq(idx.total(), next_add)
  let ids = idx.ids()
  assert_eq(ids.length(), alive_total)
  for id in ids {
    assert_true(alive_mirror[id])
  }
  // The surviving index still round-trips.
  Hnsw::load(idx.save()).verify()
}
