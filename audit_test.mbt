///|
test "public api: cosine helper and negative radius use search distances" {
  let a : FixedArray[Float] = FixedArray::from_array([2.0, 0.0])
  let b : FixedArray[Float] = FixedArray::from_array([5.0, 0.0])
  let c : FixedArray[Float] = FixedArray::from_array([0.0, 3.0])
  assert_eq(@moonhnsw.distance(@moonhnsw.Distance::Cosine, a, b), -1.0)
  assert_eq(@moonhnsw.distance(@moonhnsw.Distance::Cosine, a, c), 0.0)
  let cosine = @moonhnsw.Hnsw::new(2, @moonhnsw.Distance::Cosine)
  let _ = cosine.add(b)
  let _ = cosine.add(c)
  let close = cosine.search_radius(a, -0.5)
  assert_eq(close.length(), 1)
  assert_eq(close[0].id, 0)
  let ip = @moonhnsw.Hnsw::new(2, @moonhnsw.Distance::InnerProduct)
  let _ = ip.add(b)
  let _ = ip.add(c)
  assert_eq(ip.search_radius(a, -5.0).length(), 1)
}

///|
test "public api: invalid upsert preserves the old labeled vector" {
  let idx = @moonhnsw.Hnsw::new(2, @moonhnsw.Distance::L2)
  let old = idx.add_labeled(FixedArray::from_array([1.0, 2.0]), 7UL)
  let err : Result[Int32, @moonhnsw.HnswError] = Ok(
    idx.upsert_labeled(FixedArray::from_array([3.0]), 7UL),
  ) catch {
    e => Err(e)
  }
  assert_true(err is Err(_))
  assert_eq(idx.size(), 1)
  assert_true(idx.contains(old))
  assert_eq(idx.find_by_label(7UL), [old])
}

///|
test "public api: batch distance count sums every query" {
  let idx = @moonhnsw.Hnsw::new(2, @moonhnsw.Distance::L2)
  for i in 0..<20 {
    let _ = idx.add(
      FixedArray::from_array([Float::from_double(i.to_double()), 0.0]),
    )
  }
  let a : FixedArray[Float] = FixedArray::from_array([2.0, 0.0])
  let b : FixedArray[Float] = FixedArray::from_array([17.0, 0.0])
  let _ = idx.search(a, 3)
  let count_a = idx.distance_computations()
  let _ = idx.search(b, 3)
  let count_b = idx.distance_computations()
  let batch = idx.search_batch([a, b], 3)
  assert_eq(batch.length(), 2)
  assert_eq(idx.distance_computations(), count_a + count_b)
  let _ = idx.search_batch([], 3)
  assert_eq(idx.distance_computations(), 0L)
  let _ = idx.search(a, 3)
  assert_true(idx.distance_computations() > 0L)
  assert_eq(idx.search_mmr(a, 0).length(), 0)
  assert_eq(idx.distance_computations(), 0L)
}

///|
test "public api: inserted vectors are isolated from caller mutation" {
  for
    metric in [
      @moonhnsw.Distance::L2,
      @moonhnsw.Distance::InnerProduct,
      @moonhnsw.Distance::Cosine,
    ] {
    let idx = @moonhnsw.Hnsw::new(2, metric)
    let input : FixedArray[Float] = FixedArray::from_array([3.0, 4.0])
    let id = idx.add(input)
    let stored = idx.get_vector(id)
    input[0] = 99.0
    assert_eq(idx.get_vector(id), stored)
  }
  let sq8 = @moonhnsw.Hnsw::new(
    2,
    @moonhnsw.Distance::L2,
    params=@moonhnsw.HnswParams::new(storage=@moonhnsw.Storage::SQ8),
  )
  for i in 0..<256 {
    let _ = sq8.add(
      FixedArray::from_array([Float::from_double(i.to_double()), 3.0]),
    )
  }
  let input : FixedArray[Float] = FixedArray::from_array([17.0, 3.0])
  let id = sq8.add(input)
  let stored = sq8.get_vector(id)
  input[0] = 99.0
  assert_eq(sq8.get_vector(id), stored)
}

///|
test "public api: top-k retains a finite vector with infinite L2 distance" {
  let idx = @moonhnsw.Hnsw::new(1, @moonhnsw.Distance::L2)
  let far : FixedArray[Float] = FixedArray::from_array([1.0e20])
  let query : FixedArray[Float] = FixedArray::from_array([0.0])
  let _ = idx.add(far)
  let hits = idx.search(query, 1)
  assert_eq(hits.length(), 1)
  assert_eq(hits[0].id, 0)
  assert_true(hits[0].distance > 1.0e308)
  assert_eq(idx.search_filtered(query, 1, filter=fn(_) { true }), hits)
  assert_eq(idx.search_mmr(query, 1), hits)
  assert_eq(idx.search_radius(query, 1.0e308).length(), 0)
  assert_eq(idx.search_radius(query, hits[0].distance), hits)
}
