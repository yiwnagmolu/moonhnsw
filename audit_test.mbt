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
}
