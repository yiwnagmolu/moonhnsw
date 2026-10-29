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

///|
test "public api: all deleted vectors leave exact and adaptive search empty" {
  let idx = @moonhnsw.Hnsw::new(1, @moonhnsw.Distance::L2)
  let q : FixedArray[Float] = FixedArray::from_array([0.0])
  let id = idx.add_labeled(q, 42UL)
  idx.delete(id)
  assert_eq(idx.search_exact(q, 10).length(), 0)
  assert_eq(idx.search_radius_exact(q, 0.0).length(), 0)
  let adaptive = idx.search_filtered_adaptive(q, 1, filter=fn(_) { true })
  assert_eq(adaptive.rounds, 0)
  assert_eq(adaptive.distance_computations, 0L)
  let calibration : Result[@moonhnsw.EfCalibration, @moonhnsw.HnswError] = Ok(
    idx.calibrate_ef([q], 1, 1.0, [1]),
  ) catch {
    e => Err(e)
  }
  assert_true(calibration is Err(@moonhnsw.HnswError::InvalidParams(_)))
  let restored = @moonhnsw.Hnsw::load(idx.save())
  assert_eq(restored.search_exact(q, 1).length(), 0)
  let new_id = restored.add_labeled(q, 42UL)
  assert_eq(restored.search_exact(q, 1)[0].id, new_id)
}

///|
test "public api: SQ8 upsert, compact, and persistence keep labels searchable" {
  let idx = @moonhnsw.Hnsw::new(
    1,
    @moonhnsw.Distance::L2,
    params=@moonhnsw.HnswParams::new(storage=@moonhnsw.Storage::SQ8),
  )
  for i in 0..<256 {
    let _ = idx.add(FixedArray::from_array([Float::from_double(i.to_double())]))
  }
  let q : FixedArray[Float] = FixedArray::from_array([18.0])
  let old_id = idx.add_labeled(FixedArray::from_array([17.0]), 500UL)
  let new_id = idx.upsert_labeled(q, 500UL)
  assert_eq(idx.find_by_label(500UL), [new_id])
  assert_true(!idx.contains(old_id))
  let compacted = idx.compact()
  assert_eq(compacted.mapping[old_id], -1)
  let mapped_id = compacted.mapping[new_id]
  let loaded = @moonhnsw.Hnsw::load(compacted.index.save())
  assert_eq(loaded.find_by_label(500UL), [mapped_id])
  let hits = loaded.search_exact(q, 10, filter=fn(id) { id == mapped_id })
  assert_eq(hits.length(), 1)
  assert_eq(hits[0].id, mapped_id)
  assert_eq(
    loaded.search_radius_exact(q, 1.0, filter=fn(id) { id == mapped_id }),
    hits,
  )
}
