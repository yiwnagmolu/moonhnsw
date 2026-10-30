///|
test "exact search orders equal distances by id and honors tombstones" {
  let idx = @moonhnsw.Hnsw::new(1, @moonhnsw.Distance::L2)
  let values : Array[Float] = [1.0, -1.0, 2.0, -2.0]
  for x in values {
    let _ = idx.add(FixedArray::from_array([x]))
  }
  let q : FixedArray[Float] = FixedArray::from_array([0.0])
  let hits = idx.search_exact(q, 3)
  assert_eq(hits.map(fn(x) { x.id }), [0, 1, 2])
  assert_eq(idx.distance_computations(), 4L)
  assert_eq(idx.search_exact(q, 1)[0].id, 0)
  idx.delete(0)
  assert_eq(idx.search_exact(q, 10).map(fn(x) { x.id }), [1, 2, 3])
  assert_eq(idx.distance_computations(), 3L)
  let restored = @moonhnsw.Hnsw::load(idx.save())
  assert_eq(restored.search_exact(q, 10), idx.search_exact(q, 10))
}

///|
test "exact scan handles selective filters and exhaustive radius" {
  let idx = @moonhnsw.Hnsw::new(1, @moonhnsw.Distance::L2)
  let values : Array[Float] = [0.0, 1.0, -1.0, 2.0, -2.0]
  for x in values {
    let _ = idx.add(FixedArray::from_array([x]))
  }
  let q : FixedArray[Float] = FixedArray::from_array([0.0])
  let filtered = idx.search_exact(q, 5, filter=fn(id) { id % 2 == 1 })
  assert_eq(filtered.map(fn(x) { x.id }), [1, 3])
  assert_eq(idx.distance_computations(), 2L)
  let inside = idx.search_radius_exact(q, 1.0)
  assert_eq(inside.map(fn(x) { x.id }), [0, 1, 2])
  assert_eq(
    idx.search_radius_exact(q, 1.0, max_results=2).map(fn(x) { x.id }),
    [0, 1],
  )
  assert_eq(idx.search_radius_exact(q, 0.0).map(fn(x) { x.id }), [0])
  assert_eq(
    idx
    .search_radius_exact(q, 1.0, filter=fn(id) { id > 1 })
    .map(fn(x) { x.id }),
    [2],
  )
}

///|
test "exact scan shares metric semantics and handles large finite L2 distances" {
  let q : FixedArray[Float] = FixedArray::from_array([1.0, 0.0])
  for metric in [@moonhnsw.Distance::InnerProduct, @moonhnsw.Distance::Cosine] {
    let idx = @moonhnsw.Hnsw::new(2, metric)
    let _ = idx.add(FixedArray::from_array([1.0, 0.0]))
    let _ = idx.add(FixedArray::from_array([0.0, 1.0]))
    let _ = idx.add(FixedArray::from_array([-1.0, 0.0]))
    assert_eq(idx.search_exact(q, 3).map(fn(x) { x.id }), [0, 1, 2])
    assert_eq(idx.search_radius_exact(q, -0.5).map(fn(x) { x.id }), [0])
  }
  let huge = @moonhnsw.Hnsw::new(1, @moonhnsw.Distance::L2)
  let _ = huge.add(FixedArray::from_array([1.0e20]))
  let _ = huge.add(FixedArray::from_array([-1.0e20]))
  let zero : FixedArray[Float] = FixedArray::from_array([0.0])
  assert_eq(huge.search_exact(zero, 2).map(fn(x) { x.id }), [0, 1])
  assert_true(huge.search_exact(zero, 1)[0].distance < @double.infinity)
}

///|
test "finite f32 L2 overflow falls back to finite distances across search and load" {
  let large = Float::from_double(3.0e38)
  let query : FixedArray[Float] = FixedArray::from_array([-large])
  let far : FixedArray[Float] = FixedArray::from_array([large])
  let near : FixedArray[Float] = FixedArray::from_array([-large])
  let difference = large.to_double() - (-large).to_double()
  let expected = difference * difference
  assert_true(expected < @double.infinity)
  assert_eq(@moonhnsw.distance(@moonhnsw.Distance::L2, query, far), expected)

  let idx = @moonhnsw.Hnsw::new(1, @moonhnsw.Distance::L2)
  let _ = idx.add(far)
  let _ = idx.add(near)
  let exact = idx.search_exact(query, 2)
  assert_eq(exact.map(fn(x) { x.id }), [1, 0])
  assert_eq(exact[1].distance, expected)
  let approximate = idx.search(query, 2, ef=16)
  assert_eq(approximate.map(fn(x) { x.id }), [1, 0])
  assert_eq(approximate[1].distance, expected)

  let restored = @moonhnsw.Hnsw::load(idx.save())
  assert_eq(restored.search_exact(query, 2), exact)
  assert_eq(restored.search(query, 2, ef=16), approximate)
}

///|
test "finite f32 inner products recover from infinity cancellation" {
  let large = Float::from_double(1.0e20)
  let query : FixedArray[Float] = FixedArray::from_array([large, large])
  let cancel : FixedArray[Float] = FixedArray::from_array([large, -large])
  let strong : FixedArray[Float] = FixedArray::from_array([large, large])
  let product = large.to_double() * large.to_double()
  let expected = -(product + product)
  assert_eq(
    @moonhnsw.distance(@moonhnsw.Distance::InnerProduct, query, cancel),
    0.0,
  )
  assert_eq(
    @moonhnsw.distance(@moonhnsw.Distance::InnerProduct, query, strong),
    expected,
  )

  let idx = @moonhnsw.Hnsw::new(2, @moonhnsw.Distance::InnerProduct)
  let _ = idx.add(cancel)
  let _ = idx.add(strong)
  let exact = idx.search_exact(query, 2)
  assert_eq(exact.map(fn(x) { x.id }), [1, 0])
  assert_eq(exact[0].distance, expected)
  assert_eq(exact[1].distance, 0.0)
  let approximate = idx.search(query, 2, ef=16)
  assert_eq(approximate.map(fn(x) { x.id }), [1, 0])
  assert_eq(approximate[0].distance, expected)
  assert_eq(approximate[1].distance, 0.0)

  let restored = @moonhnsw.Hnsw::load(idx.save())
  assert_eq(restored.search_exact(query, 2), exact)
  assert_eq(restored.search(query, 2, ef=16), approximate)
}

///|
test "exact scan over SQ8 scores stored quantized values" {
  let idx = @moonhnsw.Hnsw::new(
    1,
    @moonhnsw.Distance::L2,
    params=@moonhnsw.HnswParams::new(storage=@moonhnsw.Storage::SQ8),
  )
  for i in 0..<257 {
    let _ = idx.add(FixedArray::from_array([Float::from_double(i.to_double())]))
  }
  let q : FixedArray[Float] = FixedArray::from_array([255.0])
  let hits = idx.search_exact(q, 2)
  assert_eq(hits.map(fn(x) { x.id }), [255, 256])
  assert_eq(idx.distance_computations(), 257L)
  assert_eq(idx.search_radius_exact(q, 0.0).map(fn(x) { x.id }), [255, 256])
  let loaded = @moonhnsw.Hnsw::load(idx.save())
  assert_eq(loaded.search_exact(q, 2), hits)
}

///|
test "exact scan validates dimensions and radius even on empty indexes" {
  let idx = @moonhnsw.Hnsw::new(1, @moonhnsw.Distance::L2)
  let q : FixedArray[Float] = FixedArray::from_array([0.0])
  assert_eq(idx.search_exact(q, 3).length(), 0)
  assert_eq(idx.search_radius_exact(q, 0.0).length(), 0)
  let wrong : FixedArray[Float] = FixedArray::make(2, 0.0)
  let e1 : Result[Array[@moonhnsw.Neighbor], @moonhnsw.HnswError] = Ok(
    idx.search_exact(wrong, 0),
  ) catch {
    e => Err(e)
  }
  assert_true(e1 is Err(@moonhnsw.HnswError::DimMismatch(_, _)))
  let e2 : Result[Array[@moonhnsw.Neighbor], @moonhnsw.HnswError] = Ok(
    idx.search_radius_exact(q, -1.0, max_results=0),
  ) catch {
    e => Err(e)
  }
  assert_true(e2 is Err(@moonhnsw.HnswError::InvalidParams(_)))
  let _ = idx.add(q)
  assert_eq(idx.search_exact(q, 0).length(), 0)
  assert_eq(idx.distance_computations(), 0L)
  assert_eq(idx.search_radius_exact(q, 1.0, max_results=0).length(), 0)
}

///|
test "exact top-k agrees with an independent full sort across metrics" {
  let vectors : Array[FixedArray[Float]] = []
  for i in 0..<80 {
    vectors.push(
      FixedArray::from_array([
        Float::from_double((i % 17).to_double()),
        Float::from_double((i * 7 % 13).to_double()),
      ]),
    )
  }
  let q : FixedArray[Float] = FixedArray::from_array([4.0, 9.0])
  for
    metric in [
      @moonhnsw.Distance::L2,
      @moonhnsw.Distance::InnerProduct,
      @moonhnsw.Distance::Cosine,
    ] {
    let idx = @moonhnsw.Hnsw::new(2, metric)
    for row in vectors {
      let _ = idx.add(row)
    }
    for k in [1, 7, 100] {
      let oracle : Array[(Double, Int32)] = []
      for id in 0..<vectors.length() {
        if id % 3 != 0 {
          oracle.push((@moonhnsw.distance(metric, q, vectors[id]), id))
        }
      }
      oracle.sort_by(fn(a, b) {
        if a.0 < b.0 {
          -1
        } else if a.0 > b.0 {
          1
        } else {
          a.1.compare(b.1)
        }
      })
      let got = idx.search_exact(q, k, filter=fn(id) { id % 3 != 0 })
      let expected = if k < oracle.length() { k } else { oracle.length() }
      assert_eq(got.length(), expected)
      for rank in 0..<expected {
        assert_eq(got[rank].id, oracle[rank].1)
        assert_eq(got[rank].distance, oracle[rank].0)
      }
    }
  }
}
