///|
test "public api: adaptive filtered search widens and reports its work" {
  let idx = @moonhnsw.Hnsw::new(1, @moonhnsw.Distance::L2)
  for i in 0..<30 {
    let _ = idx.add(FixedArray::from_array([Float::from_double(i.to_double())]))
  }
  let q : FixedArray[Float] = FixedArray::from_array([0.0])
  let result = idx.search_filtered_adaptive(q, 2, ef_start=2, ef_max=16, filter=fn(
    id,
  ) {
    id >= 10
  })
  assert_eq(result.neighbors.length(), 2)
  assert_true(result.rounds > 1)
  assert_true(result.final_ef <= 16)
  assert_true(result.neighbors[0].id >= 10)
  assert_true(result.neighbors[1].id >= 10)
  assert_eq(idx.distance_computations(), result.distance_computations)
  let pool = idx.search(q, result.final_ef, ef=result.final_ef)
  let expected : Array[@moonhnsw.Neighbor] = []
  for hit in pool {
    if hit.id >= 10 && expected.length() < 2 {
      expected.push(hit)
    }
  }
  assert_eq(result.neighbors, expected)
  let none = idx.search_filtered_adaptive(q, 2, ef_start=2, ef_max=8, filter=fn(
    _,
  ) {
    false
  })
  assert_eq(none.neighbors.length(), 0)
  assert_eq(none.final_ef, 8)
  let invalid : Result[@moonhnsw.AdaptiveSearchResult, @moonhnsw.HnswError] = Ok(
    idx.search_filtered_adaptive(q, 2, ef_start=8, ef_max=4, filter=fn(_) {
      true
    }),
  ) catch {
    e => Err(e)
  }
  assert_true(invalid is Err(_))
}
