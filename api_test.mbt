// Blackbox tests: verify the package works from an external user's
// perspective, using only the public surface with package-qualified names.

///|
test "public api: build, search, serialize" {
  let idx = @moonhnsw.Hnsw::new(2, @moonhnsw.Distance::L2)
  assert_eq(idx.size(), 0)
  let _ = idx.add(FixedArray::from_array([0.0, 0.0]))
  let _ = idx.add(FixedArray::from_array([1.0, 0.0]))
  let _ = idx.add(FixedArray::from_array([0.0, 1.0]))
  assert_eq(idx.size(), 3)
  let res = idx.search(FixedArray::from_array([0.9, 0.1]), 2, ef=32)
  assert_eq(res.length(), 2)
  assert_eq(res[0].id, 1)
  // Save/load through the public surface.
  let restored = @moonhnsw.Hnsw::load(idx.save())
  assert_eq(restored.size(), 3)
  let again = restored.search(FixedArray::from_array([0.9, 0.1]), 2, ef=32)
  assert_eq(again[0].id, res[0].id)
}

///|
test "public api: distance helper and error message" {
  let a : FixedArray[Float] = FixedArray::from_array([0.0, 0.0])
  let b : FixedArray[Float] = FixedArray::from_array([3.0, 4.0])
  assert_eq(@moonhnsw.distance(@moonhnsw.Distance::L2, a, b), 25.0)
  // Dimension mismatches surface as typed errors with readable messages.
  let err : Result[Double, @moonhnsw.HnswError] = Ok(
    @moonhnsw.distance(@moonhnsw.Distance::L2, a, FixedArray::make(3, 0.0)),
  ) catch {
    e => Err(e)
  }
  match err {
    Err(e) =>
      assert_eq(e.message(), "dimension mismatch: index expects 2, got 3")
    Ok(_) => fail("expected a dimension mismatch error")
  }
}

///|
test "public api: invalid params raise" {
  let err : Result[@moonhnsw.Hnsw, @moonhnsw.HnswError] = Ok(
    @moonhnsw.Hnsw::new(0, @moonhnsw.Distance::Cosine),
  ) catch {
    e => Err(e)
  }
  assert_true(err is Err(_))
}

///|
test "public api: delete hides vectors from search" {
  let idx = @moonhnsw.Hnsw::new(2, @moonhnsw.Distance::L2)
  let _ = idx.add(FixedArray::from_array([0.0, 0.0]))
  let _ = idx.add(FixedArray::from_array([1.0, 0.0]))
  let _ = idx.add(FixedArray::from_array([0.0, 1.0]))
  idx.delete(1)
  assert_eq(idx.size(), 2)
  assert_eq(idx.total(), 3)
  let res = idx.search(FixedArray::from_array([1.0, 0.0]), 3, ef=32)
  for r in res {
    assert_true(r.id != 1)
  }
  // Deleting twice raises a typed error with a readable message.
  let err : Result[Unit, @moonhnsw.HnswError] = Ok(idx.delete(1)) catch {
    e => Err(e)
  }
  match err {
    Err(e) =>
      assert_eq(
        e.message(),
        "invalid vector id 1 (out of range or already deleted)",
      )
    Ok(_) => fail("expected an invalid id error")
  }
}
