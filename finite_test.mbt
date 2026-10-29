///|
test "non-finite insertions fail before changing index or tombstones" {
  let nan = Float::from_double(@double.not_a_number)
  let inf = Float::from_double(@double.infinity)
  for storage in [@moonhnsw.Storage::Float32, @moonhnsw.Storage::SQ8] {
    let idx = @moonhnsw.Hnsw::new(
      1,
      @moonhnsw.Distance::L2,
      params=@moonhnsw.HnswParams::new(storage~),
    )
    let good : FixedArray[Float] = FixedArray::from_array([1.0])
    let id = idx.add_labeled(good, 7UL)
    let before = idx.save()
    for value in [nan, inf, -inf] {
      let bad : FixedArray[Float] = FixedArray::from_array([value])
      let add : Result[Int32, @moonhnsw.HnswError] = Ok(idx.add(bad)) catch {
        e => Err(e)
      }
      assert_true(add is Err(@moonhnsw.HnswError::InvalidParams(_)))
      let upsert : Result[Int32, @moonhnsw.HnswError] = Ok(
        idx.upsert_labeled(bad, 7UL),
      ) catch {
        e => Err(e)
      }
      assert_true(upsert is Err(@moonhnsw.HnswError::InvalidParams(_)))
      assert_eq(idx.find_by_label(7UL), [id])
      assert_eq(idx.save(), before)
    }
  }
}

///|
test "non-finite queries and distance helpers return typed errors" {
  let idx = @moonhnsw.Hnsw::new(1, @moonhnsw.Distance::L2)
  let good : FixedArray[Float] = FixedArray::from_array([1.0])
  let _ = idx.add(good)
  let bad : FixedArray[Float] = FixedArray::from_array([
    Float::from_double(@double.not_a_number),
  ])
  let search : Result[Array[@moonhnsw.Neighbor], @moonhnsw.HnswError] = Ok(
    idx.search(bad, 0),
  ) catch {
    e => Err(e)
  }
  assert_true(search is Err(@moonhnsw.HnswError::InvalidParams(_)))
  let exact : Result[Array[@moonhnsw.Neighbor], @moonhnsw.HnswError] = Ok(
    idx.search_exact(bad, 1),
  ) catch {
    e => Err(e)
  }
  assert_true(exact is Err(@moonhnsw.HnswError::InvalidParams(_)))
  let radius : Result[Array[@moonhnsw.Neighbor], @moonhnsw.HnswError] = Ok(
    idx.search_radius_exact(bad, 1.0),
  ) catch {
    e => Err(e)
  }
  assert_true(radius is Err(@moonhnsw.HnswError::InvalidParams(_)))
  let helper : Result[Double, @moonhnsw.HnswError] = Ok(
    @moonhnsw.distance(@moonhnsw.Distance::L2, bad, good),
  ) catch {
    e => Err(e)
  }
  assert_true(helper is Err(@moonhnsw.HnswError::InvalidParams(_)))
  let calibration : Result[@moonhnsw.EfCalibration, @moonhnsw.HnswError] = Ok(
    idx.calibrate_ef([bad], 1, 1.0, [1]),
  ) catch {
    e => Err(e)
  }
  assert_true(calibration is Err(@moonhnsw.HnswError::InvalidParams(_)))
}
