///|
test "public api: ef calibration survives SQ8 save and load" {
  let idx = @moonhnsw.Hnsw::new(
    2,
    @moonhnsw.Distance::L2,
    params=@moonhnsw.HnswParams::new(storage=@moonhnsw.Storage::SQ8),
  )
  for i in 0..<260 {
    let v = FixedArray::from_array([
      Float::from_double((i % 20).to_double()),
      Float::from_double((i / 20).to_double()),
    ])
    let _ = idx.add(v)
  }
  idx.delete(0)
  let queries : Array[FixedArray[Float]] = [
    FixedArray::from_array([1.1, 2.2]),
    FixedArray::from_array([9.3, 8.7]),
  ]
  let result = idx.calibrate_ef(queries, 4, 0.75, [4, 16, 64])
  assert_eq(result.trials.length(), 3)
  assert_eq(result.trials[0].ef, 4)
  assert_eq(result.trials[1].ef, 16)
  assert_eq(result.trials[2].ef, 64)
  assert_true(result.trials[0].recall >= 0.0)
  assert_true(result.trials[2].recall <= 1.0)
  assert_true(result.trials[2].avg_distance_computations > 0.0)
  assert_true(idx.distance_computations() > 0L)
  if result.met_target {
    assert_true(
      result.selected_ef == 4 ||
      result.selected_ef == 16 ||
      result.selected_ef == 64,
    )
  } else {
    assert_eq(result.selected_ef, 64)
  }
  let loaded = @moonhnsw.Hnsw::load(idx.save())
  assert_eq(loaded.calibrate_ef(queries, 4, 0.75, [4, 16, 64]), result)
  let invalid : Result[@moonhnsw.EfCalibration, @moonhnsw.HnswError] = Ok(
    idx.calibrate_ef(queries, 4, 0.9, [3, 16]),
  ) catch {
    e => Err(e)
  }
  assert_true(invalid is Err(_))
}
