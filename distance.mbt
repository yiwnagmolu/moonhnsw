///|
/// Distance between two equal-length vectors under `metric`, using the same
/// convention as search results. Useful for brute-force ground truth and
/// benchmarks.
pub fn distance(
  metric : Distance,
  a : FixedArray[Float],
  b : FixedArray[Float],
) -> Double raise HnswError {
  if a.length() != b.length() {
    raise DimMismatch(a.length(), b.length())
  }
  validate_finite_vector(a)
  validate_finite_vector(b)
  if metric is Cosine {
    // Match copy_normalized's f32 rounding without allocating two temporary
    // arrays for every pair. Zero vectors keep their original components.
    let mut norm_a_sq = 0.0
    let mut norm_b_sq = 0.0
    for i in 0..<a.length() {
      let av = a.unsafe_get(i).to_double()
      let bv = b.unsafe_get(i).to_double()
      norm_a_sq += av * av
      norm_b_sq += bv * bv
    }
    let normalize_a = norm_a_sq > 0.0
    let normalize_b = norm_b_sq > 0.0
    let inv_a = if normalize_a { 1.0 / norm_a_sq.sqrt() } else { 0.0 }
    let inv_b = if normalize_b { 1.0 / norm_b_sq.sqrt() } else { 0.0 }
    let mut sum : Float = 0.0
    for i in 0..<a.length() {
      let av = if normalize_a {
        Float::from_double(a.unsafe_get(i).to_double() * inv_a)
      } else {
        a.unsafe_get(i)
      }
      let bv = if normalize_b {
        Float::from_double(b.unsafe_get(i).to_double() * inv_b)
      } else {
        b.unsafe_get(i)
      }
      sum += av * bv
    }
    return -sum.to_double()
  }
  distance_between(metric, a, 0, b, 0, a.length())
}

///|
/// NaN or infinity makes graph ordering, SQ8 bounds, and persistence invalid.
/// Check inputs before any index mutation or distance work begins.
fn validate_finite_vector(vector : FixedArray[Float]) -> Unit raise HnswError {
  for value in vector {
    let v = value.to_double()
    if !(v > -@double.infinity && v < @double.infinity) {
      raise InvalidParams("vector must contain only finite values")
    }
  }
}

///|
/// Distance between two vector slices of length `dim`, under `metric`.
/// For `InnerProduct` and `Cosine` the inner product is negated so that
/// smaller always means more similar.
///
/// Accumulation normally happens in `Float`, matching hnswlib. If finite
/// inputs overflow that path, recompute in `Double` so graph ordering still
/// has a finite distance.
fn distance_between(
  metric : Distance,
  a : FixedArray[Float],
  a_off : Int32,
  b : FixedArray[Float],
  b_off : Int32,
  dim : Int32,
) -> Double {
  match metric {
    L2 => {
      let mut s : Float = 0.0
      let mut i = 0
      while i < dim {
        let d = a.unsafe_get(a_off + i) - b.unsafe_get(b_off + i)
        s += d * d
        i += 1
      }
      let result = s.to_double()
      if result > -@double.infinity && result < @double.infinity {
        result
      } else {
        let mut recovered = 0.0
        let mut j = 0
        while j < dim {
          let d = a.unsafe_get(a_off + j).to_double() -
            b.unsafe_get(b_off + j).to_double()
          recovered += d * d
          j += 1
        }
        recovered
      }
    }
    InnerProduct | Cosine => {
      let mut s : Float = 0.0
      let mut i = 0
      while i < dim {
        s += a.unsafe_get(a_off + i) * b.unsafe_get(b_off + i)
        i += 1
      }
      let result = s.to_double()
      if result > -@double.infinity && result < @double.infinity {
        -result
      } else {
        let mut recovered = 0.0
        let mut j = 0
        while j < dim {
          recovered += a.unsafe_get(a_off + j).to_double() *
            b.unsafe_get(b_off + j).to_double()
          j += 1
        }
        -recovered
      }
    }
  }
}

///|
/// Copies `src[src_off .. src_off+dim]` into `dst[dst_off .. dst_off+dim]`,
/// L2-normalizing when `normalize` is true and the norm is non-zero. Zero
/// vectors are copied as-is.
fn copy_normalized(
  src : FixedArray[Float],
  src_off : Int32,
  dst : FixedArray[Float],
  dst_off : Int32,
  dim : Int32,
  normalize : Bool,
) -> Double {
  let mut norm_sq = 0.0
  let mut i = 0
  while i < dim {
    let v = src.unsafe_get(src_off + i).to_double()
    norm_sq += v * v
    i += 1
  }
  if normalize && norm_sq > 0.0 {
    let inv = 1.0 / norm_sq.sqrt()
    i = 0
    while i < dim {
      dst[dst_off + i] = Float::from_double(
        src.unsafe_get(src_off + i).to_double() * inv,
      )
      i += 1
    }
  } else {
    i = 0
    while i < dim {
      dst[dst_off + i] = src.unsafe_get(src_off + i)
      i += 1
    }
  }
  norm_sq.sqrt()
}

///|
/// Sorts (distance, id) pairs by distance, ascending. Insertion sort: the
/// inputs here are beam-sized, so O(n^2) is negligible next to the distance
/// computations that produced them.
fn sort_pairs_ascending(
  pairs : Array[(Double, Int32)],
) -> Array[(Double, Int32)] {
  let arr = pairs
  for x in 1..<arr.length() {
    let v = arr[x]
    let mut y = x - 1
    while y >= 0 && arr[y].0 > v.0 {
      arr[y + 1] = arr[y]
      y -= 1
    }
    arr[y + 1] = v
  }
  arr
}
