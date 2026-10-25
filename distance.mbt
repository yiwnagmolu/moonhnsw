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
  distance_between(metric, a, 0, b, 0, a.length())
}

///|
/// Distance between two vector slices of length `dim`, under `metric`.
/// For `InnerProduct` and `Cosine` the inner product is negated so that
/// smaller always means more similar.
///
/// Accumulation happens in `Float`, matching hnswlib: the input is f32, the
/// result only feeds comparisons, and skipping the per-element f64 widening
/// is measurably faster on every backend.
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
      s.to_double()
    }
    InnerProduct | Cosine => {
      let mut s : Float = 0.0
      let mut i = 0
      while i < dim {
        s += a.unsafe_get(a_off + i) * b.unsafe_get(b_off + i)
        i += 1
      }
      -s.to_double()
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
