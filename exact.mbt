///|
/// Bounded max-heap for exact scans. The worst (distance, id) pair is at the
/// root; smaller ids win equal-distance ties so results are reproducible.
priv struct ExactHeap {
  items : Array[(Double, Int32)]
}

///|
fn exact_worse(a : (Double, Int32), b : (Double, Int32)) -> Bool {
  a.0 > b.0 || (a.0 == b.0 && a.1 > b.1)
}

///|
fn ExactHeap::offer(
  self : ExactHeap,
  distance : Double,
  id : Int32,
  limit : Int32,
) -> Unit {
  let items = self.items
  let pair = (distance, id)
  if items.length() < limit {
    items.push(pair)
    let mut i = items.length() - 1
    while i > 0 {
      let parent = (i - 1) / 2
      if !exact_worse(items[i], items[parent]) {
        break
      }
      let old = items[parent]
      items[parent] = items[i]
      items[i] = old
      i = parent
    }
  } else if exact_worse(items[0], pair) {
    items[0] = pair
    let mut i = 0
    let mut sinking = true
    while sinking {
      let left = i * 2 + 1
      let right = left + 1
      let mut worst = i
      if left < items.length() && exact_worse(items[left], items[worst]) {
        worst = left
      }
      if right < items.length() && exact_worse(items[right], items[worst]) {
        worst = right
      }
      if worst == i {
        sinking = false
      } else {
        let old = items[i]
        items[i] = items[worst]
        items[worst] = old
        i = worst
      }
    }
  }
}

///|
fn Hnsw::exact_scan(
  self : Hnsw,
  query : FixedArray[Float],
  limit : Int32,
  cutoff : Double,
  filter : (Int32) -> Bool,
) -> Array[Neighbor] {
  self.dist_calls = 0L
  if limit <= 0 || self.alive_count == 0 {
    return []
  }
  let q : FixedArray[Float] = if self.normalize {
    let normalized : FixedArray[Float] = FixedArray::make(self.dim, 0.0)
    let _ = copy_normalized(query, 0, normalized, 0, self.dim, true)
    normalized
  } else {
    query
  }
  let heap = ExactHeap::{ items: [], }
  for id in 0..<self.count {
    if self.alive[id] && filter(id) {
      let d = self.dist_query(q, id)
      if d <= cutoff {
        heap.offer(d, id, limit)
      }
    }
  }
  heap.items.sort_by(fn(a, b) {
    if a.0 < b.0 {
      -1
    } else if a.0 > b.0 {
      1
    } else {
      a.1.compare(b.1)
    }
  })
  let out : Array[Neighbor] = []
  for (distance, id) in heap.items {
    out.push({ id, distance, })
  }
  out
}

///|
/// Scans every live stored vector and returns the exact nearest `k`, sorted
/// by distance then id. Unlike graph search, this guarantees the best stored
/// matches, including after deletions and with selective filters, at O(n*dim)
/// distance work. For SQ8 it is exact over the quantized vectors that the
/// index actually stores, not the original pre-quantization inputs.
/// `distance_computations()` reports the number of live vectors passing the
/// filter that were scored.
pub fn Hnsw::search_exact(
  self : Hnsw,
  query : FixedArray[Float],
  k : Int32,
  filter~ : (Int32) -> Bool = fn(_) { true },
) -> Array[Neighbor] raise HnswError {
  if query.length() != self.dim {
    raise DimMismatch(self.dim, query.length())
  }
  validate_finite_vector(query)
  self.exact_scan(query, k, @double.infinity, filter)
}

///|
/// Exhaustive radius search over live stored vectors. Returns up to
/// `max_results` hits within `radius`, sorted by distance then id. This
/// guarantees completeness within the returned limit; it uses the same SQ8
/// distance as `search_exact` when storage is quantized.
pub fn Hnsw::search_radius_exact(
  self : Hnsw,
  query : FixedArray[Float],
  radius : Double,
  max_results~ : Int32 = 64,
  filter~ : (Int32) -> Bool = fn(_) { true },
) -> Array[Neighbor] raise HnswError {
  if query.length() != self.dim {
    raise DimMismatch(self.dim, query.length())
  }
  validate_finite_vector(query)
  if radius != radius || (self.distance is L2 && radius < 0.0) {
    raise InvalidParams("radius must be a valid distance for this metric")
  }
  self.exact_scan(query, max_results, radius, filter)
}
