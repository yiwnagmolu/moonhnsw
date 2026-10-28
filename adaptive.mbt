///|
/// Outcome of an adaptive filtered search.
pub struct AdaptiveSearchResult {
  /// Matching neighbors, sorted by distance ascending.
  neighbors : Array[Neighbor]
  /// Last beam width used; zero when no search was needed.
  final_ef : Int32
  /// Number of widths tried.
  rounds : Int32
  /// Total distance computations across all rounds.
  distance_computations : Int64
} derive(Eq, @debug.Debug, ToJson)

///|
pub extend AdaptiveSearchResult with Eq::{not_equal, equal}

///|
pub extend AdaptiveSearchResult with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AdaptiveSearchResult with ToJson::{to_json}

///|
/// Searches an unfiltered HNSW candidate pool, applies `filter` to those
/// neighbors, and doubles the beam width until `k` matches are found or
/// `ef_max` is reached. The first width is `ef_start`; both widths are
/// raised to at least `k`, and widening stops at the number of live vectors.
///
/// Each round limits the candidate pool to its beam width, which can be useful
/// when a selective filter would otherwise make filter-aware traversal inspect
/// most of the graph. Results remain approximate: an unseen matching vector
/// may be nearer, and the call may return fewer than `k` matches at the cap.
/// `distance_computations()` reports the sum across rounds after this call.
pub fn Hnsw::search_filtered_adaptive(
  self : Hnsw,
  query : FixedArray[Float],
  k : Int32,
  ef_start~ : Int32 = 32,
  ef_max~ : Int32 = 512,
  filter~ : (Int32) -> Bool,
) -> AdaptiveSearchResult raise HnswError {
  if query.length() != self.dim {
    raise DimMismatch(self.dim, query.length())
  }
  if ef_start < 1 || ef_max < ef_start {
    raise InvalidParams("ef_start must be >= 1 and ef_max >= ef_start")
  }
  if k <= 0 || self.alive_count == 0 {
    self.dist_calls = 0L
    return { neighbors: [], final_ef: 0, rounds: 0, distance_computations: 0L, }
  }
  let mut cap = if ef_max < k { k } else { ef_max }
  let min_useful = if self.alive_count < k { k } else { self.alive_count }
  if cap > min_useful {
    cap = min_useful
  }
  let mut width = if ef_start < k { k } else { ef_start }
  if width > cap {
    width = cap
  }
  let mut total_calls = 0L
  let mut rounds = 0
  let mut matches : Array[Neighbor] = []
  let mut done = false
  while !done {
    rounds += 1
    matches = []
    let pool = self.search(query, width, ef=width)
    total_calls += self.dist_calls
    for neighbor in pool {
      if filter(neighbor.id) {
        matches.push(neighbor)
        if matches.length() >= k {
          break
        }
      }
    }
    if matches.length() >= k || width >= cap {
      done = true
    } else if width > cap / 2 {
      width = cap
    } else {
      width *= 2
    }
  }
  self.dist_calls = total_calls
  {
    neighbors: matches,
    final_ef: width,
    rounds,
    distance_computations: total_calls,
  }
}
