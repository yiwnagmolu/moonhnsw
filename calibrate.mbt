///|
/// One measured beam width from `Hnsw::calibrate_ef`.
pub struct EfTrial {
  /// Beam width tested (must be at least k).
  ef : Int32
  /// Mean recall@k across the representative queries, against an exact scan
  /// of live vectors using the index's own distance semantics. Neighbors tied
  /// at the exact kth distance count as correct.
  recall : Double
  /// Mean distance computations per approximate query, excluding the exact
  /// ground-truth scan.
  avg_distance_computations : Double
} derive(Eq, @debug.Debug, ToJson)

///|
pub extend EfTrial with Eq::{not_equal, equal}

///|
pub extend EfTrial with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend EfTrial with ToJson::{to_json}

///|
/// Empirical recall/cost curve for a set of query samples.
pub struct EfCalibration {
  /// Smallest tested ef that met the target, or the largest tested ef when
  /// no trial met it.
  selected_ef : Int32
  /// Whether the selected trial reached the requested target recall.
  met_target : Bool
  /// Trial results in ascending ef order.
  trials : Array[EfTrial]
} derive(Eq, @debug.Debug, ToJson)

///|
pub extend EfCalibration with Eq::{not_equal, equal}

///|
pub extend EfCalibration with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend EfCalibration with ToJson::{to_json}

///|
/// Measures recall@k and distance work for each candidate `ef`, then picks
/// the smallest width meeting `target_recall`. When none meets it, selects
/// the largest width and sets `met_target` to false.
///
/// `queries` must be nonempty and representative of production traffic;
/// measured recall is specific to those queries and this index state. `efs`
/// must be nonempty, strictly increasing, and each value at least `k`.
/// Ground truth is an exact top-k scan of non-deleted vectors using the same
/// metric and, for frozen SQ8, the same byte-distance computation as search.
/// Any result tied at the exact kth distance counts as a correct neighbor.
/// The stored `distance_computations()` after this call is the total work of
/// the approximate trial searches, excluding ground-truth scans.
pub fn Hnsw::calibrate_ef(
  self : Hnsw,
  queries : Array[FixedArray[Float]],
  k : Int32,
  target_recall : Double,
  efs : Array[Int32],
) -> EfCalibration raise HnswError {
  if k <= 0 || self.alive_count == 0 || queries.is_empty() || efs.is_empty() {
    raise InvalidParams("calibration needs live vectors, queries, k, and efs")
  }
  if !(target_recall >= 0.0 && target_recall <= 1.0) {
    raise InvalidParams("target_recall must be within [0, 1]")
  }
  let mut previous = 0
  for ef in efs {
    if ef < k || ef <= previous {
      raise InvalidParams("efs must be strictly increasing and >= k")
    }
    previous = ef
  }
  for q in queries {
    if q.length() != self.dim {
      raise DimMismatch(self.dim, q.length())
    }
    validate_finite_vector(q)
  }
  let truth_k = if k < self.alive_count { k } else { self.alive_count }
  let thresholds : Array[Double] = []
  for query in queries {
    let nearest = self.search_exact(query, truth_k)
    thresholds.push(nearest[nearest.length() - 1].distance)
  }
  let trials : Array[EfTrial] = []
  let mut selected_ef = efs[efs.length() - 1]
  let mut met_target = false
  let mut total_trial_calls = 0L
  for ef in efs {
    let mut hits = 0
    let mut calls = 0L
    for i in 0..<queries.length() {
      let results = self.search(queries[i], k, ef~)
      calls += self.dist_calls
      for r in results {
        if r.distance <= thresholds[i] {
          hits += 1
        }
      }
    }
    total_trial_calls += calls
    let recall = hits.to_double() /
      (queries.length().to_double() * truth_k.to_double())
    trials.push({
      ef,
      recall,
      avg_distance_computations: calls.to_double() /
      queries.length().to_double(),
    })
    if !met_target && recall >= target_recall {
      selected_ef = ef
      met_target = true
    }
  }
  self.dist_calls = total_trial_calls
  { selected_ef, met_target, trials, }
}
