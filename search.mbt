///|
/// Grows the visited-marker array to cover at least `needed` node ids.
fn Hnsw::ensure_visited(self : Hnsw, needed : Int32) -> Unit {
  if self.visited.length() >= needed {
    return
  }
  let mut cap = if self.visited.length() == 0 {
    64
  } else {
    self.visited.length() * 2
  }
  while cap < needed {
    cap = cap * 2
  }
  self.visited = FixedArray::make(cap, 0)
  self.stamp = 0
}

///|
/// Returns the next traversal generation, resetting markers on overflow.
fn Hnsw::next_stamp(self : Hnsw) -> Int32 {
  let mut s = self.stamp + 1
  if s <= 0 {
    s = 1
    self.visited.fill(0)
  }
  self.stamp = s
  s
}

///|
/// Distance between a query slice and a stored vector. Under SQ8 the query
/// is quantized into a reused scratch buffer once per search (detected by
/// reference identity) and the distance runs over bytes.
fn Hnsw::dist_query(self : Hnsw, q : FixedArray[Float], id : Int32) -> Double {
  self.dist_calls += 1L
  if self.q_active() {
    if !physical_equal(q, self.q_scratch_source) {
      self.quantize_into(q, 0, self.q_scratch, 0)
      self.q_scratch_source = q
    }
    return self.dist_qq(self.q_scratch, 0, id * self.dim)
  }
  distance_between(self.distance, q, 0, self.vectors, id * self.dim, self.dim)
}

///|
/// Distance between two stored vectors.
fn Hnsw::dist_ids(self : Hnsw, a : Int32, b : Int32) -> Double {
  self.dist_calls += 1L
  if self.q_active() {
    return self.dist_qq(self.quants, a * self.dim, b * self.dim)
  }
  distance_between(
    self.distance,
    self.vectors,
    a * self.dim,
    self.vectors,
    b * self.dim,
    self.dim,
  )
}

///|
/// Greedy hill-climb from `ep` on one layer; returns the local minimum.
fn Hnsw::greedy_descent(
  self : Hnsw,
  q : FixedArray[Float],
  ep : Int32,
  ep_dist : Double,
  layer : Int32,
) -> (Int32, Double) {
  let mut cur = ep
  let mut cur_dist = ep_dist
  let mut improved = true
  while improved {
    improved = false
    for e in self.links[cur][layer] {
      let d = self.dist_query(q, e)
      if d < cur_dist {
        cur_dist = d
        cur = e
        improved = true
      }
    }
  }
  (cur, cur_dist)
}

///|
/// Beam search on one layer (Algorithm 2 of the HNSW paper). `eps` are
/// (distance, id) entry points; returns up to `ef` results sorted by
/// distance, ascending.
///
/// Deleted (tombstone) nodes are traversed for connectivity but excluded
/// from the results, unless `include_dead` is set 闁?index construction uses
/// that mode so new nodes stay linked into the graph even when their
/// surroundings are deleted.
///
/// When `cutoff` is finite, expansion stops once every remaining candidate
/// is farther than the cutoff (radius search); entry points are always
/// expanded once so that neighbors inside the radius are still discovered
/// when every entry point itself lies outside it.
///
/// `accept` restricts which nodes may enter the result set (traversal is
/// unaffected); `include_dead`, `cutoff` and `accept` combine.
fn Hnsw::search_layer(
  self : Hnsw,
  q : FixedArray[Float],
  eps : Array[(Double, Int32)],
  ef : Int32,
  layer : Int32,
  include_dead~ : Bool = false,
  cutoff~ : Double = @double.infinity,
  accept~ : (Int32) -> Bool = fn(_) { true },
) -> Array[(Double, Int32)] {
  let stamp = self.next_stamp()
  let visited = self.visited
  // Hoist the mutable field loads out of the per-hop loops; nothing in this
  // function mutates them, and dist_query only touches distance state.
  let alive = self.alive
  let links = self.links
  let count = self.count
  // Reuse the heap backing stores: to_sorted_ascending drains them, so they
  // come back empty and keep their grown capacity.
  let candidates = Heap::{ items: self.cand_items, is_min: true, }
  let results = Heap::{ items: self.result_items, is_min: false, }
  let seeds : Array[Int32] = []
  for (d, e) in eps {
    if e >= 0 && e < count && visited[e] != stamp {
      visited[e] = stamp
      candidates.push(d, e)
      if d <= cutoff && (include_dead || alive[e]) && accept(e) {
        results.push(d, e)
      }
      seeds.push(e)
    }
  }
  // Expand the entry points once unconditionally.
  for e in seeds {
    for nb in links[e][layer] {
      if visited[nb] != stamp {
        visited[nb] = stamp
        let d = self.dist_query(q, nb)
        candidates.push(d, nb)
        if d <= cutoff && (include_dead || alive[nb]) && accept(nb) {
          results.push(d, nb)
        }
      }
    }
  }
  while results.size() > ef {
    let _ = results.pop()
  }
  while candidates.size() > 0 {
    let c = candidates.top()
    if c.0 > cutoff {
      break
    }
    if results.size() >= ef && c.0 > results.top().0 {
      break
    }
    let _ = candidates.pop()
    for e in links[c.1][layer] {
      if visited[e] != stamp {
        visited[e] = stamp
        let d = self.dist_query(q, e)
        if d <= cutoff && (results.size() < ef || d < results.top().0) {
          candidates.push(d, e)
          if (include_dead || alive[e]) && accept(e) {
            results.push(d, e)
            if results.size() > ef {
              let _ = results.pop()
            }
          }
        }
      }
    }
  }
  // The remaining candidates are never read. Clear the reused backing store
  // directly instead of heap-popping every element.
  candidates.items.clear()
  results.to_sorted_ascending()
}

///|
/// Diverse neighbor selection (Algorithm 4 of the HNSW paper): from a
/// candidate list sorted by distance to the base point, keep up to `m` points
/// that are closer to the base than to any point already selected, then top
/// up with the pruned candidates.
fn Hnsw::select_heuristic(
  self : Hnsw,
  sorted : Array[(Double, Int32)],
  m : Int32,
) -> Array[Int32] {
  let selected : Array[Int32] = []
  let pruned : Array[(Double, Int32)] = []
  for (d, id) in sorted {
    if selected.length() >= m {
      pruned.push((d, id))
      continue
    }
    let mut good = true
    for r in selected {
      if self.dist_ids(id, r) < d {
        good = false
        break
      }
    }
    if good {
      selected.push(id)
    } else {
      pruned.push((d, id))
    }
  }
  if selected.length() < m {
    for (_, id) in pruned {
      if selected.length() >= m {
        break
      }
      selected.push(id)
    }
  }
  selected
}

///|
/// Approximate top-k search. Returns at most `k` neighbors sorted by
/// distance, ascending. `ef` is the search beam width; larger values give
/// higher recall at a higher cost. The search is deterministic for a given
/// index state.
pub fn Hnsw::search(
  self : Hnsw,
  query : FixedArray[Float],
  k : Int32,
  ef~ : Int32 = 64,
) -> Array[Neighbor] raise HnswError {
  if query.length() != self.dim {
    raise DimMismatch(self.dim, query.length())
  }
  validate_finite_vector(query)
  self.dist_calls = 0L
  if self.count == 0 || self.alive_count == 0 || k <= 0 {
    return []
  }
  self.q_scratch_source = self.q_scratch_reset
  // L2/IP search the caller's array directly; only Cosine needs a
  // normalized copy.
  let q : FixedArray[Float] = if self.normalize {
    let tmp : FixedArray[Float] = FixedArray::make(self.dim, 0.0)
    let _ = copy_normalized(query, 0, tmp, 0, self.dim, true)
    tmp
  } else {
    query
  }
  let r = self.descend_entry(q)
  let width = if ef < k { k } else { ef }
  let cands = self.search_layer(q, [r], width, 0)
  let out : Array[Neighbor] = []
  for (d, id) in cands {
    if out.length() >= k {
      break
    }
    out.push({ id, distance: d, })
  }
  out
}

///|
/// Approximate top-k search restricted to vectors whose id passes `filter`.
///
/// Traversal walks the graph as usual; only the result set is filtered, so
/// with a highly selective filter the call may return fewer than `k` results
/// even when more matches exist. Filter closures
/// receive the internal id; combine with `label_of(id)` for label-based
/// filtering. Deleted vectors never pass, regardless of the filter.
pub fn Hnsw::search_filtered(
  self : Hnsw,
  query : FixedArray[Float],
  k : Int32,
  ef~ : Int32 = 64,
  filter~ : (Int32) -> Bool,
) -> Array[Neighbor] raise HnswError {
  if query.length() != self.dim {
    raise DimMismatch(self.dim, query.length())
  }
  validate_finite_vector(query)
  self.dist_calls = 0L
  if self.count == 0 || self.alive_count == 0 || k <= 0 {
    return []
  }
  self.q_scratch_source = self.q_scratch_reset
  // L2/IP search the caller's array directly; only Cosine needs a
  // normalized copy.
  let q : FixedArray[Float] = if self.normalize {
    let tmp : FixedArray[Float] = FixedArray::make(self.dim, 0.0)
    let _ = copy_normalized(query, 0, tmp, 0, self.dim, true)
    tmp
  } else {
    query
  }
  let ep = self.descend_entry(q)
  let width = if ef < k { k } else { ef }
  let cands = self.search_layer(q, [ep], width, 0, accept=filter)
  let out : Array[Neighbor] = []
  for (d, id) in cands {
    if out.length() >= k {
      break
    }
    out.push({ id, distance: d, })
  }
  out
}

///|
/// Runs `search` for every query in `queries` with the same `k` and `ef`,
/// returning one result list per query, in order. A query with a wrong
/// dimension raises on its turn, after earlier results were produced.
pub fn Hnsw::search_batch(
  self : Hnsw,
  queries : Array[FixedArray[Float]],
  k : Int32,
  ef~ : Int32 = 64,
) -> Array[Array[Neighbor]] raise HnswError {
  let out : Array[Array[Neighbor]] = []
  let mut total_calls = 0L
  for q in queries {
    out.push(self.search(q, k, ef~))
    total_calls += self.dist_calls
  }
  self.dist_calls = total_calls
  out
}

///|
/// Maximal Marginal Relevance search: retrieves a candidate pool with an
/// `ef` beam search, then greedily selects `k` results that balance
/// relevance to the query against redundancy with the already-selected set.
///
/// `lambda` blends the two objectives: 1.0 keeps only the most relevant
/// candidates (identical to `search`), 0.0 picks the nearest result first
/// and then prioritizes dissimilarity to the chosen results; 0.3-0.7 is the usual
/// working range for de-duplicated RAG contexts. `filter` restricts the
/// candidate pool. Similarity is the negated index distance, so every metric
/// works unchanged.
pub fn Hnsw::search_mmr(
  self : Hnsw,
  query : FixedArray[Float],
  k : Int32,
  lambda~ : Double = 0.5,
  ef~ : Int32 = 128,
  filter~ : (Int32) -> Bool = fn(_) { true },
) -> Array[Neighbor] raise HnswError {
  if !(lambda >= 0.0 && lambda <= 1.0) {
    raise InvalidParams("lambda must be within [0, 1]")
  }
  if k <= 0 {
    return self.search_filtered(query, k, ef~, filter~)
  }
  if lambda == 1.0 {
    return self.search_filtered(query, k, ef~, filter~)
  }
  // The pool must be wider than k, or near-duplicate candidates crowd out
  // any diversity the greedy pass could achieve.
  let pool_size = if ef < k { k } else { ef }
  let pool = self.search_filtered(query, pool_size, ef~, filter~)
  if pool.length() <= 1 {
    let out : Array[Neighbor] = []
    for r in pool {
      if out.length() >= k {
        break
      }
      out.push(r)
    }
    return out
  }
  let n = pool.length()
  let taken : Array[Bool] = Array::make(n, false)
  // Each candidate only needs its maximum similarity to the selected set.
  // Update it once per newly selected neighbor instead of recomputing every
  // previously selected pair on every round.
  let max_sims : Array[Double] = Array::make(n, -1.0e308)
  let out : Array[Neighbor] = []
  for _ in 0..<k {
    let mut best = -1
    let mut best_score = 0.0
    for i in 0..<n {
      if !taken[i] {
        // Similarity is the negated distance; the first pick has no
        // redundancy penalty.
        let relevance = -pool[i].distance
        let penalty = if out.length() == 0 {
          0.0
        } else {
          (1.0 - lambda) * max_sims[i]
        }
        let score = lambda * relevance - penalty
        if best < 0 || score > best_score {
          best = i
          best_score = score
        }
      }
    }
    if best < 0 {
      break
    }
    taken[best] = true
    out.push(pool[best])
    if out.length() < k && out.length() < n {
      for i in 0..<n {
        if !taken[i] {
          let sim = -self.dist_ids(pool[i].id, pool[best].id)
          if sim > max_sims[i] {
            max_sims[i] = sim
          }
        }
      }
    }
  }
  out
}

///|
/// Approximate radius search: returns up to `max_results` non-deleted
/// neighbors with distance <= `radius`, sorted by distance ascending.
///
/// Like top-k search this is graph-approximate: expansion stops once every
/// remaining candidate is farther than `radius`, so a neighbor just inside
/// the radius can be missed when it is only reachable through farther nodes.
/// Entry points are always expanded, which keeps exact-match queries
/// (`radius = 0`) working. `filter` restricts the result set (traversal is
/// unaffected); deleted vectors never pass.
pub fn Hnsw::search_radius(
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
  self.dist_calls = 0L
  if max_results <= 0 {
    return []
  }
  if self.count == 0 || self.alive_count == 0 {
    return []
  }
  self.q_scratch_source = self.q_scratch_reset
  // L2/IP search the caller's array directly; only Cosine needs a
  // normalized copy.
  let q : FixedArray[Float] = if self.normalize {
    let tmp : FixedArray[Float] = FixedArray::make(self.dim, 0.0)
    let _ = copy_normalized(query, 0, tmp, 0, self.dim, true)
    tmp
  } else {
    query
  }
  let ep = self.descend_entry(q)
  let cands = self.search_layer(
    q,
    [ep],
    max_results,
    0,
    cutoff=radius,
    accept=filter,
  )
  let out : Array[Neighbor] = []
  for (d, id) in cands {
    out.push({ id, distance: d, })
  }
  out
}

///|
/// Greedy descent from the entry point down to layer 0; returns the
/// (distance, id) local minimum used to seed the layer-0 beam search.
fn Hnsw::descend_entry(self : Hnsw, q : FixedArray[Float]) -> (Double, Int32) {
  let mut ep = self.entry
  let mut ep_dist = self.dist_query(q, ep)
  let mut cur = self.max_level
  while cur > 0 {
    let r = self.greedy_descent(q, ep, ep_dist, cur)
    ep = r.0
    ep_dist = r.1
    cur -= 1
  }
  (ep_dist, ep)
}
