///|
/// Distance metric used by the index.
///
/// Search results always report a *smaller-is-closer* distance:
///
/// - `L2`: squared Euclidean distance.
/// - `InnerProduct`: negated inner product (smaller = more similar).
/// - `Cosine`: negated inner product of L2-normalized vectors; vectors are
///   normalized once on insertion and queries are normalized per search.
pub(all) enum Distance {
  L2
  InnerProduct
  Cosine
}

///|
/// Tuning parameters of the index. Construct with `HnswParams::new` to get
/// defaults; values outside the supported range are clamped by `Hnsw::new`.
pub struct HnswParams {
  /// Maximum number of links kept per node on layers above 0. Larger values
  /// give higher recall at the cost of memory and build time. Clamped to >= 2.
  m : Int32
  /// Size of the candidate list kept while inserting a node. Larger values
  /// build a higher-quality graph at a higher build cost. Clamped to >= 1.
  ef_construction : Int32
  /// Seed of the internal PRNG used to draw node levels.
  seed : UInt64
}

///|
/// Default parameters: `m = 16`, `ef_construction = 200`, fixed random seed.
pub fn HnswParams::new(
  m~ : Int32 = 16,
  ef_construction~ : Int32 = 200,
  seed~ : UInt64 = 0x9E3779B97F4A7C15UL,
) -> HnswParams {
  { m, ef_construction, seed, }
}

///|
/// Errors reported by the index.
pub suberror HnswError {
  /// Constructor arguments are out of range.
  InvalidParams(String)
  /// A vector dimension does not match the index dimension.
  DimMismatch(Int32, Int32)
  /// Serialized data is malformed.
  CorruptData(String)
  /// A vector id does not exist or refers to a deleted vector.
  InvalidId(Int32)
}

///|
/// Human-readable description of the error.
pub fn HnswError::message(self : HnswError) -> String {
  match self {
    InvalidParams(msg) => "invalid params: " + msg
    DimMismatch(expected, got) =>
      "dimension mismatch: index expects \{expected}, got \{got}"
    CorruptData(msg) => "corrupt data: " + msg
    InvalidId(id) => "invalid vector id \{id} (out of range or already deleted)"
  }
}

///|
/// One search result: the vector id and its distance to the query under the
/// index metric (see `Distance` for sign conventions).
pub struct Neighbor {
  id : Int32
  distance : Double
}

///|
/// An approximate nearest-neighbor index over fixed-dimension `f32` vectors,
/// using the HNSW navigable small-world graph (Malkov & Yashunin, 2016).
///
/// The index supports incremental insertion and top-k search on all backends
/// (native / wasm / js). It is single-threaded: do not share one index across
/// concurrent writers or readers.
pub struct Hnsw {
  /// Dimension of the stored vectors.
  dim : Int32
  /// Distance metric.
  distance : Distance
  /// Effective parameters (after clamping).
  params : HnswParams
  /// True when the metric pre-normalizes vectors (Cosine).
  normalize : Bool
  /// 1 / ln(m) — level assignment factor.
  level_mult : Double
  /// Id of the entry point, -1 when the index is empty.
  mut entry : Int32
  /// Highest layer any node participates in, -1 when the index is empty.
  mut max_level : Int32
  /// Total number of vectors ever inserted (including deleted tombstones).
  mut count : Int32
  /// Number of non-deleted (searchable) vectors.
  mut alive_count : Int32
  /// Alive markers for tombstone deletion, keyed by node id.
  mut alive : FixedArray[Bool]
  /// User-facing label per node; defaults to the internal id.
  labels : Array[UInt64]
  /// Flattened vector storage: `count * dim` floats.
  mut vectors : FixedArray[Float]
  /// Number of vector slots allocated in `vectors`.
  mut capacity : Int32
  /// `links[node][layer]` is the neighbor id list of `node` on `layer`.
  links : Array[Array[Array[Int32]]]
  /// Visited markers for graph traversal, keyed by node id.
  mut visited : FixedArray[Int32]
  /// Current traversal generation.
  mut stamp : Int32
  /// PRNG state used for level assignment.
  mut rng_state : UInt64
}

///|
/// Creates an empty index over vectors of dimension `dim`.
pub fn Hnsw::new(
  dim : Int32,
  distance : Distance,
  params~ : HnswParams = HnswParams::new(),
) -> Hnsw raise HnswError {
  if dim < 1 {
    raise InvalidParams("dim must be >= 1")
  }
  let m = if params.m < 2 { 2 } else { params.m }
  let efc = if params.ef_construction < 1 { 1 } else { params.ef_construction }
  {
    dim,
    distance,
    params: HnswParams::new(m~, ef_construction=efc, seed=params.seed),
    normalize: match distance {
      Cosine => true
      _ => false
    },
    level_mult: 1.0 / @math.ln(m.to_double()),
    entry: -1,
    max_level: -1,
    count: 0,
    alive_count: 0,
    alive: FixedArray::make(0, false),
    labels: [],
    vectors: FixedArray::make(0, 0.0),
    capacity: 0,
    links: [],
    visited: FixedArray::make(0, 0),
    stamp: 0,
    rng_state: params.seed,
  }
}

///|
/// Number of non-deleted vectors currently searchable in the index.
pub fn Hnsw::size(self : Hnsw) -> Int32 {
  self.alive_count
}

///|
/// Total number of vectors ever inserted, including deleted tombstones.
pub fn Hnsw::total(self : Hnsw) -> Int32 {
  self.count
}

///|
/// Returns true when `id` refers to a stored, non-deleted vector.
pub fn Hnsw::contains(self : Hnsw, id : Int32) -> Bool {
  id >= 0 && id < self.count && self.alive[id]
}

///|
/// Returns the label stored for `id` — the id itself by default, or the
/// value passed to `add_labeled`. Raises `InvalidId` for unknown or deleted
/// ids.
pub fn Hnsw::label_of(self : Hnsw, id : Int32) -> UInt64 raise HnswError {
  if !self.contains(id) {
    raise InvalidId(id)
  }
  self.labels[id]
}

///|
/// Returns a copy of the stored vector for `id`.
///
/// For `Cosine` indexes this is the normalized vector as used internally.
/// Raises `InvalidId` for unknown or deleted ids.
pub fn Hnsw::get_vector(
  self : Hnsw,
  id : Int32,
) -> FixedArray[Float] raise HnswError {
  if !self.contains(id) {
    raise InvalidId(id)
  }
  let out : FixedArray[Float] = FixedArray::make(self.dim, 0.0)
  FixedArray::unsafe_blit(out, 0, self.vectors, id * self.dim, self.dim)
  out
}

///|
/// Diagnostic snapshot of the index graph.
pub struct HnswStats {
  /// Total insertions, including deleted tombstones.
  count : Int32
  /// Non-deleted (searchable) vectors.
  alive : Int32
  /// Highest layer in use; -1 for an empty index.
  max_level : Int32
  /// Number of nodes participating on each layer (index 0..=max_level).
  /// Deleted nodes still participate: their edges keep the graph connected.
  nodes_per_level : Array[Int32]
  /// Average number of links per participating node on each layer.
  avg_degree_per_level : Array[Double]
}

///|
/// Collects diagnostic statistics about the graph: how many nodes live on
/// each layer and how densely they are connected. Handy when tuning `m` and
/// `ef_construction`.
pub fn Hnsw::stats(self : Hnsw) -> HnswStats {
  let levels = if self.max_level < 0 { 0 } else { self.max_level + 1 }
  let nodes : Array[Int32] = Array::make(levels, 0)
  let degree_sum : Array[Int32] = Array::make(levels, 0)
  for node in self.links {
    let top = node.length() - 1
    let mut l = 0
    while l <= top {
      nodes[l] += 1
      degree_sum[l] += node[l].length()
      l += 1
    }
  }
  let avg : Array[Double] = []
  for l in 0..<levels {
    avg.push(
      if nodes[l] == 0 {
        0.0
      } else {
        degree_sum[l].to_double() / nodes[l].to_double()
      },
    )
  }
  {
    count: self.count,
    alive: self.alive_count,
    max_level: self.max_level,
    nodes_per_level: nodes,
    avg_degree_per_level: avg,
  }
}

///|
/// Result of compacting an index.
pub struct CompactResult {
  /// The rebuilt index, containing only non-deleted vectors.
  index : Hnsw
  /// `mapping[old_id]` is the vector's new id in `index`, or -1 when the
  /// vector was deleted.
  mapping : FixedArray[Int32]
}

///|
/// Rebuilds the index without tombstones, reclaiming the space of deleted
/// vectors.
///
/// The new index uses the same parameters and seed; surviving vectors keep
/// their relative insertion order and receive consecutive new ids described
/// by `mapping`. The original index is left untouched.
pub fn Hnsw::compact(self : Hnsw) -> CompactResult raise HnswError {
  let fresh = Hnsw::new(self.dim, self.distance, params=self.params)
  let mapping : FixedArray[Int32] = FixedArray::make(self.count, -1)
  for id in 0..<self.count {
    if self.alive[id] {
      let v : FixedArray[Float] = FixedArray::make(self.dim, 0.0)
      FixedArray::unsafe_blit(v, 0, self.vectors, id * self.dim, self.dim)
      mapping[id] = fresh.add_labeled(v, self.labels[id])
    }
  }
  { index: fresh, mapping, }
}

///|
/// Deletes the vector with the given id using tombstone semantics.
///
/// The node stays in the graph (preserving connectivity and the id space)
/// but no longer appears in search results, and new insertions never link
/// to it. Deleting an unknown or already-deleted id raises `InvalidId`.
pub fn Hnsw::delete(self : Hnsw, id : Int32) -> Unit raise HnswError {
  if id < 0 || id >= self.count || !self.alive[id] {
    raise InvalidId(id)
  }
  self.alive[id] = false
  self.alive_count -= 1
}

///|
/// Grows the alive-marker array to cover at least `needed` node ids.
fn Hnsw::ensure_alive(self : Hnsw, needed : Int32) -> Unit {
  if self.alive.length() >= needed {
    return
  }
  let mut cap = if self.alive.length() == 0 {
    64
  } else {
    self.alive.length() * 2
  }
  while cap < needed {
    cap = cap * 2
  }
  let grown : FixedArray[Bool] = FixedArray::make(cap, false)
  FixedArray::unsafe_blit(grown, 0, self.alive, 0, self.alive.length())
  self.alive = grown
}

///|
/// Link budget of existing nodes on a layer: layer 0 keeps twice as many
/// links as upper layers.
fn Hnsw::max_links(self : Hnsw, layer : Int32) -> Int32 {
  if layer == 0 {
    self.params.m * 2
  } else {
    self.params.m
  }
}

///|
/// Draws the top layer of a new node from a geometric distribution.
fn Hnsw::random_level(self : Hnsw) -> Int32 {
  let (state, out) = splitmix_next(self.rng_state)
  self.rng_state = state
  let mut u = (out >> 11).to_double() * (1.0 / 9007199254740992.0)
  if u <= 0.0 {
    u = 1.0e-300
  }
  let level = (-@math.ln(u) * self.level_mult).to_int()
  if level < 0 {
    0
  } else {
    level
  }
}

///|
/// Grows the flattened vector storage to hold at least `needed` vectors.
fn Hnsw::ensure_vector_capacity(self : Hnsw, needed : Int32) -> Unit {
  if self.capacity >= needed {
    return
  }
  let mut cap = if self.capacity == 0 { 64 } else { self.capacity * 2 }
  while cap < needed {
    cap = cap * 2
  }
  let grown : FixedArray[Float] = FixedArray::make(cap * self.dim, 0.0)
  if self.count > 0 {
    FixedArray::unsafe_blit(grown, 0, self.vectors, 0, self.count * self.dim)
  }
  self.vectors = grown
  self.capacity = cap
}

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
/// Distance between a query slice and a stored vector.
fn Hnsw::dist_query(self : Hnsw, q : FixedArray[Float], id : Int32) -> Double {
  distance_between(self.distance, q, 0, self.vectors, id * self.dim, self.dim)
}

///|
/// Distance between two stored vectors.
fn Hnsw::dist_ids(self : Hnsw, a : Int32, b : Int32) -> Double {
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
/// from the results, unless `include_dead` is set — index construction uses
/// that mode so new nodes stay linked into the graph even when their
/// surroundings are deleted.
///
/// When `cutoff` is finite, expansion stops once every remaining candidate
/// is farther than the cutoff (radius search); entry points are always
/// expanded once so that neighbors inside the radius are still discovered
/// when every entry point itself lies outside it.
fn Hnsw::search_layer(
  self : Hnsw,
  q : FixedArray[Float],
  eps : Array[(Double, Int32)],
  ef : Int32,
  layer : Int32,
  include_dead~ : Bool = false,
  cutoff~ : Double = 1.0e308,
) -> Array[(Double, Int32)] {
  let stamp = self.next_stamp()
  let visited = self.visited
  let candidates = Heap::new(true)
  let results = Heap::new(false)
  let seeds : Array[Int32] = []
  for (d, e) in eps {
    if e >= 0 && e < self.count && visited[e] != stamp {
      visited[e] = stamp
      candidates.push(d, e)
      if d <= cutoff && (include_dead || self.alive[e]) {
        results.push(d, e)
      }
      seeds.push(e)
    }
  }
  // Expand the entry points once unconditionally.
  for e in seeds {
    for nb in self.links[e][layer] {
      if visited[nb] != stamp {
        visited[nb] = stamp
        let d = self.dist_query(q, nb)
        candidates.push(d, nb)
        if d <= cutoff && (include_dead || self.alive[nb]) {
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
    for e in self.links[c.1][layer] {
      if visited[e] != stamp {
        visited[e] = stamp
        let d = self.dist_query(q, e)
        if d <= cutoff && (results.size() < ef || d < results.top().0) {
          candidates.push(d, e)
          if include_dead || self.alive[e] {
            results.push(d, e)
            if results.size() > ef {
              let _ = results.pop()
            }
          }
        }
      }
    }
  }
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
/// Inserts a vector and returns its id (0-based insertion order).
///
/// The vector is copied into the index; later mutation of the input array
/// does not affect the index. The id doubles as the vector's label.
pub fn Hnsw::add(
  self : Hnsw,
  vector : FixedArray[Float],
) -> Int32 raise HnswError {
  self.insert(vector, self.count.to_uint64())
}

///|
/// Inserts a vector tagged with a user-chosen label (any `UInt64` key, e.g.
/// a document or chunk id) and returns its internal id.
///
/// Labels do not need to be unique; read them back with `label_of(id)`.
/// The vector is copied into the index.
pub fn Hnsw::add_labeled(
  self : Hnsw,
  vector : FixedArray[Float],
  label : UInt64,
) -> Int32 raise HnswError {
  self.insert(vector, label)
}

///|
fn Hnsw::insert(
  self : Hnsw,
  vector : FixedArray[Float],
  label : UInt64,
) -> Int32 raise HnswError {
  if vector.length() != self.dim {
    raise DimMismatch(self.dim, vector.length())
  }
  let id = self.count
  self.ensure_vector_capacity(id + 1)
  let off = id * self.dim
  let _ = copy_normalized(
    vector,
    0,
    self.vectors,
    off,
    self.dim,
    self.normalize,
  )
  let level = self.random_level()
  let node_links : Array[Array[Int32]] = []
  for _ in 0..<=level {
    node_links.push([])
  }
  self.links.push(node_links)
  self.ensure_visited(id + 1)
  self.ensure_alive(id + 1)
  self.labels.push(label)
  self.alive[id] = true
  self.alive_count += 1
  self.count = id + 1
  if self.entry < 0 {
    self.entry = id
    self.max_level = level
    return id
  }
  // The stored (possibly normalized) vector drives all distance computations.
  let q : FixedArray[Float] = FixedArray::make(self.dim, 0.0)
  FixedArray::unsafe_blit(q, 0, self.vectors, off, self.dim)
  // Greedy descent through layers above the node's level.
  let mut ep = self.entry
  let mut ep_dist = self.dist_query(q, ep)
  let mut cur = self.max_level
  while cur > level {
    let r = self.greedy_descent(q, ep, ep_dist, cur)
    ep = r.0
    ep_dist = r.1
    cur -= 1
  }
  // Connect the node on layers min(level, max_level) .. 0.
  let top = if level > self.max_level { self.max_level } else { level }
  let mut eps : Array[(Double, Int32)] = [(ep_dist, ep)]
  let mut layer = top
  while layer >= 0 {
    let cands = self.search_layer(
      q,
      eps,
      self.params.ef_construction,
      layer,
      include_dead=true,
    )
    let selected = self.select_heuristic(cands, self.params.m)
    for nb in selected {
      self.links[id][layer].push(nb)
      let other = self.links[nb]
      other[layer].push(id)
      let budget = self.max_links(layer)
      if other[layer].length() > budget {
        // Re-select neighbors of `nb` among its current links plus the new
        // node, keeping the most useful ones.
        let pool : Array[(Double, Int32)] = []
        for e in other[layer] {
          pool.push((self.dist_ids(nb, e), e))
        }
        let sorted = sort_pairs_ascending(pool)
        other[layer] = self.select_heuristic(sorted, budget)
      }
    }
    eps = cands
    layer -= 1
  }
  if level > self.max_level {
    self.entry = id
    self.max_level = level
  }
  id
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
  if self.count == 0 || self.alive_count == 0 || k <= 0 {
    return []
  }
  let q : FixedArray[Float] = FixedArray::make(self.dim, 0.0)
  let _ = copy_normalized(query, 0, q, 0, self.dim, self.normalize)
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
/// Approximate radius search: returns up to `max_results` non-deleted
/// neighbors with distance <= `radius`, sorted by distance ascending.
///
/// Like top-k search this is graph-approximate: expansion stops once every
/// remaining candidate is farther than `radius`, so a neighbor just inside
/// the radius can be missed when it is only reachable through farther nodes.
/// Entry points are always expanded, which keeps exact-match queries
/// (`radius = 0`) working.
pub fn Hnsw::search_radius(
  self : Hnsw,
  query : FixedArray[Float],
  radius : Double,
  max_results~ : Int32 = 64,
) -> Array[Neighbor] raise HnswError {
  if query.length() != self.dim {
    raise DimMismatch(self.dim, query.length())
  }
  if max_results <= 0 {
    return []
  }
  if self.count == 0 || self.alive_count == 0 {
    return []
  }
  if radius < 0.0 {
    raise InvalidParams("radius must be >= 0")
  }
  let q : FixedArray[Float] = FixedArray::make(self.dim, 0.0)
  let _ = copy_normalized(query, 0, q, 0, self.dim, self.normalize)
  let ep = self.descend_entry(q)
  let cands = self.search_layer(q, [ep], max_results, 0, cutoff=radius)
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
