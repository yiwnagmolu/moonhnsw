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
} derive(Eq)

///|
pub extend Distance with Eq::{not_equal, equal}

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
  /// Vector storage precision (full floats or 8-bit scalar quantization).
  storage : Storage
}

///|
/// Default parameters: `m = 16`, `ef_construction = 200`, full-precision
/// storage, fixed random seed.
pub fn HnswParams::new(
  m~ : Int32 = 16,
  ef_construction~ : Int32 = 200,
  seed~ : UInt64 = 0x9E3779B97F4A7C15UL,
  storage~ : Storage = Float32,
) -> HnswParams {
  { m, ef_construction, seed, storage, }
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
} derive(Eq, @debug.Debug, ToJson)

///|
pub extend Neighbor with Eq::{not_equal, equal}

///|
pub extend Neighbor with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend Neighbor with ToJson::{to_json}

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
  /// 1 / ln(m) 閳?level assignment factor.
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
  /// Selected storage precision.
  storage : Storage
  /// Flattened full-precision storage: `count * dim` floats. Held until the
  /// SQ8 store freezes, then released.
  mut vectors : FixedArray[Float]
  /// Number of vector slots allocated in `vectors`.
  mut f_capacity : Int32
  /// Flattened quantized storage: `count * dim` bytes, valid once frozen.
  mut quants : FixedArray[Byte]
  /// Number of vector slots allocated in `quants`.
  mut q_capacity : Int32
  /// Per-dimension lower quantization bound (running until frozen).
  q_mins : FixedArray[Float]
  /// Per-dimension upper quantization bound (running until frozen).
  q_maxs : FixedArray[Float]
  /// Per-dimension scale, (max - min) / 255; valid once frozen.
  q_scales : FixedArray[Float]
  /// True once the quantization bounds are frozen and vectors are stored as
  /// one byte per dimension.
  mut q_frozen : Bool
  /// Reusable scratch for quantizing query vectors during search.
  q_scratch : FixedArray[Byte]
  /// The query currently held in `q_scratch`; compared by reference so one
  /// search quantizes its query once instead of per visited neighbor.
  mut q_scratch_source : FixedArray[Float]
  /// Sentinel used to invalidate `q_scratch_source` between operations.
  q_scratch_reset : FixedArray[Float]
  /// Reusable scratch for the normalized copy of an inserted vector.
  f_scratch : FixedArray[Float]
  /// Backing stores of the traversal heaps, drained after every use and
  /// reused so searches do not allocate fresh arrays.
  cand_items : Array[(Double, Int32)]
  result_items : Array[(Double, Int32)]
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
    params: HnswParams::new(
      m~,
      ef_construction=efc,
      seed=params.seed,
      storage=params.storage,
    ),
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
    storage: params.storage,
    vectors: FixedArray::make(0, 0.0),
    f_capacity: 0,
    quants: FixedArray::make(0, 0),
    q_capacity: 0,
    q_mins: FixedArray::make(dim, 3.0e38),
    q_maxs: FixedArray::make(dim, -3.0e38),
    q_scales: FixedArray::make(dim, 1.0),
    q_frozen: false,
    q_scratch: FixedArray::make(dim, 0),
    q_scratch_source: FixedArray::make(0, 0.0),
    q_scratch_reset: FixedArray::make(0, 0.0),
    f_scratch: FixedArray::make(dim, 0.0),
    cand_items: [],
    result_items: [],
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
/// Returns the label stored for `id` 閳?the id itself by default, or the
/// value passed to `add_labeled`. Raises `InvalidId` for unknown or deleted
/// ids.
pub fn Hnsw::label_of(self : Hnsw, id : Int32) -> UInt64 raise HnswError {
  if !self.contains(id) {
    raise InvalidId(id)
  }
  self.labels[id]
}

///|
/// Returns the ids of all non-deleted vectors carrying `label`, ascending.
/// Labels may repeat, so this can return more than one id.
pub fn Hnsw::find_by_label(self : Hnsw, label : UInt64) -> Array[Int32] {
  let out : Array[Int32] = []
  for id in 0..<self.count {
    if self.alive[id] && self.labels[id] == label {
      out.push(id)
    }
  }
  out
}

///|
/// Inserts `vector` tagged with `label`, first tombstone-deleting every
/// non-deleted vector that already carries the same label — the usual
/// document-update flow: re-embed the doc, then upsert.
///
/// Returns the new id; the replaced ids stay reclaimable via `compact`.
pub fn Hnsw::upsert_labeled(
  self : Hnsw,
  vector : FixedArray[Float],
  label : UInt64,
) -> Int32 raise HnswError {
  for old in self.find_by_label(label) {
    let _ = self.delete(old)
  }
  self.add_labeled(vector, label)
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
  if self.q_active() {
    self.dequantize_into(id, out)
  } else {
    FixedArray::unsafe_blit(out, 0, self.vectors, id * self.dim, self.dim)
  }
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
  /// Rough estimate of the heap memory held by the index, in bytes: vector
  /// storage, edge ids, per-list overhead and traversal markers.
  estimated_bytes : Int64
} derive(Eq, @debug.Debug, ToJson)

///|
pub extend HnswStats with Eq::{not_equal, equal}

///|
pub extend HnswStats with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend HnswStats with ToJson::{to_json}

///|
/// Collects diagnostic statistics about the graph: how many nodes live on
/// each layer and how densely they are connected. Handy when tuning `m` and
/// `ef_construction`.
pub fn Hnsw::stats(self : Hnsw) -> HnswStats {
  let levels = if self.max_level < 0 { 0 } else { self.max_level + 1 }
  let nodes : Array[Int32] = Array::make(levels, 0)
  let degree_sum : Array[Int32] = Array::make(levels, 0)
  let mut lists = 0
  let mut edges = 0
  for node in self.links {
    let top = node.length() - 1
    let mut l = 0
    while l <= top {
      nodes[l] += 1
      degree_sum[l] += node[l].length()
      lists += 1
      edges += node[l].length()
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
  let vector_bytes = if self.storage is Float32 {
    self.f_capacity.to_int64() * self.dim.to_int64() * 4L
  } else if self.q_frozen {
    self.q_capacity.to_int64() * self.dim.to_int64() +
    3L * self.dim.to_int64() * 4L
  } else {
    self.f_capacity.to_int64() * self.dim.to_int64() * 4L +
    2L * self.dim.to_int64() * 4L
  }
  let estimated = vector_bytes +
    edges.to_int64() * 4L +
    lists.to_int64() * 24L +
    self.visited.length().to_int64() * 4L +
    self.alive.length().to_int64() +
    self.labels.length().to_int64() * 8L
  {
    count: self.count,
    alive: self.alive_count,
    max_level: self.max_level,
    nodes_per_level: nodes,
    avg_degree_per_level: avg,
    estimated_bytes: estimated,
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
      // get_vector dequantizes under SQ8; the fresh index re-quantizes with
      // bounds learned from the surviving data.
      mapping[id] = fresh.add_labeled(self.get_vector(id), self.labels[id])
    }
  }
  { index: fresh, mapping, }
}

///|
/// Inserts every non-deleted vector of `other` into `self`, keeping their
/// labels, and returns a mapping from `other`'s ids to the new ids in
/// `self` (-1 for deleted vectors).
///
/// Both indexes must share the same dimension and distance metric, but may
/// differ in parameters. `other` is left untouched; no deduplication
/// happens. Merging an index into itself doubles it.
pub fn Hnsw::absorb(
  self : Hnsw,
  other : Hnsw,
) -> FixedArray[Int32] raise HnswError {
  if other.dim != self.dim {
    raise DimMismatch(self.dim, other.dim)
  }
  if other.distance != self.distance {
    raise InvalidParams("distance metric mismatch")
  }
  let n = other.count
  let mapping : FixedArray[Int32] = FixedArray::make(n, -1)
  for id in 0..<n {
    if other.contains(id) {
      mapping[id] = self.add_labeled(other.get_vector(id), other.label_of(id))
    }
  }
  mapping
}

///|
/// Returns an independent deep copy of the index, tombstones and labels
/// included (implemented via serialization).
pub fn Hnsw::copy(self : Hnsw) -> Hnsw raise HnswError {
  Hnsw::load(self.save())
}

///|
/// Returns the ids of all non-deleted vectors, ascending. Handy for
/// pipelines that enumerate the searchable content of a persisted index.
pub fn Hnsw::ids(self : Hnsw) -> Array[Int32] {
  let out : Array[Int32] = []
  for id in 0..<self.count {
    if self.alive[id] {
      out.push(id)
    }
  }
  out
}

///|
/// Releases unused capacity in the vector stores, reallocating them to the
/// exact current size. Purely a memory optimization: ids, labels and search
/// behavior are unchanged, and the stores grow again on the next insert.
pub fn Hnsw::shrink_to_fit(self : Hnsw) -> Unit {
  let total = self.count * self.dim
  if self.storage is Float32 || !self.q_frozen {
    if self.f_capacity > self.count {
      let exact : FixedArray[Float] = FixedArray::make(total, 0.0)
      FixedArray::unsafe_blit(exact, 0, self.vectors, 0, total)
      self.vectors = exact
      self.f_capacity = self.count
    }
  } else if self.q_capacity > self.count {
    let exact : FixedArray[Byte] = FixedArray::make(total, 0)
    FixedArray::unsafe_blit(exact, 0, self.quants, 0, total)
    self.quants = exact
    self.q_capacity = self.count
  }
}

///|
/// Builds an index from a list of vectors in one call. Ids follow the
/// positions in `vectors`, and labels default to the ids.
pub fn Hnsw::from_vectors(
  dim : Int32,
  distance : Distance,
  vectors : Array[FixedArray[Float]],
  params~ : HnswParams = HnswParams::new(),
) -> Hnsw raise HnswError {
  let idx = Hnsw::new(dim, distance, params~)
  for v in vectors {
    let _ = idx.add(v)
  }
  idx
}

///|
/// Same payload as `save`, wrapped as an immutable `Bytes` value for
/// convenient handling alongside other binary data.
pub fn Hnsw::save_bytes(self : Hnsw) -> Bytes {
  Bytes::from_array(self.save())
}

///|
/// Restores an index from a `Bytes` payload produced by `save_bytes`.
/// Raises `CorruptData` when the payload is truncated or inconsistent.
pub fn Hnsw::load_bytes(data : Bytes) -> Hnsw raise HnswError {
  Hnsw::load(data.to_fixedarray())
}

///|
/// Checks the graph invariants: every link id in range and present on the
/// linked layer, node levels within bounds, degree budgets respected, alive
/// counters consistent, labels complete and the entry point holding the top
/// level. Raises `CorruptData` on the first violation.
///
/// Intended for tests and for auditing indexes produced by `load`; it walks
/// the whole graph, so it is O(count + edges).
pub fn Hnsw::verify(self : Hnsw) -> Unit raise HnswError {
  if self.alive_count < 0 || self.alive_count > self.count {
    raise CorruptData("alive count out of range")
  }
  if self.visited.length() < self.count || self.alive.length() < self.count {
    raise CorruptData("marker arrays too small")
  }
  if self.labels.length() != self.count || self.links.length() != self.count {
    raise CorruptData("node count mismatch")
  }
  if self.q_mins.length() != self.dim ||
    self.q_maxs.length() != self.dim ||
    self.q_scales.length() != self.dim ||
    self.q_scratch.length() != self.dim ||
    self.f_scratch.length() != self.dim {
    raise CorruptData("scratch or bounds size mismatch")
  }
  if self.storage is SQ8 && self.q_frozen {
    if self.quants.length() < self.count * self.dim || !self.vectors.is_empty() {
      raise CorruptData("quantized store inconsistent")
    }
  } else if self.vectors.length() < self.count * self.dim {
    raise CorruptData("vector store too small")
  }
  let mut alive = 0
  for id in 0..<self.count {
    let node = self.links[id]
    let top = node.length() - 1
    if top < 0 || top > self.max_level {
      raise CorruptData("node level out of range")
    }
    for l in 0..<=top {
      let lst = node[l]
      if lst.length() > self.max_links(l) {
        raise CorruptData("degree exceeds the link budget")
      }
      for e in lst {
        if e < 0 || e >= self.count {
          raise CorruptData("link id out of range")
        }
        if self.links[e].length() <= l {
          raise CorruptData("link target absent from this layer")
        }
      }
    }
    if self.alive[id] {
      alive += 1
    }
  }
  if alive != self.alive_count {
    raise CorruptData("alive count mismatch")
  }
  if self.count > 0 {
    if self.entry < 0 || self.entry >= self.count {
      raise CorruptData("entry point out of range")
    }
    if self.links[self.entry].length() - 1 != self.max_level {
      raise CorruptData("entry point does not hold the top level")
    }
  } else if self.entry != -1 || self.max_level != -1 {
    raise CorruptData("empty index must have no entry point")
  }
}

///|
/// Exports one graph layer in Graphviz DOT format for visualization and
/// debugging (render with e.g. `dot -Tsvg layers.dot -o layers.svg`).
///
/// Nodes carry their id and user label; deleted nodes are drawn dashed.
/// Edges are directed exactly as stored in the graph.
pub fn Hnsw::export_dot(self : Hnsw, layer : Int32) -> String raise HnswError {
  if layer < 0 || layer > self.max_level {
    raise InvalidParams("layer must be within 0..=\{self.max_level}")
  }
  let b = StringBuilder()
  b.write_string("digraph moonhnsw_layer\{layer} {\n  rankdir=\"LR\";\n")
  for id in 0..<self.count {
    if self.links[id].length() > layer {
      let style = if self.alive[id] { "" } else { ", style=dashed" }
      b.write_string(
        "  n\{id} [label=\"\{id} (label \{self.labels[id]})\"\{style}];\n",
      )
    }
  }
  for id in 0..<self.count {
    if self.links[id].length() > layer {
      for e in self.links[id][layer] {
        b.write_string("  n\{id} -> n\{e};\n")
      }
    }
  }
  b.write_string("}\n")
  b.to_string()
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
/// Grows the full-precision storage to hold at least `needed` vectors.
fn Hnsw::ensure_vector_capacity(self : Hnsw, needed : Int32) -> Unit {
  if self.f_capacity >= needed {
    return
  }
  let mut cap = if self.f_capacity == 0 { 64 } else { self.f_capacity * 2 }
  while cap < needed {
    cap = cap * 2
  }
  let grown : FixedArray[Float] = FixedArray::make(cap * self.dim, 0.0)
  if self.count > 0 && self.f_capacity > 0 {
    FixedArray::unsafe_blit(grown, 0, self.vectors, 0, self.count * self.dim)
  }
  self.vectors = grown
  self.f_capacity = cap
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
/// Distance between a query slice and a stored vector. Under SQ8 the query
/// is quantized into a reused scratch buffer once per search (detected by
/// reference identity) and the distance runs over bytes.
fn Hnsw::dist_query(self : Hnsw, q : FixedArray[Float], id : Int32) -> Double {
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
/// from the results, unless `include_dead` is set 閳?index construction uses
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
  cutoff~ : Double = 1.0e308,
  accept~ : (Int32) -> Bool = fn(_) { true },
) -> Array[(Double, Int32)] {
  let stamp = self.next_stamp()
  let visited = self.visited
  // Reuse the heap backing stores: to_sorted_ascending drains them, so they
  // come back empty and keep their grown capacity.
  let candidates = Heap::{ items: self.cand_items, is_min: true, }
  let results = Heap::{ items: self.result_items, is_min: false, }
  let seeds : Array[Int32] = []
  for (d, e) in eps {
    if e >= 0 && e < self.count && visited[e] != stamp {
      visited[e] = stamp
      candidates.push(d, e)
      if d <= cutoff && (include_dead || self.alive[e]) && accept(e) {
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
        if d <= cutoff && (include_dead || self.alive[nb]) && accept(nb) {
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
          if (include_dead || self.alive[e]) && accept(e) {
            results.push(d, e)
            if results.size() > ef {
              let _ = results.pop()
            }
          }
        }
      }
    }
  }
  // The candidate heap is never drained by the caller; empty it in place so
  // the reused backing store starts clean next time.
  while candidates.size() > 0 {
    let _ = candidates.pop()
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
  self.q_scratch_source = self.q_scratch_reset
  let off = id * self.dim
  // Normalize (when the metric needs it) into scratch, then store at the
  // active precision.
  let _ = copy_normalized(
    vector,
    0,
    self.f_scratch,
    0,
    self.dim,
    self.normalize,
  )
  if self.storage is Float32 {
    self.ensure_vector_capacity(id + 1)
    FixedArray::unsafe_blit(self.vectors, off, self.f_scratch, 0, self.dim)
  } else if !self.q_frozen {
    // Full floats until the quantization bounds freeze at FREEZE_AT.
    self.ensure_vector_capacity(id + 1)
    FixedArray::unsafe_blit(self.vectors, off, self.f_scratch, 0, self.dim)
    self.observe_bounds(self.f_scratch, 0)
    if id + 1 == FREEZE_AT {
      self.freeze_and_quantize_all(self.f_scratch)
    }
  } else {
    self.ensure_quant_capacity(id + 1)
    self.quantize_into(self.f_scratch, 0, self.quants, off)
  }
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
  // Under frozen SQ8 the full-precision copy in f_scratch serves as the
  // query; the byte store only holds the indexed side.
  let q : FixedArray[Float] = if self.q_active() {
    self.f_scratch
  } else {
    let qc : FixedArray[Float] = FixedArray::make(self.dim, 0.0)
    FixedArray::unsafe_blit(qc, 0, self.vectors, off, self.dim)
    qc
  }
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
/// even when more matches exist 閳?raise `ef` to compensate. Filter closures
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
  for q in queries {
    out.push(self.search(q, k, ef~))
  }
  out
}

///|
/// Maximal Marginal Relevance search: retrieves a candidate pool with an
/// `ef` beam search, then greedily selects `k` results that balance
/// relevance to the query against redundancy with the already-selected set.
///
/// `lambda` blends the two objectives: 1.0 keeps only the most relevant
/// candidates (identical to `search`), 0.0 orders purely by relevance too on
/// the first pick and by anti-redundancy afterwards; 0.3-0.7 is the usual
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
  if lambda < 0.0 || lambda > 1.0 {
    raise InvalidParams("lambda must be within [0, 1]")
  }
  // The pool must be wider than k, or near-duplicate candidates crowd out
  // any diversity the greedy pass could achieve.
  let pool = self.search_filtered(query, ef, ef~, filter~)
  if lambda <= 0.0 || pool.length() <= 1 {
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
  let out : Array[Neighbor] = []
  for _ in 0..<k {
    let mut best = -1
    let mut best_score = 0.0
    for i in 0..<n {
      if !taken[i] {
        // Similarity is the negated distance; the redundancy penalty is the
        // largest similarity to anything already selected (none on the
        // first pick).
        let relevance = -pool[i].distance
        let mut max_sim = -1.0e308
        for sel in out {
          let sim = -self.dist_ids(pool[i].id, sel.id)
          if sim > max_sim {
            max_sim = sim
          }
        }
        let penalty = if out.length() == 0 { 0.0 } else { lambda * max_sim }
        let score = (1.0 - lambda) * relevance - penalty
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
  if max_results <= 0 {
    return []
  }
  if self.count == 0 || self.alive_count == 0 {
    return []
  }
  if radius < 0.0 {
    raise InvalidParams("radius must be >= 0")
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
