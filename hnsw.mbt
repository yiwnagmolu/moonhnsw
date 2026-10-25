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
  /// 1 / ln(m) 闁?level assignment factor.
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
/// Returns the label stored for `id` 闁?the id itself by default, or the
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
/// Link budget of existing nodes on a layer: layer 0 keeps twice as many
/// links as upper layers.
fn Hnsw::max_links(self : Hnsw, layer : Int32) -> Int32 {
  if layer == 0 {
    self.params.m * 2
  } else {
    self.params.m
  }
}
