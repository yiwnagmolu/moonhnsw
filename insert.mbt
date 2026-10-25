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
