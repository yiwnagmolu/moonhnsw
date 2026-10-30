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
/// non-deleted vector that already carries the same label; the usual
/// document-update flow: re-embed the doc, then upsert.
///
/// Returns the new id; the replaced ids stay reclaimable via `compact`.
pub fn Hnsw::upsert_labeled(
  self : Hnsw,
  vector : FixedArray[Float],
  label : UInt64,
) -> Int32 raise HnswError {
  if vector.length() != self.dim {
    raise DimMismatch(self.dim, vector.length())
  }
  validate_finite_vector(vector)
  for old in self.find_by_label(label) {
    let _ = self.delete(old)
  }
  self.add_labeled(vector, label)
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
    self.f_scratch.length() != self.dim {
    raise CorruptData("scratch or bounds size mismatch")
  }
  if self.storage is SQ8 {
    for j in 0..<self.dim {
      let lo = self.q_mins[j].to_double()
      let hi = self.q_maxs[j].to_double()
      let scale = self.q_scales[j].to_double()
      if !(lo > -@double.infinity &&
        lo < @double.infinity &&
        hi > -@double.infinity &&
        hi < @double.infinity &&
        scale > -@double.infinity &&
        scale < @double.infinity) {
        raise CorruptData("non-finite SQ8 bounds or scale")
      }
    }
    if self.q_frozen != (self.count >= FREEZE_AT) {
      raise CorruptData("SQ8 frozen state inconsistent with vector count")
    }
    if self.q_frozen {
      for j in 0..<self.dim {
        let lo = self.q_mins[j].to_double()
        let hi = self.q_maxs[j].to_double()
        let scale = self.q_scales[j].to_double()
        if !(lo > -1.0e308 &&
          lo < 1.0e308 &&
          hi > -1.0e308 &&
          hi < 1.0e308 &&
          lo <= hi &&
          scale > 0.0 &&
          scale < 1.0e308) {
          raise CorruptData("invalid SQ8 quantization bounds or scale")
        }
      }
    }
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
/// but no longer appears in search results. New insertions may still link to
/// it to preserve connectivity. Deleting an unknown or already-deleted id
/// raises `InvalidId`.
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
