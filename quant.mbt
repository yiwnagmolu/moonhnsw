///|
/// Vector storage precision.
///
/// - `Float32`: full-precision storage (default).
/// - `SQ8`: per-dimension scalar quantization to one byte. Bounds are
///   learned from the first 256 inserted vectors, after which the store is
///   frozen and later values clamp into range. Memory drops ~4x; recall
///   drops slightly.
pub(all) enum Storage {
  Float32
  SQ8
} derive(Eq)

///|
pub extend Storage with Eq::{not_equal, equal}

///|
/// Number of vectors observed before the quantization bounds freeze.
const FREEZE_AT : Int32 = 256

///|
/// Number of vectors whose storage is quantized right now. Pre-freeze the
/// store still holds full floats; post-freeze it holds one byte per
/// dimension.
fn Hnsw::q_active(self : Hnsw) -> Bool {
  self.storage is SQ8 && self.q_frozen
}

///|
/// Grows the quantized store to hold at least `needed` vectors.
fn Hnsw::ensure_quant_capacity(self : Hnsw, needed : Int32) -> Unit {
  if self.q_capacity >= needed {
    return
  }
  let mut cap = if self.q_capacity == 0 { 64 } else { self.q_capacity * 2 }
  while cap < needed {
    cap = cap * 2
  }
  let grown : FixedArray[Byte] = FixedArray::make(cap * self.dim, 0)
  if self.q_capacity > 0 && self.q_frozen {
    FixedArray::unsafe_blit(
      grown,
      0,
      self.quants,
      0,
      self.q_capacity * self.dim,
    )
  }
  self.quants = grown
  self.q_capacity = cap
}

///|
/// Updates the running per-dimension bounds with one vector.
fn Hnsw::observe_bounds(
  self : Hnsw,
  src : FixedArray[Float],
  src_off : Int32,
) -> Unit {
  let mut j = 0
  while j < self.dim {
    let v = src.unsafe_get(src_off + j)
    if v < self.q_mins.unsafe_get(j) {
      self.q_mins[j] = v
    }
    if v > self.q_maxs.unsafe_get(j) {
      self.q_maxs[j] = v
    }
    j += 1
  }
}

///|
/// Derives per-dimension scales from the observed ranges and re-quantizes
/// every stored vector plus the not-yet-counted `current` one, dropping the
/// full-precision copies.
fn Hnsw::freeze_and_quantize_all(
  self : Hnsw,
  current : FixedArray[Float],
) -> Unit {
  let mut j = 0
  while j < self.dim {
    let range = self.q_maxs.unsafe_get(j).to_double() -
      self.q_mins.unsafe_get(j).to_double()
    let s = if range <= 1.0e-12 { 1.0 } else { range / 255.0 }
    self.q_scales[j] = Float::from_double(s)
    j += 1
  }
  self.ensure_quant_capacity(self.count + 1)
  let mut id = 0
  while id < self.count {
    self.quantize_into(self.vectors, id * self.dim, self.quants, id * self.dim)
    id += 1
  }
  self.quantize_into(current, 0, self.quants, self.count * self.dim)
  self.q_frozen = true
  self.vectors = FixedArray::make(0, 0.0)
  self.f_capacity = 0
}

///|
/// Quantizes `src[src_off .. src_off+dim]` into `dst[dst_off ..]` using the
/// frozen bounds. Values outside the observed range clamp into it.
fn Hnsw::quantize_into(
  self : Hnsw,
  src : FixedArray[Float],
  src_off : Int32,
  dst : FixedArray[Byte],
  dst_off : Int32,
) -> Unit {
  let mut j = 0
  while j < self.dim {
    let t = ((
      src.unsafe_get(src_off + j).to_double() -
      self.q_mins.unsafe_get(j).to_double()
    ) /
    self.q_scales.unsafe_get(j).to_double() +
    0.5).to_int()
    let c = if t < 0 { 0 } else if t > 255 { 255 } else { t }
    dst[dst_off + j] = c.to_byte()
    j += 1
  }
}

///|
/// Distance between two quantized vector slices: `a` from any u8 store at
/// `a_off`, `b` from the index store at `b_off`.
fn Hnsw::dist_qq(
  self : Hnsw,
  a : FixedArray[Byte],
  a_off : Int32,
  b_off : Int32,
) -> Double {
  match self.distance {
    L2 => {
      let mut s = 0.0
      let mut j = 0
      while j < self.dim {
        let d = (
            a.unsafe_get(a_off + j).to_double() -
            self.quants.unsafe_get(b_off + j).to_double()
          ) *
          self.q_scales.unsafe_get(j).to_double()
        s += d * d
        j += 1
      }
      s
    }
    InnerProduct | Cosine => {
      let mut s = 0.0
      let mut j = 0
      while j < self.dim {
        let av = a.unsafe_get(a_off + j).to_double() *
          self.q_scales.unsafe_get(j).to_double() +
          self.q_mins.unsafe_get(j).to_double()
        let bv = self.quants.unsafe_get(b_off + j).to_double() *
          self.q_scales.unsafe_get(j).to_double() +
          self.q_mins.unsafe_get(j).to_double()
        s += av * bv
        j += 1
      }
      -s
    }
  }
}

///|
/// Restores the full-precision (dequantized) vector for `id` into `dst`.
fn Hnsw::dequantize_into(
  self : Hnsw,
  id : Int32,
  dst : FixedArray[Float],
) -> Unit {
  let off = id * self.dim
  let mut j = 0
  while j < self.dim {
    dst[j] = Float::from_double(
      self.quants.unsafe_get(off + j).to_double() *
      self.q_scales.unsafe_get(j).to_double() +
      self.q_mins.unsafe_get(j).to_double(),
    )
    j += 1
  }
}
