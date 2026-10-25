///|
/// Growable little-endian byte sink.
priv struct ByteBuf {
  mut buf : FixedArray[Byte]
  mut len : Int32
}

///|
fn ByteBuf::new() -> ByteBuf {
  { buf: FixedArray::make(256, b'\x00'), len: 0, }
}

///|
fn ByteBuf::ensure(self : ByteBuf, extra : Int32) -> Unit {
  if self.len + extra <= self.buf.length() {
    return
  }
  let mut cap = self.buf.length() * 2
  while cap < self.len + extra {
    cap = cap * 2
  }
  let grown : FixedArray[Byte] = FixedArray::make(cap, b'\x00')
  FixedArray::unsafe_blit(grown, 0, self.buf, 0, self.len)
  self.buf = grown
}

///|
fn ByteBuf::put_byte(self : ByteBuf, v : Byte) -> Unit {
  self.ensure(1)
  self.buf[self.len] = v
  self.len += 1
}

///|
fn ByteBuf::put_u32(self : ByteBuf, v : Int32) -> Unit {
  self.ensure(4)
  self.buf[self.len] = (v & 0xFF).to_byte()
  self.buf[self.len + 1] = ((v >> 8) & 0xFF).to_byte()
  self.buf[self.len + 2] = ((v >> 16) & 0xFF).to_byte()
  self.buf[self.len + 3] = ((v >> 24) & 0xFF).to_byte()
  self.len += 4
}

///|
fn ByteBuf::put_u64(self : ByteBuf, v : UInt64) -> Unit {
  self.put_byte(v.to_byte())
  self.put_byte((v >> 8).to_byte())
  self.put_byte((v >> 16).to_byte())
  self.put_byte((v >> 24).to_byte())
  self.put_byte((v >> 32).to_byte())
  self.put_byte((v >> 40).to_byte())
  self.put_byte((v >> 48).to_byte())
  self.put_byte((v >> 56).to_byte())
}

///|
fn ByteBuf::put_f32(self : ByteBuf, v : Float) -> Unit {
  let bits = v.reinterpret_as_uint()
  self.put_byte(bits.to_byte())
  self.put_byte((bits >> 8).to_byte())
  self.put_byte((bits >> 16).to_byte())
  self.put_byte((bits >> 24).to_byte())
}

///|
/// Appends `count` f32 values in one bounds-checked pass.
fn ByteBuf::put_f32_bulk(
  self : ByteBuf,
  src : FixedArray[Float],
  count : Int32,
) -> Unit {
  self.ensure(count * 4)
  let mut i = 0
  while i < count {
    let bits = src.unsafe_get(i).reinterpret_as_uint()
    self.buf[self.len] = bits.to_byte()
    self.buf[self.len + 1] = (bits >> 8).to_byte()
    self.buf[self.len + 2] = (bits >> 16).to_byte()
    self.buf[self.len + 3] = (bits >> 24).to_byte()
    self.len += 4
    i += 1
  }
}

///|
/// Appends `count` bytes in one bounds-checked pass.
fn ByteBuf::put_bytes_bulk(
  self : ByteBuf,
  src : FixedArray[Byte],
  count : Int32,
) -> Unit {
  self.ensure(count)
  let mut i = 0
  while i < count {
    self.buf[self.len + i] = src.unsafe_get(i)
    i += 1
  }
  self.len += count
}

///|
fn ByteBuf::to_fixed(self : ByteBuf) -> FixedArray[Byte] {
  let out : FixedArray[Byte] = FixedArray::make(self.len, b'\x00')
  FixedArray::unsafe_blit(out, 0, self.buf, 0, self.len)
  out
}

///|
/// Little-endian cursor over a byte buffer.
priv struct ByteReader {
  data : FixedArray[Byte]
  mut pos : Int32
}

///|
fn ByteReader::new(data : FixedArray[Byte]) -> ByteReader {
  { data, pos: 0, }
}

///|
fn ByteReader::byte(self : ByteReader) -> Int32 raise HnswError {
  if self.pos >= self.data.length() {
    raise CorruptData("unexpected end of data")
  }
  let b = self.data[self.pos]
  self.pos += 1
  b.to_int() & 0xFF
}

///|
fn ByteReader::u32(self : ByteReader) -> Int32 raise HnswError {
  let b0 = self.byte()
  let b1 = self.byte()
  let b2 = self.byte()
  let b3 = self.byte()
  b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

///|
fn ByteReader::u64(self : ByteReader) -> UInt64 raise HnswError {
  let mut v : UInt64 = 0UL
  let mut i = 0
  while i < 8 {
    v = v | (self.byte().to_uint64() << (i * 8))
    i += 1
  }
  v
}

///|
fn ByteReader::f32(self : ByteReader) -> Float raise HnswError {
  let bits = self.u32()
  Float::reinterpret_from_int(bits)
}

///|
/// Reads `count` f32 values in one bounds-checked pass into `dst`.
fn ByteReader::f32_bulk(
  self : ByteReader,
  dst : FixedArray[Float],
  count : Int32,
) -> Unit raise HnswError {
  if self.pos + count * 4 > self.data.length() {
    raise CorruptData("unexpected end of data")
  }
  let mut i = 0
  while i < count {
    let b0 = self.data.unsafe_get(self.pos).to_int() & 0xFF
    let b1 = self.data.unsafe_get(self.pos + 1).to_int() & 0xFF
    let b2 = self.data.unsafe_get(self.pos + 2).to_int() & 0xFF
    let b3 = self.data.unsafe_get(self.pos + 3).to_int() & 0xFF
    dst[i] = Float::reinterpret_from_int(
      b0 | (b1 << 8) | (b2 << 16) | (b3 << 24),
    )
    self.pos += 4
    i += 1
  }
}

///|
/// Reads `count` bytes in one bounds-checked pass into `dst`.
fn ByteReader::bytes_bulk(
  self : ByteReader,
  dst : FixedArray[Byte],
  count : Int32,
) -> Unit raise HnswError {
  if self.pos + count > self.data.length() {
    raise CorruptData("unexpected end of data")
  }
  let mut i = 0
  while i < count {
    dst[i] = self.data.unsafe_get(self.pos + i)
    i += 1
  }
  self.pos += count
}

///|
fn read_magic(r : ByteReader) -> Unit raise HnswError {
  if r.byte() != b'M'.to_int() ||
    r.byte() != b'H'.to_int() ||
    r.byte() != b'N'.to_int() ||
    r.byte() != b'S'.to_int() ||
    r.byte() != b'W'.to_int() {
    raise CorruptData("bad magic, not a moonhnsw payload")
  }
}

///|
/// Serializes the whole index (parameters, vectors and graph) into a
/// self-contained byte buffer. The payload can be written to a file or sent
/// over the network and restored with `Hnsw::load`; searching a loaded index
/// yields exactly the same results as the original.
pub fn Hnsw::save(self : Hnsw) -> FixedArray[Byte] {
  let b = ByteBuf::new()
  b.put_byte(b'M')
  b.put_byte(b'H')
  b.put_byte(b'N')
  b.put_byte(b'S')
  b.put_byte(b'W')
  b.put_byte(b'\x05')
  b.put_u32(self.dim)
  b.put_byte(
    match self.distance {
      L2 => b'\x00'
      InnerProduct => b'\x01'
      Cosine => b'\x02'
    },
  )
  b.put_u32(self.params.m)
  b.put_u32(self.params.ef_construction)
  b.put_u64(self.params.seed)
  b.put_u32(self.entry)
  b.put_u32(self.max_level)
  b.put_u32(self.count)
  b.put_u64(self.rng_state)
  let mut i = 0
  while i < self.count {
    b.put_byte(if self.alive[i] { b'\x01' } else { b'\x00' })
    i += 1
  }
  for lbl in self.labels {
    b.put_u64(lbl)
  }
  // v5: storage precision. SQ8 stores its bounds and, once frozen, one byte
  // per dimension instead of full floats.
  b.put_byte(
    match self.storage {
      Float32 => b'\x00'
      SQ8 => b'\x01'
    },
  )
  if self.storage is SQ8 {
    b.put_byte(if self.q_frozen { b'\x01' } else { b'\x00' })
    let mut j = 0
    while j < self.dim {
      b.put_f32(self.q_mins.unsafe_get(j))
      j += 1
    }
    j = 0
    while j < self.dim {
      b.put_f32(self.q_maxs.unsafe_get(j))
      j += 1
    }
    j = 0
    while j < self.dim {
      b.put_f32(self.q_scales.unsafe_get(j))
      j += 1
    }
  }
  let total = self.count * self.dim
  if self.q_active() {
    b.put_bytes_bulk(self.quants, total)
  } else {
    b.put_f32_bulk(self.vectors, total)
  }
  for node in self.links {
    b.put_u32(node.length())
    for lst in node {
      b.put_u32(lst.length())
      for e in lst {
        b.put_u32(e)
      }
    }
  }
  // v4 trailer: CRC-32 over everything above, so flipped bits and silent
  // truncation are detected on load instead of surfacing as bad searches.
  b.put_u32(crc32_of(b.buf, b.len).reinterpret_as_int())
  b.to_fixed()
}

///|
/// Restores an index produced by `Hnsw::save`. Raises `CorruptData` when the
/// payload is truncated or inconsistent. The returned index keeps accepting
/// new insertions.
pub fn Hnsw::load(data : FixedArray[Byte]) -> Hnsw raise HnswError {
  let r = ByteReader::new(data)
  read_magic(r)
  let version = r.byte()
  if version != 1 &&
    version != 2 &&
    version != 3 &&
    version != 4 &&
    version != 5 {
    raise CorruptData("unsupported format version \{version}")
  }
  let dim = r.u32()
  if dim < 1 {
    raise CorruptData("dim must be >= 1")
  }
  let metric_code = r.byte()
  let distance = match metric_code {
    0 => L2
    1 => InnerProduct
    2 => Cosine
    _ => raise CorruptData("unknown distance metric code \{metric_code}")
  }
  let m = r.u32()
  let ef_construction = r.u32()
  let seed = r.u64()
  let entry = r.u32()
  let max_level = r.u32()
  let count = r.u32()
  let rng_state = r.u64()
  // Version 2 carries one alive byte per vector (tombstone deletion);
  // version 1 payloads predate deletion and are fully alive.
  let alive : FixedArray[Bool] = FixedArray::make(count, true)
  let mut alive_count = count
  if version >= 2 {
    alive_count = 0
    let mut i = 0
    while i < count {
      let flag = r.byte()
      if flag > 1 {
        raise CorruptData("invalid alive flag")
      }
      alive[i] = flag == 1
      if flag == 1 {
        alive_count += 1
      }
      i += 1
    }
  }
  // Version 3 carries one label per vector; earlier versions used the id.
  let labels : Array[UInt64] = []
  if version >= 3 {
    let mut i = 0
    while i < count {
      labels.push(r.u64())
      i += 1
    }
  } else {
    let mut i = 0
    while i < count {
      labels.push(i.to_uint64())
      i += 1
    }
  }
  if m < 2 || ef_construction < 1 {
    raise CorruptData("invalid graph parameters")
  }
  if count == 0 {
    if entry != -1 || max_level != -1 {
      raise CorruptData("empty index must have no entry point")
    }
  } else {
    if entry < 0 || entry >= count {
      raise CorruptData("entry point out of range")
    }
    if max_level < 0 {
      raise CorruptData("non-empty index must have a level")
    }
  }
  // Version 5 carries the storage precision and, for SQ8, the quantization
  // bounds and byte-per-dimension values.
  let storage : Storage = if version >= 5 {
    match r.byte() {
      0 => Float32
      1 => SQ8
      code => raise CorruptData("unknown storage code \{code}")
    }
  } else {
    Float32
  }
  let mut q_frozen = false
  let q_mins : FixedArray[Float] = FixedArray::make(dim, 3.0e38)
  let q_maxs : FixedArray[Float] = FixedArray::make(dim, -3.0e38)
  let q_scales : FixedArray[Float] = FixedArray::make(dim, 1.0)
  if version >= 5 && storage is SQ8 {
    q_frozen = r.byte() == 1
    let mut j = 0
    while j < dim {
      q_mins[j] = r.f32()
      j += 1
    }
    j = 0
    while j < dim {
      q_maxs[j] = r.f32()
      j += 1
    }
    j = 0
    while j < dim {
      q_scales[j] = r.f32()
      j += 1
    }
  }
  let total = count * dim
  let quantized = storage is SQ8 && q_frozen
  let vectors : FixedArray[Float] = if quantized {
    FixedArray::make(0, 0.0)
  } else {
    FixedArray::make(total, 0.0)
  }
  let quants : FixedArray[Byte] = if quantized {
    FixedArray::make(total, 0)
  } else {
    FixedArray::make(0, 0)
  }
  if quantized {
    r.bytes_bulk(quants, total)
  } else {
    r.f32_bulk(vectors, total)
  }
  let links : Array[Array[Array[Int32]]] = []
  for _ in 0..<count {
    let n_levels = r.u32()
    if n_levels < 1 || n_levels > max_level + 1 {
      raise CorruptData("node level out of range")
    }
    let node : Array[Array[Int32]] = []
    for _ in 0..<n_levels {
      let n_links = r.u32()
      if n_links < 0 || n_links > count {
        raise CorruptData("link list length out of range")
      }
      let lst : Array[Int32] = []
      for _ in 0..<n_links {
        let e = r.u32()
        if e < 0 || e >= count {
          raise CorruptData("link id out of range")
        }
        lst.push(e)
      }
      node.push(lst)
    }
    links.push(node)
  }
  // v4 payloads end with a CRC-32 over every preceding byte.
  if version >= 4 {
    let body_crc = crc32_of(r.data, r.pos)
    let stored = r.u32()
    if stored != body_crc.reinterpret_as_int() {
      raise CorruptData("checksum mismatch")
    }
  }
  let visited_len = if count < 64 { 64 } else { count }
  {
    dim,
    distance,
    params: HnswParams::new(m~, ef_construction~, seed~, storage~),
    normalize: match distance {
      Cosine => true
      _ => false
    },
    level_mult: 1.0 / @math.ln(m.to_double()),
    entry,
    max_level,
    count,
    alive_count,
    alive,
    labels,
    storage,
    vectors,
    f_capacity: if quantized {
      0
    } else {
      count
    },
    quants,
    q_capacity: if quantized {
      count
    } else {
      0
    },
    q_mins,
    q_maxs,
    q_scales,
    q_frozen,
    q_scratch: FixedArray::make(dim, 0),
    q_scratch_source: FixedArray::make(0, 0.0),
    q_scratch_reset: FixedArray::make(0, 0.0),
    f_scratch: FixedArray::make(dim, 0.0),
    links,
    visited: FixedArray::make(visited_len, 0),
    stamp: 0,
    rng_state,
  }
}
