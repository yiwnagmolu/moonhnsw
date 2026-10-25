///|
/// IEEE CRC-32 (polynomial 0xEDB88320) over the first `len` bytes of `data`.
/// Recomputes the 256-entry table per call: negligible next to scanning the
/// payload, and keeps the crate stateless.
fn crc32_of(data : FixedArray[Byte], len : Int32) -> UInt32 {
  let mut crc : UInt32 = 0xFFFFFFFFU
  let poly : UInt32 = 0xEDB88320U
  let mut i = 0
  while i < len {
    crc = crc ^ (data.unsafe_get(i).to_int() & 0xFF).reinterpret_as_uint()
    let mut bit = 0
    while bit < 8 {
      crc = if (crc & 1U) != 0U { (crc >> 1) ^ poly } else { crc >> 1 }
      bit += 1
    }
    i += 1
  }
  crc ^ 0xFFFFFFFFU
}
