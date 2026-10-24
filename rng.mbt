///|
/// Advances the SplitMix64 generator, returning (new_state, output).
fn splitmix_next(state : UInt64) -> (UInt64, UInt64) {
  let state = state + 0x9E3779B97F4A7C15UL
  let mut z = state
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL
  (state, z ^ (z >> 31))
}
