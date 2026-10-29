// The real-data runner uses Node only for dataset loading. On other backends
// this package is an empty executable so the root test/build matrix stays
// backend-neutral; the HNSW library itself has no JavaScript dependency.

///|
fn main {
  let _ = @moonhnsw.Distance::L2
  let _ = @bench.monotonic_clock_start()
}
