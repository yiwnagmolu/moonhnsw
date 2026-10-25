///|
/// Binary heap over (distance, id) pairs. With `is_min = true` the smallest
/// distance sits on top (candidate queue); otherwise the largest (bounded
/// result set).
priv struct Heap {
  items : Array[(Double, Int32)]
  is_min : Bool
}

///|
fn Heap::size(self : Heap) -> Int32 {
  self.items.length()
}

///|
/// Returns the top pair without removing it. The heap must be non-empty.
fn Heap::top(self : Heap) -> (Double, Int32) {
  self.items[0]
}

///|
/// True when `a` should be popped before `b`.
fn Heap::before(self : Heap, a : (Double, Int32), b : (Double, Int32)) -> Bool {
  if self.is_min {
    a.0 < b.0
  } else {
    a.0 > b.0
  }
}

///|
fn Heap::push(self : Heap, dist : Double, id : Int32) -> Unit {
  let items = self.items
  items.push((dist, id))
  let mut i = items.length() - 1
  while i > 0 {
    let parent = (i - 1) / 2
    if self.before(items[i], items[parent]) {
      let t = items[i]
      items[i] = items[parent]
      items[parent] = t
      i = parent
    } else {
      break
    }
  }
}

///|
/// Removes and returns the top pair. The heap must be non-empty.
fn Heap::pop(self : Heap) -> (Double, Int32) {
  let items = self.items
  let top = items[0]
  let last = items.pop().unwrap()
  let n = items.length()
  if n > 0 {
    items[0] = last
    let mut i = 0
    let mut swapping = true
    while swapping {
      let l = i * 2 + 1
      let r = i * 2 + 2
      let mut best = i
      if l < n && self.before(items[l], items[best]) {
        best = l
      }
      if r < n && self.before(items[r], items[best]) {
        best = r
      }
      if best == i {
        swapping = false
      } else {
        let t = items[i]
        items[i] = items[best]
        items[best] = t
        i = best
      }
    }
  }
  top
}

///|
/// Drains the heap and returns all pairs sorted by distance, ascending.
fn Heap::to_sorted_ascending(self : Heap) -> Array[(Double, Int32)] {
  let drained : Array[(Double, Int32)] = []
  while self.size() > 0 {
    drained.push(self.pop())
  }
  let out : Array[(Double, Int32)] = []
  let mut i = drained.length() - 1
  while i >= 0 {
    out.push(drained[i])
    i -= 1
  }
  out
}
