import ctypes
from opt_einsum.paths import ssa_to_linear

# Fetch the absolute path of the netzwerk.
def fetch_abs_path(rel_path):
  import os
  netzwerk_path = os.path.dirname(os.path.abspath(__file__))
  return os.path.join(netzwerk_path, rel_path)

class Sequence(ctypes.Structure):
  _fields_ = [
    ("i", ctypes.c_int),
    ("j", ctypes.c_int)
  ]

class WrappedSequence(ctypes.Structure):
  _fields_ = [
    ("size", ctypes.c_int),
    ("result", ctypes.POINTER(Sequence))
  ]

def _make_array(arr, data_type, limit=None):
# Convert `arr to C. `limit` specifies how many elements to take from `arr`. For all, let it be default.
  if limit is None:
    limit = len(arr)
  assert (limit is not None) or (0 <= limit and limit < len(arr))
  c_arr = (data_type * min(limit, len(arr)))()
  for i in range(min(limit, len(arr))):
    c_arr[i] = arr[i]
  return c_arr

def _make_array_slice(arr, data_type, field=None):
# Convert `arr to C. `field` specifies which element to take.
  c_arr = (data_type * len(arr))()
  for i in range(len(arr)):
    c_arr[i] = arr[i] if field is None else arr[i][field]
  return c_arr

def _make_dict(d, key_type, value_type):
# Convert a dictionary to `C`. Make sure that keys are in increasing order when converting.
  return _make_array(list(sorted(d.keys())), key_type), _make_array(list([d[key] for key in sorted(d.keys())]), value_type)

def flush(n, edges, open_legs, filename):
  with open(fetch_abs_path(f'{filename}.in'), 'w') as f:
    # Flush `n`, `m`, and `o`.
    f.write(f'{n} {len(edges)} {len(open_legs)}\n')

    # Flush the simple edges.
    f.write('\n'.join([' '.join([str(elem) for elem in edge]) for edge in edges]))

    # Flush the open legs.
    for v in open_legs:
      f.write(f'\n{v} {open_legs[v]}')

def _transform(graph, output, size_dict):
  # Collect the edges.
  d = {}
  vertexIndex = 0
  for elem in graph:
    for edge in elem:
      if edge not in d:
        d[edge] = []
      d[edge].append(vertexIndex)
    vertexIndex += 1

  # Freeze an edge.
  def freeze(u, v):
    if u < v:
      return (u, v)
    return (v, u)

  # Build the actual edges.
  edges = []
  edge_map = {}
  open_legs = {}
  for edge in d:
    # Fetch the dimension.
    legDimension = size_dict[edge]

    # Is it an open leg?
    if edge in output:
      # An open leg should be contained in only one tensor.
      assert len(d[edge]) == 1
      
      # Update the dimension of the *unique* corresponding open leg of this tensor.
      # Note: we multiply the dimensions of the open legs.
      vertex = d[edge][0]
      if vertex not in open_legs:
        open_legs[vertex] = 1.0
      open_legs[vertex] *= legDimension
      continue

    # Simple edge.
    assert len(d[edge]) == 2

    # Check whether this edge already occurs.
    frozen_edge = freeze(d[edge][0], d[edge][1])

    # No? Then register it.
    if frozen_edge not in edge_map:
      edges.append([frozen_edge[0], frozen_edge[1], 1])
      edge_map[frozen_edge] = len(edges) - 1

    # In case we have multiple legs between two tensors, we multiply their dimensions.
    # TODO: Is this actually optimal? Maybe we want to treat each leg separately.
    edges[edge_map[frozen_edge]][2] *= legDimension

  # Build the maximum spanning tree.
  def build_tree(n, m, edges):
    from .spanning_tree import max_st
    _, tree_edges = max_st(n, m, edges)
    return tree_edges

  # Fetch the tree edges.
  tree_edges = build_tree(len(graph), len(edges), edges)

  # Flush.
  flush(len(graph), edges, open_legs, 'graph')
  flush(len(graph), tree_edges, open_legs, 'tree')

  # Convert to C.
  # `limit` specifies how many elements from each `xs[i]` to take.
  def convert_to_c(xs, limit=None):
    c_xs = ((ctypes.POINTER(ctypes.c_int)) * len(xs))()
    for i in range(len(xs)):
      c_xs[i] = _make_array(xs[i], ctypes.c_int, limit=limit)
    return c_xs

  # Convert the edges.
  c_edges = convert_to_c(edges, limit=2)
  c_tree_edges = convert_to_c(tree_edges, limit=2)

  # Convert the costs.
  c_edge_costs = _make_array_slice(edges, ctypes.c_double, field=2)
  c_tree_edge_costs = _make_array_slice(tree_edges, ctypes.c_double, field=2)

  # Convert the open legs.
  # First set an artificial open leg for each tensor which does not have any open leg.
  for vertex in range(len(graph)):
    if vertex not in open_legs:
      open_legs[vertex] = 1.0

  # Convert the update dict to C.
  _, c_open_costs = _make_dict(open_legs, ctypes.c_int, ctypes.c_double)

  # And return.
  return len(graph), len(edges), c_edges, c_tree_edges, c_edge_costs, c_tree_edge_costs, c_open_costs

# Register optimizers.

import platform

relpaths = {
  'Linux' : 'libnetzwerk.so',
  'Darwin' : 'libnetzwerk.dylib',
  'Windows' : 'libnetzwerk.dll'
}

libname = fetch_abs_path(f'build/{relpaths[platform.system()]}')
netzwerk = ctypes.CDLL(libname)
netzwerk.tensor_ikkbz.restype = WrappedSequence
netzwerk.lindp.restype = WrappedSequence
netzwerk.greedy.restype = WrappedSequence

# Parallel implementations.
netzwerk.tensor_ikkbz_parallel.restype = WrappedSequence
netzwerk.lindp_parallel.restype = WrappedSequence

def _unwrap_result(wrapped, use_ssa):
  result = []
  for i in range(wrapped.size):
    row = wrapped.result[i]
    result.append((row.i, row.j))
  
  # Should we simply use SSA? Then directly return.
  if use_ssa:
    return result

  # Otherwise, convert.
  return ssa_to_linear(result)

fn = {
  # Sequential impls.
  'tensor-ikkbz' : netzwerk.tensor_ikkbz,
  'lindp' : netzwerk.lindp,
  'goo' : netzwerk.greedy,

  # Parallel impls.
  'tensor-ikkbz-parallel' : netzwerk.tensor_ikkbz_parallel,
  'lindp-parallel' : netzwerk.lindp_parallel,
}

def contraction_order(graph, output, size_dict, algo, use_ssa=False):
  input = _transform(graph, output, size_dict)
  wrapper = fn[algo](*input)
  return _unwrap_result(wrapper, use_ssa)