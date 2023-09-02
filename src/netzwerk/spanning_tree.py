def read_file(filename):
  f = open(filename, 'r')
  n, m = [int(x) for x in next(f).split()]
  edges = []
  for line in f.readlines():
    u, v, c = [int(x) for x in line.split()]
    edges.append((u, v, c))
  f.close()
  return n, m, edges

class UnionFind:
  def __init__(self, n):
    self.n = n
    self.boss = [0] * n
    self.size = [1] * n
    for i in range(n):
      self.boss[i] = i
      self.size[i] = 1
  
  def find(self, u):
    r = u
    while r != self.boss[r]:
      r = self.boss[r]
    while u != self.boss[u]:
      tmp = self.boss[u]
      self.boss[u] = r
      u = tmp
    return r

  def unify(self, u, v):
    def link(x, y):
      self.boss[y] = x
      self.size[x] += self.size[y]

    ru = self.find(u)
    rv = self.find(v)
    assert ru != rv
    if self.size[ru] > self.size[rv]:
      link(ru, rv)
    else:
      link(rv, ru)
    pass

def flush(filename, n, tree, st_type):
  import os

  with open(f'{(os.path.dirname(filename) + "/") if "/" in filename else ""}{st_type}-{os.path.basename(filename)}', 'w') as f:
    f.write(f'{n} {n - 1}\n')
    edges = set()
    for i in range(n):
      for elem in tree[i]:
        curr_edge = (min(i, elem), max(i, elem))
        if curr_edge in edges:
          continue
        f.write(f'{curr_edge[0]} {curr_edge[1]} {2}\n')
        edges.add(curr_edge)

    f.close()

def min_degree_spanning_tree(n, m, edges):
  assert len(edges) == m
  for i in range(len(edges)):
    assert max(edges[i][0], edges[i][1]) < n

  def is_tree(config):
    uf = UnionFind(n)

    def add_edge(edge):
      u, v, c = edge
      if uf.find(u) == uf.find(v):
        return
      uf.unify(u, v)

    for i in range(m):
      if (config >> i) & 1:
        add_edge(edges[i])
    for i in range(1, n):
      if uf.find(0) != uf.find(i):
        return False
    return True

  def build_tree_deg(config, best_so_far = 0):
    tree_deg = [0] * n
    def add_tree_edge(edge):
      u, v, c = edge
      tree_deg[u] += 1
      tree_deg[v] += 1

      if best_so_far and tree_deg[u] > best_so_far:
        return False
      if best_so_far and tree_deg[v] > best_so_far:
        return False
      return True

    for i in range(m):
      if (config >> i) & 1:
        if not add_tree_edge(edges[i]):
          return None
    return max(tree_deg)
  
  def build_tree(config, best_so_far = 0):
    tree = []
    for i in range(n):
      tree.append([])
    def add_tree_edge(edge):
      u, v, c = edge
      tree[u].append(v)
      tree[v].append(u)
      if best_so_far and len(tree[u]) > best_so_far:
        return False
      if best_so_far and len(tree[v]) > best_so_far:
        return False
      return True

    for i in range(m):
      if (config >> i) & 1:
        if not add_tree_edge(edges[i]):
          return None
    return tree

  # def compute_max_degree(tree):
  #   max_deg = -1
  #   for i in range(n):
  #     max_deg = max(max_deg, len(tree[i]))
  #   assert max_deg != -1
  #   return max_deg

  def k_subsets(K, N):
    mask = (1 << K) - 1
    while mask < (1 << N):
      if not mask:
        break
      yield mask

      # determine next mask with Gosper's hack
      a = mask & -mask                # determine rightmost 1 bit
      b = mask + a                    # determine carry bit
      mask = (((mask ^ b) >> 2) // a) | b # produce block of ones that begins at the least-significant bit

  best_ret, best_config = n, None
  for config in k_subsets(n - 1, m):
    assert bin(config).count("1") == n - 1
    if not is_tree(config):
      continue

    max_degree = build_tree_deg(config, best_ret)
    if max_degree is None:
      continue

    if max_degree < best_ret:
      best_ret = max_degree
      best_config = config
      
    # print(f'-> best_ret={best_ret}')
    if best_ret == 2:
      break

  assert best_config is not None
  print(f'best_ret={best_ret}')
  best_tree = build_tree(best_config)
  return best_tree, []

def general_st(n, m, edges):
  uf = UnionFind(n)
  tree_edges = []
  for i in range(len(edges)):
    u, v, c = edges[i]
    if uf.find(u) == uf.find(v):
      continue

    tree_edges.append(edges[i])
    uf.unify(u, v)

  tree = []
  for i in range(n):
    tree.append([])
  def add_tree_edge(edge):
    u, v, c = edge
    tree[u].append(v)
    tree[v].append(u)
    return True

  for edge in tree_edges:
    add_tree_edge(edge)
  return tree, tree_edges

def min_st(n, m, edges):
  return general_st(n, m, sorted(edges, key = lambda x: x[2]))

def max_st(n, m, edges):
  return general_st(n, m, sorted(edges, key = lambda x: -x[2]))

def main(filename, st_type):
  n, m, edges = read_file(filename)

  tree = None
  if st_type == 'mst':
    tree, tree_edges = mst(n, m, edges)
  elif st_type == 'mdst':
    tree, tree_edges = min_degree_spanning_tree(n, m, edges)
  else:
    assert 0
  assert tree is not None

  flush(filename, n, tree, st_type)
  pass

if __name__ == '__main__':
  import sys
  assert len(sys.argv) == 3
  filename = sys.argv[1]
  st_type = sys.argv[2]
  main(filename, st_type)