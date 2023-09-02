import argparse

seed = 123
import random
random.seed(seed)

def random_int(lower, upper):
  return random.randint(lower, upper)

def random_leg_dim(f):
  c = random_int(1, 100)
  if c <= 50:
    return random_int(f, f**2)
  elif c <= 90:
    return random_int(f**2, f**3)
  return random_int(f**3, f**4)

class Generator:
  def __init__(self, tn_type, size, leg_type, factor):
    self.tn_type = tn_type
    self.max_size = size
    self.leg_type = leg_type
    self.factor = factor
    self.cache = {}
    self.reset()

  def makeEdge(self, u, v, leg_dim):
    leg = (min(u, v), max(u, v))
    if leg not in self.legs:
      self.legs[leg] = leg_dim
    else:
      assert 0
      # TODO: later put here min(., .) for powers of two.
      pass

  def set_leg_dim(self, leg_dim, count=1, open=False):
    # Choose random leg dimension.
    assert count >= 1
    
    # Open leg?
    if open:
      return self.factor**count
    
    # Otherwise, choose randomly.
    if leg_dim is None:
      leg_dim = random_leg_dim(self.factor)**count
    return leg_dim

  def get_vertex(self, c):
    if c not in self.vertices:
      self.vertices[c] = self.vertex_index
      self.vertex_index += 1
    return self.vertices[c]

  def open(self, c, count, leg_dim = None):
    # Set the leg dimension.
    leg_dim = self.set_leg_dim(leg_dim, count, open=True)

    # Get the vertex.
    u = self.get_vertex(c)

    # Insert the open leg.
    self.open_legs[u] = leg_dim

  def connect(self, c1, c2, leg_dim = None):
    # Set the leg dimension.
    leg_dim = self.set_leg_dim(leg_dim)

    # Get the vertices. Note: the order matters for a nice output.
    u = self.get_vertex(c2)
    v = self.get_vertex(c1)

    # Make the edge.
    self.makeEdge(u, v, leg_dim)

  def genFTPS(self):
    print(f'Generate FTPS')

    # The expected number of vertices.
    def expected_size(w, h):
      return w * (h + 1)

    def buildFTPS(w, h):
      print(f'[ftps] w={w}, h={h}')

      # Reset.
      self.reset()

      # Compute the expected number of nodes.
      num_nodes = expected_size(w, h)

      for i in range(w):
        # Bind to left on the main spine.
        if i:
          self.connect((i, 0), (i - 1, 0))

        # Add an open leg for each node on the main spine.
        self.open((i, 0), 1)

        # Build the chain.
        for j in range(1, h + 1):
          # Connect with the previous node.
          self.connect((i, j), (i, j - 1))

          # Add an open leg.
          self.open((i, j), 1)

      # Assert correctness of the build process.
      assert len(self.vertices) == num_nodes
      assert len(self.legs) == num_nodes - 1
      assert len(self.open_legs) == num_nodes

    # Build all FTPS with size under `max_size`.
    for w in range(2, self.max_size + 1):
      for h in range(1, self.max_size + 1):
        # Check if we need to build this.
        if expected_size(w, h) > self.max_size:
          continue

        # Build.
        buildFTPS(w, h)

        # And flush.
        self.flush(params = {
          'w' : w,
          'h' : h
        })

    # Flush all.
    self.flush_all()

  def genMERA(self):
    print(f'Generate MERA')

    # The expected number of vertices.
    def expected_size(h):
      num_nodes = 0
      for i in range(h):
        num_nodes += 2**i
      extra_nodes = 0
      for i in range(1, h):
        extra_nodes += 2**i - 1
      return num_nodes + extra_nodes, extra_nodes

    def buildMERA(h):
      print(f'[mera] h={h}')
      # Reset.
      self.reset()
      num_nodes, extra_nodes = expected_size(h)

      for i in range(h):
        for j in range(2**i):
          # Skip the first level.
          if not i:
            continue

          # Connect to parent only if we are on the border.
          if not j or j == (2**i) - 1:
            self.connect((i, j), (i - 1, j // 2))

          # Insert open leg only if we are on the border of the last layer.
          if i == h - 1 and (not j or j == (2**i) - 1):
            self.open((i, j), 1)

      # Insert the intermediate nodes.
      for i in range(1, h):
        for j in range(1, 2**i):
          # Connect to layer above.
          self.connect((h + i, j), (i, j - 1))
          self.connect((h + i, j), (i, j))

          # Connect to layer below, if we are not on the last layer.
          if i != h - 1:
            self.connect((h + i, j), (i + 1, j))
            self.connect((h + i, j), (i + 1, j + 1))
          else:
            # Otherwise insert an open leg.
            self.open((h + i, j), 2)
      
      assert len(self.vertices) == num_nodes
      assert len(self.legs) == (num_nodes - extra_nodes) + 2 * extra_nodes - 1
      assert len(self.open_legs) == 2**(h - 1) + 1

    for h in range(2, self.max_size + 1):
      # Check if we need to build this.
      if expected_size(h)[0] > self.max_size:
        continue
      buildMERA(h)
      self.flush(params = {
        'h' : h
      })

    # Flush all.
    self.flush_all()

  def genTTN(self):
    print(f'Generate TTN')

    # The expected number of vertices.
    def expected_size(h):
      num_nodes = 0
      for i in range(h):
        num_nodes += 2**i
      return num_nodes

    def buildTTN(h):
      print(f'[ttn] h={h}')
      # Reset.
      self.reset()
      num_nodes = expected_size(h)

      for i in range(h):
        for j in range(2**i):
          # Skip the first level.
          if not i:
            continue
          
          print(f'i={i}, j={j}')

          # Connect with the parent.
          self.connect((i, j), (i - 1, j // 2))

          if i == h - 1:
            self.open((i, j), 2)

      assert len(self.vertices) == num_nodes
      assert len(self.legs) == num_nodes - 1
      assert len(self.open_legs) == 2**(h - 1)

    for h in range(2, self.max_size + 1):
      # Check if we need to build this.
      if expected_size(h) > self.max_size:
        continue
      buildTTN(h)
      self.flush(params = {
        'h' : h
      })
    
    # Flush all.
    self.flush_all()
    
  def genMPS(self):
    print(f'Generate MEPS')
    pass

  def genPEPS(self):
    # The expected number of vertices.
    def expected_size(w, h):
      return w * h

    print(f'Generate PEPS')
    def buildPEPS(w, h):
      print(f'[peps] w={w}, h={h}')
      # Reset.
      self.reset()
      num_nodes = expected_size(w, h)

      for i in range(w):
        for j in range(h):
          if i:
            self.connect((i, j), (i - 1, j))
          if j:
            self.connect((i, j), (i, j - 1))

          # Add the open leg.
          self.open((i, j), 1)

      print(f'v={len(self.vertices)}, num_nodes={num_nodes}')
      assert len(self.vertices) == num_nodes
      assert len(self.legs) == (w - 1) * h + (h - 1) * w
      assert len(self.open_legs) == num_nodes

    for w in range(2, self.max_size + 1):
      for h in range(2, self.max_size + 1):
        # Check if we need to build this.
        if expected_size(w, h) > self.max_size:
          continue

        # Build.
        buildPEPS(w, h)

        # Flush.
        self.flush(params = {
          'w' : w,
          'h' : h
        })

    # Flush all.
    self.flush_all()

  def reset(self):
    self.vertices = {}
    self.vertex_index = 0
    self.legs = {}
    self.open_legs = {}
    
  def flush_file(self, params, vertices, legs, open_legs):
    # Ensure the directory exists.
    import os
    os.system(f'mkdir -p data/{self.tn_type}')

    # Build the parameters.
    params_str = '_'.join([f'{elem[0]}-{elem[1]}' for elem in params.items()])

    # Create the file name.
    file_name = f'{len(vertices)}_{len(legs)}_{self.tn_type}_{self.leg_type}_{params_str}'

    # Open the file.
    with open(f'data/{self.tn_type}/{file_name}.in', 'w') as f:
      # Flush the number of vertices and legs.
      f.write(f'{len(vertices)} {len(legs)} {len(open_legs) if self.leg_type == "open" else 0}\n')

      # Flush the closed legs.
      for edge in legs:
        f.write(f'{edge[0]} {edge[1]} {legs[edge]}\n')

      # Flush the open legs, if required.
      if self.leg_type == 'open':
        for leg in open_legs:
          f.write(f'{leg} {open_legs[leg]}\n')

  def flush_all(self):
    print(f'Flushing to disk..')
    for n in self.cache:
      self.flush_file(self.cache[n]['params'],
        self.cache[n]['vertices'],
        self.cache[n]['legs'],
        self.cache[n]['open_legs']
      )

  def flush(self, params):
    assert len(self.vertices) <= self.max_size

    # Do we store only the optimal parameters for special classes of tensor networks?
    if self.tn_type in ['ftps', 'peps']:
      if params['w'] > params['h']:
        return

      n = len(self.vertices)
      if ((n in self.cache) and (params['w'] > self.cache[n]['params']['w'])) or (n not in self.cache):
        self.cache[n] = {
          'params' : params,
          'vertices' : self.vertices,
          'legs' : self.legs,
          'open_legs' : self.open_legs
        }
      return

    # Flush file.
    self.flush_file(params, self.vertices, self.legs, self.open_legs)

def gen(tn_type, size, leg_type, factor):
  # Create the generator.
  gen = Generator(tn_type, size, leg_type, factor)

  # Run the corresponding function.
  getattr(gen, f'gen{tn_type.upper()}')()
  
def main():
  parser = argparse.ArgumentParser(description='Generate Tensor Networks')

  parser._action_groups.pop()
  required = parser.add_argument_group('required arguments')
  optional = parser.add_argument_group('optional arguments')

  # Required.
  required.add_argument('-t', '--type', type=str,
    choices=['mera', 'peps', 'mps', 'mpo', 'ftps', 'ttn'], help='Type of tensor network',
    required=True)
  
  required.add_argument('-s', '--size', type=int,
    help='Maximal number of tensors',
    required=True)
  
  # Optional.
  optional.add_argument('-l', '--leg_type', type=str, default='closed',
    choices=['open', 'closed'], help='Type of legs')
  optional.add_argument('-f', '--factor', type=int, default=2,
    help='Start factor for bond dimensions')
  
  # Parse.
  args = parser.parse_args()

  # Generate.
  gen(args.type, args.size, args.leg_type, args.factor)

if __name__ == '__main__':
  main()