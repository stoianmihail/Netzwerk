import time
import os
import opt_einsum as oe

def get_files(path):
  from os import listdir
  from os.path import isfile, join
  return [f for f in listdir(path) if isfile(join(path, f))]

def parse_line(line):
  return [int(elem) for elem in line.strip().split(' ')]

def parse_filename(filename):
  is_qsim = '.qsim' in filename
  split = filename.split('/')[-1].replace('.in', '').replace('.out', '').split('_')

  if not is_qsim:
    params = {}
    for elem in split[4:]:
      l, r = elem.split('-')
      params[l] = int(r)
    return {
      'size' : int(split[0]),
      'legs' : int(split[1]),
      'type' : split[2],
      'leg'  : split[3],
      'params' : params
    }
  
  # Special case for qsim circuit.
  import re
  p = re.compile('m([0-9]+)')
  for elem in split:
    m = p.match(elem)
    if m:
      cycle_size = int(m.group(1))
      return {
        'qsim' : True,
        'size' : cycle_size
      }

def build_adj(n, edges, open_legs):
  # Init the adjacency list.
  d = {}
  for i in range(n):
    d[i] = []

  # Build the adjacency list.
  for i in range(len(edges)):
    u, v, _ = edges[i]
    d[u].append(i)
    d[v].append(i)

  # Update the adjacency list with open legs.
  for vertex in open_legs:
    d[vertex].append(len(edges) + vertex)
  return d

def get_einsum(n, edges, open_legs):
  d = build_adj(n, edges, open_legs)
  expr = []
  for vertex in d:
    tensor = ''.join([oe.get_symbol(edge_index) for edge_index in d[vertex]])
    expr.append(tensor)
  return ','.join(expr) + '->' + ''.join([oe.get_symbol(len(edges) + vertex) for vertex in open_legs])

def get_shapes(n, edges, open_legs):
  d = build_adj(n, edges, open_legs)
  shapes = []
  for vertex in d:
    shapes.append(tuple([edges[edge_index][2] if edge_index < len(edges) else open_legs[edge_index - len(edges)] for edge_index in d[vertex]]))
  return shapes

def translate_edges(n, edges, open_legs):
  assert not open_legs
  inputs, output, size_dict = [], [], {}

  # Build the inputs.
  d = build_adj(n, edges, open_legs)
  for vertex in d:
    input = [oe.get_symbol(edge_index) for edge_index in d[vertex]]
    inputs.append(input)

  # Build `size_dict`.
  for vertex in d:
    for edge_index in d[vertex]:
      symbol = oe.get_symbol(edge_index)
      if symbol in size_dict:
        assert edges[edge_index][2] == size_dict[symbol]
      else:
        size_dict[symbol] = edges[edge_index][2]
  return inputs, output, size_dict

def parse_net(filename):
  with open(filename) as f:
    n, m, o = parse_line(f.readline())
    line_index = 1
    edges, open_legs = [], {}
    for line in f.readlines():
      if line_index <= m:
        edges.append(parse_line(line))
      else:
        vertex, cost = parse_line(line)
        open_legs[vertex] = cost
      line_index += 1
    assert o == len(open_legs)
  
    # Compute the einsum.
    eqs = get_einsum(n, edges, open_legs)

    # Compute the shapes.
    shapes = get_shapes(n, edges, open_legs)
    
    return n, m, o, edges, open_legs, eqs, shapes

def format_elems(elems, pretty=True):
# Format the dict `elems`.
  def fill(maxlen, s, pretty):
    if pretty:
      return ' ' * (maxlen - len(s))
    else:
      return ''

  maxlen = 0
  for opt in elems:
    maxlen = max(maxlen, len(opt))
  ret = ''

  for opt, val in dict(sorted(elems.items(), key=lambda item: item[1])).items():
    ret += f'{opt}{fill(1 + maxlen, opt, pretty)}: {val}\n'
  return ret

def debug_elems(title, elems):
# Debug the elements.
  print(title)
  print(format_elems(elems))

class ChronoTool:
  def __init__(self, file):
    self.file = file
    self.active = None
    self.start = None
    self.all = {}

  def time(self):
    return time.time_ns() // 1_000_000

  def log(self, fn=''):
    assert fn
    assert self.active is None
    self.start = self.time()
    self.active = fn

  def finish(self):
    assert self.active is not None
    stop = self.time()
    self.all[self.active] = stop - self.start
    self.active = None

  def register(self, title, content):
    self.extra = {
      'title' : title,
      'content' : content
    }

  def format(self, should_clean=False, withExtra=True):
    def cleanse(s):
      if should_clean:
        pos = 0
        while pos != len(key) and key[pos] != ']':
          pos += 1
        return key[pos + 1:].strip()
      return s

    opt_ret, cot_ret = '', ''
    for key, val in dict(sorted(self.all.items(), key=lambda item: -item[1])).items():
      if 'optimize' in key:
        opt_ret += f'{cleanse(key)}: {"{:.2f}".format(val)} ms\n'
      else:
        cot_ret += f'{cleanse(key)}: {"{:.2f}".format(val)} ms\n'
    extra = f'{self.extra["title"]}:\n{self.extra["content"]}'
    if not withExtra:
      return f'Optimization:\n{opt_ret}\nContraction:\n{cot_ret}'
    return f'Optimization:\n{opt_ret}\nContraction:\n{cot_ret}\n{extra}\n'

  def debug(self, package, withExtra):
    print(f'\n[Benchmark::{package}]\n')
    print(self.format(should_clean=False, withExtra=withExtra), end='')

  def flush(self, package):
    type = self.file.split('/')[1]
    filename = self.file.split('/')[2].replace('.in', '')

    print(f'type={type}, filename={filename}')

    # Create the directory if it doesn't exist.
    os.system(f'mkdir -p results/{type}')

    # Write.
    with open(f'results/{type}/{filename}.out', 'w') as f:
      f.write(f'Benchmark:\n{package}\n\n')
      f.write(self.format(True))
    pass