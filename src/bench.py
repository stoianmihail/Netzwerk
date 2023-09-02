import sys
import argparse
import numpy as np
import opt_einsum as oe
from concurrent.futures import ThreadPoolExecutor
from util import ChronoTool, parse_filename, parse_net, format_elems, debug_elems, translate_edges

import jax

# Import netzwerk.
import netzwerk

# Which tensor networks are trees?
isTree = {
  'mps'  : True,
  'ftps' : True,
  'bttn' : True,
  'mera' : False,
  'peps' : False
}

def assess(n, tn_type, costs):
# Assert correctness of speed-ups.
  if 'dpccp' in costs:
    for key in costs:
      assert costs[key] >= costs['dpccp']
  if 'dpsizelinear' in costs:
    if isTree[tn_type]:
      assert costs['dpsizelinear'] == costs['tensor-ikkbz']
  if 'tensor-ikkbz' in costs and ('lindp' in costs or 'lindp-parallel' in costs):
    if 'lindp' in costs:
      assert costs['lindp'] <= costs['tensor-ikkbz']
    if 'lindp-parallel' in costs:
      assert costs['lindp-parallel'] <= costs['tensor-ikkbz']
    if 'lindp' in costs and 'lindp-parallel' in costs:
      assert costs['lindp'] == costs['lindp-parallel']
  if 'idp' in costs and ('lindp' in costs or 'lindp-parallel' in costs):
    if 'lindp' in costs:
      assert costs['lindp'] >= costs['idp']
    if 'lindp-parallel' in costs:
      assert costs['lindp-parallel'] >= costs['idp']

class Benchmark:
  def __init__(self, file, package, size, should_contract):
    self.package = package
    self.file = file
    self.max_size = size
    self.should_contract = should_contract

    # Init the benchmark mapper.
    self.bench_mapper = {
      'opt_einsum' : self.run_opt_einsum,
      'cotengra' : self.run_cotengra
    }
    assert self.package in self.bench_mapper

    # Parse the tensor networks and build the einsum expressions.
    self.info = parse_filename(file)
    self.is_valid = bool(size == -1 or self.info['size'] <= size)
    self.parse()

  def parse(self):
    if not self.is_valid:
      return
    self.tn = parse_net(self.file)

  def run_opt_einsum(self, edges, open_legs, eq, arrays, optimizer, tool):
    tool.log(f'[optimize] {optimizer}')
    path, path_info = oe.contract_path(eq, *arrays, optimize=netzwerk.fetch_optimizer(optimizer))
    tool.finish()
    
    # Contract, if requested.
    if self.should_contract:
      tool.log(f'[contract] {optimizer}')
      oe.contract(eq, *arrays, optimize=path)
      tool.finish()

    # And return the cost.
    return path_info.opt_cost, path_info

  def run_cotengra(self, edges, open_legs, eq, arrays, optimizer, tool):
    # Convert to same input as for `opt_einsum`.
    inputs, output, size_dict = translate_edges(len(arrays), edges, open_legs)
    
    # Create the optimizer.
    from cotengra.hyper import HyperOptimizer
    opt = HyperOptimizer(
      methods=optimizer,
      parallel=False,
      # Enable subtree reconfiguration.
      reconf_opts={},
      max_repeats=1
    )

    # Extract the contraction tree.
    tool.log(f'[optimize] {optimizer}')
    tree = opt.search(inputs, output, size_dict)
    tool.finish()

    # Contract, if requested.
    if self.should_contract:
      tool.log(f'[contract] {optimizer}')
      
      # we'll run the GPU contraction on a separate single thread, which mostly
      # serves as an example of how one might distribute contractions to multi-GPUs
      pool = ThreadPoolExecutor(1)

      # we'll compile the core contraction optimizer (other options here are
      # tensorflow and torch) since we call it possibly many times
      contract_core_jit = jax.jit(tree.contract_core)

      # eagerly submit all the contractions to the thread pool
      fs = [
        pool.submit(contract_core_jit, tree.slice_arrays(arrays, i))
        for i in range(tree.nslices)
      ]

      # lazily gather all the slices in the main process with progress bar
      slices = (np.array(f.result()) for f in fs)

      tree.gather_slices(slices, progbar=True)
      
      tool.finish()

    # Record the contraction cost.
    return tree.contraction_cost(), None

  def run(self, optimizer, tool):
    if not self.is_valid:
      return

    # Check if we can run this optimizer.
    # if not isTree[self.info['type']] and optimizer == 'treelindp-iks':
    #   import math
    #   return math.inf

    # Fetch the tensor network.
    n, m, o, edges, open_legs, eq, shapes = self.tn

    # Materialize it.
    arrays = list(map(np.ones, shapes))

    # Choose the benchmark type.
    cost, path = self.bench_mapper[self.package](edges, open_legs, eq, arrays, optimizer, tool)

    return cost, path

def qsim_bench(circuit_file, package):
  assert package == 'cotengra'
  print(f'>>> Benchmark {circuit_file} <<<')

  import quimb.tensor as qtn
  import cotengra as ctg

  def run_optimizer(algorithm_name, circuit, tool):
    optimizer_name = algorithm_name

    is_cotengra_algorithm = algorithm_name.startswith('cotengra-')
    algorithm_name = algorithm_name.replace('cotengra-', '')

    def get_opt(is_raw):
      reconf_opts = None if is_raw else {}

      # One of our algorithms.
      if not is_cotengra_algorithm:
        return ctg.HyperOptimizer(
          methods=algorithm_name,
          # We already have internal parallelization.
          parallel=False,
          # Repeat once.
          max_repeats=1,
          # Set reconfiguration.
          reconf_opts=reconf_opts,
          progbar=True,
        )
      
      # Cotengra algorithm.
      return ctg.HyperOptimizer(
        methods=algorithm_name,
        parallel=True,
        # Set reconfiguration.
        reconf_opts=reconf_opts,
        progbar=True,
      )

    # Raw.
    tool.log(f'[optimize] raw-{optimizer_name}')
    raw_rehs = circuit.amplitude_rehearse(optimize=get_opt(True))
    tool.finish()

    # Normal.
    tool.log(f'[optimize] {optimizer_name}')
    rehs = circuit.amplitude_rehearse(optimize=get_opt(False))
    tool.finish()

    # And return.
    return raw_rehs, rehs

  algorithms = [
    'cotengra-greedy',
    # 'custom-cotengra-greedy',
    # 'custom-cotengra-kahypar',
    'cotengra-kahypar',
    # 'idp-parallel',
    'lindp-parallel',
    'tensor-ikkbz-parallel'
  ]

  # Read the circuit.
  circuit = qtn.Circuit.from_qasm_file(circuit_file)

  # Init the time tool.
  tool = ChronoTool(circuit_file)

  # Optimize.
  costs = {}
  for optimizer in algorithms:
    print(f'Optimizer: {optimizer}')

    # Run the optimizer.
    ret1, ret2 = run_optimizer(optimizer, circuit, tool)
    assert ret1 is not None and ret2 is not None
    
    # Collect the costs.
    costs[f'raw-{optimizer}'] = ret1['C']
    costs[optimizer] = ret2['C']

  # Register the costs.
  tool.register('Costs', format_elems(costs, pretty=False))

  # Flush the tool.
  tool.debug(package, withExtra=True)
  tool.flush(package)
  pass
 
def bench(file, package, size, contract):
  # A qsim file? Then use the corresponding benchmark.
  if '.qsim' in file:
    qsim_bench(file, package)
    return

  print(f'>>> Benchmark {file} <<<')

  # Create the generator.
  tool = ChronoTool(file)

  # Parse the information from the filename.
  info = parse_filename(file)
  n, m = info['size'], info['legs']

  # And run the optimizers.
  costs = {}
  for opt in netzwerk.fetch_package_optimizers(package):
    print(f'Optimizer: {opt}')

    # Deny exponential optimizers if the number of legs is large.
    if (opt == 'dp' or opt == 'dpccp' or opt == 'dpsizelinear') and m > 30:
      continue

    # Run the optimizer.
    bench = Benchmark(file, package, size, contract)
    cost_info, _ = bench.run(opt, tool)

    costs[opt] = cost_info
    del bench

  # Register the costs.
  tool.register('Costs', format_elems(costs, pretty=False))

  # Flush the tool.
  tool.debug(package, withExtra=True)
  tool.flush(package)

  # Assess.
  assess(n, info['type'], costs)

  # And print.
  debug_elems(f'Type: {info["type"]}\n#Tensors: {n}\n#Legs: {m}', costs)
  pass

def main():
  parser = argparse.ArgumentParser(description='Performance benchmark')

  parser._action_groups.pop()
  required = parser.add_argument_group('Required arguments')
  optional = parser.add_argument_group('Optional arguments')

  # Required.
  required.add_argument('-p', '--package', type=str,
    choices=['opt_einsum', 'cotengra'], help='Package to use',
    required=True)
  
  required.add_argument('-f', '--file', type=str, help='Tensor network descriptor',
    required=True)
  
  # Optional.
  optional.add_argument('-c', '--contract', type=int, default=0,
    choices=[0, 1], help='Perform the actual contractions')

  optional.add_argument('-s', '--size', type=int, default=-1,
    help='Maximal number of tensors in the current benchmark')
  
  # Parse.
  args = parser.parse_args()

  if not args.file.endswith('.in') and not args.file.endswith('.qsim'):
    print(f'The input should be a file!')
    sys.exit(-1)

  # Generate.
  bench(args.file, args.package, args.size, args.contract)

if __name__ == '__main__':
  main()