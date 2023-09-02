
import opt_einsum as oe
import multiprocessing
import framework.wrapper as framework

class TensorIKKBZ(oe.paths.PathOptimizer):
  def __call__(self, inputs, output, size_dict, memory_limit=None):
    print(f'%%%%%%%%%%%%%% TensorIKKBZ %%%%%%%%%%%%%%')
    return framework.contraction_order(inputs, output, size_dict, 'tensor-ikkbz')

class LinDP(oe.paths.PathOptimizer):
  def __call__(self, inputs, output, size_dict, memory_limit=None):
    print(f'%%%%%%%%%%%%%% LinDP %%%%%%%%%%%%%%')
    return framework.contraction_order(inputs, output, size_dict, 'lindp')

class Greedy(oe.paths.PathOptimizer):
  def __call__(self, inputs, output, size_dict, memory_limit=None):
    print(f'%%%%%%%%%%%%%% Greedy %%%%%%%%%%%%%%')
    return framework.contraction_order(inputs, output, size_dict, 'goo')

class ParallelTensorIKKBZ(oe.paths.PathOptimizer):
  def __call__(self, inputs, output, size_dict, memory_limit=None):
    print(f'%%%%%%%%%%%%%% TensorIKKBZ [parallel] %%%%%%%%%%%%%%')
    return framework.contraction_order(inputs, output, size_dict, 'tensor-ikkbz-parallel')

class ParallelLinDP(oe.paths.PathOptimizer):
  def __call__(self, inputs, output, size_dict, memory_limit=None):
    print(f'%%%%%%%%%%%%%% LinDP [parallel] %%%%%%%%%%%%%%')
    return framework.contraction_order(inputs, output, size_dict, 'lindp-parallel')

class CustomRandomGreedy(oe.paths.PathOptimizer):
  def __call__(self, inputs, output, size_dict, memory_limit=None):
    optimizer = oe.RandomGreedy(max_repeats=len(inputs), parallel=multiprocessing.cpu_count())
    return optimizer(inputs, output, size_dict, memory_limit)

class DynamicProgramming(oe.paths.PathOptimizer):
  def __call__(self, inputs, output, size_dict, memory_limit=None):
    optimizer = oe.DynamicProgramming()
    return optimizer(inputs, output, size_dict, memory_limit)

# Map the name of the optimizers to their implementation.

optimizers = {
  # 'tensor-ikkbz' : TensorIKKBZ(),
  'greedy' : 'greedy',
  # 'lindp' : LinDP(),
  # 'goo' : Greedy(),
  # 'custom' : Custom(),
  'tensor-ikkbz-parallel' : ParallelTensorIKKBZ(),
  'lindp-parallel' : ParallelLinDP(),
  'custom-random-greedy' : CustomRandomGreedy(),
  'dp' : DynamicProgramming()
}

def fetch_optimizer(name):
  if name in optimizers:
    return optimizers[name]
  return name

# Opt_einsum optimizers.
opt_einsum_optimizers = optimizers.copy()

# Cotengra optimizers.
cotengra_optimizers = list(optimizers.keys())

def fetch_package_optimizers(package):
  if package == 'opt_einsum':
    return opt_einsum_optimizers.keys()
  if package == 'cotengra':
    return cotengra_optimizers
  assert 0
  return {}

# Setup Cotengra.

import functools
from cotengra import hyper
from cotengra.core import ContractionTree

def general_optimizer(meta_fn, algo, inputs, output, size_dict):
  print(f'%%%%%%%%%%%%%% {algo.upper()} %%%%%%%%%%%%%%')
  ssa_path = framework.contraction_order(inputs, output, size_dict, algo, use_ssa=True)
  return meta_fn(inputs, output, size_dict, ssa_path=ssa_path)

# Register the algorithms.
for opt in fetch_package_optimizers('cotengra'):
  if opt == 'greedy':
    continue
  hyper.register_hyper_function(
    name=opt,
    ssa_func=functools.partial(general_optimizer, ContractionTree.from_path, opt),
    space={}
  )

# Utils for registering custom solution.
def register_custom_solution(path_info):
  from opt_einsum.paths import linear_to_ssa
  print(f'register custom solution')
  import os
  curr_path = os.path.abspath('.')
  if '/src' not in curr_path:
    curr_path = os.path.join(curr_path, 'src')
  print(curr_path)
  with open(os.path.join(curr_path, 'custom_ssa_solution.in'), 'w') as f:
    print(f'test={linear_to_ssa([item[0] for item in path_info.contraction_list])}')
    for (x, y) in linear_to_ssa([item[0] for item in path_info.contraction_list]):
      f.write(f'{x} {y}\n')
    f.close()