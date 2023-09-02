import sys
import math

# def build_min_deg_spanning_tree(n, m, edges):
#   import spanning_tree
#   best_tree = spanning_tree.min_degree_spanning_tree(n, m, edges)
#   for u in best_tree:
#     for v in best_tree[u]:
#       tree_edges.append([u, v, 2])
#   return tree_edges

def main():
  assert len(sys.argv) == 2
  dirname = sys.argv[1]

  import os
  files = [f for f in os.listdir(os.path.abspath(dirname)) if os.path.isfile(os.path.join(os.path.abspath(dirname), f))]
  
  if dirname[-1] == '/':
    dirname = dirname[:-1]

  import spanning_tree
  import enhance
  global_ratio = 0.0
  for file in files:
    local = {}
    # for variant in ['mst']:#, 'mdst']:
    variant = 'mst'
    spanning_tree.main(os.path.join(dirname, file), variant)
    fetched = os.popen(f'./build/main lindp {dirname}/{file} {dirname}/{variant}-{file}').read()
    curr_cost = float(fetched.split('=')[1])
    local[variant] = curr_cost

    # fetched = os.popen(f'./build/main tensor-ikkbz {dirname}/{file} {dirname}/{variant}-{file}').read()
    # linear_curr_cost = float(fetched.split('=')[1])
    # # local[variant] = linear_curr_cost

    enhanced_local = {}
    for enh in ['node', 'contraction']:
      enhance.main(os.path.join(dirname, file), f'{dirname}/{variant}-{file}', enh)
      enhanced_fetched = os.popen(f'./build/main lindp {dirname}/{file} {dirname}/enhanced_{enh}-{variant}-{file}').read()
      enhanced_curr_cost = float(enhanced_fetched.split('=')[1])
      enhanced_local[enh] = enhanced_curr_cost

    print(f'>>>> curr_cost={curr_cost} vs enhanced_local={enhanced_local}')
    ratio = curr_cost / enhanced_local['node']

      # print(f'[{variant}] normal={curr_cost} vs enhanced={enhanced_curr_cost}')

    # ratio = local['mst'] / local['mdst']
    # enhanced_ratio = enhanced_local['mst'] / enhanced_local['mdst']
    # print(f'>>>> ratio={ratio}')
    # print(f'#### enhanced={enhanced_ratio}')
    global_ratio += math.log10(ratio)

  global_ratio /= len(files)
  print(f'global_ratio={global_ratio}')

if __name__ == '__main__':
  main()