#include <unordered_map>
#include <cassert>
#include <string>

#include "include/TensorNetwork.hpp"
#include "include/Optimizer.hpp"
#include "include/Util.hpp"

struct WrappedSequence {
  int size;
  Sequence *result;
};

template <class BitSet>
Sequence* runAlgorithm(const char* algorithm, int n, int m,
  int **edges, int **tree_edges,
  double *costs, double *tree_costs,
  double *open_costs) {
  
  auto ttn = TensorNetwork<BitSet>(n, n - 1, tree_edges, tree_costs, open_costs);
  auto tn = TensorNetwork<BitSet>(n, m, edges, costs, open_costs);
  tn.setTreeView(ttn);
  auto opt = Optimizer<BitSet>(tn);
  const auto plan = opt.optimize(algorithm, 1);
  return opt.translatePlanToSequence(plan);
}

template <typename... Args>
Sequence* runAlgorithm(const char* algorithm, int n, int m, Args... args) {
  Sequence* ret = nullptr;
  auto safe_size = m + n;
  if (safe_size <= 64) {
    ret = runAlgorithm<BitSet64>(algorithm, n, m, args...);
  } else if (safe_size <= 128) {
    ret = runAlgorithm<BitSet128>(algorithm, n, m, args...);
  } else if (safe_size <= 256) {
    ret = runAlgorithm<BitSet256>(algorithm, n, m, args...);
  } else if (safe_size <= 512) {
    ret = runAlgorithm<BitSet512>(algorithm, n, m, args...);
  } else if (safe_size <= 1024) {
    ret = runAlgorithm<BitSet1024>(algorithm, n, m, args...);
  } else if (safe_size <= 2048) {
    ret = runAlgorithm<BitSet2048>(algorithm, n, m, args...);
  } else {
    assert(0);
  }
  assert(ret);
  return ret;
}

extern "C" {
// A generic optimizer.
#define OPTIMIZER(algorithm) \
  Timer timer(algorithm);    \
  auto ret = runAlgorithm(   \
    algorithm,               \
    n, m, edges, tree_edges, \
    costs, tree_costs,       \
    open_costs               \
  );                         \
  timer.summary();           \
  return {n - 1, ret};       \

  WrappedSequence tensor_ikkbz(int n, int m, int **edges, int **tree_edges, double *costs, double *tree_costs, double *open_costs) { OPTIMIZER("tensor-ikkbz") }

  WrappedSequence lindp(int n, int m, int **edges, int **tree_edges, double *costs, double *tree_costs, double *open_costs) { OPTIMIZER("lindp") }

  WrappedSequence greedy(int n, int m, int **edges, int **tree_edges, double *costs, double *tree_costs, double *open_costs) { OPTIMIZER("greedy") }

  WrappedSequence custom(int n, int m, int **edges, int **tree_edges, double *costs, double *tree_costs, double *open_costs) { OPTIMIZER("custom") }

  WrappedSequence tensor_ikkbz_parallel(int n, int m, int **edges, int **tree_edges, double *costs, double *tree_costs, double *open_costs) { OPTIMIZER("tensor-ikkbz-parallel") }

  WrappedSequence lindp_parallel(int n, int m, int **edges, int **tree_edges, double *costs, double *tree_costs, double *open_costs) { OPTIMIZER("lindp-parallel") }

#undef OPTIMIZER
}