#include "Optimizer.hpp"
#include "TensorNetwork.hpp"

#include <set>

template <class BitSet>
const std::shared_ptr<typename Optimizer<BitSet>::Plan> Optimizer<BitSet>::runGreedy()
// Run a greedy algorithm.
{
  // Init the base tensors.
  initBaseTensors();

  // Init the mappings.
  std::unordered_map<unsigned, BitSet> mapping;
  for (unsigned index = 0, limit = tensorNetwork->n; index != limit; ++index)
    mapping[index] = {index};

  // Cost of an edge contraction.
  auto cost = [&](unsigned edgeId) -> long double {
    auto [u, v] = tensorNetwork->edgeInfo[edgeId].edge;
    auto l = mapping[u], r = mapping[v];
    if (l == r)
      return -1.0;
    return tensorNetwork->computeContractionCost(l, r);
  };

  auto cmp = [&](unsigned lhs, unsigned rhs) {
    // If the costs are the same, decide based on lexicographic order.
    auto l = cost(lhs), r = cost(rhs);
    if (isClose(l, r))
      return lhs < rhs;

    // Otherwise, compare the costs.
    return l < r;
  };

  // Setup the heap.
  std::set<unsigned, decltype(cmp)> heap(cmp);
  
  // Init the heap.
  for (unsigned index = 0, limit = tensorNetwork->edgeInfo.size(); index != limit; ++index) {
    heap.insert(index);
  }

  while (!heap.empty()) {
    // Fetch the min edge.
    auto minEdgeId = *heap.begin();
    heap.erase(heap.begin());

    // Is the edge already inside a contracted tensor?
    if (cost(minEdgeId) < 0)
      continue;

    // If not, the create a new plan.
    auto [u, v] = tensorNetwork->edgeInfo[minEdgeId].edge;

    auto l = mapping[u], r = mapping[v];
    createPlan(getPlan(l), getPlan(r));

    // Update the mappings.
    auto cum = l + r;
    for (auto elem : cum)
      mapping[elem] = cum;
  }

  // Return the final plan.
  auto finalPlan = getPlan(BitSet::fill(tensorNetwork->n));

#if DEBUG_COSTS
  std::cerr << "[greedy] cost=" << finalPlan->totalCost << std::endl;
#endif

  return finalPlan;
}

template class Optimizer<BitSet64>;
template class Optimizer<BitSet128>;
template class Optimizer<BitSet256>;
template class Optimizer<BitSet512>;
template class Optimizer<BitSet1024>;
template class Optimizer<BitSet2048>;