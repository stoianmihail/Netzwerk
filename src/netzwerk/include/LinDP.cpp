#include "Optimizer.hpp"
#include "TensorNetwork.hpp"

template <class BitSet>
std::pair<long double, std::vector<RangeNode>> Optimizer<BitSet>::runLocalLinDP(const std::vector<unsigned>& baseSol, const long double /* cost */)
// Run general LinDP.
{
  // Disable the tree view.
  tensorNetwork->setTreeViewStatus(false);

  // Init the variables.
  auto n = tensorNetwork->n;
  std::vector<std::vector<long double>> dp(n, std::vector<long double>(n, kInf));
  std::vector<std::vector<unsigned>> ptr(n, std::vector<unsigned>(n, -1));
  std::vector<std::vector<BitSet>> tensorLegs(n, std::vector<BitSet>(n));
  std::vector<std::vector<long double>> tensorSizes(n, std::vector<long double>(n));

  // Init the tensor legs and sizes.
  for (unsigned i = n - 1; i < n; --i) {
    long double currTensorSize = 1.0;
    BitSet currLegs;
    for (unsigned j = i; j != n; ++j) {
      // Compute the legs to be added.
      auto incomingLegs = tensorNetwork->vertexLegs[baseSol[j]];

      // Compute the common size.
      auto commonSize = tensorNetwork->computeLegDimProduct(currLegs & incomingLegs);
      assert(isLessOrEqualThan(commonSize, currTensorSize));
      
      // Update the current tensor size. 
      currTensorSize = (currTensorSize / commonSize) * (tensorNetwork->vertexSizes[baseSol[j]] / commonSize);
      
      // And the current legs.
      currLegs ^= incomingLegs;

      // Store persistently.
      tensorSizes[i][j] = currTensorSize;
      tensorLegs[i][j] = currLegs;
    }
  }

  // Init the DP table for windows of length 1.
  for (unsigned index = 0; index != n; ++index) {
    dp[index][index] = 0;
    ptr[index][index] = -1;
  }

  // Compute for windows of length >= 2.
  for (unsigned d = 1; d != n; ++d) {
    for (unsigned i = 0; i + d != n; ++i) {
      // Fix the right end, `j`.
      auto j = i + d;

      // Try all splits.
      for (unsigned k = i; k != j; ++k) {
        auto l = dp[i][k], r = dp[k + 1][j];

        // Check whether we have solutions for the subranges.
        if ((l == kInf) || (r == kInf)) continue;

        // Check for any common legs.
        auto commonLegs = tensorLegs[i][k] & tensorLegs[k + 1][j];
        // TODO:
#if ENABLE_LinDP_OUTER_PRODUCTS
        // Outer products are enabled.
#else
        // Deny outer products.
        if (!commonLegs) continue;
#endif

        // Update the cost, if better.
        auto commonSize = tensorNetwork->computeLegDimProduct(commonLegs);
        auto leftSize = tensorSizes[i][k], rightSize = tensorSizes[k + 1][j];

        // Compute the contraction cost: left * right / common.
        // Note: we use a trick to reduce the magnitude of intermediate multiplications.
        // To this end, we first divide the larger size by the common size and only do we multiply by the smaller one. 
        auto contractionCost = (std::max(leftSize, rightSize) / commonSize) * std::min(leftSize, rightSize);
        if (l + r + contractionCost < dp[i][j]) {
          // Update the dp cost.
          dp[i][j] = l + r + contractionCost;

          // And store the split point.
          ptr[i][j] = k;
        }
      }
    }
  }

  // Build the range solution.
  std::vector<RangeNode> sol;
  std::function<unsigned(unsigned, unsigned, unsigned)> buildSolution = [&](unsigned i, unsigned j, unsigned d) {
    assert(i <= j);
    
#if 0
    std::cerr << indent(d) << " build: " << "(" << i << ", " << j << ")" << std::endl;
#endif

    if (i == j) {
      // Put the current node from the linear solution.
      sol.push_back({baseSol[i], NIL, NIL});
      return sol.size() - 1;
    }

    assert(ptr[i][j] != -1);
    auto k = ptr[i][j];
    assert((i <= k) && (k <= j));
    auto l = buildSolution(i, k, d + 1);
    auto r = buildSolution(k + 1, j, d + 1);
    sol.push_back({baseSol[k], l, r});
    return sol.size() - 1;
  };

  // TODO: for IDP, maybe improve here.
  // TODO: we don't have to comput the bushy solution each time!
  auto _ = buildSolution(0, n - 1, 0);
  auto cost = tensorNetwork->computeBushyCost(sol);

#if DEBUG_COSTS
  std::cerr << "[lindp::local] cost=" << cost << std::endl;
#endif
  return {cost, sol};
}