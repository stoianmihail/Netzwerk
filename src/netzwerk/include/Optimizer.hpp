#ifndef H_Optimizer
#define H_Optimizer

template <class BitSet>
class TensorNetwork;

#include "BitSet.hpp"
#include "Util.hpp"
#include <unordered_map>

#define DEBUG_COSTS 0
#define ENABLE_LinDP_OUTER_PRODUCTS 0

template <class BitSet>
class Optimizer {
public:
  struct Plan {
    Plan(long double totalCost, BitSet set, std::shared_ptr<Plan> left, std::shared_ptr<Plan> right)
    : totalCost(totalCost), set(set), left(left), right(right) {}
    
    long double totalCost;
    BitSet set;
    std::shared_ptr<Plan> left;
    std::shared_ptr<Plan> right;
  };
  
  using TensorNetwork = ::TensorNetwork<BitSet>;

  class PrecedenceGraph;
  friend class PrecedenceGraph;

private:
  // The tensor network to be optimized.
  TensorNetwork* tensorNetwork = nullptr;
  // The current left plan. Used in DPccp.
  std::shared_ptr<Plan> leftPlan = nullptr;

  class SetNode {
  public:
    // The acumulated representatives of the compound tensors.
    BitSet representatives;
    // The open legs. This includes the *real* open legs and the virtual ones.
    BitSet openLegs;
    // The total cost of the subtree.
    long double totalCost;
    // The contraction cost.
    long double contraction;
    // The size of the represented tensor.
    long double size;
    // How many compound tensors we span.
    unsigned span;
    // The parent in the plan.
    unsigned parent;
    // The left child in the plan.
    unsigned left;
    // The right child in the plan.
    unsigned right;
    // Mark whether this is a compound tensor.
    bool isCompound = false;

    void debug() {
      std::cerr
        << "\tisCompound=" << isCompound << std::endl
        << "\tspan=" << span << std::endl
        << "\ttotalCost=" << totalCost << std::endl
        << "\trepresentatives=" << representatives << std::endl;
    }
  };

  using localOptType = std::function<std::pair<long double, std::vector<RangeNode>>(const std::vector<unsigned>& baseSol, long double cost)>;

  // Operator.
  const std::shared_ptr<Plan> opImpl(std::string name, localOptType fn);
  // Parallel operator.
  const std::shared_ptr<Plan> parallelOpImpl(std::string name, localOptType fn, unsigned numThreads);
  // Dummy helper which simply translates a linear solution to a general one.
  std::pair<long double, std::vector<RangeNode>> runDummy(const std::vector<unsigned>& baseSol, long double cost);
  // LinDP helpers.
  std::pair<long double, std::vector<RangeNode>> runLocalLinDP(const std::vector<unsigned>& baseSol, long double cost = 0);

  std::shared_ptr<Plan> getPlan(typename BitSet::arg_type s) {
    auto iter = plans.find(s);
    assert(iter != plans.end());
    return iter->second;
  }

  std::vector<unsigned> flattenPlan(const std::shared_ptr<Plan>& plan) {
    std::vector<unsigned> ret(plan->set.size());

    std::function<void(const std::shared_ptr<Plan>, unsigned)> flatten = [&](const std::shared_ptr<Plan>& p, unsigned startIndex) {
      // A single relation>
      if (p->set.size() == 1) {
        ret[startIndex] = p->set.front();
        return;
      }

      // Otherwise, split the plan.
      flatten(p->left, startIndex);
      flatten(p->right, startIndex + p->left->set.size());
    };

    // Flatten.
    flatten(plan, 0);
    return ret;
  }

  // Init the base tensors.
  void initBaseTensors();
  // Create a plan. Assumes `leftPlan` is already set.
  std::shared_ptr<Plan> createPlan(std::shared_ptr<Plan> l, std::shared_ptr<Plan> r);
  // Translate a linear solution to the corresponding plan.
  const std::shared_ptr<Plan> translateSolutionToPlan(const std::vector<unsigned>& solution);
  // Translate a `RangeNode` solution to the corresponding plan.
  const std::shared_ptr<Plan> translateSolutionToPlan(const std::vector<RangeNode>& solution);
  // The plans.
  std::unordered_map<BitSet, std::shared_ptr<Plan>, typename BitSet::hasher> plans;
  // Debug a plan.
  static void debugPlan(const std::shared_ptr<Plan> plan);

public:
  // The constructor. For the actual optimization, run `optimize`.
  Optimizer(TensorNetwork& tensorNetwork) { this->tensorNetwork = &tensorNetwork; }
  ~Optimizer() {}

  // The algorithms.
  const std::shared_ptr<Plan> runDPSizeLinear();
  const std::shared_ptr<Plan> runGreedy();
  const std::shared_ptr<Plan> runCustom();

  // Operators.
  const std::shared_ptr<Plan> runTensorIKKBZ() { return opImpl("tensor-ikkbz", std::bind(&Optimizer::runDummy, this, std::placeholders::_1, std::placeholders::_2)); }
  const std::shared_ptr<Plan> runLinDP() { return opImpl("lindp", std::bind(&Optimizer::runLocalLinDP, this, std::placeholders::_1, std::placeholders::_2)); }

  // Parallel implementations.
  const std::shared_ptr<Plan> runParallelTensorIKKBZ(unsigned numThreads) { return parallelOpImpl("tensor-ikkbz-parallel", std::bind(&Optimizer::runDummy, this, std::placeholders::_1, std::placeholders::_2), numThreads); }
  const std::shared_ptr<Plan> runParallelLinDP(unsigned numThreads) { return parallelOpImpl("lindp-parallel", std::bind(&Optimizer::runLocalLinDP, this, std::placeholders::_1, std::placeholders::_2), numThreads); }

  // Run the selected algorithm.
  const std::shared_ptr<Plan> optimize(const char* algorithm, unsigned numThreads);
  
  // Translate a plan to a sequence (used in the framework).
  Sequence* translatePlanToSequence(const std::shared_ptr<Plan> solution);
};

#include "Optimizer.tcc"

#endif
