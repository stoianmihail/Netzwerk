#include "Optimizer.hpp"
#include "TensorNetwork.hpp"

template <class BitSet>
std::shared_ptr<typename Optimizer<BitSet>::Plan> Optimizer<BitSet>::createPlan(std::shared_ptr<Plan> l, std::shared_ptr<Plan> r) {
  assert((l) && (r));

  // Fetch the problems.
  auto rightProblem = r->set;
  auto leftProblem = l->set;

  assert((leftProblem & rightProblem).empty());
  auto totalProblem = leftProblem + rightProblem;
  auto iter = plans.find(totalProblem);
  
  // Compute the contraction cost.
  // Note: this has to be recomputed, as the same `totalProblem := leftProblem + rightProblem` has different contraction costs.
  // This is different to join ordering, where the contraction cost is *always* the same.
  auto contractionCost = tensorNetwork->computeContractionCost(leftProblem, rightProblem);
  auto currCost = contractionCost + l->totalCost + r->totalCost;

  // Is this plan new?
  if (iter == plans.end()) {
    plans[totalProblem] = std::make_shared<Plan>(
      currCost, totalProblem, l, r
    );
    return plans[totalProblem];
  }

  // Update the current cost.
  auto old = iter->second;
  assert(old->set == totalProblem);

  // Is our cost better?
  if (currCost < old->totalCost) {
    // Then update.
    old->totalCost = currCost;
    old->left = l;
    old->right = r;
  }
  return old;
}

template <class BitSet>
void Optimizer<BitSet>::initBaseTensors() {
  for (unsigned index = 0, limit = tensorNetwork->n; index != limit; ++index) {
    plans[BitSet({index})] = std::make_shared<Plan>(
      0, BitSet({index}), nullptr, nullptr
    );
  }
}

template <class BitSet>
void Optimizer<BitSet>::debugPlan(const std::shared_ptr<Optimizer<BitSet>::Plan> plan)
// Debug a plan.
{
  std::function<void(const std::shared_ptr<Plan>, unsigned)> debug = [&](const std::shared_ptr<Plan> plan, unsigned depth) {
    assert(plan);
    std::cerr << indent(depth) << " " << plan->set << std::endl;
    if ((!plan->left) && (!plan->right)) {
      assert(plan->set.size() == 1);
      std::cerr << plan->set.front() << std::endl;
      return;
    }
    debug(plan->left, depth + 1);
    debug(plan->right, depth + 1);
  };

  debug(plan, 1);
}

template <class BitSet>
const std::shared_ptr<typename Optimizer<BitSet>::Plan> Optimizer<BitSet>::translateSolutionToPlan(const std::vector<unsigned>& solution) {
  // Init the base plans.
  initBaseTensors();

  // Init the left problem and plan, respectively.
  BitSet leftProblem = {solution.front()};
  auto l = plans[leftProblem];

  // And build the left-deep plan.
  for (unsigned index = 1, limit = tensorNetwork->n; index != limit; ++index) {
    // Set the right problem.
    auto rightProblem = BitSet({solution[index]});

    // Create the plan and update the current one.
    auto r = getPlan(rightProblem);
    assert((l->set == leftProblem) && (r->set == rightProblem));

    // Create the plan.
    l = createPlan(l, r);

    // Update the problem.
    leftProblem += rightProblem;
    assert(l->set == leftProblem);
  }
  assert(leftProblem.size() == tensorNetwork->n);
  return l;
};

template <class BitSet>
const std::shared_ptr<typename Optimizer<BitSet>::Plan> Optimizer<BitSet>::translateSolutionToPlan(const std::vector<RangeNode>& solution) {
  // Init the base plans.
  initBaseTensors();

  auto debug = [&](unsigned depth) {
    std::string ret = "[";
    for (unsigned i = 0; i != depth; ++i)
      ret += "*";
    ret += "] ";
    return ret;
  };

  // Build the plan from the range solution.
  int currIndex = tensorNetwork->n;
  std::function<std::shared_ptr<Plan>(unsigned, unsigned)> build = [&](unsigned index, unsigned depth) {
    if (solution[index].left == NIL) {
      assert(solution[index].right == NIL);
      return plans[{solution[index].nodeIndex}];
    }

    auto l = build(solution[index].left, 1 + depth);
    auto r = build(solution[index].right, 1 + depth);
    ++currIndex;
    return createPlan(l, r);
  };

  // Start with the last node of the solution.
  auto ret = build(solution.size() - 1, 0);
  assert(currIndex == 2 * tensorNetwork->n - 1);
  return ret;
}

template <class BitSet>
Sequence* Optimizer<BitSet>::translatePlanToSequence(const std::shared_ptr<Plan> solution)
// Translate a plan to the corresponding sequence.
{
  auto n = tensorNetwork->n;
  Sequence *ret = new Sequence[n - 1];
  int currIndex = n;
  std::function<int(const std::shared_ptr<Plan>)> build = [&](const std::shared_ptr<Plan> plan) -> int {
    if (plan->set.size() == 1) {
      assert((plan->left == nullptr) && (plan->right == nullptr));
      return static_cast<int>(*plan->set.begin());
    }

    auto l = build(plan->left);
    auto r = build(plan->right);
    ret[(currIndex++) - n] = {l, r};
    return currIndex - 1;
  };

  auto tmp = build(solution);
  assert(currIndex == 2 * n - 1);
  return ret;
}

template <class BitSet>
const std::shared_ptr<typename Optimizer<BitSet>::Plan> Optimizer<BitSet>::optimize(const char* algorithm, unsigned numThreads) {
  assert(!!tensorNetwork);
  tensorNetwork->prepareForOptimization();
  assert(tensorNetwork->isConnected(BitSet::fill(tensorNetwork->n)));
  if (isEqual(algorithm, "tensor-ikkbz"))
    return runTensorIKKBZ();
  if (isEqual(algorithm, "lindp"))
    return runLinDP();
  if (isEqual(algorithm, "greedy"))
    return runGreedy();
  
  // Parallel implementations.
  if (isEqual(algorithm, "tensor-ikkbz-parallel"))
    return runParallelTensorIKKBZ(numThreads);
  if (isEqual(algorithm, "lindp-parallel"))
    return runParallelLinDP(numThreads);

  std::cerr << "Optimizer " << algorithm << " not supported yet!" << std::endl;
  assert(0);
  return nullptr;
}

template class Optimizer<BitSet64>;
template class Optimizer<BitSet128>;
template class Optimizer<BitSet256>;
template class Optimizer<BitSet512>;
template class Optimizer<BitSet1024>;
template class Optimizer<BitSet2048>;
