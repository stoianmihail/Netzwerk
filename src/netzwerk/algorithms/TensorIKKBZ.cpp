#include "Optimizer.hpp"
#include "TensorNetwork.hpp"
#include <barrier>

template <class BitSet>
class Optimizer<BitSet>::PrecedenceGraph {
public:
  struct Node {
    PrecedenceGraph *master;
    // The id. This represents the leading node id in `compound`.
    unsigned vertexId;
    // The id of the incoming edge towards this node
    unsigned incomingEdgeId;
    // The size of the open legs of the current (compound) node. These are the *real* open legs.
    // Storing this attribute improves the repeated computation when updating the symbolic rank.
    long double openSize;
    // The outer legs of the current (compound) node. These are only wrt the algorithmic logic.
    BitSet outerLegs;
    // The accumulated cost. This represents the numerator of the symbolic rank.
    long double acc;
    // The number of *vertices* (could be already contracted) from the chain which have been melded into this node
    unsigned contracted;
    // The children
    std::vector<unsigned> children;
    // The chain (contains the *vertices* in its subtree, which may be *represent* relations, yet they are not directly stored herein)
    std::vector<unsigned> chain;
    // The compound relation (contains the *relations* - may *not* be compound relations - from the chains which it merged with)
    std::vector<unsigned> compound;
  
    std::string debug() const {
      std::string ret;
      ret += "acc=" + std::to_string(acc);
      ret += ", ";
      ret += "incomingEdgeId=" + std::to_string(incomingEdgeId);
      ret += ", ";
      ret += "outerLegs=" + outerLegs.toString();
      ret += ", ";
      ret += "compound=" + debugVector(compound);
      return ret;
    }

    bool operator<(const Node& o) const {
      auto [a, b] = computeRank();
      auto [c, d] = o.computeRank();
      return a * d < b * c;
    }

    bool shouldMergeWith(const Node& o) const {
      auto [a, b] = computeRank();
      auto [c, d] = o.computeRank();
      return a * d > b * c;
    }

  private:
    std::pair<long double, long double> computeRank() const {
      return {acc, master->net->fetchEdge(incomingEdgeId).cost - openSize * master->net->computeLegDimProduct(outerLegs)};
    }
  };

  // The constructor. It builds the precedence graph of node `rootIndex`.
  PrecedenceGraph(Optimizer& optimizer, unsigned rootIndex);

  // Run a local TensorIKKBZ. Assumes the precedence graph has already been built.
  std::pair<long double, std::vector<unsigned>> runLocalTensorIKKBZ();

private:
  // The tensor network.
  TensorNetwork* net = nullptr;
  // The actual precedence graph.
  std::vector<Node> tree;
  // The BFS traversal.
  std::vector<unsigned> bfs;
  // The current root.
  unsigned currRootIndex;
};

template <class BitSet>
Optimizer<BitSet>::PrecedenceGraph::PrecedenceGraph(Optimizer& optimizer, unsigned root) : net(optimizer.tensorNetwork)
// Build the precedence graph.
{
  // View the network as a tree for now.
  assert(!!net);
  net->setTreeViewStatus(true);

  // Init the tree.
  tree.clear();
  bfs.clear();
  tree.resize(net->n);
  bfs.reserve(net->n);

  std::queue<unsigned> q;
  currRootIndex = root;
  tree[root].openSize = net->openLegSize[root];
  tree[root].outerLegs = net->collectVertexLegs(root, true);
  tree[root].acc = tree[root].openSize * net->computeLegDimProduct(tree[root].outerLegs);
  tree[root].incomingEdgeId = -1;
  tree[root].master = this;
  tree[root].vertexId = root;
  q.push(root);

  // Add a child.
  auto addChild = [&](unsigned parent, unsigned v, unsigned edgeId) -> void {
    // Push the child.
    tree[parent].children.push_back(v);

    // Store the incoming edge id.
    tree[v].incomingEdgeId = edgeId;
    
    // Store the open size.
    tree[v].openSize = net->openLegSize[v];

    // Store current outer legs.
    tree[v].outerLegs = net->collectVertexLegs(v, true) - edgeId;

    // Compute the current size.
    tree[v].acc = tree[v].openSize * net->computeLegDimProduct(tree[v].outerLegs + edgeId);

    // Set the master.
    tree[v].master = this;

    // And the vertex id.
    tree[v].vertexId = v;

    // And push.
    q.push(v);
  };

  // Run BFS.
  BitSet seen;
  while (!q.empty()) {
    // Pop.
    auto curr = q.front();
    q.pop();
  
    // Update bfs.
    bfs.push_back(curr);
    
    // Update the set of seen nodes.
    seen.insert(curr);

    // And find the children.
    assert((net->getTreeViewStatus()) && (!!net->treeView));
    for (auto pos = net->treeView->adj[curr]; pos; pos = net->treeView->list[pos].next) {
      // Have we already seen this node?
      if (seen.count(net->treeView->list[pos].v)) continue;

      // If not, add as child.
      addChild(curr, net->treeView->list[pos].v, net->treeView->list[pos].edgeId); 
    }
  }
  // std::cerr << "seen.size=" << seen.size() << " n=" << net->n << std::endl;
  // std::cerr << "bfs.size=" << bfs.size() << std::endl;
  assert((seen.size() == net->n) && (bfs.size() == net->n));
}

template <class BitSet>
std::pair<long double, std::vector<unsigned>> Optimizer<BitSet>::PrecedenceGraph::runLocalTensorIKKBZ()
// Run a local TensorIKKBZ.
{
  std::vector<unsigned> heap;
  std::vector<std::optional<unsigned>> pointer;

#define getChild(pos) tree[curr].children[pos]
  for (unsigned index = net->n - 1; index < net->n; --index) {
    auto curr = bfs[index];

    // Init the compound relation
    tree[curr].compound.push_back(curr);
    
    // Leaf? Then its chain remains empty
    if (tree[curr].children.empty())
      continue;
        
    // Init the heap and the pointers within the chains
    pointer.resize(tree[curr].children.size());
    heap.resize(tree[curr].children.size());
    for (unsigned i = 0, sz = tree[curr].children.size(); i != sz; ++i) {
      heap[i] = i;
      pointer[i] = std::nullopt;
    }

    // Heap
    auto cmp = [&](auto a, auto b) {
      // a, b represent the indices of the children, i.e., [0, children.size()[, not the children itself
      auto l = pointer[a].has_value() ? tree[getChild(a)].chain[pointer[a].value()] : getChild(a);
      auto r = pointer[b].has_value() ? tree[getChild(b)].chain[pointer[b].value()] : getChild(b);
      return tree[r] < tree[l];
    };
    
    // Init the heap
    make_heap(heap.begin(), heap.end(), cmp);
    
    // And merge
    while (!heap.empty()) {
      // Pop
      pop_heap(heap.begin(), heap.end(), cmp);

      // Analyze the popped element
      auto currChildIndex = heap.back();

      // Update the chain of the current node
      tree[curr].chain.push_back(pointer[currChildIndex].has_value() ? tree[getChild(currChildIndex)].chain[pointer[currChildIndex].value()] : getChild(currChildIndex));
      
      // Forward the pointer on the chain of this child
      // Is it the first time we extracted this child?
      if (!pointer[currChildIndex].has_value()) {
        // The child is a leaf, i.e., its chain is empty?
        if (tree[getChild(currChildIndex)].chain.empty()) {
          assert(tree[getChild(currChildIndex)].children.empty());
          
          // Then erase it from the heap
          heap.pop_back();
          continue;
        } else {
          // Does the contracted vertex occupy the entire chain?
          if (tree[getChild(currChildIndex)].contracted == tree[getChild(currChildIndex)].chain.size()) {
            // Then erase it from the heap
            heap.pop_back();
            continue;
          }
          
          // Otherwise set the pointer to the beginning of the chain, which has been cut (since we contracted the vertex)
          pointer[currChildIndex] = tree[getChild(currChildIndex)].contracted;
        }
      } else {
        // Simply increment the pointer within the chain
        pointer[currChildIndex].value()++;
      }

      // Is the chain fully consumed?
      assert(pointer[currChildIndex].has_value());
      if (pointer[currChildIndex].value() == tree[getChild(currChildIndex)].chain.size()) {
        // Then erase it from the heap
        heap.pop_back();
        continue;
      }
      
      // Only heapify, in this case the heap has not been touched
      push_heap(heap.begin(), heap.end(), cmp);
    }

    // Skip if root.
    if (curr == currRootIndex)
      break;

    // Compute the contracted vertex
    tree[curr].contracted = 0;
    for (unsigned i = 0, sz = tree[curr].chain.size(); (i != sz) && (tree[curr].shouldMergeWith(tree[tree[curr].chain[i]])); ++i) {
      // Note: the order of instructions matters
      auto next = tree[curr].chain[i];

      // Update the accumulated contraction cost.
      // TODO: one problem here is that for many open legs, this is bad.
      // TODO: since we do the product each time.
      // TODO: maybe really store only the open legs in a variable and collect them.
      // TODO: since they're only added each time.
      tree[curr].acc += tree[curr].openSize * net->computeLegDimProduct(tree[curr].outerLegs - tree[next].incomingEdgeId) * tree[next].acc;

      // Update the outer legs. Take into consideration also the incoming edge of `next`.
      tree[curr].outerLegs ^= (tree[next].outerLegs + tree[next].incomingEdgeId);
      
      // Update the open size.
      tree[curr].openSize *= tree[next].openSize;

      // And update the contracted vertex.
      tree[curr].contracted++;
      
      // Update the compound relation.
      tree[curr].compound.insert(tree[curr].compound.end(), tree[next].compound.begin(), tree[next].compound.end());
    }
  }
#undef getChild

  std::vector<unsigned> order(net->n);
  unsigned ptr = 0;
  order[ptr++] = currRootIndex;
  for (unsigned index = 0, limit = tree[currRootIndex].chain.size(); index != limit; ++index) {
    for (auto elem : tree[tree[currRootIndex].chain[index]].compound) {
      order[ptr++] = elem;
    }
  }
  assert(ptr == net->n);

  // Disable tree view.
  // TODO: when rotating, make sure we always come back to tree view.
  net->setTreeViewStatus(false);
  return {net->computeLinearCost(order), order};
}

template <class BitSet>
const std::shared_ptr<typename Optimizer<BitSet>::Plan> Optimizer<BitSet>::opImpl(std::string name, localOptType fn)
// Run `fn` on TensorIKKBZ linearizations.
{
  // And run the algorithm for each tensor.
  long double minCost = kInf;
  std::vector<RangeNode> bestSol;

  for (unsigned index = tensorNetwork->n - 1; index < tensorNetwork->n; --index) {
    // Build the precedence graph.
    PrecedenceGraph pg(*this, index);
    assert(tensorNetwork->getTreeViewStatus());

    // Run TensorIKKBZ.
    auto [cost, sol] = pg.runLocalTensorIKKBZ();

    // Run `fn`.
    auto [bushyCost, bushySol] = fn(sol, cost);
    assert(bushyCost <= cost + 1e-6);
    if (bushyCost < minCost) {
      minCost = bushyCost;
      bestSol = bushySol;
    }
  }

  // And translate the bushy solution to its corresponding plan.
  assert(!tensorNetwork->getTreeViewStatus());
  auto plan = translateSolutionToPlan(bestSol);
  assert(isClose(std::log10(plan->totalCost), std::log10(minCost)));

#if DEBUG_COSTS
  std::cerr << "[" << name << "] cost=" << minCost << std::endl;
#endif

  return plan;
}

template <class BitSet>
const std::shared_ptr<typename Optimizer<BitSet>::Plan> Optimizer<BitSet>::parallelOpImpl(std::string name, localOptType fn, unsigned /* numThreads */)
// Run a parallel implementation of `fn`.
{
  unsigned numThreads = std::thread::hardware_concurrency() - 1;
  std::cerr << "!!!!!!!!!!! numThreads=" << numThreads<< " !!!!!!!!!!!!!! " << std::endl;

  long double minCost = kInf;
  unsigned bestIndex;
  std::vector<RangeNode> bestSol;

  std::mutex updateMutex;
  auto updateSolutionIfBetter = [&](unsigned index, long double currCost, std::vector<RangeNode>& currSol) {
    const std::lock_guard<std::mutex> _(updateMutex);
    if (currCost < minCost) {
      minCost = currCost;
      bestSol = currSol;
      bestIndex = index;
    }
  };

  // The barrier.
  std::barrier sync(numThreads, [&] {});
 
  // The consumer which runs an iteration of LinDP.
  std::atomic<unsigned> taskIndex = 0;
  auto consume = [&]() {
    // Register this thread.
    tensorNetwork->registerThread();

    // Wait for all threads to be registered before moving forward.
    sync.arrive_and_wait();

    // Select the next tensor index to optimize for.
    while (taskIndex.load() < tensorNetwork->n) {
      unsigned index = taskIndex++;
      if (index >= tensorNetwork->n)
        return;

      // Build the precedence graph.
      PrecedenceGraph pg(*this, index);
      assert(tensorNetwork->getTreeViewStatus());

      // Run TensorIKKBZ.
      auto [cost, sol] = pg.runLocalTensorIKKBZ();

      // Run `fn`.
      auto [bushyCost, bushySol] = fn(sol, cost);
      assert(bushyCost <= cost + 1e-6);
      updateSolutionIfBetter(index, bushyCost, bushySol);
    }
  };
    
  // Start the threads.
  std::vector<std::thread> threads;
  for (unsigned threadIndex = 0; threadIndex != numThreads; ++threadIndex)
    threads.emplace_back(consume);
  
  // Collect them.
  for (auto& thread : threads)
    thread.join();

  // Unregister the threads.
  tensorNetwork->unregisterThreads();

  // And translate the bushy solution to its corresponding plan.
  assert(!tensorNetwork->getTreeViewStatus());

  auto plan = translateSolutionToPlan(bestSol);
  assert(isClose(std::log10(plan->totalCost), std::log10(minCost)));

#if DEBUG_COSTS
  std::cerr << "[" << name << "] FINAL cost=" << minCost << " bestIndex=" << bestIndex << std::endl;
#endif

  return plan;
}

template <class BitSet>
std::pair<long double, std::vector<RangeNode>> Optimizer<BitSet>::runDummy(const std::vector<unsigned>& baseSol, const long double cost)
// Translate a linear solution into a general one.
{
  unsigned n = baseSol.size();
  std::vector<RangeNode> sol(2 * n - 1);
  for (unsigned index = 0, limit = n; index != limit; ++index)
    sol[index] = {baseSol[index], NIL, NIL};
  for (unsigned ptr = 0, index = n, limit = sol.size(); index != limit; ptr = index++)
    sol[index] = {NIL, ptr, index - n + 1};
  return {cost, sol};
}