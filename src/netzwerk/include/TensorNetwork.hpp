#ifndef H_TensorNetwork
#define H_TensorNetwork

#include <fstream>
#include <queue>
#include <optional>
#include <mutex>
#include <thread>

#include "BitSet.hpp"
#include "Common.hpp"

#define ENABLE_TWO 0

template <class BitSet>
class TensorNetwork {  
public:
  TensorNetwork() = default;

  TensorNetwork(int n, int m, int **edges, double *costs, double *open_costs) {
    this->n = static_cast<unsigned>(n), this->m = static_cast<unsigned>(m);

    // Account for the open legs as well.
    edgeInfo.resize(m + n);
    for (unsigned index = 0; index != this->m; ++index) {
      auto u = static_cast<unsigned>(edges[index][0]);
      auto v = static_cast<unsigned>(edges[index][1]);
      edgeInfo[index] = {static_cast<long double>(costs[index]), {u, v}};
    }

    // Init the open legs.
    openLegSize.assign(n, 1.0);
    for (unsigned index = 0; index != this->n; ++index)
      openLegSize[index] = static_cast<long double>(open_costs[index]);

    // Init graph structure.
    initGraphStructure();
  }

  TensorNetwork(std::string filepath)
  : filepath(filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
      std::cerr << "File " << filepath << " does not exist!" << std::endl;
      assert(0);
    }

    unsigned o;
    in >> n >> m >> o;

    // Account for the open legs as well.
    edgeInfo.resize(m + n);
    for (unsigned index = 0; index != m; ++index) {
      unsigned u, v;
      in >> u >> v >> edgeInfo[index].cost;
      edgeInfo[index].edge = {u, v};
    }

    // Read the open legs and init them.
    openLegSize.assign(n, 1.0);
    while (o--) {
      unsigned u;
      in >> u;
      assert(u < n);
      in >> openLegSize[u];
    }    

    // Init the graph structure.
    initGraphStructure();
  }

  // Copy the relevant information of this tensor network.
  void copy(TensorNetwork<BitSet>& newTensorNetwork);
  // Slice the tensor network based on the ids provided.
  void slice(TensorNetwork<BitSet>& slice, const BitSet& ids);
  // Update the subgraph by contracting it.
  void updateSubgraph(const BitSet& ids);
  // Set the tree view.
  void setTreeView(TensorNetwork<BitSet>& ttn) { treeView = &ttn; }
  // Prepare for optimization.
  void prepareForOptimization();
  // Test whether the tensor network is actually a tree.
  bool isTree() const;
  // Fetch the edge.
  EdgeInfo fetchEdge(unsigned edgeId) const;
  // Compute cost of contraction between `set1` and `set2`.
  long double computeContractionCost(BitSet set1, BitSet set2) const;
  // Compute the neighbors for a given set, forbidding those from `x`.
  BitSet computeNeighbors(typename BitSet::arg_type s, typename BitSet::arg_type x) const;
  // Compute the cost of a linear solution.
  long double computeLinearCost(const std::vector<unsigned>& sol) const;
  // Compute the cost of a bushy solution.
  long double computeBushyCost(const std::vector<RangeNode>& sol) const;
  // Check if node is neighbor to set.
  bool reaches(unsigned u, BitSet set) const;
  // Check whether the tensors in `set` are connected.
  bool isConnected(BitSet set, bool verbose = false) const;
  // Check whether the range is connected.
  bool isRangeConnected(unsigned i, unsigned j, const std::vector<unsigned>& linearSol) const;
  // Collect the vertex legs of node `u`. Mode `strict` does not take the open legs.
  BitSet collectVertexLegs(unsigned u, bool strict = false) const;
  // Collect the open legs of `set`.
  BitSet collectOpenLegs(BitSet set) const;
  // Compute the product of the leg dimensions.
  long double computeLegDimProduct(BitSet set) const;
  // Extract a spanning tree.
  void extractSpanningTree();
  // Extract a path tree.
  void extractPathTree(unsigned root);

  void registerThread() {
    std::unique_lock<std::mutex> lock(statusMutex);
    auto threadId = std::this_thread::get_id();
    threadStatus[threadId] = false;
  }

  void unregisterThreads() {
    threadStatus.clear();
  }

  void setTreeViewStatus(bool value) {
    // No registered thread?
    if (threadStatus.empty()) {
      treeViewIsEnabled = value;
    } else {
      auto threadId = std::this_thread::get_id();
      assert(threadStatus.find(threadId) != threadStatus.end());
      threadStatus[threadId] = value;
    }
  }

  bool getTreeViewStatus() const {
    // No registered thread?
    if (threadStatus.empty())
      return treeViewIsEnabled;

    auto threadId = std::this_thread::get_id();
    auto iter = threadStatus.find(threadId);
    assert(iter != threadStatus.end());
    return iter->second;
  }

private:
  void initGraphStructure(bool initNeighborSets = true)
  // Init the graph structure.
  {
    auto addEdge = [&](unsigned u, unsigned v, unsigned edgeId) {
      // Build the bitsets, if required.
      if (initNeighborSets)
        N[u].insert(v);
      else
        assert(N[u].count(v));
      
      // Insert the edge.
      list[bufPtr] = {v, edgeId, adj[u]};
      adj[u] = bufPtr;
      ++bufPtr;
    };

    if (initNeighborSets) N.assign(n, {});
    bufPtr = 1;
    adj.assign(n, 0);
    list.resize(1 + 2 * m);
    assert(edgeInfo.size() == m + n);
    for (unsigned index = 0; index != m; ++index) {
      auto edge = edgeInfo[index].edge;
      addEdge(edge.first, edge.second, index);
      addEdge(edge.second, edge.first, index);
    }

    // Init the open legs.
    for (unsigned index = 0; index != n; ++index) {
      assert(openLegSize[index] > std::numeric_limits<long double>::epsilon());
      edgeInfo[m + index].cost = openLegSize[index];
    }
  }

  void reach(unsigned u, BitSet& cum, const BitSet& allowedSet) const
  // Determine the reachable set from node `u`, by allowind nodes only from `allowedSet`.
  {
    if (getTreeViewStatus()) {
      assert(!!treeView);
      treeView->reach(u, cum, allowedSet);
      return;
    }

    cum.insert(u);
    for (unsigned pos = adj[u]; pos; pos = list[pos].next) {
      // Explore the next neighbor.
      if ((!cum.count(list[pos].v)) && (allowedSet.count(list[pos].v))) {
        reach(list[pos].v, cum, allowedSet);
      }
    }
  }

  BitSet formRangeSet(unsigned i, unsigned j, const std::vector<unsigned>& base) const
  // Form the set of range `[i, j]`.
  {
    assert(j < base.size());
    BitSet ret;
    for (unsigned index = i; index <= j; ++index)
      ret.insert(base[index]);
    return ret;
  }

public:
  unsigned n, m;
  unsigned bufPtr = 1;
  std::string filepath;
  std::vector<BitSet> N;
  std::vector<unsigned> adj;
  std::vector<EdgeInfo> edgeInfo;
  std::vector<Cell> list;
  BitSetHashMap<BitSet, BitSet> legs;
  std::vector<BitSet> vertexLegs;
  std::vector<long double> vertexSizes;
  TensorNetwork<BitSet>* treeView;
  std::vector<long double> openLegSize;

private:
  // Thread utils for setting `treeViewIsEnabled` on each thread.
  bool treeViewIsEnabled = false;
  std::mutex statusMutex;
  std::unordered_map<std::thread::id, bool> threadStatus;
};

template class TensorNetwork<BitSet64>;
template class TensorNetwork<BitSet128>;
template class TensorNetwork<BitSet256>;
template class TensorNetwork<BitSet512>;
template class TensorNetwork<BitSet1024>;
template class TensorNetwork<BitSet2048>;

#include "TensorNetwork.inl"

#endif
