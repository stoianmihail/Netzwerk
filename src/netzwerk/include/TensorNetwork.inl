#include "TensorNetwork.hpp"
#include <functional>

template <class BitSet>
void TensorNetwork<BitSet>::copy(TensorNetwork<BitSet>& newTensorNetwork)
// Copy the tensor network by value.
{
  newTensorNetwork.n = n;
  newTensorNetwork.m = m;
  newTensorNetwork.adj = adj;
  newTensorNetwork.list = list;
  newTensorNetwork.edgeInfo = edgeInfo;
  newTensorNetwork.openLegSize = openLegSize;
}

template <class BitSet>
void TensorNetwork<BitSet>::slice(TensorNetwork<BitSet>& slice, const BitSet& ids)
// Slice the tensor network based on the ids provided.
{
  slice.n = ids.size();
  slice.N.assign(slice.n, {});
  slice.openLegSize.assign(slice.n, 1.0);
  // TODO: maybe reserve `edgeInfo`.

  // Init the mapping andthe open legs.
  std::vector<unsigned> mapping(n, NIL);
  unsigned index = 0;
  for (auto elem : ids) {
    mapping[elem] = index;
    slice.openLegSize[index] = openLegSize[elem];
    ++index;
  }

  // Build the edges.
  assert(slice.edgeInfo.empty());
  for (auto u : ids) {
    for (unsigned pos = adj[u]; pos; pos = list[pos].next) {
      // Fetch the neighbor.
      auto v = list[pos].v;

      auto edgeCost = edgeInfo[list[pos].edgeId].cost;

      // Contained in this slice?
      if (ids.count(v)) {
        auto v1 = mapping[u], v2 = mapping[v];
        assert(v1 != v2);
        if (slice.N[v1].count(v2)) {
          assert(slice.N[v2].count(v1));
          continue;
        }

        // Update the neighbors.
        slice.N[v1].insert(v2);
        slice.N[v2].insert(v1);
        slice.edgeInfo.push_back({edgeCost, {v1, v2}});
        continue;
      }

      // Update the open leg size with the cost of the edge.
      slice.openLegSize[mapping[u]] *= edgeCost;
    }
  }

  // Init the number of edges.
  slice.m = slice.edgeInfo.size();

  // Resize `edgeInfo` before introducing the open legs therein.
  slice.edgeInfo.resize(slice.m + slice.n);

  // Init the graph structure without initializing the neighbor sets.
  slice.initGraphStructure(false);
}

template <class BitSet>
void TensorNetwork<BitSet>::updateSubgraph(const BitSet& ids)
// Update the subgraph by contracting it.
{
  // Fetch the representative.
  auto repr = ids.min();

  // TODO: this is way too complicated.
  // TODO: first try to use the `initGraphStructure`. Then optimize this path.
#if 0
  // Update the open legs.
  for (auto u : ids) {
    if (u == repr) continue;
    openLegSize[repr] *= openLegSize[u];
  }

  // Update the graph structure.
  std::vector<unsigned> incomingEdgeId(n, NIL);
  for (auto u : ids) {
    // The representative?
    if (u == repr) {
      // Update the head of the adjacency list until there is no neighbor which is contained in the subgraph.
      updateAdj: {
        // Done.
        if (!adj[u]) continue;

        // Inside the subgraph?
        if (ids.count(list[adj[u]].v)) {
          adj[u] = list[adj[u]].next;
          goto updateAdj;
        }
      }

      // Erase from the adjacency list the neighbors which are contained in the subgraph.
      unsigned pos = adj[u], prev = NIL;
      do {
        // In our subgraph?
        if (ids.count(list[pos].v)) {
          // Make sure this is not the first time.
          assert(prev != NIL);

          // Update the next pointer, i.e., erase this connection.
          list[prev].next = list[pos].next;
        } else {          
          // No, then update the previous position.
          prev = pos;
        }

        // Go to the next position.
        pos = list[pos].next;
      } while (pos);
      continue;
    }

    // A non-representative.
    for (unsigned pos = adj[u]; pos; pos = list[pos].next) {
      auto v = list[pos].v;
      if (ids.count(v)) continue;

      // First time?
      if (incomingEdgeId[v] == NIL) {
        incomingEdgeId[v] = list[pos].edgeId;

        // Update the edge itself.
        edgeInfo[incomingEdgeId[v]].edge = {repr, }
      }
        incomingEdgeId[v] = list[pos].edgeId;
    
      if (edgeInfo[])
      edgeInfo[incomingEdgeId[v]].cost *= edgeInfo[in]
    }
  }

  // Update the neighbor set.
  N[u] -= ids;
#else
  unsigned currPtr = 0;
  std::vector<long double> incomingCost(n, -1);
  for (unsigned index = 0; index != m; ++index) {
    auto [u, v] = edgeInfo[index].edge;
    if (ids.count(u) && ids.count(v))
      continue;
    if (ids.count(u)) {
      if (incomingCost[v] < 0) incomingCost[v] = 1.0;
      incomingCost[v] *= edgeInfo[index].cost;
    } else if (ids.count(v)) {
      if (incomingCost[u] < 0) incomingCost[u] = 1.0;
      incomingCost[u] *= edgeInfo[index].cost;
    } else {
      edgeInfo[currPtr++] = edgeInfo[index];
    }
  }
  for (unsigned index = 0; index != n; ++index) {
    if (incomingCost[index] > 0) {
      assert(index != repr);
      edgeInfo[currPtr++] = {incomingCost[index], {index, repr}};
    }
  }

  // Update the number of edges and resize the edegs.
  m = currPtr;
  edgeInfo.resize(m + n);

  // Update the open legs.
  for (auto u : ids) {
    if (u == repr) continue;
    openLegSize[repr] *= openLegSize[u];
  }

  // And init the graph structure.
  initGraphStructure();
#endif
}

template <class BitSet>
void TensorNetwork<BitSet>::prepareForOptimization()
// Prepare for optimization.
{
  assert(!getTreeViewStatus());
  vertexLegs.resize(n);
  vertexSizes.resize(n);
  for (unsigned index = 0; index != n; ++index) {
    vertexLegs[index] = collectVertexLegs(index);
    vertexSizes[index] = computeLegDimProduct(vertexLegs[index]);
  }
}

template <class BitSet>
bool TensorNetwork<BitSet>::isTree() const
// Checks whether the tensor network is a tree.
{
  // Check whether it is connected and `m = n - 1`.
  BitSet tmp;
  reach(0, tmp, BitSet::fill(n));
  return (tmp.size() == n) && (edgeInfo.size() == n - 1);
}

template <class BitSet>
EdgeInfo TensorNetwork<BitSet>::fetchEdge(unsigned edgeId) const
// Fetch the edge `edgeId`.
{
  // Tree?
  if (getTreeViewStatus()) {
    assert(!!treeView);
    return treeView->fetchEdge(edgeId);
  }
  assert(edgeId < m);
  return edgeInfo[edgeId];
}

template <class BitSet>
BitSet TensorNetwork<BitSet>::computeNeighbors(typename BitSet::arg_type s, typename BitSet::arg_type x) const {
  BitSet result;
  for (unsigned index = 0, limit = edgeInfo.size(); index != limit; ++index) {
    auto [u, v] = edgeInfo[index].edge;
    auto l = BitSet({u}), r = BitSet({v});
      
    if (l.isSubsetOf(s) && (!r.doesIntersectWith(s + x)) && (!r.doesIntersectWith(result)))
      result.insert(*r.begin());
    if (r.isSubsetOf(s) && (!l.doesIntersectWith(s + x)) && (!l.doesIntersectWith(result)))
      result.insert(*l.begin());
  }
  return result;
}

template <class BitSet>
bool TensorNetwork<BitSet>::reaches(unsigned u, BitSet set) const {
  for (unsigned pos = adj[u]; pos; pos = list[pos].next) {
    if (set.count(list[pos].v)) return true;
  }
  return false;
}

template <class BitSet>
long double TensorNetwork<BitSet>::computeContractionCost(BitSet set1, BitSet set2) const
// Compute cost of contraction between `set1` and `set2`.
{
  auto e1 = collectOpenLegs(set1), e2 = collectOpenLegs(set2);
  return computeLegDimProduct(e1 + e2);
}

template <class BitSet>
BitSet TensorNetwork<BitSet>::collectVertexLegs(unsigned u, bool strict) const
// Collect the vertex legs of node `u`.
{
  // Tree?
  assert(u < n);
  if (getTreeViewStatus()) {
    assert(!!treeView);
    return treeView->collectVertexLegs(u, strict);
  }
  
  // Collect the simple edges.
  BitSet ret;
  for (auto pos = adj[u]; pos; pos = list[pos].next)
    ret.insert(list[pos].edgeId);

  // Insert the open leg as well, in case we are not strict.
  // The `strict` mode is used in TensorIKKBZ to differentiate between virtual legs and the *actual* open legs.
  if (!strict)
    ret.insert(m + u);
  return ret;
}

template <class BitSet>
BitSet TensorNetwork<BitSet>::collectOpenLegs(BitSet set) const
// Collect the open legs of `set`.
{
  BitSet ret;
  for (auto elem : set)
    ret ^= collectVertexLegs(elem);
  return ret;
}

template <class BitSet>
long double TensorNetwork<BitSet>::computeLegDimProduct(BitSet set) const
// Compute the product of the leg dimensions.
{
  if (getTreeViewStatus()) {
    assert(!!treeView);
    return treeView->computeLegDimProduct(set);
  }
  
  long double ret = 1.0;
  for (auto edgeIndex : set) {
    assert(edgeIndex < edgeInfo.size());
    ret *= edgeInfo[edgeIndex].cost;
  }
  return ret;
}

template <class BitSet>
bool TensorNetwork<BitSet>::isConnected(BitSet set, bool verbose) const
// Check whether the tensors in `set` are connected.
{
  BitSet cum;
  reach(set.front(), cum, set);
  return set.isSubsetOf(cum);
}

template <class BitSet>
bool TensorNetwork<BitSet>::isRangeConnected(unsigned i, unsigned j, const std::vector<unsigned>& linearSol) const
// Check whether range `[i, j]` is connected.
{
  // TODO: improve this.
  BitSet set;
  for (unsigned k = i; k <= j; ++k)
    set.insert(linearSol[k]);
  return isConnected(set);
}

template <class BitSet>
long double TensorNetwork<BitSet>::computeLinearCost(const std::vector<unsigned>& sol) const
// Compute the cost of a linear solution.
{
  // TODO: when rotating, make sure we always come back to tree view.
  long double ret = 0;
  auto currEdgeSet = collectVertexLegs(sol.front());
  for (unsigned index = 1, limit = sol.size(); index != limit; ++index) {
    auto vertexEdges = collectVertexLegs(sol[index]);
    ret += computeLegDimProduct(currEdgeSet + vertexEdges);
    currEdgeSet ^= vertexEdges;
  }

  // And return.
  return ret;
}

template <class BitSet>
long double TensorNetwork<BitSet>::computeBushyCost(const std::vector<RangeNode>& sol) const
// Compute the cost of a bushy solution.
{
  assert(!getTreeViewStatus());
  std::function<std::pair<long double, BitSet>(unsigned)> rec = [&](unsigned index) -> std::pair<long double, BitSet> {
    if (sol[index].left == NIL) {
      assert(sol[index].right == NIL);
      return {0, {sol[index].nodeIndex}};
    }
    auto [lc, ls] = rec(sol[index].left);
    auto [rc, rs] = rec(sol[index].right);
    auto contractionCost = computeContractionCost(ls, rs);
    assert(!(ls & rs));
    return {lc + rc + contractionCost, ls + rs};
  };
  return rec(sol.size() - 1).first;
}