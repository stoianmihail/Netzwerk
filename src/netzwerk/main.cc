#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iomanip>

#include "include/TensorNetwork.hpp"
#include "include/Optimizer.hpp"

template <class BitSet>
std::pair<long double, Sequence*> runAlgorithm(std::string tn_filepath, std::string ttn_filepath, const char* algorithm, unsigned numThreads) {
  auto ttn = TensorNetwork<BitSet>(ttn_filepath);
  auto tn = TensorNetwork<BitSet>(tn_filepath);
  tn.setTreeView(ttn);
  auto opt = Optimizer<BitSet>(tn);
  const auto plan = opt.optimize(algorithm, numThreads);
  return {plan->totalCost, opt.translatePlanToSequence(plan)};
}

int main(int argc, char** argv) {
  if ((argc != 2) && (argc != 4)) {
    std::cerr << "Usage: " << argv[0] << " <approach:string> [<graph_input:file> <tree_input:file>]" << std::endl;
    exit(-1);
  }

  const char* approach = static_cast<const char*>(argv[1]);
  std::string graph_filename, tree_filename;
  unsigned numThreads = 1;
  if (argc != 4) {
    assert(argc == 2);
    graph_filename = "../graph.in";
    tree_filename = "../tree.in";
  } else {
    graph_filename = argv[2];
    tree_filename = argv[3];
  }

  std::cerr << std::fixed << std::setprecision(2);

  auto tn_filepath = std::string(graph_filename);
  auto ttn_filepath = std::string(tree_filename);

  std::cerr << "tn_filepath=" << tn_filepath << " ttn_filepath=" << ttn_filepath << std::endl;

  std::ifstream input(tn_filepath);
  assert(input.is_open());
  unsigned n, m;
  input >> n >> m;
  input.close();

  std::cerr << "n=" << n << " m=" << m << std::endl;

  std::pair<long double, Sequence*> ret;
  auto safe_size = m + n;
  if (safe_size <= 64) {
    ret = runAlgorithm<BitSet64>(tn_filepath, ttn_filepath, approach, numThreads);
  } else if (safe_size <= 128) {
    ret = runAlgorithm<BitSet128>(tn_filepath, ttn_filepath, approach, numThreads);
  } else if (safe_size <= 256) {
    ret = runAlgorithm<BitSet256>(tn_filepath, ttn_filepath, approach, numThreads);
  } else if (safe_size <= 512) {
    ret = runAlgorithm<BitSet512>(tn_filepath, ttn_filepath, approach, numThreads);
  } else if (safe_size <= 1024) {
    ret = runAlgorithm<BitSet1024>(tn_filepath, ttn_filepath, approach, numThreads);
  } else if (safe_size <= 2048) {
    ret = runAlgorithm<BitSet2048>(tn_filepath, ttn_filepath, approach, numThreads);  
  } else {
    assert(0);
  }

  // debugSequence(n, ret.second);
  std::cout << "[" << approach << "] cost=" << ret.first << std::endl;
}