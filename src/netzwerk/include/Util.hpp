#ifndef H_Util
#define H_Util

#include "Common.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <limits>
#include <filesystem>
#include <ctime>

namespace fs = std::filesystem;
using namespace std::chrono;

// enum class ContractionType : unsigned {
//   Linear,
//   General
// };

struct Sequence {
  int i, j;
};

// Sequence* convertLinearSolution(unsigned n, std::vector<unsigned>& sol) {
//   Sequence *ret = new Sequence[n - 1];
//   int curr = static_cast<int>(n);
//   for (unsigned index = 1, limit = n; index != limit; ++index) {
//     int node = static_cast<int>(sol[index]);//mapper[sol[index]];
//     if (index == 1) {
//       ret[index - 1] = {static_cast<int>(sol[0]), node};
//     } else {
//       ret[index - 1] = {curr++, node};
//     }
//   }
//   return ret;
// }

// Sequence* convertBushySolution(unsigned n, std::vector<RangeBushyNode>& sol) {
//   // TODO: comment function.
//   Sequence *ret = new Sequence[n - 1];
//   int currIndex = n;
//   std::function<int(unsigned)> build = [&](unsigned index) -> int {
//     if (sol[index].left == NIL) {
//       assert(sol[index].right == NIL);
//       return static_cast<int>(sol[index].nodeIndex);
//     }

//     auto l = build(sol[index].left);
//     auto r = build(sol[index].right);
//     ret[(currIndex++) - n] = {l, r};
//     return currIndex - 1;
//   };

//   auto tmp = build(sol.size() - 1);
//   assert(currIndex == 2 * n - 1);
//   return ret;
// }

// template <class BitSet>
// Sequence* convertBushySolution(unsigned n, CcpBushyNode<BitSet>* sol) {
//   // TODO: comment function.
//   Sequence *ret = new Sequence[n - 1];
//   int currIndex = n;
//   std::function<int(CcpBushyNode<BitSet>*)> build = [&](CcpBushyNode<BitSet>* plan) -> int {
//     if (plan->set.size() == 1) {
//       assert((plan->left == nullptr) && (plan->right == nullptr));
//       return static_cast<int>(*plan->set.begin());
//     }

//     auto l = build(plan->left);
//     auto r = build(plan->right);
//     ret[(currIndex++) - n] = {l, r};
//     return currIndex - 1;
//   };

//   auto tmp = build(sol);
//   assert(currIndex == 2 * n - 1);
//   return ret;
// }

// void debugBushyTree(std::vector<RangeBushyNode>& tree) {
//   for (unsigned index = 0, limit = tree.size(); index != limit; ++index) {
//     std::cerr << "index=" << index
//               << " split node=" << tree[index].nodeIndex
//               << " range=(" << tree[index].range.first << ", " << tree[index].range.second << ")"
//               << " left=" << tree[index].left << " right=" << tree[index].right << std::endl;
//   }
// }

static std::string indent(unsigned d) {
  std::string ret = "[";
  while (d--)
    ret += "*";
  ret += "]";
  return ret;
}

static bool isEqual(const char* str1, const char* str2) {
  return !strcmp(str1, str2);
}

static void debugSequence(unsigned n, Sequence* seq) {
  for (unsigned index = 0, limit = n - 1; index != limit; ++index) {
    std::cerr << "index=" << index << " -> (" << seq[index].i << ", " << seq[index].j << ")" << std::endl;
  }
}

static std::vector<std::string> parseDirectory(std::string dir) {
  std::vector<std::string> ret;
  for (const auto &entry : fs::directory_iterator(dir))
    ret.push_back(entry.path());
  return ret;
}

static std::pair<unsigned, unsigned> getInfo(std::string filepath) {
  auto filename = fs::path(filepath).filename().string().substr(4);

  auto split = [&](std::string str, char delim) {
    std::vector<std::string> ret = {};
    std::string curr = "";
    for (char c : str) {
      if (c == delim) {
        ret.push_back(curr);
        curr.clear();
      } else {
        curr += c;
      }
    }
    ret.push_back(curr);
    return ret;
  };

  auto tmp = split(filename, '-');
  auto size = std::stoi(tmp[0]);
  auto index = std::stoi(tmp[1]);
  return {size, index};
}

class Timer {
public:
  Timer(std::string approach, unsigned size = 0)
  : approach_(approach),
    size_(size),
    duration_(0),
    isStopped(false),
    timeout(std::numeric_limits<double>::max()) {
      start();
    }

  void setTimeout(double timeout) {
    // Set timeout in seconds.
    this->timeout = timeout;
  }

  void start() {
    start_ = ::high_resolution_clock::now();
    duration_ = 0;
  }

  void stop() {
    isStopped = true;
    stop_ = ::high_resolution_clock::now();
    duration_ = duration_cast<microseconds>(stop_ - start_).count();
    ++counter_;
  }

  void debug() {
    if (!isStopped) stop();
    std::cerr << "Approach: " << approach_ << " took " << duration_ / 1e3 << " ms" << std::endl;
  }

  bool isTimeout() const {
    auto tmp = ::high_resolution_clock::now();
    return (duration_cast<microseconds>(tmp - start_).count() > timeout * 1e6);
  }

  void merge(const Timer& o) {
    reports.push_back({o.size_, 1.0 * o.duration_ / o.counter_ / 1000});
  }

  void summary() {}

  void flush() {
    const std::time_t now = std::time(nullptr);
    const std::tm ct = *std::localtime( std::addressof(now) ) ;
    std::string time = std::to_string(ct.tm_hour) + "-" + std::to_string(ct.tm_min) + "-" + std::to_string(ct.tm_sec);

    auto filename = "../results/" + approach_ + "_" + time + ".out";
    std::ofstream out(filename);
    assert(out.is_open());
    for (const auto& [s, t] : reports) {
      out << s << ": " << t << std::endl;
    }
  }

private:
  double timeout;
  high_resolution_clock::time_point start_, stop_;
  std::string approach_;
  unsigned size_;
  double duration_ = 0.0;
  unsigned counter_ = 0;
  bool isStopped = false;
  std::vector<std::tuple<unsigned, double>> reports;
};

#endif