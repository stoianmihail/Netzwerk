#ifndef H_Common
#define H_Common

#include <unordered_map>
#include <limits>
#include <cmath>

#include "BitSet.hpp"

#define DEBUG_IKKBZ 0
#define DEBUG_LINDP 0

// TODO: use hopscotch_map
template <class BitSet, typename T>
using BitSetHashMap = std::unordered_map<BitSet, T, typename BitSet::hasher>;

static constexpr unsigned NIL = std::numeric_limits<unsigned>::max();
static constexpr long double kInf = std::numeric_limits<long double>::max();

struct Cell {
  unsigned v, edgeId;
  unsigned next;
};

struct EdgeInfo {
  long double cost;
  std::pair<unsigned, unsigned> edge;
};

struct RangeNode {
  unsigned nodeIndex;
  unsigned left;
  unsigned right;
};

static int sign(double x) {
  if (x > 1e-6) return +1;
  if (x < -1e6) return -1;
  return 0;
}

static uint64_t randomState = 123;

template <class T>
static bool isClose(T a, T b) {
  return std::fabs(a - b) < std::numeric_limits<T>::epsilon();
}

template <class T>
static bool isLessThan(T a, T b) {
  return a + std::numeric_limits<T>::epsilon() < b;
}

template <class T>
static bool isGreaterThan(T a, T b) {
  return b + std::numeric_limits<T>::epsilon() < a;
}

template <class T>
static bool isLessOrEqualThan(T a, T b) {
  return (a + std::numeric_limits<T>::epsilon() < b) || isClose(a, b);
}

static uint64_t nextRandom() {
  uint64_t x = randomState;
  x = x ^ (x >> 12);
  x = x ^ (x << 25);
  x = x ^ (x >> 27);
  randomState = x;
  return x * 0x2545F4914F6CDD1Dull;
}

static unsigned randomInt(unsigned lower, unsigned upper) {
  return (upper <= lower) ? lower : (lower + (nextRandom() % (upper - lower + 1)));
}

static std::string debugVector(const std::vector<unsigned>& vs) {
  std::string ret = "{";
  unsigned index = 0;
  for (auto elem : vs) {
    if (index++)
      ret += ", ";
    ret += std::to_string(elem);
  }
  ret += "}";
  return ret;
}

#endif