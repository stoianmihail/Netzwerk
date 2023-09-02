#ifndef H_BitSet
#define H_BitSet

#include <vector>
#include <cstdint>
#include <cassert>
#include <string>
#include <iostream>

// Shift the value.
#define SHIFT(x) ((x < half) ? (x) : (x - half))

// Select the corresponding set.
#define SET(x) ((x < half) ? set1 : set2)

// Apply function on the corresponding set.
#define applyFn(fn) (SET(v).fn(SHIFT(v)))

// Functions, whose results are combined by `comb`.
#define applyUnaryFn(fn, comb) (set1.fn() comb set2.fn())
#define applyBinaryFn(fn, comb) (set1.fn(o.set1) comb set2.fn(o.set2))

// Operators.
#define applyUnaryOp(op) (BitSet(op set1, op set2))
#define applyBinaryOp(op) (BitSet(set1 op o.set1, set2 op o.set2))
#define applyUpdateOp(op) \
  (set1 op o.set1);       \
  (set2 op o.set2);       \

// Bool operators, whose results are combined by `comb`.
#define applyBoolOp(op, comb) ((set1 op o.set1) comb (set2 op o.set2))

// Operators with values.
#define applyVal(op, x) (BitSet((x < half) ? (set1 op x) : set1, (x >= half) ? (set2 op (x - half)) : set2))
#define applyUpdateVal(op, x) (SET(x) op SHIFT(x))

// Prime factors used in the hash function.
static constexpr uint64_t primes[] = {1, 10007, 10009, 10037, 10039, 10061, 10067, 10069, 10079, 10091, 10093, 10099, 10103, 10111, 10133, 10139, 10141, 10151, 10159, 10163, 10169};

template <unsigned N>
struct BitSet {
  static constexpr unsigned half = (1u << (N + 6 - 1));
  using bitset_t = BitSet<N - 1>;

  // Positions: 7654   3210
  // Structure: set2 | set1
  // `set1` represents the lower half of the bits.
  // `set2` represents the 
  bitset_t set1 = bitset_t(), set2 = bitset_t();
public:
  explicit BitSet(bitset_t set1, bitset_t set2) : set1(set1), set2(set2) {}
  BitSet() = default;
  BitSet(std::initializer_list<unsigned> elements) { for (auto e : elements) insert(e); }

  void clear() { set1.clear(); set2.clear(); }
  void insert(unsigned v) { assert(!count(v)); assert(v != -1); applyFn(insert); }
  bool count(unsigned v) const { return applyFn(count); }
  
  bool empty() const { return applyUnaryFn(empty, &&); }
  unsigned size() const { return applyUnaryFn(size, +); }
  unsigned min() const { assert(!empty()); return (!set1.empty()) ? set1.min() : (set2.min() + half); }
  unsigned max() const { assert(!empty()); return (!set2.empty()) ? (set2.max() + half) : set1.max(); }

  // Inclusion and intersection.
  bool isSubsetOf(BitSet o) const { return applyBinaryFn(isSubsetOf, &&); }
  bool doesIntersectWith(BitSet o) const { return applyBinaryFn(doesIntersectWith, ||); }

  using arg_type = BitSet<N>;

  // Unary.
  BitSet operator~() const { return applyUnaryOp(~); }
  BitSet operator-() const { return (~(*this)).increment(); }
  bool operator!() const { return empty(); }

  // Binary.
  BitSet operator&(BitSet o) const { return applyBinaryOp(&); }
  BitSet operator+(BitSet o) const { return applyBinaryOp(+); }
  BitSet operator-(BitSet o) const { return applyBinaryOp(-); }
  BitSet operator^(BitSet o) const { return applyBinaryOp(^); }
  BitSet operator>>(unsigned index) const {
    assert(index == 1);

    // Is `set2` empty?
    if (set2.empty()) {
      return BitSet(set1 >> 1, set2);
    } else if (set2.min() == 0) {
      // Does `set2` have bit 0 set? Then put it into `set1`.
      return BitSet((set1 >> 1) + (half - 1), set2 >> 1);
    } else {
      return BitSet(set1 >> 1, set2 >> 1);
    }
  }

  // Update.
  void operator&=(BitSet o) { applyUpdateOp(&=); }
  void operator+=(BitSet o) { applyUpdateOp(+=); }
  void operator-=(BitSet o) { applyUpdateOp(-=); }
  void operator^=(BitSet o) { applyUpdateOp(^=); }

  // Custom functions. Useful for expressions like `set + {index}`.
  BitSet operator+(unsigned index) const {
    assert(!count(index));
    assert(index != -1);
    return applyVal(+, index);
  }
  BitSet operator-(unsigned index) const {
    assert(count(index));
    assert(index != -1);
    return applyVal(-, index);
  }
  void operator+=(unsigned index) {
    assert(!count(index));
    assert(index != -1);
    applyUpdateVal(+=, index);
  }
  void operator-=(unsigned index) {
    assert(count(index));
    assert(index != -1);
    applyUpdateVal(-=, index);
  }
  
  // Increment. Note: this is literally `set + 1`, i.e., we do not simply insert `1` into `set`.
  BitSet increment() {
    // Corner case, which also occurs for BitSet64.
    if (size() == 2 * half) return BitSet(bitset_t::fill(0), bitset_t::fill(0));

    // Is the lower set full?
    if (set1.size() == half) return BitSet(bitset_t::fill(0), set2.increment());

    // Otherwise, increment it.
    return BitSet(set1.increment(), set2);
  }

  // Decrement. Note: this is literally `set - 1`, i.e., we do not simply remove `1` from `set`.
  BitSet decrement() {
    if (!set1.empty()) return BitSet(set1.decrement(), set2);

    assert(!set2.empty());
    return BitSet(bitset_t::fill(half), set2.decrement());
  }

  // Build the first `n` bits.
  static BitSet fill(unsigned n) {
    assert(n <= 2 * half);
    return BitSet(bitset_t::fill((n >= half) ? half : n), bitset_t::fill((n >= half) ? (n - half) : 0));
  }

  // Create the range `[begin, end[`.
  static BitSet range(unsigned begin, unsigned end) {
    assert(begin <= end);
    auto upper = BitSet::fill(end);
    auto lower = BitSet::fill(begin);
    auto ret = upper - lower;
    assert(ret.size() == end - begin);
    return ret;
  }

  class iterator {
  private:
    bitset_t set1, set2;

    constexpr iterator(bitset_t set1, bitset_t set2) : set1(set1), set2(set2) {}

    friend class BitSet;

    public:
    unsigned operator*() const { return BitSet(set1, set2).min(); }

    iterator& operator++() {
      if (!set1.empty())
        set1 &= (set1.decrement());
      else
        set2 &= (set2.decrement());
      return *this;
    }

    bool operator==(const iterator& o) const { return applyBoolOp(==, &&); }
    bool operator!=(const iterator& o) const { return applyBoolOp(!=, ||); }
  };
  
  iterator begin() const { return iterator(set1, set2); }
  iterator end() const { return iterator(bitset_t::fill(0), bitset_t::fill(0)); }
  unsigned front() const { return *begin(); }

  class reverse_iterator {
    private:
    bitset_t set1, set2;

    constexpr reverse_iterator(bitset_t set1, bitset_t set2) : set1(set1), set2(set2) {}

    friend class BitSet;

    public:
    unsigned operator*() const { return BitSet(set1, set2).max(); }
    
    reverse_iterator& operator++() {
      if (!set2.empty())
        set2 -= set2.max();
      else
        set1 -= set1.max();
      return *this;
    }

    bool operator==(const reverse_iterator& o) const { return applyBoolOp(==, &&); }
    bool operator!=(const reverse_iterator& o) const { return applyBoolOp(!=, ||); }
  };
  
  reverse_iterator rbegin() const { return reverse_iterator(set1, set2); }
  reverse_iterator rend() const { return reverse_iterator(bitset_t::fill(0), bitset_t::fill(0)); }
  
  class reverseorder_adapter {
    bitset_t set1, set2;

    public:
    constexpr reverseorder_adapter(bitset_t set1, bitset_t set2) : set1(set1), set2(set2) {}

    constexpr reverse_iterator begin() const { return reverse_iterator(set1, set2); }
    constexpr reverse_iterator end() const { return reverse_iterator(bitset_t::fill(0), bitset_t::fill(0)); }
  };
  
  reverseorder_adapter reverseorder() const { return reverseorder_adapter(set1, set2); }

  class subset_iterator {
  private:
    bitset_t current1, current2, total1, total2;

    public:
    subset_iterator(bitset_t current1, bitset_t current2, bitset_t total1, bitset_t total2) : current1(current1), current2(current2), total1(total1), total2(total2) {}

    BitSet operator*() const { return BitSet(current1, current2); }
    subset_iterator& operator++() {
      current1 = ((current1 + (~total1)).increment()) & total1;
      if (current1.empty())
        current2 = ((current2 + (~total2)).increment()) & total2;
      return *this;
    }

    bool operator==(const subset_iterator& o) const { return (current1 == o.current1) && (current2 == o.current2); }
    bool operator!=(const subset_iterator& o) const { return (current1 != o.current1) || (current2 != o.current2); }
  };

  class subsets_adapter {
    bitset_t set1, set2;

    subsets_adapter(bitset_t set1, bitset_t set2) : set1(set1), set2(set2) {}

    friend class BitSet;

  public:
    subset_iterator begin() const {
      bitset_t c1 = (set1 & (-set1)), c2 = (!c1.empty()) ? bitset_t::fill(0) : (set2 & (-set2));
      return subset_iterator(c1, c2, set1, set2);
    }
    subset_iterator end() const { return subset_iterator(bitset_t::fill(0), bitset_t::fill(0), set1, set2); }
  };

  subsets_adapter subsets() const { return subsets_adapter(set1, set2); }

  std::string rawString(unsigned offset = 0) const {
    if (empty())
      return "";
    std::string ret;
    if (!set1.empty()) {
      ret += set1.rawString();
      if (!set2.empty()) {
        ret += ", ";
        ret += set2.rawString(half);
      }
    } else {
      assert(!set2.empty());
      ret += set2.rawString(half);
    }
    return ret; 
  }

  std::string toString() const { return std::string("{" + rawString() + "}"); }
  friend std::ostream& operator<< (std::ostream& os, const BitSet& o) { os << o.toString(); return os; }

  bool operator==(BitSet o) const { return applyBoolOp(==, &&); }
  bool operator!=(BitSet o) const { return applyBoolOp(!=, ||); }
  bool operator<(BitSet o) const {
    return (set2 < o.set2) || ((set2 == o.set2) && (set1 < o.set1));
  }
  
  // The hash function.
  uint64_t hash(bool primeIndex = 0) {
    // Since we define each bitset recursively, each uint64_t the class splits into will receive a prime multiplicative factor.
    return set1.hash(2 * primeIndex) ^ set2.hash(2 * primeIndex + 1);
  }

  // Hasher.
  struct hasher {
    using argument_type = BitSet<N>;
    using result_type = std::size_t;
    inline result_type operator()(argument_type a) const noexcept {
      return a.hash();
    }
  };
};

template<>
struct BitSet<0> {
#define BitSet BitSet
  uint64_t set = 0;

public:
  explicit BitSet(uint64_t set) : set(set) {}
  BitSet() = default;
  BitSet(std::initializer_list<unsigned> elements) {
    for (auto e : elements)
      insert(e);
  }

  void clear() { set = 0; }
  constexpr void insert(unsigned v) { set |= 1ull << v; }
  bool count(unsigned v) const { return set & (1ull << v); }
  bool empty() const { return !set; }
  unsigned size() const { return __builtin_popcountll(set); }
  unsigned min() const { return __builtin_ctzll(set); }
  unsigned max() const { return 63 - __builtin_clzll(set); }

  // Inclusion and intersection.
  bool isSubsetOf(BitSet o) const { return (set & o.set) == set; }
  bool doesIntersectWith(BitSet o) const { return set & o.set; }

  // Unary.
  BitSet operator~() const { return BitSet(~set); }
  BitSet operator-() const { return BitSet(-set); }
  bool operator!() const { return empty(); }

  // Binary.
  BitSet operator&(BitSet o) const { return BitSet(set & o.set); }
  BitSet operator+(BitSet o) const { return BitSet(set | o.set); }
  BitSet operator-(BitSet o) const { return BitSet(set & (~o.set)); }
  BitSet operator^(BitSet o) const { return BitSet(set ^ o.set); }
  
  // Update.
  void operator&=(BitSet o) { set &= o.set; }
  void operator+=(BitSet o) { set |= o.set; }
  void operator-=(BitSet o) { set = set & (~o.set); }
  void operator^=(BitSet o) { set ^= o.set; }

  // Custom functions. Useful for expressions like `set + {index}`.
  BitSet operator+(unsigned index) const {
    assert(!count(index));
    return BitSet(set | (1ull << index));
  }
  BitSet operator-(unsigned index) const {
    assert(count(index));
    return BitSet(set ^ (1ull << index));
  }
  BitSet operator>>(unsigned index) const {
    assert(index == 1);
    return BitSet(set >> 1);
  }
  void operator+=(unsigned index) {
    assert(!count(index));
    set |= (1ull << index);
  }
  void operator-=(unsigned index) {
    assert(count(index));
    set ^= (1ull << index);
  }

  BitSet increment() {
    return BitSet(set + 1);
  }

  BitSet decrement() {
    assert(set);
    return BitSet(set - 1);
  }

  using arg_type = BitSet<0>;

  // Build the first `n` bits.
  static BitSet fill(unsigned n) {
    assert(n <= 64);
    return BitSet((n >= 64) ? (~0ull) : ((1ull << n) - 1));
  }

  // Create the range `[begin, end[`.
  static BitSet range(unsigned begin, unsigned end) {
    assert(begin <= end);
    auto upper = BitSet::fill(end);
    auto lower = BitSet::fill(begin);
    auto ret = upper - lower;
    assert(ret.size() == end - begin);
    return ret;
  }
  
  class iterator {
  private:
    uint64_t set;

    explicit iterator(uint64_t set) : set(set) {}

    friend class BitSet;

    public:
    unsigned operator*() const { return BitSet(set).min(); }
    iterator& operator++() { set &= (set - 1); return *this; }

    bool operator==(const iterator& o) const { return set == o.set; }
    bool operator!=(const iterator& o) const { return set != o.set; }
  };

  iterator begin() const { return iterator(set); }
  iterator end() const { return iterator(0); }
  unsigned front() const { return *begin(); }

  class reverse_iterator {
  private:
    uint64_t set;

    constexpr explicit reverse_iterator(uint64_t set) : set(set) {}

    friend class BitSet;

  public:
    unsigned operator*() const { return BitSet(set).max(); }
    reverse_iterator& operator++() { set ^= (1ull << operator*()); return *this; }

    bool operator==(const reverse_iterator& o) const { return set == o.set; }
    bool operator!=(const reverse_iterator& o) const { return set != o.set; }
  };

  reverse_iterator rbegin() const { return reverse_iterator(set); }
  reverse_iterator rend() const { return reverse_iterator(0); }
  
  class reverseorder_adapter {
  private:
    uint64_t set;

  public:
    constexpr explicit reverseorder_adapter(uint64_t set) : set(set) {}

    constexpr reverse_iterator begin() const { return reverse_iterator(set); }
    constexpr reverse_iterator end() const { return reverse_iterator(0); }
  };

  reverseorder_adapter reverseorder() const { return reverseorder_adapter(set); }

  class subset_iterator {
    private:
    uint64_t current, total;

    public:
    subset_iterator(uint64_t current, uint64_t total) : current(current), total(total) {}

    BitSet operator*() const { return BitSet(current); }
    subset_iterator& operator++() {
      current = ((current | (~total)) + 1) & total;
      return *this;
    }

    bool operator==(const subset_iterator& o) const { return current == o.current; }
    bool operator!=(const subset_iterator& o) const { return current != o.current; }
  };
  
  class subsets_adapter {
    uint64_t set;

    explicit subsets_adapter(uint64_t set) : set(set) {}

    friend class BitSet;

    public:
    subset_iterator begin() const { return subset_iterator(set & (-set), set); }
    subset_iterator end() const { return subset_iterator(0, set); }
  };

  subsets_adapter subsets() const { return subsets_adapter(set); }

  std::string rawString(unsigned offset = 0) const {
    if (!size())
      return "";
    std::string ret;
    auto iter = begin();
    for (unsigned index = 0, limit = size(); index != limit; ++index, ++iter) {
      ret += std::to_string(offset + (*iter));
      if (index != limit - 1)
        ret += ", ";
    }
    return ret;
  }

  std::string toString() const { return std::string("{" + rawString() + "}"); }
  friend std::ostream& operator<< (std::ostream& os, const BitSet& o) {
    os << o.toString();
    return os;
  }

  bool operator==(BitSet o) const { return set == o.set; }
  bool operator!=(BitSet o) const { return set != o.set; }
  bool operator<(BitSet o) const { return set < o.set; }
  
  // The hash function.
  uint64_t hash(unsigned primeIndex = 0) const {
    return std::hash<uint64_t>()(primes[primeIndex] * set);
  }

  // Hasher.
  struct hasher {
    inline std::size_t operator()(auto a) const noexcept {
      return a.hash();
    }
  };
#undef BitSet
};

using BitSet64 = BitSet<0>;
using BitSet128 = BitSet<1>;
using BitSet256 = BitSet<2>;
using BitSet512 = BitSet<3>;
using BitSet1024 = BitSet<4>;
using BitSet2048 = BitSet<5>;

#endif