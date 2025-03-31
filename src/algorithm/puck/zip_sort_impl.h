// Copyright 2024 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



#ifndef SCANN_UTILS_ZIP_SORT_IMPL_H_
#define SCANN_UTILS_ZIP_SORT_IMPL_H_

#include <algorithm>
#include <array>
#include <type_traits>

namespace puck {
namespace zip_sort_internal {

enum : size_t { kInsertionSortSize = 20, kSelectionSortSize = 15 };

class DefaultComparator {
 public:
  template <typename T>
  bool operator()(const T& a, const T& b) const {
    return a < b;
  }
};

template <typename T, typename Comparator>
inline size_t MedianOf3(Comparator comp, T iter, size_t index1, size_t index2) {
  const size_t mid_index = index1 + (index2 - index1) / 2;
  const auto& mid = iter[mid_index];
  //::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(&mid);
  const auto& first = iter[index1];
  //::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(&first);
  const auto& last = iter[index2 - 1];
  //::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(&last);

  if (comp(mid, first)) {
    if (comp(last, mid)) {
      return mid_index;
    } else if (comp(first, last)) {
      return index1;
    } else {
      return index2 - 1;
    }
  } else {
    if (comp(mid, last)) {
      return mid_index;
    } else if (comp(last, first)) {
      return index1;
    } else {
      return index2 - 1;
    }
  }
}

template <typename T>
void PrefetchIter(T iter) {
  //::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(&(*iter));
}

template <typename T, typename Comparator>
size_t MedianOf9(Comparator comp, T iter, size_t index1, size_t index2) {
  //DCHECK_GE(index2 - index1, 9);

  std::array<T, 9> iterators;
  iterators[0] = iter + index1;
  PrefetchIter(iterators[0]);
  iterators[1] = iter + index1 + 1;
  PrefetchIter(iterators[1]);
  iterators[2] = iter + index1 + 2;
  PrefetchIter(iterators[2]);
  const size_t mid = index1 + (index2 - index1) / 2;
  iterators[3] = iter + mid - 1;
  PrefetchIter(iterators[3]);
  iterators[4] = iter + mid;
  PrefetchIter(iterators[4]);
  iterators[5] = iter + mid + 1;
  PrefetchIter(iterators[5]);
  iterators[6] = iter + index2 - 3;
  PrefetchIter(iterators[6]);
  iterators[7] = iter + index2 - 2;
  PrefetchIter(iterators[7]);
  iterators[8] = iter + index2 - 1;
  PrefetchIter(iterators[8]);

  for (size_t i = 0; i < 5; ++i) {
    auto to_swap = iterators.begin() + i;
    for (auto it = iterators.begin() + i + 1; it != iterators.end(); ++it) {
      if (comp(**it, **to_swap)) to_swap = it;
    }
    using std::swap;
    swap(iterators[i], *to_swap);
  }
  return iterators[4] - iter;
}

template <typename T, typename... U>
inline void ZipSwap(size_t index1, size_t index2, T begin) {
  using std::swap;
  swap(begin[index1], begin[index2]);
}

template <typename T, typename... U>
void ZipSwap(size_t index1, size_t index2, T begin, U... rest);

template <typename T, typename... U>
inline void ZipSwapRecursionHelper(size_t index1, size_t index2, T begin, T end,
                                   U... rest) {
  ZipSwap(index1, index2, begin, rest...);
}

template <typename T, typename... U>
inline void ZipSwap(size_t index1, size_t index2, T begin, U... rest) {
  using std::swap;
  swap(begin[index1], begin[index2]);
  ZipSwapRecursionHelper(index1, index2, rest...);
}

template <typename Comparator, typename T, typename... U>
void SelectionZipSort(Comparator comp, size_t index1, size_t index2, T begin,
                      U... rest) {
  const T end = begin + index2;
  for (size_t i = index1; i + 1 < index2; ++i) {
    T to_swap = begin + i;
    for (T it = begin + i + 1; it != end; ++it) {
      if (comp(*it, *to_swap)) to_swap = it;
    }
    const size_t to_swap_index = to_swap - begin;
    ZipSwap(i, to_swap_index, begin, rest...);
  }
}

template <typename T>
using ConstRefIfNotPod = std::conditional_t<std::is_pod<std::decay_t<T>>::value,
                                       std::decay_t<T>, const std::decay_t<T>&>;

template <typename Comparator, typename T, typename... U>
inline size_t PivotPartition(Comparator comp, size_t index1, size_t index2,
                             T begin, U... rest) {
  const size_t median_index = MedianOf3(comp, begin, index1, index2);
  ZipSwap(median_index, index2 - 1, begin, rest...);
  using Elem = decltype(begin[0]);
  ConstRefIfNotPod<Elem> pivot = begin[index2 - 1];
  T less_iter = begin + index1 - 1;
  T greater_iter = begin + index2 - 1;

  while (true) {
    while (comp(*(++less_iter), pivot)) {
      //DCHECK(less_iter < begin + index2);
    }

    while (comp(pivot, *(--greater_iter))) {
      //DCHECK(greater_iter >= begin);
    }

    if (less_iter < greater_iter) {
      ZipSwap(less_iter - begin, greater_iter - begin, begin, rest...);
    } else {
      break;
    }
  }

  const size_t pivot_swap_index = less_iter - begin;
  ZipSwap(index2 - 1, pivot_swap_index, begin, rest...);
  return pivot_swap_index;
}

template <typename Comparator, typename T, typename... U>
size_t PivotPartitionBranchOptimized(Comparator comp, size_t index1,
                                     size_t index2, T begin, U... rest) {
  const size_t range_start = index1, range_end = index2;
  const size_t median_index = (index2 - index1 >= 1000)
                                  ? MedianOf9(comp, begin, index1, index2)
                                  : MedianOf3(comp, begin, index1, index2);
  ZipSwap(median_index, index2 - 1, begin, rest...);
  using Elem = decltype(begin[0]);
  const size_t pivot_index = index2 - 1;

  ConstRefIfNotPod<Elem> pivot = begin[pivot_index];

  index2 -= 2;

  static constexpr ssize_t kNumSwapSlots = 32;
  std::array<size_t, kNumSwapSlots> left_swaps, right_swaps;
  ssize_t left_swaps_slot;
  ssize_t right_swaps_slot;
  for (;;) {
    left_swaps_slot = -kNumSwapSlots;
    right_swaps_slot = -kNumSwapSlots;
    do {
      left_swaps.end()[left_swaps_slot] = index1;
      right_swaps.end()[right_swaps_slot] = index2;

      left_swaps_slot += !comp(begin[index1++], pivot);
      right_swaps_slot += !comp(pivot, begin[index2--]);

      if (left_swaps_slot == 0) {
        if (right_swaps_slot == 0) break;
        while (index1 < index2) {
          right_swaps.end()[right_swaps_slot] = index2;
          if ((right_swaps_slot += !comp(pivot, begin[index2--])) == 0) break;
        }
        break;
      } else if (right_swaps_slot == 0) {
        while (index1 < index2) {
          left_swaps.end()[left_swaps_slot] = index1;
          if ((left_swaps_slot += !comp(begin[index1++], pivot)) == 0) break;
        }
        break;
      }
    } while (index1 < index2);

    const ssize_t num_swaps = std::min(kNumSwapSlots + left_swaps_slot,
                                       kNumSwapSlots + right_swaps_slot);
    for (ssize_t i = 0; i < num_swaps; ++i) {
      const size_t left_swap = left_swaps[i];
      const size_t right_swap = right_swaps[i];
      ZipSwap(left_swap, right_swap, begin, rest...);
    }

    if (index1 >= index2) break;

  }

  size_t pivot_swap_index = index1;

  pivot_swap_index +=
      (index1 == index2 && comp(begin[pivot_swap_index], pivot));

  for (; left_swaps_slot > right_swaps_slot; --left_swaps_slot) {
    --pivot_swap_index;
    const size_t left_swap = left_swaps.end()[left_swaps_slot - 1];

    ZipSwap(pivot_swap_index, left_swap, begin, rest...);
  }

  for (; right_swaps_slot > left_swaps_slot; --right_swaps_slot) {
    const size_t right_swap = right_swaps.end()[right_swaps_slot - 1];

    ZipSwap(pivot_swap_index, right_swap, begin, rest...);
    ++pivot_swap_index;
  }

  ZipSwap(pivot_swap_index, pivot_index, begin, rest...);
  return pivot_swap_index;
}

template <typename Comparator, typename T, typename... U>
void ZipNthElementImplBranchOptimized(Comparator comp, size_t n, T begin, T end,
                                      U... rest) {
  size_t index1 = 0;
  size_t index2 = end - begin;

  while (index2 - index1 > 3) {
    const size_t pivot_index =
        PivotPartitionBranchOptimized<Comparator, T, U...>(comp, index1, index2,
                                                           begin, rest...);
    if (pivot_index == n) {
      return;
    } else if (pivot_index < n) {
      index1 = pivot_index + 1;
    } else {
      index2 = pivot_index;
    }
  }

  SelectionZipSort<Comparator, T, U...>(comp, index1, index2, begin, rest...);
}

}  // namespace zip_sort_internal


}  // namespace research_scann

#endif
