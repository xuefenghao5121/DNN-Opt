/// @file aligned_alloc.cpp
/// Cache-line-aligned memory allocation with huge page support.

#include "dnnopt/aligned_alloc.h"

#include <cstdlib>
#include <cstdio>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace dnnopt {

void* aligned_malloc(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) {
        fprintf(stderr, "aligned_malloc failed: size=%zu align=%zu\n",
                size, alignment);
        return nullptr;
    }
    return ptr;
}

void* aligned_malloc_huge(size_t size, size_t alignment) {
    if (size == 0) return nullptr;

#ifdef __linux__
    // For large allocations, try MAP_HUGETLB (2MB huge pages)
    if (size >= kHugePageThreshold) {
        // Round up to 2MB boundary for huge pages
        size_t huge_size = (size + (2 * 1024 * 1024 - 1)) & ~(size_t)(2 * 1024 * 1024 - 1);
        void* p = mmap(nullptr, huge_size,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                       -1, 0);
        if (p != MAP_FAILED) {
            return p;
        }
        // Fallback: try transparent huge pages via mmap + madvise
        p = mmap(nullptr, huge_size,
                 PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS,
                 -1, 0);
        if (p != MAP_FAILED) {
#ifdef MADV_HUGEPAGE
            madvise(p, huge_size, MADV_HUGEPAGE);
#endif
            return p;
        }
    }
#endif

    // Fallback to regular aligned allocation
    return aligned_malloc(size, alignment);
}

void aligned_free(void* ptr) {
    free(ptr);
}

void aligned_free_huge(void* ptr, size_t size) {
    if (!ptr) return;

#ifdef __linux__
    if (size >= kHugePageThreshold) {
        // This was likely allocated via mmap — try munmap
        size_t huge_size = (size + (2 * 1024 * 1024 - 1)) & ~(size_t)(2 * 1024 * 1024 - 1);
        if (munmap(ptr, huge_size) == 0) return;
        // If munmap fails, it was allocated via posix_memalign — fall through
    }
#else
    (void)size;
#endif

    free(ptr);
}

}  // namespace dnnopt
