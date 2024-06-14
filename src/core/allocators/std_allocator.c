#include "ww/allocators/allocator.h"
#include <ww/allocators/std_allocator.h>
#include <stdlib.h>

typedef struct StdAllocatorSelf {
} StdAllocatorSelf;

static StdAllocatorSelf self = {};

static WwAllocationResult __ww_must_check std_allocator_alloc(ww_allocator_ptr self, usize size, const char* file, i32 line) {
    void* ptr = malloc(size);
    return (WwAllocationResult){
        .failed = ptr == NULL,
        .ptr = ptr
    };
}

static void std_allocator_free(ww_allocator_ptr self, void* ptr, const char* file, i32 line) {
    free(ptr);
}


inline WwAllocator ww_std_allocator() {
    const static ww_allocator_vtable vtable = {
        .alloc = std_allocator_alloc,
        .free = std_allocator_free
    };

    return (WwAllocator) {
        .ptr = (ww_allocator_ptr)&self,
        .vtable = &vtable,
    };
}
