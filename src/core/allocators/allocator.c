#include <ww/allocators/allocator.h>
#include <assert.h>

static inline void assert_allocator(WwAllocator allocator) {
    assert(allocator.ptr);
    assert(allocator.vtable);
    assert(allocator.vtable->alloc);
    assert(allocator.vtable->free);
}

WwAllocationResult _ww_allocator_alloc(WwAllocator self, usize size, const char* file, i32 line) {
    assert_allocator(self);
    return self.vtable->alloc(self.ptr, size, file, line);
}

void _ww_allocator_free(WwAllocator self, void* ptr, const char* file, i32 line) {
    assert_allocator(self);
    self.vtable->free(self.ptr, ptr, file, line);
}