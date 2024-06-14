#pragma once

#include <ww/prim_types.h>
#include <ww/defines.h>

WW_DEFINE_HANDLE(ww_allocator_ptr);

typedef struct WwAllocationResult {
    b8 failed;
    void* ptr;
} WwAllocationResult;

typedef struct ww_allocator_vtable {
    WwAllocationResult (*alloc)(ww_allocator_ptr, usize, const char*, i32);
    void (*free)(ww_allocator_ptr, void*, const char*, i32);
} ww_allocator_vtable;

typedef struct WwAllocator {
    ww_allocator_ptr ptr;
    const ww_allocator_vtable* vtable;
} WwAllocator;

WwAllocationResult __ww_must_check _ww_allocator_alloc(WwAllocator self, usize size, const char* file, i32 line);
void _ww_allocator_free(WwAllocator self, void* ptr, const char* file, i32 line);

#define ww_allocator_alloc(self, size) _ww_allocator_alloc(self, size, __FILE__, __LINE__)
#define ww_allocator_free(self, ptr) _ww_allocator_free(self, ptr, __FILE__, __LINE__)

#define ww_allocator_alloc_type(self, type) \
    ({ WwAllocationResult _res = _ww_allocator_alloc(self, sizeof(type), __FILE__, __LINE__); \
       struct { b8 failed; type* ptr; } _generic_res = { .failed = _res.failed, .ptr = (type*)_res.ptr, }; \
        _generic_res;    })
    

#define WW_MAX(a, b) \
    ({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b);    \
    _MAX(_a, _b); })
