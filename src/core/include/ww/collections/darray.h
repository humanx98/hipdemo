#pragma once

#include <ww/prim_types.h>
#include <ww/allocators/allocator.h>

typedef struct _WwDArray {
    WwAllocator allocator;
    void* ptr;
    usize capacity;
    usize len;
    usize elem_size;
} _WwDArray; 

void ww_darray_deinit(_WwDArray* self);
b8 __ww_must_check ww_darray_ensure_total_capacity_precise(_WwDArray* self, usize new_capacity);
b8 __ww_must_check ww_darray_ensure_total_capacity(_WwDArray* self, usize new_capacity);
b8 __ww_must_check ww_darray_resize(_WwDArray* self, usize new_len);
void ww_darray_resize_assume_capacity(_WwDArray* self, usize new_len);
b8 __ww_must_check ww_darray_append_many(_WwDArray* self, const void* elems, usize n);
void ww_darray_append_many_assume_capacity(_WwDArray* self, const void* elems, usize n);
usize __ww_must_check ww_darray_elem_size(const _WwDArray* self);
usize __ww_must_check ww_darray_capacity(const _WwDArray* self);
usize __ww_must_check ww_darray_len(const _WwDArray* self);
void* __ww_must_check ww_darray_ptr(const _WwDArray* self); 
void* __ww_must_check _ww_darray_get(const _WwDArray* self, usize index, const char* file, i32 line);

#define WwDArray(type) _WwDArray
#define ww_darray_init_with_elem_size(alloc, el_sz) (WwDArray(type)){ .allocator = alloc, .elem_size = el_sz }
#define ww_darray_init(allocator, type) ww_darray_init_with_elem_size(allocator, sizeof(type))
#define ww_darray_get_void_ref(self, index) _ww_darray_get(self, index, __FILE__, __LINE__) 
#define ww_darray_get_ref(self, type, index) (type*)ww_darray_get_void_ref(self, index) 
#define ww_darray_get(self, type, index) *ww_darray_get_ref(self, type, index) 
#define ww_darray_foreach_by_ref(self, type, var) for (type* var = (type*)(self)->ptr; var < (type*)(self)->ptr + (self)->len; ++var) 
#define ww_darray_append(self, elem) ww_darray_append_many(self, &elem, 1)
#define ww_darray_append_assume_capacity(self, elem) ww_darray_append_many_assume_capacity(self, &elem, 1)
