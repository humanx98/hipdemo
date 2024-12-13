#include <ww/collections/darray.h>
#include <ww/exit.h>
#include <string.h>

static inline usize __ww_must_check grow_capacity(usize current, usize minimum);
static inline void memory_copy(void* dst, const void* src, usize len, usize elem_size);

static inline void assert_darray(const _WwDArray* self) {
    assert(self);
    assert(self->elem_size > 0);
}

usize grow_capacity(usize current, usize minimum) {
    usize new = current;
    while (true) {
        new += new / 2 + 8;
        if (new >= minimum)
            return new;
    }
}

void memory_copy(void* dst, const void* src, usize len, usize elem_size) {
    memcpy(dst, src, elem_size * len);
}

void ww_darray_deinit(_WwDArray* self) {
    assert_darray(self);

    if (self->ptr) {
        ww_allocator_free(self->allocator, self->ptr);
        *self = (_WwDArray){0};
    }
}

b8 ww_darray_ensure_total_capacity_precise(_WwDArray* self, usize new_capacity) {
    assert_darray(self);

    if (self->capacity >= new_capacity) {
        return true;
    }

    WwAllocationResult alloc_result = ww_allocator_alloc(self->allocator, self->elem_size * new_capacity);
    if (alloc_result.failed) {
        return false;
    }

    if (self->ptr) {
        memory_copy(alloc_result.ptr, self->ptr, self->len, self->elem_size);
        ww_allocator_free(self->allocator, self->ptr);
    }
    self->ptr = alloc_result.ptr;
    self->capacity = new_capacity;
    return true;
}

b8 ww_darray_ensure_total_capacity(_WwDArray* self, usize new_capacity) {
    assert_darray(self);

    if (self->capacity >= new_capacity) {
        return true;
    }

    usize better_capacity = grow_capacity(self->capacity, new_capacity);
    if (ww_darray_ensure_total_capacity_precise(self, better_capacity)) {
        return true;
    }
    return ww_darray_ensure_total_capacity_precise(self, new_capacity);
}

b8 ww_darray_resize(_WwDArray* self, usize new_len) {
    assert_darray(self);

    if (ww_darray_ensure_total_capacity(self, new_len)) {
        ww_darray_resize_assume_capacity(self, new_len);
        return true;
    }

    return false;
}

void ww_darray_resize_assume_capacity(_WwDArray* self, usize new_len) {
    assert_darray(self);
    assert(new_len <= self->capacity);    

    self->len = new_len;
}

b8 ww_darray_append_many(_WwDArray* self, const void* elems, usize n) {
    assert_darray(self);

    if (n == 0) {
        return true;
    }

    if (ww_darray_ensure_total_capacity(self, self->len + n)) {
        ww_darray_append_many_assume_capacity(self, elems, n);
        return true;
    }

    return false;
}

void ww_darray_append_many_assume_capacity(_WwDArray* self, const void* elems, usize n) {
    assert_darray(self);
    usize newlen = self->len + n;
    assert(newlen <= self->capacity);

    usize prevlen = self->len;
    self->len = newlen;
    memory_copy(ww_darray_get_void_ref(self, prevlen), elems, n, self->elem_size);
}

inline usize ww_darray_elem_size(const _WwDArray* self) {
    assert_darray(self);
    return self->elem_size;
}

inline usize ww_darray_capacity(const _WwDArray* self) {
    assert_darray(self);
    return self->capacity;
}

inline usize ww_darray_len(const _WwDArray* self) {
    assert_darray(self);
    return self->len;
}

inline void* ww_darray_ptr(const _WwDArray* self) {
    assert_darray(self);
    return self->ptr;
}

inline void* _ww_darray_get(const _WwDArray* self, usize index, const char* file, i32 line) {
    assert_darray(self);
#ifndef NDEBUG
    if (index >= self->len) {
        WW_EXIT_WITH_MSG("%s:%d: attemp to acces an item out of range, elem index = %zu, WwDArray len = %zu\n", file, line, index, self->len);
    }
#endif
    return self->ptr + index * self->elem_size;
}
