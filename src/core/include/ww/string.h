#pragma once

#include <ww/collections/darray.h>

typedef struct WwString {
    WwDArray(char) darr;
} WwString;

void ww_string_deinit(WwString* self);
b8 __ww_must_check ww_string_ensure_total_capacity_precise(WwString* self, usize new_capacity);
b8 __ww_must_check ww_string_ensure_total_capacity(WwString* self, usize new_capacity);
b8 __ww_must_check ww_string_resize(WwString* self, usize new_len);
void ww_string_resize_assume_capacity(WwString* self, usize new_len);
b8 __ww_must_check ww_string_append(WwString* self, const char* chars, usize len);
void ww_string_append_assume_capacity(WwString* self, const char* chars, usize len);
usize __ww_must_check ww_string_len(const WwString* self);

#define ww_string_init(allocator) (WwString) { .darr = ww_string_init(allocator, char); }
