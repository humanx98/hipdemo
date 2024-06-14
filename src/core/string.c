#include <ww/string.h>

void ww_string_deinit(WwString* self) {
    assert(self);
    ww_darray_deinit(&self->darr);
}

b8 ww_string_ensure_total_capacity_precise(WwString* self, usize new_capacity) {
    assert(self);
    return ww_darray_ensure_total_capacity_precise(&self->darr, new_capacity);
}

b8 ww_string_ensure_total_capacity(WwString* self, usize new_capacity) {
    assert(self);
    return ww_darray_ensure_total_capacity(&self->darr, new_capacity);
}

b8 ww_string_resize(WwString* self, usize new_len) {
    assert(self);
    return ww_darray_resize(&self->darr, new_len);
}

void ww_string_resize_assume_capacity(WwString* self, usize new_len) {
    assert(self);
    ww_darray_resize_assume_capacity(&self->darr, new_len);
}

b8 ww_string_append(WwString* self, const char* chars, usize len) {
    assert(self);
    return ww_darray_append_many(&self->darr, chars, len);
}

void ww_string_append_assume_capacity(WwString* self, const char* chars, usize len) {
    assert(self);
    ww_darray_append_many_assume_capacity(&self->darr, chars, len);
}

usize ww_string_len(const WwString* self) {
    assert(self);
    return ww_darray_len(&self->darr);
}
