#include <string.h>
#include <ww/collections/hash_map.h>
#include <ww/exit.h>

static inline __ww_must_check i32* ww_hash_map_get_bucket(_WwHashMap* self, usize hash_code);
static inline __ww_must_check usize grow_capacity(usize current, usize minimum);

void ww_hash_map_deinit(_WwHashMap* self) {
    assert(self);
    ww_darray_deinit(&self->buckets);
    ww_darray_deinit(&self->entries);
}

b8 ww_hash_map_ensure_capacity(_WwHashMap* self, usize capacity) {
    usize buckets_old_len = ww_darray_len(&self->buckets);
    if (buckets_old_len >= capacity) {
        return true;
    }

    if (!ww_darray_ensure_total_capacity_precise(&self->buckets, capacity)) {
        return false;
    }

    if (!ww_darray_ensure_total_capacity_precise(&self->entries, capacity)) {
        return false; 
    }
    
    usize bucket_elem_size = ww_darray_elem_size(&self->buckets);
    usize entry_elem_size = ww_darray_elem_size(&self->entries);
    ww_darray_resize_assume_capacity(&self->buckets, capacity);
    ww_darray_resize_assume_capacity(&self->entries, capacity);
    memset(ww_darray_get_void_ref(&self->buckets, buckets_old_len), 0, (capacity - buckets_old_len) * bucket_elem_size);
    memset(ww_darray_get_void_ref(&self->entries, buckets_old_len), 0, (capacity - buckets_old_len) * entry_elem_size);
    return true;
}

b8 _ww_hash_map_get(const _WwHashMap* self, WwHashMapKey key) {
    assert(self);
    return false;
}

b8 _ww_hash_map_put(_WwHashMap* self, WwHashMapKey key, WwHashMapValue value) {
    assert(self);
    WwHashMapGerOrPutResult gop = _ww_hash_map_get_or_put(self, key);
    if (gop.failed) {
        return false;
    }

    assert(self->value_size == value.size);
    memcpy(gop.value, value.ptr, value.size);
    return true;
}

WwHashMapGerOrPutResult _ww_hash_map_get_or_put(_WwHashMap* self, WwHashMapKey key) {
    assert(self);

    if (ww_darray_capacity(&self->buckets) == 0 && !ww_hash_map_ensure_capacity(self, 8)) {
        return (WwHashMapGerOrPutResult){ .failed = true };
    }

    u32 hash_code = self->get_hash(key.ptr);
    usize collision_count = 0;
    i32* bucket = ww_hash_map_get_bucket(self, hash_code);
    i32 i = *bucket - 1;

    while ((usize)i < ww_darray_len(&self->entries)) {
        void* entry = ww_darray_get_void_ref(&self->entries, i);
        WwEntryHeader* entry_header = entry;
        void* entry_key = entry + sizeof(WwEntryHeader);
        void* entry_value = entry_key + self->key_size;
        if (entry_header->hash_code == hash_code && self->compare_keys(entry_key, key.ptr)) {
            return (WwHashMapGerOrPutResult) {
                .key = entry_key,
                .value = entry_value,
                .found_existing = true,
            };
        }

        i = entry_header[i].next;
        collision_count++;

        if (collision_count > ww_darray_len(&self->entries)) {
            //  Break out of the loop and throw, rather than looping forever.
            WW_EXIT_WITH_MSG("[WwHashMap] The chain of entries forms a loop; which means a concurrent update has happened.\n");
        }
    }

    i32 index = 0;
    if (false) {
        // TODO: implement removing items
    } else {
        if (self->count == ww_darray_len(&self->entries)) {
            if (!ww_hash_map_ensure_capacity(self, grow_capacity(self->count, self->count + 1))) {
                return (WwHashMapGerOrPutResult){ .failed = true };
            }
            bucket = ww_hash_map_get_bucket(self, hash_code);
        }
        index = self->count;
        self->count += 1;
    }

    void* entry = ww_darray_get_void_ref(&self->entries, index);
    void* entry_key = entry + sizeof(WwEntryHeader);
    void* entry_value = entry_key + self->key_size;
    
    WwEntryHeader* entry_header = entry;
    entry_header->hash_code = hash_code;
    entry_header->next = *bucket - 1;
    memcpy(entry_key, key.ptr, key.size);

    *bucket = index + 1;

    return (WwHashMapGerOrPutResult){
        .key = entry_key,
        .value = entry_value,
    };
}

i32* ww_hash_map_get_bucket(_WwHashMap* self, usize hash_code) {
    return ww_darray_get_ref(&self->buckets, i32, hash_code % ww_darray_len(&self->buckets));
}

usize grow_capacity(usize current, usize minimum) {
    usize new = current;
    while (true) {
        new += new / 2 + 8;
        if (new >= minimum)
            return new;
    }
}
