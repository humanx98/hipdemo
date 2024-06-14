#pragma once

#include <ww/allocators/allocator.h>
#include <ww/collections/darray.h>

typedef u32 (*ww_get_hash_fn)(const void*);
typedef b8 (*ww_compare_keys_fn)(const void*, const void*);

typedef struct _WwHashMap {
    WwDArray(i32) buckets;
    WwDArray(u8) entries;
    usize key_size;
    usize value_size;
    ww_get_hash_fn get_hash;
    ww_compare_keys_fn compare_keys;
    usize count;
} _WwHashMap;


typedef struct WwEntryHeader {
    i32 next;
    u32 hash_code;
} WwEntryHeader;

typedef struct WwHashMapKey {
    const void* ptr;
    usize size;
} WwHashMapKey;

typedef struct WwHashMapValue {
    void* ptr;
    usize size;
} WwHashMapValue;

typedef struct WwHashMapGerOrPutResult {
    const void* key;
    void* value;
    b8 failed;
    b8 found_existing;
} WwHashMapGerOrPutResult;

void ww_hash_map_deinit(_WwHashMap* self);
b8 __ww_must_check ww_hash_map_ensure_capacity(_WwHashMap* self, usize capacity);
b8 __ww_must_check _ww_hash_map_get(const _WwHashMap* self, WwHashMapKey key);
b8 __ww_must_check _ww_hash_map_put(_WwHashMap* self, WwHashMapKey key, WwHashMapValue value);
WwHashMapGerOrPutResult __ww_must_check _ww_hash_map_get_or_put(_WwHashMap* self, WwHashMapKey key);

#define WwHashMap(K, V) _WwHashMap
#define ww_to_hash_map_key(k) (WwHashMapKey) { .ptr = &k, .size = sizeof(k) }
#define ww_to_hash_map_value(v) (WwHashMapValue) { .ptr = &v, .size = sizeof(v) }
#define ww_hash_map_entry_size(k, v) (sizeof(WwEntryHeader) + sizeof(k) + sizeof(v))
#define ww_hash_map_init(allocator, k, v, gh, cmpk) (_WwHashMap) {                     \
    .buckets = ww_darray_init(allocator, i32),                                         \
    .entries = ww_darray_init_with_elem_size(allocator, ww_hash_map_entry_size(k, v)), \
    .key_size = sizeof(k),                                                             \
    .value_size = sizeof(v),                                                           \
    .get_hash = gh,                                                                    \
    .compare_keys = cmpk,                                                              \
}

#define ww_hash_map_get(self, k) _ww_hash_map_get(self, ww_to_hash_map_key(k))
#define ww_hash_map_put(self, k, v) _ww_hash_map_put(self, ww_to_hash_map_key(k), ww_to_hash_map_value(v));
#define ww_hash_map_get_or_put(self, k) _ww_hash_map_get_or_put(self, ww_to_hash_map_key(k));
