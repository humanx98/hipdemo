#include "app.h"
#include "ww/allocators/allocator.h"
#include <ww/prim_types.h>
#include <ww/math.h>
#include <ww/collections/hash_map.h>

// u32 i32_get_hash(const void* key) {
//     return *(i32*)key;
// }

// b8 i32_compare_keys(const void* a, const void* b) {
//     return *(i32*)a == *(i32*)b;
// }

i32 main() {
    i32 retvalue = 0;
#ifdef NDEBUG
#include <ww/allocators/std_allocator.h>
    WwAllocator allocator = std_allocator();
#else
#include <ww/allocators/safe_allocator.h>
    WwSafeAllocator safe_allocator = ww_safe_allocator_init();
    WwAllocator allocator = ww_safe_allocator_get_interface(&safe_allocator);
#endif
    //ww_allocator_free(allocator, alloc1);

    // WwHashMap(i32, f32) map = ww_hash_map_init(allocator, i32, vec3, i32_get_hash, i32_compare_keys);
    // for (i32 i = 0; i < 10; i++) {
    //     WwHashMapGerOrPutResult res = ww_hash_map_get_or_put(&map, i);
    //     if (res.found_existing) {
    //         vec3 value = *(vec3*)res.value;
    //         WW_LOG_INFO("found %f %f %f\n", value.x, value.y, value.z);
    //     } else if (!res.failed) {
    //         vec3 value = make_vec3(i, -i, 0);
    //         WW_LOG_INFO("set %f %f %f\n", value.x, value.y, value.z);
    //         *(vec3*)res.value = value;
    //     }
    // }
    // for (i32 i = 0; i < 10; i++) {
    //     WwHashMapGerOrPutResult res = ww_hash_map_get_or_put(&map, i);
    //     if (res.found_existing) {
    //         vec3 value = *(vec3*)res.value;
    //         WW_LOG_INFO("found %f %f %f\n", value.x, value.y, value.z);
    //     } else if (!res.failed) {
    //         vec3 value = make_vec3(i, -i, 0);
    //         WW_LOG_INFO("set %f %f %f\n", value.x, value.y, value.z);
    //         *(vec3*)res.value = value;
    //     }
    // }
    // ww_hash_map_deinit(&map);
    App* app;
    AppCreationProperties creation_properties = {
        .allocator = allocator,
        .width = 1200,
        .height = 800,
        .device_index = 0,
    };
    if (app_create(creation_properties, &app).failed) {
        retvalue = -1;
        goto leave;
    }

    if (app_run(app).failed) {
        retvalue = -1;
        goto destroy;
    }

destroy:
    app_destroy(app);
leave:
#ifndef NDEBUG
    ww_safe_allocator_deinit(&safe_allocator);
#endif
    return retvalue;
}
