set(HEADERS
    include/ww/allocators/allocator.h
    include/ww/allocators/safe_allocator.h
    include/ww/allocators/std_allocator.h
    include/ww/collections/darray.h
    include/ww/collections/hash_map.h
    include/ww/defines.h
    include/ww/exit.h
    include/ww/file.h
    include/ww/log.h
    include/ww/prim_types.h
    include/ww/math.h
    include/ww/string.h
)

set(SOURCES
    allocators/allocator.c
    allocators/safe_allocator.c
    allocators/std_allocator.c
    collections/darray.c
    collections/hash_map.c
    file.c
    string.c
)

add_library(ww_core STATIC ${HEADERS} ${SOURCES})
target_include_directories(ww_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
