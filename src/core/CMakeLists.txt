set(ALLOCATORS_HEADERS
    include/ww/allocators/allocator.h
    include/ww/allocators/safe_allocator.h
    include/ww/allocators/std_allocator.h
)
set(ALLOCATORS_SOURCES
    allocators/allocator.c
    allocators/safe_allocator.c
    allocators/std_allocator.c
)

set(COLLECTIONS_HEADERS
    include/ww/collections/darray.h
    include/ww/collections/hash_map.h
)
set(COLLECTIONS_SOURCES
    collections/darray.c
    collections/hash_map.c
)

set(RENDERER_HEADERS
    include/ww/renderer/camera_def_impl.h
    include/ww/renderer/camera.h
    include/ww/renderer/object_instance.h
    include/ww/renderer/renderer.h
    include/ww/renderer/result.h
    include/ww/renderer/scene.h
    include/ww/renderer/triangle_mesh.h
)
set(RENDERER_SOURCES
    renderer/camera_def_impl.c
    renderer/camera.c
    renderer/object_instance.c
    renderer/renderer.c
    renderer/scene.c
    renderer/triangle_mesh.c
)

set(HEADERS
    ${ALLOCATORS_HEADERS}
    ${COLLECTIONS_HEADERS}
    ${RENDERER_HEADERS}
    include/ww/defines.h
    include/ww/exit.h
    include/ww/file.h
    include/ww/log.h
    include/ww/prim_types.h
    include/ww/math.h
    include/ww/string.h
    include/ww/viewport.h
)

set(SOURCES
    ${ALLOCATORS_SOURCES}
    ${COLLECTIONS_SOURCES}
    ${RENDERER_SOURCES}
    file.c
    string.c
    viewport.c
)

add_library(ww_core STATIC ${HEADERS} ${SOURCES})
target_include_directories(ww_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
