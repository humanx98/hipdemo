
set(HEADERS
    include/ww/renderer/camera_def_impl.h
    include/ww/renderer/camera.h
    include/ww/renderer/object_instance.h
    include/ww/renderer/result.h
    include/ww/renderer/scene.h
    include/ww/renderer/triangle_mesh.h
    include/ww/renderer.h
)

set(SOURCES
    camera_def_impl.c
    camera.c
    object_instance.c
    renderer.c
    scene.c
    triangle_mesh.c
)

add_library(renderer STATIC ${HEADERS} ${SOURCES})
target_include_directories(renderer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(renderer ww_core)