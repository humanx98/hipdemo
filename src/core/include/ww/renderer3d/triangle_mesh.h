#pragma once

#include <ww/math.h>
#include <ww/defines.h>

WW_DEFINE_HANDLE(ww_triangle_mesh_ptr);
typedef void (*ww_triangle_mesh_destroy_fn)(ww_triangle_mesh_ptr ptr);


typedef struct ww_triangle_mesh_vtable {
    ww_triangle_mesh_destroy_fn destroy;
} ww_triangle_mesh_vtable;

typedef struct WwTriangleMeshCreationProperties {
    usize vertex_count;
    vec3* vertices;
    usize triangle_count; 
    u32* triangle_indices;
} WwTriangleMeshCreationProperties;

typedef struct WwTriangleMesh {
    ww_triangle_mesh_ptr ptr;
    const ww_triangle_mesh_vtable* vtable;
} WwTriangleMesh;

void ww_triangle_mesh_destroy(WwTriangleMesh self);

