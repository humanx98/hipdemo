#pragma once

#include <ww/math.h>
#include <ww/defines.h>

WW_DEFINE_HANDLE(triangle_mesh_ptr);
typedef void (*triangle_mesh_destroy_fn)(triangle_mesh_ptr ptr);


typedef struct triangle_mesh_vtable {
    triangle_mesh_destroy_fn destroy;
} triangle_mesh_vtable;

typedef struct TriangleMeshCreationProperties {
    usize vertex_count;
    vec3* vertices;
    usize triangle_count; 
    u32* triangle_indices;
} TriangleMeshCreationProperties;

typedef struct TriangleMesh {
    triangle_mesh_ptr ptr;
    const triangle_mesh_vtable* vtable;
} TriangleMesh;

void triangle_mesh_destroy(TriangleMesh self);

