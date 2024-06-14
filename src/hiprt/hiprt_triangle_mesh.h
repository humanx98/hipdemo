#pragma once

#include "hiprt_common.h"

extern "C" {
#include <ww/renderer/triangle_mesh.h>
}

typedef struct triangle_mesh_ptr_impl {
    HipRTRenderContext context;
    hiprtTriangleMeshPrimitive mesh;
    hiprtDevicePtr geometry_temp;
    hiprtGeometry geometry;
} triangle_mesh_ptr_impl;

RendererResult __ww_must_check hiprt_triangle_mesh_create(HipRTRenderContext context, TriangleMeshCreationProperties creation_properties, TriangleMesh* triangle_mesh);