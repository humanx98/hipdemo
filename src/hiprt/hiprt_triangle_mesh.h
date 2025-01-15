#pragma once

#include "hiprt_common.h"

extern "C" {
#include <ww/renderer3d/triangle_mesh.h>
}

typedef struct ww_triangle_mesh_ptr_impl {
    HipRTRenderContext context;
    hiprtTriangleMeshPrimitive mesh;
    hiprtDevicePtr geometry_temp;
    hiprtGeometry geometry;
} ww_triangle_mesh_ptr_impl;

WwRenderer3DResult __ww_must_check hiprt_triangle_mesh_create(HipRTRenderContext context, WwTriangleMeshCreationProperties creation_properties, WwTriangleMesh* triangle_mesh);