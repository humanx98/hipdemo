#pragma once

#include "hiprt_common.h"
#include "hiprt_triangle_mesh.h"

extern "C" {
#include <ww/renderer/object_instance.h>
}

typedef struct object_instance_ptr_impl {
    HipRTRenderContext context;
    hiprtInstance instance;
    mat4 transform;
} object_instance_ptr_impl;

RendererResult __ww_must_check hiprt_object_instance_create(HipRTRenderContext context, const triangle_mesh_ptr triangle_mesh, ObjectInstance* object_instance);
