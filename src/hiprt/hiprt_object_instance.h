#pragma once

#include "hiprt_common.h"

extern "C" {
#include <ww/collections/darray.h>
#include <ww/renderer/object_instance.h>
#include <ww/renderer/triangle_mesh.h>
#include <ww/renderer/scene.h>
}

typedef struct object_instance_ptr_impl {
    HipRTRenderContext context;
    hiprtInstance instance;
    mat4 transform;
    WwDArray(scene_ptr) scenes;
} object_instance_ptr_impl;

RendererResult __ww_must_check hiprt_object_instance_create(HipRTRenderContext context, const triangle_mesh_ptr triangle_mesh, ObjectInstance* object_instance);
