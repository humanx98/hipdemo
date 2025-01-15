#pragma once

#include "hiprt_common.h"

extern "C" {
#include <ww/collections/darray.h>
#include <ww/renderer3d/object_instance.h>
#include <ww/renderer3d/triangle_mesh.h>
#include <ww/renderer3d/scene.h>
}

typedef struct ww_object_instance_ptr_impl {
    HipRTRenderContext context;
    hiprtInstance instance;
    mat4 transform;
    WwDArray(ww_scene_ptr) scenes;
} ww_object_instance_ptr_impl;

WwRenderer3DResult __ww_must_check hiprt_object_instance_create(HipRTRenderContext context, const ww_triangle_mesh_ptr triangle_mesh, WwObjectInstance* object_instance);
