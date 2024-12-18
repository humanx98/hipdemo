#pragma once

#include "hiprt_common.h"

extern "C" {
#include <ww/collections/darray.h>
#include <ww/renderer/scene.h>
}

typedef struct scene_ptr_impl {
    HipRTRenderContext context;
    hiprtSceneBuildInput input;
    hiprtDevicePtr input_buff;
    usize input_buff_size;
    hiprtScene scene;
    hiprtDevicePtr scene_buff;
    usize scene_buff_size;
    b8 rebuild;
    camera_ptr camera;
    WwDArray(object_instance_ptr) attached_object_instances;
} scene_ptr_impl;

RendererResult __ww_must_check hiprt_scene_create(HipRTRenderContext context, Scene* scene);
RendererResult __ww_must_check hiprt_scene_rebuild(scene_ptr self);