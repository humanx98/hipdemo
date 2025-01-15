#pragma once

#include "hiprt_common.h"

extern "C" {
#include <ww/collections/darray.h>
#include <ww/renderer3d/scene.h>
}

typedef struct ww_scene_ptr_impl {
    HipRTRenderContext context;
    hiprtSceneBuildInput input;
    hiprtDevicePtr input_buff;
    usize input_buff_size;
    hiprtScene scene;
    hiprtDevicePtr scene_buff;
    usize scene_buff_size;
    b8 rebuild;
    ww_camera_ptr camera;
    WwDArray(ww_object_instance_ptr) attached_object_instances;
} ww_scene_ptr_impl;

WwRenderer3DResult __ww_must_check hiprt_scene_create(HipRTRenderContext context, WwScene* scene);
WwRenderer3DResult __ww_must_check hiprt_scene_rebuild(ww_scene_ptr self);