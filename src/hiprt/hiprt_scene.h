#pragma once

#include <vector>
#include "hiprt_common.h"

extern "C" {
#include <ww/defines.h>
#include <ww/renderer/scene.h>
}

typedef struct scene_ptr_impl {
    HipRTRenderContext context;
    hiprtSceneBuildInput scene_input;
    hiprtDevicePtr scene_temp;
    hiprtScene scene;
    std::vector<object_instance_ptr> attached_object_instances;
} scene_ptr_impl;

RendererResult __ww_must_check hiprt_scene_create(HipRTRenderContext context, Scene* scene);