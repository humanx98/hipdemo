#include "hiprt_scene.h"
#include <hiprt/hiprt.h>
#include <cassert>

extern "C" {
#include <ww/hip/common.h>
#include <ww/exit.h>
}

static RendererResult __ww_must_check hiprt_scene_init(scene_ptr self, HipRTRenderContext context);
static RendererResult __ww_must_check hiprt_scene_set_camera(scene_ptr self, camera_ptr camera);
static RendererResult __ww_must_check hiprt_scene_attach_object_instance(scene_ptr self, object_instance_ptr object_instance);
static void hiprt_scene_destroy(scene_ptr self);

RendererResult hiprt_scene_create(HipRTRenderContext context, Scene* scene) {
    assert(scene);

    ww_auto_type alloc_res = ww_allocator_alloc_type(context.allocator, scene_ptr_impl);
    if (alloc_res.failed) {
        return renderer_result(RENDERER_ERROR_OUT_OF_HOST_MEMORY);
    }

    scene_ptr self = alloc_res.ptr;
    RendererResult res = hiprt_scene_init(self, context);
    if (res.failed) {
        hiprt_scene_destroy(self);
        return res;
    }

    static scene_vtable vtable = {
        .set_camera = hiprt_scene_set_camera,
        .attach_object_instance = hiprt_scene_attach_object_instance,
        .destroy = hiprt_scene_destroy,
    };

    *scene = {
        .ptr = (scene_ptr)self,
        .vtable = &vtable,
    };

    return res;
}

RendererResult hiprt_scene_init(scene_ptr self, HipRTRenderContext context) {
    *self = {
        .context = context,
    };
    return renderer_result(RENDERER_SUCCESS);
}

void hiprt_scene_destroy(scene_ptr self) {
    assert(self);

    RendererResult res = {};
    self->attached_object_instances.~vector();

    if (self->scene_input.instanceTransformHeaders) {
        res = HIP_CHECK(hipFree(self->scene_input.instanceTransformHeaders));
    }

    if (self->scene_input.instanceFrames) {
        res = HIP_CHECK(hipFree(self->scene_input.instanceFrames));
    }

    if (self->scene_input.instances) {
        res = HIP_CHECK(hipFree(self->scene_input.instances));
    }

    if (self->scene_temp) {
        res = HIP_CHECK(hipFree(self->scene_temp));
    }

    if (self->scene) {
        res = HIPRT_CHECK(hiprtDestroyScene(self->context.hiprt, self->scene));
    }
    ww_allocator_free(self->context.allocator, self);
}

RendererResult hiprt_scene_set_camera(scene_ptr self, camera_ptr camera) {
    assert(self);
    self->camera = camera;
    return renderer_result(RENDERER_SUCCESS);
}

RendererResult hiprt_scene_attach_object_instance(scene_ptr self, object_instance_ptr object_instance) {
    assert(self);
    assert(object_instance);
    self->attached_object_instances.push_back(object_instance);
    return renderer_result(RENDERER_SUCCESS);
}
