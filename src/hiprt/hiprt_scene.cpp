#include "hiprt_scene.h"
#include "hiprt_object_instance.h"
#include <hiprt/hiprt.h>
#include <cassert>
#include <vector>

extern "C" {
#include <ww/hip/common.h>
#include <ww/exit.h>
}

static RendererResult __ww_must_check hiprt_scene_init(scene_ptr self, HipRTRenderContext context);
static RendererResult __ww_must_check hiprt_scene_rebuild_input(scene_ptr self);
static RendererResult __ww_must_check hiprt_scene_set_camera(scene_ptr self, camera_ptr camera);
static RendererResult __ww_must_check hiprt_scene_attach_object_instance(scene_ptr self, object_instance_ptr object_instance);
static RendererResult __ww_must_check hiprt_scene_detach_object_instance(scene_ptr self, object_instance_ptr object_instance);
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
        .detach_object_instance = hiprt_scene_detach_object_instance,
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
        .rebuild = true,
        .attached_object_instances = ww_darray_init(context.allocator, object_instance_ptr),
    };
    return renderer_result(RENDERER_SUCCESS);
}

void hiprt_scene_destroy(scene_ptr self) {
    assert(self);

    RendererResult res = {};
    ww_darray_deinit(&self->attached_object_instances);

    if (self->input_buff) {
        res = HIP_CHECK(hipFree(self->input_buff));
    }

    if (self->scene_buff) {
        res = HIP_CHECK(hipFree(self->scene_buff));
    }

    if (self->scene) {
        res = HIPRT_CHECK(hiprtDestroyScene(self->context.hiprt, self->scene));
    }
    ww_allocator_free(self->context.allocator, self);
}

RendererResult hiprt_scene_rebuild(scene_ptr self) {
    assert(self);

    RendererResult res = hiprt_scene_rebuild_input(self);
    if (res.failed) {
        return res;
    }

    usize new_scene_buff_size;
    hiprtBuildOptions options = {
        .buildFlags = hiprtBuildFlagBitPreferFastBuild,
    };
    res = HIPRT_CHECK(hiprtGetSceneBuildTemporaryBufferSize(self->context.hiprt, self->input, options, new_scene_buff_size));
    if (res.failed) {
        return res;
    }

    if (self->scene_buff_size < new_scene_buff_size) {
        if (self->scene_buff) {
            res = HIP_CHECK(hipFree(self->scene_buff));
            if (res.failed) {
                return res;
            }

            self->scene_buff= NULL;
            self->scene_buff_size = 0;
        }

        res = HIP_CHECK(hipMalloc(&self->scene_buff, new_scene_buff_size));
        if (res.failed) {
            return res;
        }

        self->scene_buff_size = new_scene_buff_size;
    }

    if (!self->scene) {
        res = HIPRT_CHECK(hiprtCreateScene(self->context.hiprt, self->input, options, self->scene));
        if (res.failed) {
            return res;
        }
    }

    res = HIPRT_CHECK(hiprtBuildScene(self->context.hiprt, hiprtBuildOperationBuild, self->input, options, self->scene_buff, 0, self->scene));
    if (res.failed) {
        return res;
    }

    self->rebuild = false;
    return res;
}

RendererResult hiprt_scene_rebuild_input(scene_ptr self) {
    std::vector<hiprtInstance> instances;
    std::vector<hiprtTransformHeader> transform_headers;
    std::vector<hiprtFrameMatrix> frame_matrices;
    usize instances_count = ww_darray_len(&self->attached_object_instances);
    instances.reserve(instances_count);
    transform_headers.reserve(instances_count);
    frame_matrices.reserve(instances_count);

    ww_darray_foreach_by_ref(&self->attached_object_instances, object_instance_ptr, aoi) {
        instances.push_back((*aoi)->instance);
        hiprtTransformHeader header = {
            .frameIndex = (u32)transform_headers.size(),
            .frameCount = 1,
        };
        transform_headers.push_back(header);
        hiprtFrameMatrix matrix = {
            .matrix = {
                { (*aoi)->transform.e[0][0], (*aoi)->transform.e[0][1], (*aoi)->transform.e[0][2], (*aoi)->transform.e[0][3] },
                { (*aoi)->transform.e[1][0], (*aoi)->transform.e[1][1], (*aoi)->transform.e[1][2], (*aoi)->transform.e[1][3] },
                { (*aoi)->transform.e[2][0], (*aoi)->transform.e[2][1], (*aoi)->transform.e[2][2], (*aoi)->transform.e[2][3] },
            },
        };
        frame_matrices.push_back(matrix);
    }

    usize new_input_buff_size =  instances.size() * sizeof(hiprtInstance)
        + transform_headers.size() * sizeof(hiprtTransformHeader)
        + frame_matrices.size() * sizeof(hiprtFrameMatrix);

    RendererResult res;
    if (self->input_buff_size < new_input_buff_size) {
        if (self->input_buff) {
            res = HIP_CHECK(hipFree(self->input_buff));
            if (res.failed) {
                return res;
            }

            self->input_buff = NULL;
            self->input_buff_size = 0;
        }

        res = HIP_CHECK(hipMalloc(&self->input_buff, new_input_buff_size));
        if (res.failed) {
            return res;
        }

        self->input_buff_size = new_input_buff_size;
    }

    self->input.instanceCount = instances.size();
    self->input.instanceMasks = nullptr;
    self->input.frameType = hiprtFrameTypeMatrix;
    self->input.frameCount = frame_matrices.size();
    self->input.instances = self->input_buff;
    self->input.instanceTransformHeaders = (u8*)self->input.instances + instances.size() * sizeof(hiprtInstance);
    self->input.instanceFrames = (u8*)self->input.instanceTransformHeaders + transform_headers.size() * sizeof(hiprtTransformHeader);

    res = HIP_CHECK(hipMemcpyHtoD(self->input.instances, instances.data(), instances.size() * sizeof(hiprtInstance)));
    if (res.failed) {
        return res;
    }

    res = HIP_CHECK(hipMemcpyHtoD(self->input.instanceTransformHeaders, transform_headers.data(), transform_headers.size() * sizeof(hiprtTransformHeader)));
    if (res.failed) {
        return res;
    }

    return HIP_CHECK(hipMemcpyHtoD(self->input.instanceFrames, frame_matrices.data(), frame_matrices.size() * sizeof(hiprtFrameMatrix)));
}

RendererResult hiprt_scene_set_camera(scene_ptr self, camera_ptr camera) {
    assert(self);
    self->camera = camera;
    return renderer_result(RENDERER_SUCCESS);
}

RendererResult hiprt_scene_attach_object_instance(scene_ptr self, object_instance_ptr object_instance) {
    assert(self);
    assert(object_instance);

    for (usize i = 0; i < ww_darray_len(&self->attached_object_instances); i++) {
        if (object_instance == ww_darray_get(&self->attached_object_instances, object_instance_ptr, i)) {
            WW_LOG_WARN("[hiprt_scene] The same object instance has been attached twice.\n");
            return renderer_result(RENDERER_SUCCESS);
        }
    }

    if (!ww_darray_append(&self->attached_object_instances, object_instance)) {
        return renderer_result(RENDERER_ERROR_OUT_OF_HOST_MEMORY);
    }

    if (!ww_darray_append(&object_instance->scenes, self)) {
        return renderer_result(RENDERER_ERROR_OUT_OF_HOST_MEMORY);
    }

    self->rebuild = true;
    return renderer_result(RENDERER_SUCCESS);
}

RendererResult hiprt_scene_detach_object_instance(scene_ptr self, object_instance_ptr object_instance) {
    assert(self);
    assert(object_instance);

    for (usize i = 0; i < ww_darray_len(&self->attached_object_instances); i++) {
        if (object_instance == ww_darray_get(&self->attached_object_instances, object_instance_ptr, i)) {
            ww_darray_swap_remove(&self->attached_object_instances, i);
            self->rebuild = true;
            for (usize j = 0; j < ww_darray_len(&object_instance->scenes); j++) {
                if (self == ww_darray_get(&object_instance->scenes, scene_ptr, j)) {
                    ww_darray_swap_remove(&object_instance->scenes, j);
                    break;
                }
            }
            return renderer_result(RENDERER_SUCCESS);
        }
    }

    WW_LOG_WARN("[hiprt_scene] Detach non existing object instance.\n");
    return renderer_result(RENDERER_SUCCESS);
}