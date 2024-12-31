#include <ww/hiprt/renderer.h>
#include <cassert>
#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>
#include <string>
#include "hiprt_common.h"
#include "hiprt_object_instance.h"
#include "hiprt_triangle_mesh.h"
#include "hiprt_scene.h"
#include "kernels/types/camera.h"

extern "C" {
#include <ww/hip/common.h>
#include <ww/renderer/renderer.h>
#include <ww/renderer/camera_def_impl.h>
#include <ww/collections/darray.h>
#include <ww/file.h>
#include <ww/log.h>
#include <ww/exit.h>
#include <ww/math.h>
}

typedef struct renderer_ptr_impl {
    u32 width;
    u32 height;
    HipRTRenderContext context;
    struct {
        hipExternalMemory_t external_memory;
        hipDeviceptr_t ptr;
    } pixels;
    u32 frames_in_flight;
    u32 current_frame;
    WwDArray(hipExternalSemaphore_t) wait_semaphores;
    WwDArray(hipExternalSemaphore_t) signal_semaphores;
    hipModule_t module;
    hipFunction_t kernel_func;
    hipStream_t stream;
    scene_ptr scene;
} renderer_ptr_impl;

static void hiprt_renderer_destroy(renderer_ptr self);
static RendererResult __ww_must_check hiprt_renderer_set_target_resolution(renderer_ptr self, u32 width, u32 height);
static RendererResult __ww_must_check hiprt_renderer_set_external_memory(renderer_ptr self, ViewportExternalHandle external_memory, u32 width, u32 height);
static RendererResult __ww_must_check hiprt_renderer_render(renderer_ptr self);
static RendererResult __ww_must_check hiprt_renderer_copy_target_to(renderer_ptr self, void* dst);
static RendererResult __ww_must_check hiprt_renderer_set_scene(renderer_ptr self, scene_ptr scene);
static RendererResult __ww_must_check hiprt_renderer_create_scene(renderer_ptr self, Scene* scene);
static RendererResult __ww_must_check hiprt_renderer_create_camera(renderer_ptr self, Camera* camera);
static RendererResult __ww_must_check hiprt_renderer_create_object_instance(renderer_ptr self, const triangle_mesh_ptr triangle_mesh, ObjectInstance* object_instance);
static RendererResult __ww_must_check hiprt_renderer_create_triangle_mesh(renderer_ptr self, TriangleMeshCreationProperties creation_properties, TriangleMesh* triangle_mesh);
static RendererResult __ww_must_check hiprt_renderer_init(renderer_ptr self, HipRTCreationProperties creation_properties);
static RendererResult __ww_must_check hiprt_renderer_free_target(renderer_ptr self);

RendererResult hiprt_renderer_create(HipRTCreationProperties creation_properties, Renderer* renderer) {
    assert(renderer);

    ww_auto_type alloc_result = ww_allocator_alloc_type(creation_properties.allocator, renderer_ptr_impl);
    if (alloc_result.failed) {
        return renderer_result(RENDERER_ERROR_OUT_OF_HOST_MEMORY);
    }

    renderer_ptr self = alloc_result.ptr;
    RendererResult res = hiprt_renderer_init(self, creation_properties);
    if (res.failed) {
        hiprt_renderer_destroy(self);
        return res;
    }

    static const renderer_vtable vtable = {
        .set_target_resolution = hiprt_renderer_set_target_resolution,
        .set_target_external_memory = hiprt_renderer_set_external_memory,
        .render = hiprt_renderer_render,
        .copy_target_to = hiprt_renderer_copy_target_to,
        .set_scene = hiprt_renderer_set_scene,
        .create_camera = hiprt_renderer_create_camera,
        .create_object_instance = hiprt_renderer_create_object_instance,
        .create_scene = hiprt_renderer_create_scene,
        .create_triangle_mesh = hiprt_renderer_create_triangle_mesh,
        .destroy = hiprt_renderer_destroy,
    };
    *renderer = (Renderer){
        .ptr = self,
        .vtable = &vtable,
    };
    return res;
}

RendererResult hiprt_renderer_init(renderer_ptr self, HipRTCreationProperties creation_properties) {
    *self = (renderer_ptr_impl){
        .context = {
            .allocator = creation_properties.allocator,
        },
        .frames_in_flight = creation_properties.viewport_external_memory_semaphores.frames_in_flight,
        .wait_semaphores = ww_darray_init(creation_properties.allocator, hipExternalSemaphore_t),
        .signal_semaphores = ww_darray_init(creation_properties.allocator, hipExternalSemaphore_t),
    };

    RendererResult res = HIP_CHECK(hipInit(0));
    if (res.failed) {
        return res;
    }

    res = HIP_CHECK(hipSetDevice(creation_properties.device_index));
    if (res.failed) {
        return res;
    }

    res = HIP_CHECK(hipCtxCreate(&self->context.hip, 0, creation_properties.device_index));
    if (res.failed) {
        return res;
    }

    hipDeviceProp_t props;
    res = HIP_CHECK(hipGetDeviceProperties(&props, creation_properties.device_index));
    if (res.failed) {
        return res;
    }

    hiprtContextCreationInput ctx_creation_input;
    if (std::string(props.name).find("NVIDIA") != std::string::npos) {
        ctx_creation_input.deviceType = hiprtDeviceNVIDIA;
    } else {
        ctx_creation_input.deviceType = hiprtDeviceAMD;
    }
    ctx_creation_input.ctxt = self->context.hip;
    ctx_creation_input.device = creation_properties.device_index;

    res = HIPRT_CHECK(hiprtCreateContext(HIPRT_API_VERSION, ctx_creation_input, self->context.hiprt));
    if (res.failed) {
        return res;
    }

    hiprtSetCacheDirPath(self->context.hiprt, "hip_spv_bin");
    res = HIP_CHECK(hipModuleLoad(&self->module, "hip_spv_bin/hiprt_renderer.hipfb"));
    if (res.failed) {
        return res;
    }

    res = HIP_CHECK(hipModuleGetFunction(&self->kernel_func, self->module, "SceneIntersectionKernel"));
    if (res.failed) {
        return res;
    }

    if (self->frames_in_flight > 0) {
        if (!ww_darray_ensure_total_capacity_precise(&self->wait_semaphores, self->frames_in_flight)
            || !ww_darray_ensure_total_capacity_precise(&self->signal_semaphores, self->frames_in_flight)) {
            return renderer_result(RENDERER_ERROR_OUT_OF_HOST_MEMORY);
        }

        for (usize i = 0; i < self->frames_in_flight; i++) {
            hipExternalSemaphore_t semaphore;
            res = HIP_CHECK(hip_import_viewport_external_semaphore(&semaphore, creation_properties.viewport_external_memory_semaphores.wait_for_signal_external_memory_from_viewport[i]));
            if (res.failed) {
                return res;
            }

            ww_darray_append_assume_capacity(&self->wait_semaphores, semaphore);

            res = HIP_CHECK(hip_import_viewport_external_semaphore(&semaphore, creation_properties.viewport_external_memory_semaphores.signal_external_memory_for_viewport[i]));
            if (res.failed) {
                return res;
            }

            ww_darray_append_assume_capacity(&self->signal_semaphores, semaphore);
        }
    }

    return HIP_CHECK(hipStreamCreateWithFlags(&self->stream, hipStreamNonBlocking));
}

void hiprt_renderer_destroy(renderer_ptr self) {
    assert(self);
    RendererResult res = hiprt_renderer_free_target(self);

    if (self->stream) {
        res = HIP_CHECK(hipStreamDestroy(self->stream));
    }

    ww_darray_foreach_by_ref(&self->wait_semaphores, hipExternalSemaphore_t, s) {
        res = HIP_CHECK(hipDestroyExternalSemaphore(*s));
    }
    ww_darray_foreach_by_ref(&self->signal_semaphores, hipExternalSemaphore_t, s) {
        res = HIP_CHECK(hipDestroyExternalSemaphore(*s));
    }
    ww_darray_deinit(&self->wait_semaphores);
    ww_darray_deinit(&self->signal_semaphores);

    if (self->module) {
        res = HIP_CHECK(hipModuleUnload(self->module));
    }

    if (self->context.hiprt) {
        res = HIPRT_CHECK(hiprtDestroyContext(self->context.hiprt));
    }

    if (self->context.hip) {
        res = HIP_CHECK(hipCtxDestroy(self->context.hip));
    }

    ww_allocator_free(self->context.allocator, self);
}

RendererResult hiprt_renderer_set_target_resolution(renderer_ptr self, u32 width, u32 height) {
    assert(self);

    RendererResult res = HIP_CHECK(hipStreamSynchronize(self->stream));
    if (res.failed) {
        return res;
    }
    
    res = hiprt_renderer_free_target(self);
    if (res.failed) {
        return res;
    }

    self->width = width;
    self->height = height;
    res = HIP_CHECK(hipMalloc(&self->pixels.ptr, self->width * self->height * 4 * sizeof(f32)));
    return res;
}

RendererResult hiprt_renderer_set_external_memory(renderer_ptr self, ViewportExternalHandle external_memory, u32 width, u32 height) {
    assert(self);
    RendererResult res = hiprt_renderer_free_target(self);
    if (res.failed) {
        return res;
    }

    usize size = width * height * 4 * sizeof(f32);
    res = HIP_CHECK(hip_import_viewport_external_memory(&self->pixels.external_memory, external_memory, size));
    if (res.failed) {
        return res;
    }

    hipExternalMemoryBufferDesc buff_desc = { .size = size};
    res = HIP_CHECK(hipExternalMemoryGetMappedBuffer(&self->pixels.ptr, self->pixels.external_memory, &buff_desc));
    self->width = width;
    self->height = height;
    return res;
}

RendererResult hiprt_renderer_render(renderer_ptr self) {
    assert(self);
    assert(self->pixels.ptr);
    assert(self->scene);

    // scene creation
    RendererResult res;
    if (self->scene->rebuild && (res = hiprt_scene_rebuild(self->scene)).failed) {
        return res;
    }

    if (self->frames_in_flight > 0) {
        hipExternalSemaphoreWaitParams wait_params = {};
        hipExternalSemaphore_t wait_semaphore = ww_darray_get(&self->wait_semaphores, hipExternalSemaphore_t, self->current_frame);
        res = HIP_CHECK(hipWaitExternalSemaphoresAsync(&wait_semaphore, &wait_params, 1, self->stream));
        if (res.failed) {
            return res;
        }
    }

    device::Camera camera = {
        .origin = self->scene->camera->origin,
        .lower_left_corner = self->scene->camera->lower_left_corner,
        .horizontal = self->scene->camera->horizontal,
        .vertical = self->scene->camera->vertical,
        .u = self->scene->camera->u,
        .v = self->scene->camera->v,
        .w = self->scene->camera->w,
        .lens_radius = self->scene->camera->lens_radius,
        .vfov = self->scene->camera->vfov,
        .focus_dist = self->scene->camera->focus_dist,
        .aspect_ratio = self->scene->camera->aspect_ratio,
        .look_from = self->scene->camera->look_from,
        .look_at = self->scene->camera->look_at,
        .vup = self->scene->camera->vup,
    };

    hiprtInt2 resolution = {(i32)self->width, (i32)self->height};
    b8 flip_y = true;
    void *args[] = {&self->scene->scene, &camera, &self->pixels.ptr, &resolution, &flip_y};
    int3 block = { 1024, 1, 1 };
    int3 grid = { ((self->width * self->height) + block.x - 1) / block.x, 1, 1 };
    res = HIP_CHECK(hipModuleLaunchKernel(
        (hipFunction_t)self->kernel_func,
        grid.x,
        grid.y,
        grid.z,
        block.x,
        block.y,
        block.z,
        0,
        self->stream,
        args,
        0
    ));
    if (res.failed) {
        return res;
    }

    if (self->frames_in_flight > 0) {
        hipExternalSemaphoreSignalParams signal_params = {};
        hipExternalSemaphore_t signal_semaphore = ww_darray_get(&self->signal_semaphores, hipExternalSemaphore_t, self->current_frame);
        res = HIP_CHECK(hipSignalExternalSemaphoresAsync(&signal_semaphore, &signal_params, 1, self->stream));
        if (res.failed) {
            return res;
        }
        self->current_frame = (self->current_frame + 1) % self->frames_in_flight;
        return res;
    } else {
        return HIP_CHECK(hipStreamSynchronize(self->stream));
    }
}

RendererResult hiprt_renderer_copy_target_to(renderer_ptr self, void *dst) {
    assert(self);
    assert(!self->pixels.external_memory);
    RendererResult res = HIP_CHECK(hipMemcpyDtoHAsync(dst, self->pixels.ptr, self->width * self->height * 4 * sizeof(f32), self->stream));
    if (res.failed) {
        return res;
    }

    return HIP_CHECK(hipStreamSynchronize(self->stream));
}

RendererResult hiprt_renderer_set_scene(renderer_ptr self, scene_ptr scene) {
    assert(self);
    self->scene = scene;
    return renderer_result(RENDERER_SUCCESS);
}

RendererResult hiprt_renderer_create_scene(renderer_ptr self, Scene* scene) {
    assert(self);
    return hiprt_scene_create(self->context, scene);
}

RendererResult hiprt_renderer_create_camera(renderer_ptr self, Camera* camera) {
    assert(self);
    return camera_def_impl_create(self->context.allocator, camera);
}

RendererResult hiprt_renderer_create_object_instance(renderer_ptr self, const triangle_mesh_ptr triangle_mesh, ObjectInstance* object_instance) {
    assert(self);
    return hiprt_object_instance_create(self->context, triangle_mesh, object_instance);
}

RendererResult hiprt_renderer_create_triangle_mesh(renderer_ptr self, TriangleMeshCreationProperties creation_properties, TriangleMesh* triangle_mesh) {
    assert(self);
    return hiprt_triangle_mesh_create(self->context, creation_properties, triangle_mesh);
}

RendererResult hiprt_renderer_free_target(renderer_ptr self) {
    RendererResult res = renderer_result(RENDERER_SUCCESS);
    if (self->pixels.external_memory) {
        res = HIP_CHECK(hipDestroyExternalMemory(self->pixels.external_memory));
    } else if (self->pixels.ptr) {
        res = HIP_CHECK(hipFree(self->pixels.ptr));
    }

    if (res.failed) {
        return res;
    }

    self->pixels.external_memory = NULL;
    self->pixels.ptr = NULL;
    return res;
}
