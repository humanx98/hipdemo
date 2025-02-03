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
#include <ww/renderer3d/renderer3d.h>
#include <ww/renderer3d/camera_def_impl.h>
#include <ww/collections/darray.h>
#include <ww/file.h>
#include <ww/log.h>
#include <ww/exit.h>
#include <ww/math.h>
}

typedef struct ww_renderer3d_ptr_impl {
    u32 width;
    u32 height;
    HipRTRenderContext context;
    struct {
        hipExternalMemory_t external_memory;
        hipDeviceptr_t ptr;
    } pixels;
    struct {
        u64* value;
        hipExternalSemaphore_t semaphore;
    } external_semaphore;
    hipModule_t module;
    hipFunction_t kernel_func;
    hipStream_t stream;
    ww_scene_ptr scene;
} ww_renderer3d_ptr_impl;

static void hiprt_renderer_destroy(ww_renderer3d_ptr self);
static WwRenderer3DResult __ww_must_check hiprt_renderer_set_target_resolution(ww_renderer3d_ptr self, u32 width, u32 height);
static WwRenderer3DResult __ww_must_check hiprt_renderer_set_external_memory(ww_renderer3d_ptr self, WwViewportExternalHandle external_memory, u32 width, u32 height);
static WwRenderer3DResult __ww_must_check hiprt_renderer_render(ww_renderer3d_ptr self);
static WwRenderer3DResult __ww_must_check hiprt_renderer_copy_target_to(ww_renderer3d_ptr self, void* dst);
static WwRenderer3DResult __ww_must_check hiprt_renderer_set_scene(ww_renderer3d_ptr self, ww_scene_ptr scene);
static WwRenderer3DResult __ww_must_check hiprt_renderer_create_scene(ww_renderer3d_ptr self, WwScene* scene);
static WwRenderer3DResult __ww_must_check hiprt_renderer_create_camera(ww_renderer3d_ptr self, WwCamera* camera);
static WwRenderer3DResult __ww_must_check hiprt_renderer_create_object_instance(ww_renderer3d_ptr self, const ww_triangle_mesh_ptr triangle_mesh, WwObjectInstance* object_instance);
static WwRenderer3DResult __ww_must_check hiprt_renderer_create_triangle_mesh(ww_renderer3d_ptr self, WwTriangleMeshCreationProperties creation_properties, WwTriangleMesh* triangle_mesh);
static WwRenderer3DResult __ww_must_check hiprt_renderer_init(ww_renderer3d_ptr self, HipRTCreationProperties creation_properties);
static WwRenderer3DResult __ww_must_check hiprt_renderer_free_target(ww_renderer3d_ptr self);

WwRenderer3DResult hiprt_renderer_create(HipRTCreationProperties creation_properties, WwRenderer3D* renderer) {
    assert(renderer);

    ww_auto_type alloc_result = ww_allocator_alloc_type(creation_properties.allocator, ww_renderer3d_ptr_impl);
    if (alloc_result.failed) {
        return ww_renderer3d_result(WW_RENDERER3D_ERROR_OUT_OF_HOST_MEMORY);
    }

    ww_renderer3d_ptr self = alloc_result.ptr;
    WwRenderer3DResult res = hiprt_renderer_init(self, creation_properties);
    if (res.failed) {
        hiprt_renderer_destroy(self);
        return res;
    }

    static const ww_renderer3d_vtable vtable = {
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
    *renderer = (WwRenderer3D){
        .ptr = self,
        .vtable = &vtable,
    };
    return res;
}

WwRenderer3DResult hiprt_renderer_init(ww_renderer3d_ptr self, HipRTCreationProperties creation_properties) {
    *self = (ww_renderer3d_ptr_impl){
        .context = {
            .allocator = creation_properties.allocator,
        },
    };

    WwRenderer3DResult res = HIP_CHECK(hipInit(0));
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

    // doesn't work anymore
    // hiprtSetCacheDirPath(self->context.hiprt, "hip_spv_bin");
    res = HIP_CHECK(hipModuleLoad(&self->module, "hiprt_renderer.hipfb"));
    if (res.failed) {
        return res;
    }

    res = HIP_CHECK(hipModuleGetFunction(&self->kernel_func, self->module, "SceneIntersectionKernel"));
    if (res.failed) {
        return res;
    }


    if (creation_properties.use_viewport_external_semaphore) {
        self->external_semaphore.value = creation_properties.viewport_external_memory_semaphore.value;
        res = HIP_CHECK(hip_import_viewport_external_semaphore(&self->external_semaphore.semaphore, creation_properties.viewport_external_memory_semaphore.handle));
        if (res.failed) {
            return res;
        }
    }

    return HIP_CHECK(hipStreamCreateWithFlags(&self->stream, hipStreamNonBlocking));
}

void hiprt_renderer_destroy(ww_renderer3d_ptr self) {
    assert(self);
    WwRenderer3DResult res = hiprt_renderer_free_target(self);

    if (self->stream) {
        res = HIP_CHECK(hipStreamSynchronize(self->stream));
        res = HIP_CHECK(hipStreamDestroy(self->stream));
    }

    if (self->external_semaphore.semaphore) {
        res = HIP_CHECK(hipDestroyExternalSemaphore(self->external_semaphore.semaphore));
    }

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

WwRenderer3DResult hiprt_renderer_set_target_resolution(ww_renderer3d_ptr self, u32 width, u32 height) {
    assert(self);

    WwRenderer3DResult res = HIP_CHECK(hipStreamSynchronize(self->stream));
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

WwRenderer3DResult hiprt_renderer_set_external_memory(ww_renderer3d_ptr self, WwViewportExternalHandle external_memory, u32 width, u32 height) {
    assert(self);

    WwRenderer3DResult res = HIP_CHECK(hipStreamSynchronize(self->stream));
    if (res.failed) {
        return res;
    }

    res = hiprt_renderer_free_target(self);
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

WwRenderer3DResult hiprt_renderer_render(ww_renderer3d_ptr self) {
    assert(self);
    assert(self->pixels.ptr);
    assert(self->scene);

    // scene creation
    WwRenderer3DResult res;
    if (self->scene->rebuild && (res = hiprt_scene_rebuild(self->scene)).failed) {
        return res;
    }

    if (self->external_semaphore.semaphore) {
        hipExternalSemaphoreWaitParams wait_params = {
            .params = {
                .fence = {
                    .value = *self->external_semaphore.value,
                },
            }
        };
        res = HIP_CHECK(hipWaitExternalSemaphoresAsync(&self->external_semaphore.semaphore, &wait_params, 1, self->stream));
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

    if (self->external_semaphore.semaphore) {
        hipExternalSemaphoreSignalParams signal_params = {
            .params = {
                .fence = {
                    .value = *self->external_semaphore.value + 1,
                },
            }
        };
        res = HIP_CHECK(hipSignalExternalSemaphoresAsync(&self->external_semaphore.semaphore, &signal_params, 1, self->stream));
        if (res.failed) {
            return res;
        }

        *self->external_semaphore.value += 1;
        return res;
    } else {
        return HIP_CHECK(hipStreamSynchronize(self->stream));
    }
}

WwRenderer3DResult hiprt_renderer_copy_target_to(ww_renderer3d_ptr self, void *dst) {
    assert(self);
    assert(!self->pixels.external_memory);
    WwRenderer3DResult res = HIP_CHECK(hipMemcpyDtoHAsync(dst, self->pixels.ptr, self->width * self->height * 4 * sizeof(f32), self->stream));
    if (res.failed) {
        return res;
    }

    return HIP_CHECK(hipStreamSynchronize(self->stream));
}

WwRenderer3DResult hiprt_renderer_set_scene(ww_renderer3d_ptr self, ww_scene_ptr scene) {
    assert(self);
    self->scene = scene;
    return ww_renderer3d_result(WW_RENDERER3D_SUCCESS);
}

WwRenderer3DResult hiprt_renderer_create_scene(ww_renderer3d_ptr self, WwScene* scene) {
    assert(self);
    return hiprt_scene_create(self->context, scene);
}

WwRenderer3DResult hiprt_renderer_create_camera(ww_renderer3d_ptr self, WwCamera* camera) {
    assert(self);
    return ww_camera_def_impl_create(self->context.allocator, camera);
}

WwRenderer3DResult hiprt_renderer_create_object_instance(ww_renderer3d_ptr self, const ww_triangle_mesh_ptr triangle_mesh, WwObjectInstance* object_instance) {
    assert(self);
    return hiprt_object_instance_create(self->context, triangle_mesh, object_instance);
}

WwRenderer3DResult hiprt_renderer_create_triangle_mesh(ww_renderer3d_ptr self, WwTriangleMeshCreationProperties creation_properties, WwTriangleMesh* triangle_mesh) {
    assert(self);
    return hiprt_triangle_mesh_create(self->context, creation_properties, triangle_mesh);
}

WwRenderer3DResult hiprt_renderer_free_target(ww_renderer3d_ptr self) {
    WwRenderer3DResult res = ww_renderer3d_result(WW_RENDERER3D_SUCCESS);
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
