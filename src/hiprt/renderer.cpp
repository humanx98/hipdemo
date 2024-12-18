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
#include <ww/renderer.h>
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
    hipDeviceptr_t pixels;
    hipModule_t module;
    hipFunction_t kernel_func;
    scene_ptr scene;
} renderer_ptr_impl;

static RendererResult __ww_must_check hiprt_renderer_init(renderer_ptr self, HipRTCreationProperties creation_properties);
static RendererResult __ww_must_check hiprt_renderer_set_target_resolution(renderer_ptr self, u32 width, u32 height);
static RendererResult __ww_must_check hiprt_renderer_render(renderer_ptr self);
static RendererResult __ww_must_check hiprt_renderer_copy_target_to(renderer_ptr self, void* dst);
static RendererResult __ww_must_check hiprt_renderer_set_scene(renderer_ptr self, scene_ptr scene);
static RendererResult __ww_must_check hiprt_renderer_create_scene(renderer_ptr self, Scene* scene);
static RendererResult __ww_must_check hiprt_renderer_create_camera(renderer_ptr self, Camera* camera);
static RendererResult __ww_must_check hiprt_renderer_create_object_instance(renderer_ptr self, const triangle_mesh_ptr triangle_mesh, ObjectInstance* object_instance);
static RendererResult __ww_must_check hiprt_renderer_create_triangle_mesh(renderer_ptr self, TriangleMeshCreationProperties creation_properties, TriangleMesh* triangle_mesh);
static void hiprt_renderer_destroy(renderer_ptr self);

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
    return res;
}

void hiprt_renderer_destroy(renderer_ptr self) {
    assert(self);
    RendererResult res = {};

    if (self->module) {
        res = HIP_CHECK(hipModuleUnload(self->module));
    }

    if (self->pixels) {
        res = HIP_CHECK(hipFree(self->pixels));
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
    RendererResult res = {};
    if (self->pixels) {
        res = HIP_CHECK(hipFree(self->pixels));
        if (res.failed) {
        return res;
        } else {
        self->pixels = NULL;
        }
    }

    self->width = width;
    self->height = height;
    res = HIP_CHECK(hipMalloc(&self->pixels, self->width * self->height * 4 * sizeof(f32)));
    return res;
}

RendererResult hiprt_renderer_render(renderer_ptr self) {
    assert(self);
    assert(self->pixels);
    assert(self->scene);

    // scene creation
    RendererResult result;
    if (self->scene->rebuild && (result = hiprt_scene_rebuild(self->scene)).failed) {
        return result;
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
    void *args[] = {&self->scene->scene, &camera, &self->pixels, &resolution, &flip_y};
    int3 block = { 1024, 1, 1 };
    int3 grid = { ((self->width * self->height) + block.x - 1) / block.x, 1, 1 };
    return HIP_CHECK(hipModuleLaunchKernel(
        (hipFunction_t)self->kernel_func,
        grid.x,
        grid.y,
        grid.z,
        block.x,
        block.y,
        block.z,
        0,
        0,
        args,
        0
    ));
}

RendererResult hiprt_renderer_copy_target_to(renderer_ptr self, void *dst) {
    assert(self);
    RendererResult res = HIP_CHECK(hipMemcpyDtoH(dst, self->pixels, self->width * self->height * 4 * sizeof(f32)));
    return res;
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
