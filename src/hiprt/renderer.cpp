#include <ww/hiprt/renderer.h>
#include <cassert>
#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>
#include <string>
#include <vector>
#include "hiprt_common.h"
#include "hiprt_object_instance.h"
#include "hiprt_scene.h"

extern "C" {
#include <ww/hip/common.h>
#include <ww/renderer.h>
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
    if (!self->scene->scene)
    {
        std::vector<hiprtInstance> instances;
        std::vector<hiprtTransformHeader> transform_headers;
        std::vector<hiprtFrameMatrix> frame_matrices;
        instances.reserve(self->scene->attached_object_instances.size());
        transform_headers.reserve(self->scene->attached_object_instances.size());
        frame_matrices.reserve(self->scene->attached_object_instances.size());
        for (auto aoi : self->scene->attached_object_instances) {
            instances.push_back(aoi->instance);
            hiprtTransformHeader header = {
                .frameIndex = (u32)transform_headers.size(),
                .frameCount = 1,
            };
            transform_headers.push_back(header);
            hiprtFrameMatrix matrix = {
                .matrix = {
                    { aoi->transform.e[0][0], aoi->transform.e[0][1], aoi->transform.e[0][2], aoi->transform.e[0][3] },
                    { aoi->transform.e[1][0], aoi->transform.e[1][1], aoi->transform.e[1][2], aoi->transform.e[1][3] },
                    { aoi->transform.e[2][0], aoi->transform.e[2][1], aoi->transform.e[2][2], aoi->transform.e[2][3] },
                },
            };
            frame_matrices.push_back(matrix);
        }

        self->scene->scene_input.instanceCount = instances.size();
        self->scene->scene_input.instanceMasks = nullptr;
        RendererResult res = HIP_CHECK(hipMalloc(&self->scene->scene_input.instances, instances.size() * sizeof(hiprtInstance)));
        if (res.failed) {
            return res;
        }
        res = HIP_CHECK(hipMemcpyHtoD(self->scene->scene_input.instances, instances.data(), instances.size() * sizeof(hiprtInstance)));
        if (res.failed) {
            return res;
        }

        res = HIP_CHECK(hipMalloc(&self->scene->scene_input.instanceTransformHeaders, transform_headers.size() * sizeof(hiprtTransformHeader)));
        if (res.failed) {
            return res;
        }

        res = HIP_CHECK(hipMemcpyHtoD(self->scene->scene_input.instanceTransformHeaders, transform_headers.data(), transform_headers.size() * sizeof(hiprtTransformHeader)));
        if (res.failed) {
            return res;
        }

        self->scene->scene_input.frameType = hiprtFrameTypeMatrix;
        self->scene->scene_input.frameCount = frame_matrices.size();
        res = HIP_CHECK(hipMalloc(&self->scene->scene_input.instanceFrames, frame_matrices.size() * sizeof(hiprtFrameMatrix)));
        if (res.failed) {
            return res;
        }

        res = HIP_CHECK(hipMemcpyHtoD(self->scene->scene_input.instanceFrames, frame_matrices.data(), frame_matrices.size() * sizeof(hiprtFrameMatrix)));
        if (res.failed) {
            return res;
        }

        usize scene_temp_size;
        hiprtBuildOptions options = {
            .buildFlags = hiprtBuildFlagBitPreferFastBuild,
        };
        res = HIPRT_CHECK(hiprtGetSceneBuildTemporaryBufferSize(self->context.hiprt, self->scene->scene_input, options, scene_temp_size));
        if (res.failed) {
            return res;
        }

        res = HIP_CHECK(hipMalloc(&self->scene->scene_temp, scene_temp_size));
        if (res.failed) {
            return res;
        }

        res = HIPRT_CHECK(hiprtCreateScene(self->context.hiprt, self->scene->scene_input, options, self->scene->scene));
        if (res.failed) {
            return res;
        }
        res = HIPRT_CHECK(hiprtBuildScene(self->context.hiprt, hiprtBuildOperationBuild, self->scene->scene_input, options, self->scene->scene_temp, 0, self->scene->scene));
        if (res.failed) {
            return res;
        }
    }

    hiprtInt2 res = {(i32)self->width, (i32)self->height};
    void *args[] = {&self->scene->scene, &self->pixels, &res};
    i32 bx = 8;
    i32 by = 8;
    hiprtInt3 nb;
    nb.x = (self->width + bx - 1) / bx;
    nb.y = (self->height + by - 1) / by;
    return HIP_CHECK(hipModuleLaunchKernel((hipFunction_t)self->kernel_func, nb.x, nb.y, 1, bx, by, 1, 0, 0, args, 0));
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
    WW_EXIT_WITH_MSG("create camera is not implemented\n");
}

RendererResult hiprt_renderer_create_object_instance(renderer_ptr self, const triangle_mesh_ptr triangle_mesh, ObjectInstance* object_instance) {
    assert(self);
    return hiprt_object_instance_create(self->context, triangle_mesh, object_instance);
}

RendererResult hiprt_renderer_create_triangle_mesh(renderer_ptr self, TriangleMeshCreationProperties creation_properties, TriangleMesh* triangle_mesh) {
    assert(self);
    return hiprt_triangle_mesh_create(self->context, creation_properties, triangle_mesh);
}
