#include <ww/hip/renderer.h>
#include <hip/hip_runtime.h>
#include <ww/collections/darray.h>
#include <ww/file.h>
#include <ww/exit.h>
#include <ww/math.h>
#include <ww/renderer3d/renderer3d.h>
#include <ww/hip/common.h>

typedef struct ww_renderer3d_ptr_impl {
    WwAllocator allocator;
    u32 width;
    u32 height;
    hipDeviceptr_t pixels;
    hipModule_t module;
    hipFunction_t kernel_func;
} ww_renderer3d_ptr_impl;

static WwRenderer3DResult __ww_must_check hip_renderer_init(ww_renderer3d_ptr self, HipCreationProperties creation_properties);
static void hip_renderer_destroy(ww_renderer3d_ptr self);
static WwRenderer3DResult __ww_must_check hip_renderer_set_target_resolution(ww_renderer3d_ptr self, u32 width, u32 height);
static WwRenderer3DResult __ww_must_check hip_renderer_render(ww_renderer3d_ptr self);
static WwRenderer3DResult __ww_must_check hip_renderer_copy_target_to(ww_renderer3d_ptr self, void* dst);
static WwRenderer3DResult __ww_must_check hip_renderer_set_scene(ww_renderer3d_ptr self, ww_scene_ptr scene);
static WwRenderer3DResult __ww_must_check hip_renderer_create_scene(ww_renderer3d_ptr self, WwScene* scene);
static WwRenderer3DResult __ww_must_check hip_renderer_create_camera(ww_renderer3d_ptr self, WwCamera* camera);
static WwRenderer3DResult __ww_must_check hip_renderer_create_object_instance(ww_renderer3d_ptr self, const ww_triangle_mesh_ptr triangle_mesh, WwObjectInstance* object_instance);
static WwRenderer3DResult __ww_must_check hip_renderer_create_triangle_mesh(ww_renderer3d_ptr self, WwTriangleMeshCreationProperties creation_properties, WwTriangleMesh* triangle_mesh);

WwRenderer3DResult hip_renderer_create(HipCreationProperties creation_properties, WwRenderer3D* renderer) {
    assert(renderer);

    ww_auto_type alloc_result = ww_allocator_alloc_type(creation_properties.allocator, ww_renderer3d_ptr_impl);
    if (alloc_result.failed) {
        return ww_renderer3d_result(WW_RENDERER3D_ERROR_OUT_OF_HOST_MEMORY);
    }

    ww_renderer3d_ptr self = alloc_result.ptr; 
    WwRenderer3DResult res = hip_renderer_init(self, creation_properties);
    if (res.failed) {
        hip_renderer_destroy(self);
        return res;
    }

    static const ww_renderer3d_vtable vtable = {
        .set_target_resolution = hip_renderer_set_target_resolution,
        .render = hip_renderer_render,
        .copy_target_to = hip_renderer_copy_target_to,
        .set_scene = hip_renderer_set_scene,
        .create_camera = hip_renderer_create_camera,
        .create_object_instance = hip_renderer_create_object_instance,
        .create_scene = hip_renderer_create_scene,
        .create_triangle_mesh = hip_renderer_create_triangle_mesh,
        .destroy = hip_renderer_destroy,
    };

    *renderer = (WwRenderer3D){
        .ptr = self,
        .vtable = &vtable,
    };
    return res;
}

WwRenderer3DResult hip_renderer_init(ww_renderer3d_ptr self, HipCreationProperties creation_properties) {
    *self = (ww_renderer3d_ptr_impl){ 
        .allocator = creation_properties.allocator,
    };

    WwRenderer3DResult res = HIP_CHECK(hipInit(0));
    if(res.failed) {
        return res;
    }

    res = HIP_CHECK(hipSetDevice(creation_properties.device_index));
    if(res.failed) {
        return res;
    }

    if (res.failed) {
        return res;
    }

    res = HIP_CHECK(hipModuleLoad(&self->module, "hip_renderer.hipfb"));
    if (res.failed) {
        return res;
    }
    
    res = HIP_CHECK(hipModuleGetFunction(&self->kernel_func, self->module, "ray_trace"));
    return res;
}

void hip_renderer_destroy(ww_renderer3d_ptr self) {
    assert(self);
    WwRenderer3DResult res = {};

    if (self->module) {
        res = HIP_CHECK(hipModuleUnload(self->module));
    }

    if (self->pixels) {
        res = HIP_CHECK(hipFree(self->pixels));
    }

    ww_allocator_free(self->allocator, self);
}


WwRenderer3DResult hip_renderer_set_target_resolution(ww_renderer3d_ptr self, u32 width, u32 height) {
    assert(self);
    WwRenderer3DResult res = {};
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

WwRenderer3DResult hip_renderer_render(ww_renderer3d_ptr self) {
    assert(self);

    ivec2 res = { (i32)self->width, (i32)self->height };
    void* args[] = { &self->pixels, &res };
    int3 block = { 1024, 1, 1 };
    int3 grid = { ((self->width * self->height) + block.x - 1) / block.x, 1, 1 };
    return HIP_CHECK(hipModuleLaunchKernel(
        self->kernel_func,
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

WwRenderer3DResult hip_renderer_copy_target_to(ww_renderer3d_ptr self, void* dst) {
    assert(self);
    WwRenderer3DResult res = HIP_CHECK(hipMemcpyDtoH(dst, self->pixels, self->width * self->height * 4 * sizeof(f32)));
    return res;
}

WwRenderer3DResult hip_renderer_set_scene(ww_renderer3d_ptr self, ww_scene_ptr scene) {
    WW_EXIT_WITH_MSG("set scene is not implemented\n");
}

WwRenderer3DResult hip_renderer_create_scene(ww_renderer3d_ptr self, WwScene* scene) {
    WW_EXIT_WITH_MSG("create scene is not implemented\n");
}

WwRenderer3DResult hip_renderer_create_camera(ww_renderer3d_ptr self, WwCamera* camera) {
    WW_EXIT_WITH_MSG("create camera is not implemented\n");
}

WwRenderer3DResult hip_renderer_create_object_instance(ww_renderer3d_ptr self, const ww_triangle_mesh_ptr triangle_mesh, WwObjectInstance* object_instance) {
    WW_EXIT_WITH_MSG("create object instance is not implemented\n");
}

WwRenderer3DResult hip_renderer_create_triangle_mesh(ww_renderer3d_ptr self, WwTriangleMeshCreationProperties creation_properties, WwTriangleMesh* triangle_mesh) {
    WW_EXIT_WITH_MSG("create triangle mesh is not imeplemented\n");
}

