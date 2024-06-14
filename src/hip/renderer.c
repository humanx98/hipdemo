#include <ww/hip/renderer.h>
#include <hip/hip_runtime.h>
#include <ww/collections/darray.h>
#include <ww/file.h>
#include <ww/exit.h>
#include <ww/math.h>
#include <ww/renderer.h>
#include <ww/hip/common.h>

typedef struct renderer_ptr_impl {
    WwAllocator allocator;
    u32 width;
    u32 height;
    hipDeviceptr_t pixels;
    hipModule_t module;
    hipFunction_t kernel_func;
} renderer_ptr_impl;

static RendererResult __ww_must_check hip_renderer_init(renderer_ptr self, HipCreationProperties creation_properties);
static void hip_renderer_destroy(renderer_ptr self);
static RendererResult __ww_must_check hip_renderer_set_target_resolution(renderer_ptr self, u32 width, u32 height);
static RendererResult __ww_must_check hip_renderer_render(renderer_ptr self);
static RendererResult __ww_must_check hip_renderer_copy_target_to(renderer_ptr self, void* dst);
static RendererResult __ww_must_check hip_renderer_set_scene(renderer_ptr self, scene_ptr scene);
static RendererResult __ww_must_check hip_renderer_create_scene(renderer_ptr self, Scene* scene);
static RendererResult __ww_must_check hip_renderer_create_camera(renderer_ptr self, Camera* camera);
static RendererResult __ww_must_check hip_renderer_create_object_instance(renderer_ptr self, const triangle_mesh_ptr triangle_mesh, ObjectInstance* object_instance);
static RendererResult __ww_must_check hip_renderer_create_triangle_mesh(renderer_ptr self, TriangleMeshCreationProperties creation_properties, TriangleMesh* triangle_mesh);

RendererResult hip_renderer_create(HipCreationProperties creation_properties, Renderer* renderer) {
    assert(renderer);

    ww_auto_type alloc_result = ww_allocator_alloc_type(creation_properties.allocator, renderer_ptr_impl);
    if (alloc_result.failed) {
        return renderer_result(RENDERER_ERROR_OUT_OF_HOST_MEMORY);
    }

    renderer_ptr self = alloc_result.ptr; 
    RendererResult res = hip_renderer_init(self, creation_properties);
    if (res.failed) {
        hip_renderer_destroy(self);
        return res;
    }

    static const renderer_vtable vtable = {
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

    *renderer = (Renderer){
        .ptr = self,
        .vtable = &vtable,
    };
    return res;
}

RendererResult hip_renderer_init(renderer_ptr self, HipCreationProperties creation_properties) {
    *self = (renderer_ptr_impl){ 
        .allocator = creation_properties.allocator,
    };

    RendererResult res = HIP_CHECK(hipInit(0));
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

void hip_renderer_destroy(renderer_ptr self) {
    assert(self);
    RendererResult res = {};

    if (self->module) {
        res = HIP_CHECK(hipModuleUnload(self->module));
    }

    if (self->pixels) {
        res = HIP_CHECK(hipFree(self->pixels));
    }

    ww_allocator_free(self->allocator, self);
}


RendererResult hip_renderer_set_target_resolution(renderer_ptr self, u32 width, u32 height) {
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

RendererResult hip_renderer_render(renderer_ptr self) {
    assert(self);

    ivec2 res = { (i32)self->width, (i32)self->height };
    void* args[] = { &self->pixels, &res };
    i32 bx = 8;
    i32 by = 8;
    ivec3 nb;
    nb.x = (self->width + bx - 1) / bx;
    nb.y = (self->height + by - 1) / by;
    return HIP_CHECK(hipModuleLaunchKernel((hipFunction_t)self->kernel_func, nb.x, nb.y, 1, bx, by, 1, 0, 0, args, 0));
}

RendererResult hip_renderer_copy_target_to(renderer_ptr self, void* dst) {
    assert(self);
    RendererResult res = HIP_CHECK(hipMemcpyDtoH(dst, self->pixels, self->width * self->height * 4 * sizeof(f32)));
    return res;
}

RendererResult hip_renderer_set_scene(renderer_ptr self, scene_ptr scene) {
    WW_EXIT_WITH_MSG("set scene is not implemented\n");
}

RendererResult hip_renderer_create_scene(renderer_ptr self, Scene* scene) {
    WW_EXIT_WITH_MSG("create scene is not implemented\n");
}

RendererResult hip_renderer_create_camera(renderer_ptr self, Camera* camera) {
    WW_EXIT_WITH_MSG("create camera is not implemented\n");
}

RendererResult hip_renderer_create_object_instance(renderer_ptr self, const triangle_mesh_ptr triangle_mesh, ObjectInstance* object_instance) {
    WW_EXIT_WITH_MSG("create object instance is not implemented\n");
}

RendererResult hip_renderer_create_triangle_mesh(renderer_ptr self, TriangleMeshCreationProperties creation_properties, TriangleMesh* triangle_mesh) {
    WW_EXIT_WITH_MSG("create triangle mesh is not imeplemented\n");
}

