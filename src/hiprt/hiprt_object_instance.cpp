#include "hiprt_object_instance.h"
#include "hiprt_triangle_mesh.h"
#include "hiprt_scene.h"
#include <cassert>

extern "C" {
#include <ww/exit.h>
#include <ww/math.h>
}

static __ww_must_check RendererResult hiprt_object_instance_init(object_instance_ptr self, HipRTRenderContext context, const triangle_mesh_ptr triangle_mesh);
static RendererResult __ww_must_check hiprt_object_instance_set_transform(object_instance_ptr self, mat4 transform);
static void hiprt_object_instance_destroy(object_instance_ptr self);

RendererResult hiprt_object_instance_create(HipRTRenderContext context, triangle_mesh_ptr triangle_mesh, ObjectInstance* object_instance) {
    assert(object_instance);
    ww_auto_type alloc_res = ww_allocator_alloc_type(context.allocator, object_instance_ptr_impl);
    if (alloc_res.failed) {
        return renderer_result(RENDERER_ERROR_OUT_OF_HOST_MEMORY);
    }

    object_instance_ptr self = alloc_res.ptr;
    RendererResult res = hiprt_object_instance_init(self, context, triangle_mesh);
    if (res.failed) {
        hiprt_object_instance_destroy(self);
        return res;
    }

    static object_instance_vtable vtable = {
        .set_transform = hiprt_object_instance_set_transform,
        .destroy = hiprt_object_instance_destroy,
    };
    *object_instance = {
        .ptr = self,
        .vtable = &vtable,
    };
    return res;
}

RendererResult hiprt_object_instance_init(object_instance_ptr self, HipRTRenderContext context, const triangle_mesh_ptr triangle_mesh) {
    *self = {
        .context = context,
        .instance = {
            .type = hiprtInstanceTypeGeometry,
            .geometry = triangle_mesh->geometry,
        },
        .transform = MAT4_IDENTITY,
        .scenes = ww_darray_init(context.allocator, scene_ptr),
    };

    return renderer_result(RENDERER_SUCCESS);
}

void hiprt_object_instance_destroy(object_instance_ptr self) {
    assert(self);
    assert(ww_darray_len(&self->scenes) == 0);

    ww_darray_deinit(&self->scenes);
    ww_allocator_free(self->context.allocator, self);
}

RendererResult hiprt_object_instance_set_transform(object_instance_ptr self, mat4 transform) {
    assert(self);
    self->transform = transform;
    ww_darray_foreach_by_ref(&self->scenes, scene_ptr, scene) {
        (*scene)->rebuild = true;
    }
    return renderer_result(RENDERER_SUCCESS);
}
