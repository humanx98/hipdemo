#include "hiprt_triangle_mesh.h"
#include <cassert>

extern "C" {
#include <ww/hip/common.h>
}

static WwRendererResult __ww_must_check hiprt_triangle_mesh_init(ww_triangle_mesh_ptr self, WwTriangleMeshCreationProperties creation_properties, HipRTRenderContext context);
static void hiprt_triangle_mesh_destroy(ww_triangle_mesh_ptr self);

WwRendererResult hiprt_triangle_mesh_create(HipRTRenderContext context, WwTriangleMeshCreationProperties creation_properties, WwTriangleMesh* triangle_mesh) {
    assert(triangle_mesh);

    ww_auto_type alloc_res = ww_allocator_alloc_type(context.allocator, ww_triangle_mesh_ptr_impl);
    if (alloc_res.failed) {
        return ww_renderer_result(WW_RENDERER_ERROR_OUT_OF_HOST_MEMORY);
    }

    ww_triangle_mesh_ptr self = alloc_res.ptr;
    WwRendererResult res = hiprt_triangle_mesh_init(self, creation_properties, context);
    if (res.failed) {
        hiprt_triangle_mesh_destroy(self);
        return res;
    }

    static ww_triangle_mesh_vtable vtable = {
        .destroy = hiprt_triangle_mesh_destroy,
    };
    *triangle_mesh = {
        .ptr = self,
        .vtable = &vtable,
    };

    return res;
}

WwRendererResult hiprt_triangle_mesh_init(ww_triangle_mesh_ptr self, WwTriangleMeshCreationProperties creation_properties, HipRTRenderContext context) {
    *self = {
        .context = context
    };

    self->mesh.triangleCount = creation_properties.triangle_count;
    self->mesh.triangleStride = sizeof(int3);
    WwRendererResult res = HIP_CHECK(hipMalloc(&self->mesh.triangleIndices, self->mesh.triangleCount * sizeof(int3)));
    if (res.failed) {
        return res;
    }

    res = HIP_CHECK(hipMemcpyHtoD(self->mesh.triangleIndices, creation_properties.triangle_indices, self->mesh.triangleCount * sizeof(int3)));
    if (res.failed) {
        return res;
    }

    self->mesh.vertexCount = creation_properties.vertex_count;
    self->mesh.vertexStride = sizeof(vec3);
    res = HIP_CHECK(hipMalloc(&self->mesh.vertices, self->mesh.vertexCount * sizeof(vec3)));
    if (res.failed) {
        return res;
    }

    res = HIP_CHECK(hipMemcpyHtoD(self->mesh.vertices, creation_properties.vertices, self->mesh.vertexCount * sizeof(vec3)));
    if (res.failed) {
        return res;
    }

    hiprtGeometryBuildInput geom_input;
    geom_input.type = hiprtPrimitiveTypeTriangleMesh;
    geom_input.primitive.triangleMesh = self->mesh;

    usize geom_temp_size;
    hiprtBuildOptions options = {
        .buildFlags = hiprtBuildFlagBitPreferFastBuild,
    };
    res = HIPRT_CHECK(hiprtGetGeometryBuildTemporaryBufferSize(self->context.hiprt, geom_input, options, geom_temp_size));
    if (res.failed) {
        return res;
    }
    res = HIP_CHECK(hipMalloc(&self->geometry_temp, geom_temp_size));
    if (res.failed) {
        return res;
    }

    res = HIPRT_CHECK(hiprtCreateGeometry(self->context.hiprt, geom_input, options, self->geometry));
    if (res.failed) {
        return res;
    }
    res = HIPRT_CHECK(hiprtBuildGeometry(self->context.hiprt, hiprtBuildOperationBuild, geom_input, options, self->geometry_temp, 0, self->geometry));
    if (res.failed) {
        return res;
    }

    return ww_renderer_result(WW_RENDERER_SUCCESS);
}

void hiprt_triangle_mesh_destroy(ww_triangle_mesh_ptr self) {
    assert(self);

    WwRendererResult res = {};

    if (self->geometry) {
        res = HIPRT_CHECK(hiprtDestroyGeometry(self->context.hiprt, self->geometry));
    }

    if (self->geometry_temp) {
        res = HIP_CHECK(hipFree(self->geometry_temp));
    }

    if (self->mesh.triangleIndices) {
        res = HIP_CHECK(hipFree(self->mesh.triangleIndices));
    }

    if (self->mesh.vertices) {
        res = HIP_CHECK(hipFree(self->mesh.vertices));
    }

    ww_allocator_free(self->context.allocator, self);
}
