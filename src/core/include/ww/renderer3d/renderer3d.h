#pragma once

#include <ww/defines.h>
#include <ww/prim_types.h>
#include <ww/viewport.h>
#include <ww/renderer3d/result.h>
#include <ww/renderer3d/scene.h>
#include <ww/renderer3d/triangle_mesh.h>

WW_DEFINE_HANDLE(ww_renderer3d_ptr);

typedef WwRenderer3DResult __ww_must_check (*ww_renderer3d_set_target_resolution_fn)(ww_renderer3d_ptr ptr, u32 width, u32 height);
typedef WwRenderer3DResult __ww_must_check (*ww_renderer3d_set_target_external_memory_fn)(ww_renderer3d_ptr ptr, WwViewportExternalHandle external_memory, u32 width, u32 height);
typedef WwRenderer3DResult __ww_must_check (*ww_renderer3d_render_fn)(ww_renderer3d_ptr ptr);
typedef WwRenderer3DResult __ww_must_check (*ww_renderer3d_copy_target_to_fn)(ww_renderer3d_ptr ptr, void* dst);
typedef WwRenderer3DResult __ww_must_check (*ww_renderer3d_set_scene_fn)(ww_renderer3d_ptr ptr, ww_scene_ptr scene);
typedef WwRenderer3DResult __ww_must_check (*ww_renderer3d_create_scene_fn)(ww_renderer3d_ptr ptr, WwScene* scene);
typedef WwRenderer3DResult __ww_must_check (*ww_renderer3d_create_camera_fn)(ww_renderer3d_ptr ptr, WwCamera* camera);
typedef WwRenderer3DResult __ww_must_check (*ww_renderer3d_create_object_instance_fn)(ww_renderer3d_ptr ptr, const ww_triangle_mesh_ptr triangle_mesh, WwObjectInstance* object_instance);
typedef WwRenderer3DResult __ww_must_check (*ww_renderer3d_create_triangle_mesh_fn)(ww_renderer3d_ptr ptr, WwTriangleMeshCreationProperties creation_properties, WwTriangleMesh* triangle_mesh);
typedef void (*ww_renderer3d_destroy_fn)(ww_renderer3d_ptr ptr);

typedef struct ww_renderer3d_vtable {
    ww_renderer3d_set_target_resolution_fn set_target_resolution;
    ww_renderer3d_set_target_external_memory_fn set_target_external_memory;
    ww_renderer3d_render_fn render;
    ww_renderer3d_copy_target_to_fn copy_target_to;
    ww_renderer3d_set_scene_fn set_scene;
    ww_renderer3d_create_camera_fn create_camera;
    ww_renderer3d_create_object_instance_fn create_object_instance;
    ww_renderer3d_create_scene_fn create_scene;
    ww_renderer3d_create_triangle_mesh_fn create_triangle_mesh;
    ww_renderer3d_destroy_fn destroy;
} ww_renderer3d_vtable;

typedef struct WwRenderer3D {
    ww_renderer3d_ptr ptr;
    const ww_renderer3d_vtable* vtable;
} WwRenderer3D;

WwRenderer3DResult __ww_must_check ww_renderer3d_set_target_resolution(WwRenderer3D self, u32 width, u32 height);
WwRenderer3DResult __ww_must_check ww_renderer3d_set_target_external_memory(WwRenderer3D self, WwViewportExternalHandle external_memory, u32 width, u32 height);
WwRenderer3DResult __ww_must_check ww_renderer3d_render(WwRenderer3D self);
WwRenderer3DResult __ww_must_check ww_renderer3d_copy_target_to(WwRenderer3D self, void* dst);
WwRenderer3DResult __ww_must_check ww_renderer3d_set_scene(WwRenderer3D self, ww_scene_ptr scene);
WwRenderer3DResult __ww_must_check ww_renderer3d_create_scene(WwRenderer3D self, WwScene* scene);
WwRenderer3DResult __ww_must_check ww_renderer3d_create_camera(WwRenderer3D self, WwCamera* camera);
WwRenderer3DResult __ww_must_check ww_renderer3d_create_object_instance(WwRenderer3D self, const ww_triangle_mesh_ptr triangle_mesh, WwObjectInstance* object_instance);
WwRenderer3DResult __ww_must_check ww_renderer3d_create_triangle_mesh(WwRenderer3D self, WwTriangleMeshCreationProperties creation_properties, WwTriangleMesh* triangle_mesh);
void ww_renderer3d_destroy(WwRenderer3D self);