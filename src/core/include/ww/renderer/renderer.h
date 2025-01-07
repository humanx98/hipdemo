#pragma once

#include <ww/defines.h>
#include <ww/prim_types.h>
#include <ww/viewport.h>
#include <ww/renderer/result.h>
#include <ww/renderer/scene.h>
#include <ww/renderer/triangle_mesh.h>

WW_DEFINE_HANDLE(ww_renderer_ptr);

typedef WwRendererResult __ww_must_check (*ww_renderer_set_target_resolution_fn)(ww_renderer_ptr ptr, u32 width, u32 height);
typedef WwRendererResult __ww_must_check (*ww_renderer_set_target_external_memory_fn)(ww_renderer_ptr ptr, WwViewportExternalHandle external_memory, u32 width, u32 height);
typedef WwRendererResult __ww_must_check (*ww_renderer_render_fn)(ww_renderer_ptr ptr);
typedef WwRendererResult __ww_must_check (*ww_renderer_copy_target_to_fn)(ww_renderer_ptr ptr, void* dst);
typedef WwRendererResult __ww_must_check (*ww_renderer_set_scene_fn)(ww_renderer_ptr ptr, ww_scene_ptr scene);
typedef WwRendererResult __ww_must_check (*ww_renderer_create_scene_fn)(ww_renderer_ptr ptr, WwScene* scene);
typedef WwRendererResult __ww_must_check (*ww_renderer_create_camera_fn)(ww_renderer_ptr ptr, WwCamera* camera);
typedef WwRendererResult __ww_must_check (*ww_renderer_create_object_instance_fn)(ww_renderer_ptr ptr, const ww_triangle_mesh_ptr triangle_mesh, WwObjectInstance* object_instance);
typedef WwRendererResult __ww_must_check (*ww_renderer_create_triangle_mesh_fn)(ww_renderer_ptr ptr, WwTriangleMeshCreationProperties creation_properties, WwTriangleMesh* triangle_mesh);
typedef void (*ww_renderer_destroy_fn)(ww_renderer_ptr ptr);

typedef struct ww_renderer_vtable {
    ww_renderer_set_target_resolution_fn set_target_resolution;
    ww_renderer_set_target_external_memory_fn set_target_external_memory;
    ww_renderer_render_fn render;
    ww_renderer_copy_target_to_fn copy_target_to;
    ww_renderer_set_scene_fn set_scene;
    ww_renderer_create_camera_fn create_camera;
    ww_renderer_create_object_instance_fn create_object_instance;
    ww_renderer_create_scene_fn create_scene;
    ww_renderer_create_triangle_mesh_fn create_triangle_mesh;
    ww_renderer_destroy_fn destroy;
} ww_renderer_vtable;

typedef struct WwRenderer {
    ww_renderer_ptr ptr;
    const ww_renderer_vtable* vtable;
} WwRenderer;

WwRendererResult __ww_must_check ww_renderer_set_target_resolution(WwRenderer self, u32 width, u32 height);
WwRendererResult __ww_must_check ww_renderer_set_target_external_memory(WwRenderer self, WwViewportExternalHandle external_memory, u32 width, u32 height);
WwRendererResult __ww_must_check ww_renderer_render(WwRenderer self);
WwRendererResult __ww_must_check ww_renderer_copy_target_to(WwRenderer self, void* dst);
WwRendererResult __ww_must_check ww_renderer_set_scene(WwRenderer self, ww_scene_ptr scene);
WwRendererResult __ww_must_check ww_renderer_create_scene(WwRenderer self, WwScene* scene);
WwRendererResult __ww_must_check ww_renderer_create_camera(WwRenderer self, WwCamera* camera);
WwRendererResult __ww_must_check ww_renderer_create_object_instance(WwRenderer self, const ww_triangle_mesh_ptr triangle_mesh, WwObjectInstance* object_instance);
WwRendererResult __ww_must_check ww_renderer_create_triangle_mesh(WwRenderer self, WwTriangleMeshCreationProperties creation_properties, WwTriangleMesh* triangle_mesh);
void ww_renderer_destroy(WwRenderer self);