#pragma once

#include <ww/defines.h>
#include <ww/prim_types.h>
#include <ww/renderer/result.h>
#include <ww/renderer/scene.h>
#include <ww/renderer/triangle_mesh.h>

WW_DEFINE_HANDLE(renderer_ptr);

typedef RendererResult __ww_must_check (*renderer_set_target_resolution_fn)(renderer_ptr ptr, u32 width, u32 height);
typedef RendererResult __ww_must_check (*renderer_render_fn)(renderer_ptr ptr);
typedef RendererResult __ww_must_check (*renderer_copy_target_to_fn)(renderer_ptr ptr, void* dst);
typedef RendererResult __ww_must_check (*renderer_set_scene_fn)(renderer_ptr ptr, scene_ptr scene);
typedef RendererResult __ww_must_check (*renderer_create_scene_fn)(renderer_ptr ptr, Scene* scene);
typedef RendererResult __ww_must_check (*renderer_create_camera_fn)(renderer_ptr ptr, Camera* camera);
typedef RendererResult __ww_must_check (*renderer_create_object_instance_fn)(renderer_ptr ptr, const triangle_mesh_ptr triangle_mesh, ObjectInstance* object_instance);
typedef RendererResult __ww_must_check (*renderer_create_triangle_mesh_fn)(renderer_ptr ptr, TriangleMeshCreationProperties creation_properties, TriangleMesh* triangle_mesh);
typedef void (*renderer_destroy_fn)(renderer_ptr ptr);

typedef struct renderer_vtable {
    renderer_set_target_resolution_fn set_target_resolution;
    renderer_render_fn render;
    renderer_copy_target_to_fn copy_target_to;
    renderer_set_scene_fn set_scene;
    renderer_create_camera_fn create_camera;
    renderer_create_object_instance_fn create_object_instance;
    renderer_create_scene_fn create_scene;
    renderer_create_triangle_mesh_fn create_triangle_mesh;
    renderer_destroy_fn destroy;
} renderer_vtable;

typedef struct Renderer {
    renderer_ptr ptr;
    const renderer_vtable* vtable;
} Renderer;

RendererResult __ww_must_check renderer_set_target_resolution(Renderer self, u32 width, u32 height);
RendererResult __ww_must_check renderer_render(Renderer self);
RendererResult __ww_must_check renderer_copy_target_to(Renderer self, void* dst);
RendererResult __ww_must_check renderer_set_scene(Renderer self, scene_ptr scene);
RendererResult __ww_must_check renderer_create_scene(Renderer self, Scene* scene);
RendererResult __ww_must_check renderer_create_camera(Renderer self, Camera* camera);
RendererResult __ww_must_check renderer_create_object_instance(Renderer self, const triangle_mesh_ptr triangle_mesh, ObjectInstance* object_instance);
RendererResult __ww_must_check renderer_create_triangle_mesh(Renderer self, TriangleMeshCreationProperties creation_properties, TriangleMesh* triangle_mesh);
void renderer_destroy(Renderer self);