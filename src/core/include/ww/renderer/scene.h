#pragma once

#include <ww/renderer/result.h>
#include <ww/renderer/camera.h>
#include <ww/renderer/object_instance.h>

WW_DEFINE_HANDLE(scene_ptr);

typedef RendererResult __ww_must_check (*scene_set_camera_fn)(scene_ptr ptr, camera_ptr camera);
typedef RendererResult __ww_must_check (*scene_attach_detach_object_instance_fn)(scene_ptr ptr, object_instance_ptr object_instance);
typedef void (*scene_destroy_fn)(scene_ptr ptr);


typedef struct scene_vtable {
    scene_set_camera_fn set_camera;
    scene_attach_detach_object_instance_fn attach_object_instance;
    scene_attach_detach_object_instance_fn detach_object_instance;
    scene_destroy_fn destroy;
} scene_vtable;

typedef struct Scene {
    scene_ptr ptr;
    const scene_vtable* vtable;
} Scene;

RendererResult __ww_must_check scene_set_camera(Scene self, camera_ptr camera);
RendererResult __ww_must_check scene_attach_object_instance(Scene self, object_instance_ptr object_instance);
RendererResult __ww_must_check scene_detach_object_instance(Scene self, object_instance_ptr object_instance);
void scene_destroy(Scene self);
