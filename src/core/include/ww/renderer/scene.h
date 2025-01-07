#pragma once

#include <ww/renderer/result.h>
#include <ww/renderer/camera.h>
#include <ww/renderer/object_instance.h>

WW_DEFINE_HANDLE(ww_scene_ptr);

typedef WwRendererResult __ww_must_check (*ww_scene_set_camera_fn)(ww_scene_ptr ptr, ww_camera_ptr camera);
typedef WwRendererResult __ww_must_check (*ww_scene_attach_detach_object_instance_fn)(ww_scene_ptr ptr, ww_object_instance_ptr object_instance);
typedef void (*ww_scene_destroy_fn)(ww_scene_ptr ptr);


typedef struct ww_scene_vtable {
    ww_scene_set_camera_fn set_camera;
    ww_scene_attach_detach_object_instance_fn attach_object_instance;
    ww_scene_attach_detach_object_instance_fn detach_object_instance;
    ww_scene_destroy_fn destroy;
} ww_scene_vtable;

typedef struct WwScene {
    ww_scene_ptr ptr;
    const ww_scene_vtable* vtable;
} WwScene;

WwRendererResult __ww_must_check ww_scene_set_camera(WwScene self, ww_camera_ptr camera);
WwRendererResult __ww_must_check ww_scene_attach_object_instance(WwScene self, ww_object_instance_ptr object_instance);
WwRendererResult __ww_must_check ww_scene_detach_object_instance(WwScene self, ww_object_instance_ptr object_instance);
void ww_scene_destroy(WwScene self);
