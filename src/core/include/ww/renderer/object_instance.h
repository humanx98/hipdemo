#pragma once

#include <ww/math.h>
#include <ww/defines.h>
#include <ww/renderer/result.h>

WW_DEFINE_HANDLE(ww_object_instance_ptr);
typedef WwRendererResult __ww_must_check (*ww_object_instance_set_transform_fn)(ww_object_instance_ptr ptr, mat4 transform);
typedef void (*ww_object_instance_destroy_fn)(ww_object_instance_ptr ptr);

typedef struct ww_object_instance_vtable {
    ww_object_instance_set_transform_fn set_transform;
    ww_object_instance_destroy_fn destroy;
} ww_object_instance_vtable;

typedef struct WwObjectInstance {
    ww_object_instance_ptr ptr;
    const ww_object_instance_vtable* vtable;
} WwObjectInstance;

WwRendererResult __ww_must_check ww_object_instance_set_transform(WwObjectInstance self, mat4 transform);
void ww_object_instance_destroy(WwObjectInstance self);