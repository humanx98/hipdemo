#pragma once

#include <ww/math.h>
#include <ww/defines.h>
#include <ww/renderer/result.h>

WW_DEFINE_HANDLE(object_instance_ptr);
typedef RendererResult __ww_must_check (*object_instance_set_transform_fn)(object_instance_ptr ptr, mat4 transform);
typedef void (*object_instance_destroy_fn)(object_instance_ptr ptr);

typedef struct object_instance_vtable {
    object_instance_set_transform_fn set_transform;
    object_instance_destroy_fn destroy;
} object_instance_vtable;

typedef struct ObjectInstance {
    object_instance_ptr ptr;
    const object_instance_vtable* vtable;
} ObjectInstance;

RendererResult __ww_must_check object_instance_set_transform(ObjectInstance self, mat4 transform);
void object_instance_destroy(ObjectInstance self);