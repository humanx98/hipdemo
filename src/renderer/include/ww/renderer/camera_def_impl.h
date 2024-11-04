#pragma once

#include <ww/renderer/camera.h>
#include <ww/allocators/allocator.h>

typedef struct camera_ptr_impl {
    WwAllocator allocator;
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u;
    vec3 v;
    vec3 w;
    f32 lens_radius;
    f32 vfov;
    f32 focus_dist;
    f32 aspect_ratio;
    vec3 look_from;
    vec3 look_at;
    vec3 vup;
    b8 dirty;
} camera_ptr_impl;

RendererResult __ww_must_check camera_def_impl_create(WwAllocator allocator, Camera* camera);
