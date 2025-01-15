#pragma once

#include <ww/renderer3d/camera.h>
#include <ww/allocators/allocator.h>

typedef struct ww_camera_ptr_impl {
    WwAllocator allocator;
    b8 dirty;
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
} ww_camera_ptr_impl;

WwRenderer3DResult __ww_must_check ww_camera_def_impl_create(WwAllocator allocator, WwCamera* camera);
