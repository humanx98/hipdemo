#pragma once

#include <ww/defines.h>
#include <ww/math.h>
#include <ww/renderer/result.h>

WW_DEFINE_HANDLE(camera_ptr);
typedef RendererResult __ww_must_check (*camera_set_look_at_fn)(camera_ptr ptr, vec3 position, vec3 at, vec3 up);
typedef RendererResult __ww_must_check (*camera_set_f32_param_fn)(camera_ptr ptr, f32 value);
typedef void (*camera_destroy_fn)(camera_ptr ptr);

typedef struct camera_vtable {
    camera_set_look_at_fn set_look_at;
    camera_set_f32_param_fn set_aspect_ratio;
    camera_set_f32_param_fn set_aperture;
    camera_set_f32_param_fn set_focus_dist;
    camera_set_f32_param_fn set_vfov;
    camera_destroy_fn destroy;
} camera_vtable;

typedef struct Camera {
    camera_ptr ptr;
    const camera_vtable* vtable;
} Camera;

RendererResult __ww_must_check camera_set_aperture(Camera camera, f32 aperture);
RendererResult __ww_must_check camera_set_aspect_ratio(Camera camera, f32 aspect_ratio);
RendererResult __ww_must_check camera_set_focus_dist(Camera camera, f32 focus_dist);
RendererResult __ww_must_check camera_set_vfov(Camera camera, f32 vfov);
RendererResult __ww_must_check camera_set_look_at(Camera camera, vec3 position, vec3 at, vec3 up);
void camera_destroy(Camera camera);
