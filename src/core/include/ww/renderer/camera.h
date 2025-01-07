#pragma once

#include <ww/defines.h>
#include <ww/math.h>
#include <ww/renderer/result.h>

WW_DEFINE_HANDLE(ww_camera_ptr);
typedef WwRendererResult __ww_must_check (*ww_camera_set_look_at_fn)(ww_camera_ptr ptr, vec3 position, vec3 at, vec3 up);
typedef WwRendererResult __ww_must_check (*ww_camera_set_f32_param_fn)(ww_camera_ptr ptr, f32 value);
typedef void (*ww_camera_destroy_fn)(ww_camera_ptr ptr);

typedef struct ww_camera_vtable {
    ww_camera_set_look_at_fn set_look_at;
    ww_camera_set_f32_param_fn set_aspect_ratio;
    ww_camera_set_f32_param_fn set_aperture;
    ww_camera_set_f32_param_fn set_focus_dist;
    ww_camera_set_f32_param_fn set_vfov;
    ww_camera_destroy_fn destroy;
} ww_camera_vtable;

typedef struct WwCamera {
    ww_camera_ptr ptr;
    const ww_camera_vtable* vtable;
} WwCamera;

WwRendererResult __ww_must_check ww_camera_set_aperture(WwCamera camera, f32 aperture);
WwRendererResult __ww_must_check ww_camera_set_aspect_ratio(WwCamera camera, f32 aspect_ratio);
WwRendererResult __ww_must_check ww_camera_set_focus_dist(WwCamera camera, f32 focus_dist);
WwRendererResult __ww_must_check ww_camera_set_vfov(WwCamera camera, f32 vfov);
WwRendererResult __ww_must_check ww_camera_set_look_at(WwCamera camera, vec3 position, vec3 at, vec3 up);
void ww_camera_destroy(WwCamera camera);
