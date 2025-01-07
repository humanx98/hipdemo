#include "ww/math.h"
#include <ww/renderer/camera_def_impl.h>

static void ww_camera_def_impl_init(ww_camera_ptr self, WwAllocator allocator, vec3 look_from, vec3 look_at, vec3 vup, f32 aspect_ratio, f32 vfov, f32 aperture, f32 focus_dist);
static WwRendererResult __ww_must_check ww_camera_def_impl_set_aperture(ww_camera_ptr self, f32 aperture);
static WwRendererResult __ww_must_check ww_camera_def_impl_set_aspect_ratio(ww_camera_ptr self, f32 aspect_ratio);
static WwRendererResult __ww_must_check ww_camera_def_impl_set_focus_dist(ww_camera_ptr self, f32 focus_dist);
static WwRendererResult __ww_must_check ww_camera_def_impl_set_vfov(ww_camera_ptr self, f32 vfov);
static WwRendererResult __ww_must_check ww_camera_def_impl_set_look_at(ww_camera_ptr self, vec3 position, vec3 at, vec3 up);
static WwRendererResult __ww_must_check ww_camera_def_impl_set_look_at(ww_camera_ptr self, vec3 position, vec3 at, vec3 up);
static void ww_camera_def_impl_destroy(ww_camera_ptr self);

WwRendererResult ww_camera_def_impl_create(WwAllocator allocator, WwCamera* camera) {
    assert(camera);
    ww_auto_type alloc_res = ww_allocator_alloc_type(allocator, ww_camera_ptr_impl);
    if (alloc_res.failed) {
        return ww_renderer_result(WW_RENDERER_ERROR_OUT_OF_HOST_MEMORY);
    }

    ww_camera_ptr self = alloc_res.ptr;

    vec3 look_from = make_vec3(13.0f, 2.0f, 3.0f);
    vec3 look_at = make_vec3(0.0f, 0.0f, 0.0f);
    vec3 vup = make_vec3(0.0f, 1.0f, 0.0f);
    f32 aspect_ratio = 1.0f;
    f32 vfov = WW_F32_PI / 9;
    f32 aperture = 0.1f;
    f32 focus_dist = 10.0f;
    ww_camera_def_impl_init(self, allocator, look_from, look_at, vup, aspect_ratio, vfov, aperture, focus_dist);

    static ww_camera_vtable vtable = {
        .set_aperture = ww_camera_def_impl_set_aperture,
        .set_aspect_ratio = ww_camera_def_impl_set_aspect_ratio,
        .set_focus_dist = ww_camera_def_impl_set_focus_dist,
        .set_look_at = ww_camera_def_impl_set_look_at,
        .set_vfov = ww_camera_def_impl_set_vfov,
        .destroy = ww_camera_def_impl_destroy,
    };

    *camera = (WwCamera){
        .ptr = self,
        .vtable = &vtable,
    };

    return ww_renderer_result(WW_RENDERER_SUCCESS);
}

void ww_camera_def_impl_init(ww_camera_ptr self, WwAllocator allocator, vec3 look_from, vec3 look_at, vec3 vup, f32 aspect_ratio, f32 vfov, f32 aperture, f32 focus_dist) {
    f32 theta = vfov;
    f32 h = tanf(theta / 2.0f);
    f32 viewport_height = 2.0f * h;
    f32 viewport_width = aspect_ratio * viewport_height;

    vec3 w = vec3_normalize(vec3_sub(look_from, look_at));
    vec3 u = vec3_normalize(vec3_cross(vup, w));
    vec3 v = vec3_cross(w, u);

    vec3 origin = look_from;
    vec3 horizontal = vec3_mul(u, focus_dist * viewport_width);
    vec3 vertical = vec3_mul(v, focus_dist * viewport_height);
    vec3 lower_left_corner = vec3_sub(origin, vec3_div(horizontal, 2.0f));
    lower_left_corner = vec3_sub(lower_left_corner, vec3_div(vertical, 2.0f));
    lower_left_corner = vec3_sub(lower_left_corner, vec3_mul(w, focus_dist));

    float lens_radius = aperture / 2.0f;

    *self = (ww_camera_ptr_impl){
        .allocator = allocator,
        .origin = origin,
        .lower_left_corner = lower_left_corner,
        .horizontal = horizontal,
        .vertical = vertical,
        .u = u,
        .v = v,
        .w = w,
        .lens_radius = lens_radius,
        .focus_dist = focus_dist,
        .vfov = vfov,
        .aspect_ratio = aspect_ratio,
        .look_from = look_from,
        .look_at = look_at,
        .vup = vup,
        .dirty = true,
    };
}

void ww_camera_def_impl_destroy(ww_camera_ptr self) {
    assert(self);
    ww_allocator_free(self->allocator, self);
}

WwRendererResult ww_camera_def_impl_set_aperture(ww_camera_ptr self, f32 aperture) {
    assert(self);
    ww_camera_def_impl_init(self, self->allocator, self->look_from, self->look_at, self->vup, self->aspect_ratio, self->vfov, aperture, self->focus_dist);
    return ww_renderer_result(WW_RENDERER_SUCCESS);
};

WwRendererResult ww_camera_def_impl_set_aspect_ratio(ww_camera_ptr self, f32 aspect_ratio) {
    assert(self);
    ww_camera_def_impl_init(self, self->allocator, self->look_from, self->look_at, self->vup, aspect_ratio, self->vfov, 2.0f * self->lens_radius, self->focus_dist);
    return ww_renderer_result(WW_RENDERER_SUCCESS);
}

WwRendererResult ww_camera_def_impl_set_focus_dist(ww_camera_ptr self, f32 focus_dist) {
    assert(self);
    ww_camera_def_impl_init(self, self->allocator, self->look_from, self->look_at, self->vup, self->aspect_ratio, self->vfov, 2.0f * self->lens_radius, focus_dist);
    return ww_renderer_result(WW_RENDERER_SUCCESS);
}

WwRendererResult ww_camera_def_impl_set_vfov(ww_camera_ptr self, f32 vfov) {
    assert(self);
    ww_camera_def_impl_init(self, self->allocator, self->look_from, self->look_at, self->vup, self->aspect_ratio, vfov, 2.0f * self->lens_radius, self->focus_dist);
    return ww_renderer_result(WW_RENDERER_SUCCESS);
}

WwRendererResult ww_camera_def_impl_set_look_at(ww_camera_ptr self, vec3 position, vec3 at, vec3 up) {
    assert(self);
    ww_camera_def_impl_init(self, self->allocator, position, at, up, self->aspect_ratio, self->vfov, 2.0f * self->lens_radius, self->focus_dist);
    return ww_renderer_result(WW_RENDERER_SUCCESS);
}
