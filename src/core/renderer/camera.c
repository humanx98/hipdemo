#include <ww/renderer/camera.h>
#include <assert.h>

static inline void assert_camera(Camera camera) {
    assert(camera.ptr);
    assert(camera.vtable);
    assert(camera.vtable->set_aperture);
    assert(camera.vtable->set_aspect_ratio);
    assert(camera.vtable->set_focus_dist);
    assert(camera.vtable->set_look_at);
    assert(camera.vtable->set_vfov);
    assert(camera.vtable->destroy);
}

RendererResult camera_set_aperture(Camera camera, f32 aperture) {
    assert_camera(camera);
    return camera.vtable->set_aperture(camera.ptr, aperture);
}

RendererResult camera_set_aspect_ratio(Camera camera, f32 aspect_ratio) {
    assert_camera(camera);
    return camera.vtable->set_aspect_ratio(camera.ptr, aspect_ratio);
}

RendererResult camera_set_focus_dist(Camera camera, f32 focus_dist) {
    assert_camera(camera);
    return camera.vtable->set_focus_dist(camera.ptr, focus_dist);
}

RendererResult camera_set_vfov(Camera camera, f32 vfov) {
    assert_camera(camera);
    return camera.vtable->set_vfov(camera.ptr, vfov);
}

RendererResult camera_set_look_at(Camera camera, vec3 position, vec3 at, vec3 up) {
    assert_camera(camera);
    return camera.vtable->set_look_at(camera.ptr, position, at, up);
}

void camera_destroy(Camera camera) {
    assert_camera(camera);
    return camera.vtable->destroy(camera.ptr);
}