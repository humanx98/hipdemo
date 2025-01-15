#include <ww/renderer3d/camera.h>
#include <assert.h>

static inline void assert_camera(WwCamera camera) {
    assert(camera.ptr);
    assert(camera.vtable);
    assert(camera.vtable->set_aperture);
    assert(camera.vtable->set_aspect_ratio);
    assert(camera.vtable->set_focus_dist);
    assert(camera.vtable->set_look_at);
    assert(camera.vtable->set_vfov);
    assert(camera.vtable->destroy);
}

WwRenderer3DResult ww_camera_set_aperture(WwCamera camera, f32 aperture) {
    assert_camera(camera);
    return camera.vtable->set_aperture(camera.ptr, aperture);
}

WwRenderer3DResult ww_camera_set_aspect_ratio(WwCamera camera, f32 aspect_ratio) {
    assert_camera(camera);
    return camera.vtable->set_aspect_ratio(camera.ptr, aspect_ratio);
}

WwRenderer3DResult ww_camera_set_focus_dist(WwCamera camera, f32 focus_dist) {
    assert_camera(camera);
    return camera.vtable->set_focus_dist(camera.ptr, focus_dist);
}

WwRenderer3DResult ww_camera_set_vfov(WwCamera camera, f32 vfov) {
    assert_camera(camera);
    return camera.vtable->set_vfov(camera.ptr, vfov);
}

WwRenderer3DResult ww_camera_set_look_at(WwCamera camera, vec3 position, vec3 at, vec3 up) {
    assert_camera(camera);
    return camera.vtable->set_look_at(camera.ptr, position, at, up);
}

void ww_camera_destroy(WwCamera camera) {
    assert_camera(camera);
    return camera.vtable->destroy(camera.ptr);
}