#include <ww/renderer/camera.h>
#include <assert.h>

static inline void assert_camera(Camera camera) {
    assert(camera.ptr);
    assert(camera.vtable);
    assert(camera.vtable->set_look_at);
    assert(camera.vtable->destroy);
}


RendererResult camera_set_look_at(Camera camera, vec3 position, vec3 at, vec3 up) {
    assert_camera(camera);
    return camera.vtable->set_look_at(camera.ptr, position, at, up);
}

void camera_destroy(Camera camera) {
    assert_camera(camera);
    return camera.vtable->destroy(camera.ptr);
}