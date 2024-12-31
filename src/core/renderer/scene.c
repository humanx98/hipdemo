#include <ww/renderer/scene.h>
#include <assert.h>

static inline void assert_scene(Scene s) {
    assert(s.ptr);
    assert(s.vtable);
    assert(s.vtable->set_camera);
    assert(s.vtable->attach_object_instance);
    assert(s.vtable->detach_object_instance);
    assert(s.vtable->destroy);
}

RendererResult scene_set_camera(Scene self, camera_ptr camera) {
    assert_scene(self);
    return self.vtable->set_camera(self.ptr, camera);
}

RendererResult scene_attach_object_instance(Scene self, object_instance_ptr object_instance) {
    assert_scene(self);
    return self.vtable->attach_object_instance(self.ptr, object_instance);
}

RendererResult scene_detach_object_instance(Scene self, object_instance_ptr object_instance) {
    assert_scene(self);
    return self.vtable->detach_object_instance(self.ptr, object_instance);
}

void scene_destroy(Scene self) {
    assert_scene(self);
    self.vtable->destroy(self.ptr);
}