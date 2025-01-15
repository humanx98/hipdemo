#include <ww/renderer3d/scene.h>
#include <assert.h>

static inline void assert_scene(WwScene s) {
    assert(s.ptr);
    assert(s.vtable);
    assert(s.vtable->set_camera);
    assert(s.vtable->attach_object_instance);
    assert(s.vtable->detach_object_instance);
    assert(s.vtable->destroy);
}

WwRenderer3DResult ww_scene_set_camera(WwScene self, ww_camera_ptr camera) {
    assert_scene(self);
    return self.vtable->set_camera(self.ptr, camera);
}

WwRenderer3DResult ww_scene_attach_object_instance(WwScene self, ww_object_instance_ptr object_instance) {
    assert_scene(self);
    return self.vtable->attach_object_instance(self.ptr, object_instance);
}

WwRenderer3DResult ww_scene_detach_object_instance(WwScene self, ww_object_instance_ptr object_instance) {
    assert_scene(self);
    return self.vtable->detach_object_instance(self.ptr, object_instance);
}

void ww_scene_destroy(WwScene self) {
    assert_scene(self);
    self.vtable->destroy(self.ptr);
}