
#include <ww/renderer3d/object_instance.h>
#include <assert.h>

static inline void assert_object_instance(WwObjectInstance oi) {
    assert(oi.ptr);
    assert(oi.vtable);
    assert(oi.vtable->set_transform);
    assert(oi.vtable->destroy);
}

WwRenderer3DResult ww_object_instance_set_transform(WwObjectInstance self, mat4 transform) {
    assert_object_instance(self);
    return self.vtable->set_transform(self.ptr, transform);
}

void ww_object_instance_destroy(WwObjectInstance self) {
    assert_object_instance(self);
    self.vtable->destroy(self.ptr);
}