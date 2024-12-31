
#include <ww/renderer/object_instance.h>
#include <assert.h>

static inline void assert_object_instance(ObjectInstance oi) {
    assert(oi.ptr);
    assert(oi.vtable);
    assert(oi.vtable->set_transform);
    assert(oi.vtable->destroy);
}

RendererResult object_instance_set_transform(ObjectInstance self, mat4 transform) {
    assert_object_instance(self);
    return self.vtable->set_transform(self.ptr, transform);
}

void object_instance_destroy(ObjectInstance self) {
    assert_object_instance(self);
    self.vtable->destroy(self.ptr);
}