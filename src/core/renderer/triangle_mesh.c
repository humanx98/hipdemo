#include <ww/renderer/triangle_mesh.h>
#include <assert.h>

static inline void assert_triangle_mesh(TriangleMesh tm) {
    assert(tm.ptr);
    assert(tm.vtable);
    assert(tm.vtable->destroy);
}

void triangle_mesh_destroy(TriangleMesh self) {
    assert_triangle_mesh(self);
    self.vtable->destroy(self.ptr);
}