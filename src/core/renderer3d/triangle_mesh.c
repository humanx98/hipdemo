#include <ww/renderer3d/triangle_mesh.h>
#include <assert.h>

static inline void assert_triangle_mesh(WwTriangleMesh tm) {
    assert(tm.ptr);
    assert(tm.vtable);
    assert(tm.vtable->destroy);
}

void ww_triangle_mesh_destroy(WwTriangleMesh self) {
    assert_triangle_mesh(self);
    self.vtable->destroy(self.ptr);
}