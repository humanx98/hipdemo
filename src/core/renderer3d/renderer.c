#include <ww/renderer3d/renderer3d.h>
#include <assert.h>

static inline void assert_renderer(WwRenderer3D renderer) {
    assert(renderer.ptr);
    assert(renderer.vtable);
    assert(renderer.vtable->set_target_resolution);
    assert(renderer.vtable->set_target_external_memory);
    assert(renderer.vtable->render);
    assert(renderer.vtable->copy_target_to);
    assert(renderer.vtable->set_scene);
    assert(renderer.vtable->create_scene);
    assert(renderer.vtable->create_camera);
    assert(renderer.vtable->create_object_instance);
    assert(renderer.vtable->create_triangle_mesh);
    assert(renderer.vtable->destroy);
}

WwRenderer3DResult ww_renderer3d_set_target_resolution(WwRenderer3D self, u32 width, u32 height) {
    assert_renderer(self);
    return self.vtable->set_target_resolution(self.ptr, width, height);
}

WwRenderer3DResult __ww_must_check ww_renderer3d_set_target_external_memory(WwRenderer3D self, WwViewportExternalHandle external_memory, u32 width, u32 height) {
    assert_renderer(self);
    return self.vtable->set_target_external_memory(self.ptr, external_memory, width, height);
}

WwRenderer3DResult ww_renderer3d_render(WwRenderer3D self) {
    assert_renderer(self);
    return self.vtable->render(self.ptr);
}

WwRenderer3DResult ww_renderer3d_copy_target_to(WwRenderer3D self, void* dst) {
    assert_renderer(self);
    return self.vtable->copy_target_to(self.ptr, dst);
}

WwRenderer3DResult ww_renderer3d_set_scene(WwRenderer3D self, ww_scene_ptr scene) {
    assert_renderer(self);
    return self.vtable->set_scene(self.ptr, scene);
}

WwRenderer3DResult ww_renderer3d_create_scene(WwRenderer3D self, WwScene* scene) {
    assert_renderer(self);
    return self.vtable->create_scene(self.ptr, scene);
}

WwRenderer3DResult ww_renderer3d_create_camera(WwRenderer3D self, WwCamera* camera) {
    assert_renderer(self);
    return self.vtable->create_camera(self.ptr, camera);
}

WwRenderer3DResult ww_renderer3d_create_object_instance(WwRenderer3D self, const ww_triangle_mesh_ptr triangle_mesh, WwObjectInstance* object_instance) {
    assert_renderer(self);
    return self.vtable->create_object_instance(self.ptr, triangle_mesh, object_instance);
}

WwRenderer3DResult ww_renderer3d_create_triangle_mesh(WwRenderer3D self, WwTriangleMeshCreationProperties creation_properties, WwTriangleMesh* triangle_mesh) {
    assert_renderer(self);
    return self.vtable->create_triangle_mesh(self.ptr, creation_properties, triangle_mesh);
}

void ww_renderer3d_destroy(WwRenderer3D self) {
    assert_renderer(self);
    self.vtable->destroy(self.ptr);
}
