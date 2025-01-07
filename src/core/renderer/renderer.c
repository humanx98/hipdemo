#include <ww/renderer/renderer.h>
#include <assert.h>

static inline void assert_renderer(WwRenderer renderer) {
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

WwRendererResult ww_renderer_set_target_resolution(WwRenderer self, u32 width, u32 height) {
    assert_renderer(self);
    return self.vtable->set_target_resolution(self.ptr, width, height);
}

WwRendererResult __ww_must_check ww_renderer_set_target_external_memory(WwRenderer self, WwViewportExternalHandle external_memory, u32 width, u32 height) {
    assert_renderer(self);
    return self.vtable->set_target_external_memory(self.ptr, external_memory, width, height);
}

WwRendererResult ww_renderer_render(WwRenderer self) {
    assert_renderer(self);
    return self.vtable->render(self.ptr);
}

WwRendererResult ww_renderer_copy_target_to(WwRenderer self, void* dst) {
    assert_renderer(self);
    return self.vtable->copy_target_to(self.ptr, dst);
}

WwRendererResult ww_renderer_set_scene(WwRenderer self, ww_scene_ptr scene) {
    assert_renderer(self);
    return self.vtable->set_scene(self.ptr, scene);
}

WwRendererResult ww_renderer_create_scene(WwRenderer self, WwScene* scene) {
    assert_renderer(self);
    return self.vtable->create_scene(self.ptr, scene);
}

WwRendererResult ww_renderer_create_camera(WwRenderer self, WwCamera* camera) {
    assert_renderer(self);
    return self.vtable->create_camera(self.ptr, camera);
}

WwRendererResult ww_renderer_create_object_instance(WwRenderer self, const ww_triangle_mesh_ptr triangle_mesh, WwObjectInstance* object_instance) {
    assert_renderer(self);
    return self.vtable->create_object_instance(self.ptr, triangle_mesh, object_instance);
}

WwRendererResult ww_renderer_create_triangle_mesh(WwRenderer self, WwTriangleMeshCreationProperties creation_properties, WwTriangleMesh* triangle_mesh) {
    assert_renderer(self);
    return self.vtable->create_triangle_mesh(self.ptr, creation_properties, triangle_mesh);
}

void ww_renderer_destroy(WwRenderer self) {
    assert_renderer(self);
    self.vtable->destroy(self.ptr);
}
