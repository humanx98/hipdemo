#include <ww/renderer.h>
#include <assert.h>

static void assert_renderer(Renderer renderer) {
    assert(renderer.ptr);
    assert(renderer.vtable);
    assert(renderer.vtable->set_target_resolution);
    assert(renderer.vtable->render);
    assert(renderer.vtable->copy_target_to);
    assert(renderer.vtable->set_scene);
    assert(renderer.vtable->create_scene);
    assert(renderer.vtable->create_camera);
    assert(renderer.vtable->create_object_instance);
    assert(renderer.vtable->create_triangle_mesh);
    assert(renderer.vtable->destroy);
}

RendererResult renderer_set_target_resolution(Renderer self, u32 width, u32 height) {
    assert_renderer(self);
    return self.vtable->set_target_resolution(self.ptr, width, height);
}

RendererResult renderer_render(Renderer self) {
    assert_renderer(self);
    return self.vtable->render(self.ptr);
}

RendererResult renderer_copy_target_to(Renderer self, void* dst) {
    assert_renderer(self);
    return self.vtable->copy_target_to(self.ptr, dst);
}

RendererResult renderer_set_scene(Renderer self, scene_ptr scene) {
    assert_renderer(self);
    return self.vtable->set_scene(self.ptr, scene);
}

RendererResult renderer_create_scene(Renderer self, Scene* scene) {
    assert_renderer(self);
    return self.vtable->create_scene(self.ptr, scene);
}

RendererResult renderer_create_camera(Renderer self, Camera* camera) {
    assert_renderer(self);
    return self.vtable->create_camera(self.ptr, camera);
}

RendererResult renderer_create_object_instance(Renderer self, const triangle_mesh_ptr triangle_mesh, ObjectInstance* object_instance) {
    assert_renderer(self);
    return self.vtable->create_object_instance(self.ptr, triangle_mesh, object_instance);
}

RendererResult renderer_create_triangle_mesh(Renderer self, TriangleMeshCreationProperties creation_properties, TriangleMesh* triangle_mesh) {
    assert_renderer(self);
    return self.vtable->create_triangle_mesh(self.ptr, creation_properties, triangle_mesh);
}

void renderer_destroy(Renderer self) {
    assert_renderer(self);
    self.vtable->destroy(self.ptr);
}
