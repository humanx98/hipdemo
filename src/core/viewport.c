#include <ww/viewport.h>
#include <assert.h>

static inline void assert_viewport(Viewport viewport) {
    assert(viewport.ptr);
    assert(viewport.vtable);
    assert(viewport.vtable->wait_idle);
    assert(viewport.vtable->get_mapped_input);
    assert(viewport.vtable->get_external_memory);
    assert(viewport.vtable->get_external_semaphores);
    assert(viewport.vtable->set_resolution);
    assert(viewport.vtable->get_resolution);
    assert(viewport.vtable->destroy);
}

ViewportResult viewport_render(Viewport self) {
    assert_viewport(self);
    return self.vtable->render(self.ptr);
}

ViewportResult viewport_wait_idle(Viewport self) {
    assert_viewport(self);
    return self.vtable->wait_idle(self.ptr);
}

void* viewport_get_mapped_input(Viewport self) {
    assert_viewport(self);
    return self.vtable->get_mapped_input(self.ptr);
}

ViewportExternalHandle viewport_get_external_memory(Viewport self) {
    assert_viewport(self);
    return self.vtable->get_external_memory(self.ptr);
}

ViewportExternalSemaphores viewport_get_external_semaphores(Viewport self) {
    assert_viewport(self);
    return self.vtable->get_external_semaphores(self.ptr);
}

ViewportResult viewport_set_resolution(Viewport self, u32 width, u32 height) {
    assert_viewport(self);
    return self.vtable->set_resolution(self.ptr, width, height);
}

void viewport_get_resolution(Viewport self, u32* width, u32* height) {
    assert_viewport(self);
    self.vtable->get_resolution(self.ptr, width, height);
}

void viewport_destroy(Viewport self) {
    assert_viewport(self);
    self.vtable->destroy(self.ptr);
}