#include <ww/viewport.h>
#include <assert.h>

static inline void assert_viewport(WwViewport viewport) {
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

WwViewportResult ww_viewport_render(WwViewport self) {
    assert_viewport(self);
    return self.vtable->render(self.ptr);
}

WwViewportResult ww_viewport_wait_idle(WwViewport self) {
    assert_viewport(self);
    return self.vtable->wait_idle(self.ptr);
}

void* ww_viewport_get_mapped_input(WwViewport self) {
    assert_viewport(self);
    return self.vtable->get_mapped_input(self.ptr);
}

WwViewportExternalHandle ww_viewport_get_external_memory(WwViewport self) {
    assert_viewport(self);
    return self.vtable->get_external_memory(self.ptr);
}

WwViewportExternalSemaphores ww_viewport_get_external_semaphores(WwViewport self) {
    assert_viewport(self);
    return self.vtable->get_external_semaphores(self.ptr);
}

WwViewportResult ww_viewport_set_resolution(WwViewport self, u32 width, u32 height) {
    assert_viewport(self);
    return self.vtable->set_resolution(self.ptr, width, height);
}

void ww_viewport_get_resolution(WwViewport self, u32* width, u32* height) {
    assert_viewport(self);
    self.vtable->get_resolution(self.ptr, width, height);
}

void ww_viewport_destroy(WwViewport self) {
    assert_viewport(self);
    self.vtable->destroy(self.ptr);
}