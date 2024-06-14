#include <ww/allocators/std_allocator.h>
#include <ww/allocators/safe_allocator.h>
#include <ww/log.h>
#include <ww/exit.h>
#include <stdlib.h>

typedef struct WwAllocationNode {
    void* address;
    usize size;
    struct WwAllocationNode * next;
    usize index;
    const char* file;
    i32 line;
} WwAllocationNode;

static WwAllocationResult __ww_must_check safe_allocator_alloc(ww_allocator_ptr self, usize size, const char* file, const i32 line);
static void safe_allocator_free(ww_allocator_ptr self, void* ptr, const char* file, const i32 line);

inline WwAllocator ww_safe_allocator_get_interface(WwSafeAllocator* self) {
    assert(self);

    const static ww_allocator_vtable vtable = {
        .alloc = safe_allocator_alloc,
        .free = safe_allocator_free
    };

    return (WwAllocator) {
        .ptr = (ww_allocator_ptr)self,
        .vtable = &vtable,
    };
}

WwSafeAllocator ww_safe_allocator_init() {
    return (WwSafeAllocator){
        .allocator = ww_std_allocator(),
    };
}

void ww_safe_allocator_deinit(WwSafeAllocator* self) {
    assert(self);

    if (self->counter == 0) {
        return;
    }

    WwAllocationNode *node = self->first;
    while (node != NULL) {
      WW_LOG_ERROR("----------------------------------------------------------------------\n");
      WW_LOG_ERROR("%zu byte leak detected at memory address %p\n", node->size, node->address);
      WW_LOG_ERROR("        %p was allocated at %s:%d\n", node->address, node->file, node->line);
      WW_LOG_ERROR("----------------------------------------------------------------------\n");
      node = node->next;
    }

    WW_EXIT;
}

static WwAllocationResult safe_allocator_alloc(ww_allocator_ptr _self, usize size, const char* file, const i32 line) {
    assert(_self);
    WwSafeAllocator* self = (WwSafeAllocator*)_self;

    WwAllocationResult alloc_result = ww_allocator_alloc(self->allocator, sizeof(WwAllocationNode) + size);
    if (alloc_result.failed) {
        return  alloc_result;
    } 

    WwAllocationNode* new_allocation = alloc_result.ptr;
    new_allocation->address = alloc_result.ptr + sizeof(WwAllocationNode);
    new_allocation->size = size;
    new_allocation->next = NULL;
    new_allocation->index = self->counter++;
    new_allocation->file = file;
    new_allocation->line = line;

    if (self->first != NULL) {
        new_allocation->next = self->first;
        self->first = new_allocation;
    } else {
        self->first = new_allocation;
        self->last = new_allocation;
    }
    return (WwAllocationResult) { .ptr = new_allocation->address };
}

static void safe_allocator_free(ww_allocator_ptr _self, void* ptr, const char* file, const i32 line) {
    assert(_self);
    WwSafeAllocator* self = (WwSafeAllocator*)_self;

    WwAllocationNode* prev_node = NULL;
    WwAllocationNode* node = self->first;
    while (node != NULL) {
        if (node->address == ptr) {
            if (self->first == node && self->last == node) {
                self->first = NULL;
                self->last = NULL;
            } else if (prev_node == NULL) {
                self->first = node->next;
            } else {
                prev_node->next = node->next;
            }
            self->counter--;
            ww_allocator_free(self->allocator, node);
            return;
        } else if (node->address < ptr && ptr < node->address + node->size) {
          WW_EXIT_WITH_MSG("%s:%d: Attempt to free in the middle of a malloc region\n", file, line);
        }

        prev_node = node;
        node = node->next;
    }

    WW_EXIT_WITH_MSG("%s:%d: Trying to free unknown memory pointer\n", file, line);
}