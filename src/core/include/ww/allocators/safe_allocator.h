#pragma once

#include "allocator.h"
#include "ww/defines.h"

typedef struct WwAllocationNode WwAllocationNode;

typedef struct WwSafeAllocator {
    WwAllocator allocator;
    WwAllocationNode* first;
    WwAllocationNode* last;
    usize counter;
} WwSafeAllocator;

WwAllocator __ww_must_check ww_safe_allocator_get_interface(WwSafeAllocator* self);
WwSafeAllocator __ww_must_check ww_safe_allocator_init();
void ww_safe_allocator_deinit(WwSafeAllocator* self);