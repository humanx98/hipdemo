#include <ww/vulkan/viewport.h>
#include <ww/collections/darray.h>
#include <vulkan/vulkan_core.h>
#include <string.h>
#include <stdlib.h>
#include <ww/defines.h>
#include <ww/file.h>
#include <ww/log.h>
#include "vma.h"

static const char* device_extensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
static const char* validation_layers[] = {};
#else
static const char* validation_layers[] = {
    "VK_LAYER_KHRONOS_validation"
};
#endif

typedef struct QueueFamilyIndices {
    u32 present;
} QueueFamilyIndices;

typedef struct TransitionImageLayoutInfo {
    VkAccessFlags from_access;
    VkImageLayout from_layout;
    VkPipelineStageFlags from_pipeline_stage;
    VkAccessFlags to_access;
    VkImageLayout to_layout;
    VkPipelineStageFlags to_pipeline_stage;
} TransitionImageLayoutInfo;

typedef struct viewport_ptr_impl {
    WwAllocator allocator;
    u32 frames_in_flight;
    u32 current_frame;
    b8 use_cmd_blit;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkSurfaceKHR surface;
    QueueFamilyIndices queue_family_indices;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue present_queue;
    VkCommandPool command_pool;
    WwDArray(VkCommandBuffer) command_buffers;
    WwDArray(VkSemaphore) image_available_semaphores;
    WwDArray(VkSemaphore) render_finished_semaphores;
    WwDArray(VkFence) in_flight_fences;
    VmaAllocator vma_allocator;
    struct {
        struct {
            VkSurfaceCapabilitiesKHR capabilities;
            VkSurfaceFormatKHR format;
            VkPresentModeKHR present_mode;
            VkExtent2D extent;
        } details;
        VkSwapchainKHR swapchain;
        WwDArray(VkImage) images;
        WwDArray(b8) has_undefined_layout;
    } swapchain;
    struct {
        VkImage image;
        b8 has_undefined_layout;
        VkExtent3D extent;
        VmaAllocation allocation;
        VkImageView view;
        VkBuffer staging_buffer;
        VmaAllocation staging_buffer_allocation;
        VmaAllocationInfo staging_buffer_allocation_info;
    } input;
} viewport_ptr_impl;

static void vulkan_viewport_no_gp_destroy(viewport_ptr self);
static ViewportResult __ww_must_check vulkan_viewport_no_gp_render(viewport_ptr self);
static ViewportResult __ww_must_check vulkan_viewport_no_gp_wait_idle(viewport_ptr self);
static void* __ww_must_check vulkan_viewport_no_gp_get_mapped_input(viewport_ptr self);
static ViewportResult __ww_must_check vulkan_viewport_no_gp_set_resolution(viewport_ptr self, u32 width, u32 height);
static void vulkan_viewport_no_gp_get_resolution(viewport_ptr self, u32* width, u32* height);

static VulkanResult __ww_must_check vulkan_viewport_no_gp_init_vulkan(viewport_ptr self, VulkanViewportCreationProperties creation_properties);
static VulkanResult __ww_must_check vulkan_viewport_no_gp_create_instance(viewport_ptr self, VulkanViewportCreationProperties creation_properties, const VkDebugUtilsMessengerCreateInfoEXT* debug_create_info);
static VulkanResult __ww_must_check vulkan_viewport_no_gp_pick_physical_device(viewport_ptr self, u32 device_index);
static VulkanResult __ww_must_check vulkan_viewport_no_gp_get_swap_chain_details(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_no_gp_create_logical_device(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_no_gp_create_command_pool(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_no_gp_create_command_buffers(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_no_gp_create_sync_objects(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_no_gp_create_swapchain(viewport_ptr self, u32 width, u32 height);
static VulkanResult __ww_must_check vulkan_viewport_no_gp_create_input_image(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_no_gp_record_command_buffer(viewport_ptr self, VkCommandBuffer command_buffer, u32 image_index);
static void vulkan_viewport_no_gp_cleanup_swapchain_and_input_image(viewport_ptr self);
static VKAPI_ATTR VkBool32 VKAPI_CALL vulkan_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data); 
static void transition_image_layout(VkCommandBuffer command_buffer, VkImage image, TransitionImageLayoutInfo info);
static ViewportResult __ww_must_check to_viewport_result(VulkanResult res);

VulkanResult vulkan_viewport_no_gp_create(VulkanViewportCreationProperties creation_properties, Viewport* viewport) {
    assert(viewport);

    ww_auto_type alloc_result = ww_allocator_alloc_type(creation_properties.allocator, viewport_ptr_impl);
    if (alloc_result.failed) {
        return VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
    }

    viewport_ptr self = alloc_result.ptr;
    VulkanResult res = vulkan_viewport_no_gp_init_vulkan(self, creation_properties);
    if (res.failed) {
        vulkan_viewport_no_gp_destroy(self);
        return res;
    }

    const static viewport_vtable vtable = {
        .render = vulkan_viewport_no_gp_render,
        .wait_idle = vulkan_viewport_no_gp_wait_idle,
        .get_mapped_input = vulkan_viewport_no_gp_get_mapped_input,
        .set_resolution = vulkan_viewport_no_gp_set_resolution,
        .get_resolution = vulkan_viewport_no_gp_get_resolution,
        .destroy = vulkan_viewport_no_gp_destroy,
    };
    *viewport = (Viewport) {
        .ptr = self,
        .vtable = &vtable,
    };

    return res;
}

void vulkan_viewport_no_gp_destroy(viewport_ptr self) {
    assert(self);

    vulkan_viewport_no_gp_cleanup_swapchain_and_input_image(self);
    ww_darray_deinit(&self->swapchain.has_undefined_layout);
    ww_darray_deinit(&self->swapchain.images);

    if (self->vma_allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(self->vma_allocator);
    }

    ww_darray_foreach_by_ref(&self->image_available_semaphores, VkSemaphore, s)
        vkDestroySemaphore(self->device, *s, NULL);
    ww_darray_foreach_by_ref(&self->render_finished_semaphores, VkSemaphore, s)
        vkDestroySemaphore(self->device, *s, NULL);
    ww_darray_foreach_by_ref(&self->in_flight_fences, VkFence, f) 
        vkDestroyFence(self->device, *f, NULL);

    ww_darray_deinit(&self->image_available_semaphores);
    ww_darray_deinit(&self->render_finished_semaphores);
    ww_darray_deinit(&self->in_flight_fences); 
    
    ww_darray_deinit(&self->command_buffers);
    if (self->command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(self->device, self->command_pool, NULL);
    }

    if (self->device != VK_NULL_HANDLE) {
        vkDestroyDevice(self->device, NULL);
    }

    if (self->surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(self->instance, self->surface, NULL);
    }

    if (self->debug_messenger != VK_NULL_HANDLE) {
        PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(self->instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != NULL) {
            func(self->instance, self->debug_messenger, NULL);
        }
    }

    if (self->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(self->instance, NULL);
    }

    ww_allocator_free(self->allocator, self);
}

ViewportResult vulkan_viewport_no_gp_wait_idle(viewport_ptr self) {
    assert(self);
    return to_viewport_result(VULKAN_CHECK(vkQueueWaitIdle(self->present_queue)));
}

void* vulkan_viewport_no_gp_get_mapped_input(viewport_ptr self) {
    assert(self);
    return self->input.staging_buffer_allocation_info.pMappedData;
}

ViewportResult vulkan_viewport_no_gp_set_resolution(viewport_ptr self, u32 width, u32 height) {
    assert(self);

    VulkanResult res = VULKAN_CHECK(vkQueueWaitIdle(self->present_queue));
    if (res.failed) {
        return to_viewport_result(res);
    }

    vulkan_viewport_no_gp_cleanup_swapchain_and_input_image(self);

    res = vulkan_viewport_no_gp_get_swap_chain_details(self);
    if (res.failed) {
        return to_viewport_result(res);
    }

    res = vulkan_viewport_no_gp_create_swapchain(self, width, height);
    if (res.failed) {
        return to_viewport_result(res);
    }

    res = vulkan_viewport_no_gp_create_input_image(self);
    return to_viewport_result(res);
}

void vulkan_viewport_no_gp_get_resolution(viewport_ptr self, u32* width, u32* height) {
    assert(self);
    assert(width);
    assert(height);

    *width = self->swapchain.details.extent.width;
    *height = self->swapchain.details.extent.height;
}

VulkanResult vulkan_viewport_no_gp_init_vulkan(viewport_ptr self, VulkanViewportCreationProperties creation_properties) {
    *self = (viewport_ptr_impl) {
        .allocator = creation_properties.allocator,
        .frames_in_flight = creation_properties.frames_in_flight,
        .swapchain.images = ww_darray_init(creation_properties.allocator, VkImage),
        .command_buffers = ww_darray_init(creation_properties.allocator, VkCommandBuffer),
        .image_available_semaphores = ww_darray_init(creation_properties.allocator, VkSemaphore),
        .render_finished_semaphores = ww_darray_init(creation_properties.allocator, VkSemaphore),
        .in_flight_fences = ww_darray_init(creation_properties.allocator, VkFence),
        .swapchain.has_undefined_layout = ww_darray_init(creation_properties.allocator, b8),
    };

    VkDebugUtilsMessengerCreateInfoEXT debug_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = vulkan_debug_callback
    };
    const VkDebugUtilsMessengerCreateInfoEXT* p_debug_create_info = 
#ifdef NDEBUG
        NULL;
#else
        &debug_create_info;
#endif
    VulkanResult res = vulkan_viewport_no_gp_create_instance(self, creation_properties, p_debug_create_info);
    if (res.failed) {
        return res;
    }

    res = vulkan_create_debugger_messenger(self->instance, p_debug_create_info, &self->debug_messenger);
    if (res.failed) {
        return res;
    }
    
    assert(creation_properties.vulkan_create_surface);
    assert(creation_properties.window);
    res = VULKAN_CHECK(creation_properties.vulkan_create_surface(self->instance, creation_properties.window, &self->surface));
    if (res.failed) {
        return res;
    }

    res = vulkan_viewport_no_gp_pick_physical_device(self, creation_properties.device_index);
    if (res.failed) {
        return res;
    }

    res = vulkan_viewport_no_gp_create_logical_device(self);
    if (res.failed) {
        return res;
    }

    res = vulkan_viewport_no_gp_create_command_pool(self);
    if (res.failed) {
        return res;
    }

    res = vulkan_viewport_no_gp_create_command_buffers(self);
    if (res.failed) {
        return res;
    }

    res = vulkan_viewport_no_gp_create_sync_objects(self);
    if (res.failed) {
        return res;
    }

    VmaAllocatorCreateInfo allocator_create_info = {
        .vulkanApiVersion = VK_API_VERSION_1_2,
        .instance = self->instance,
        .physicalDevice = self->physical_device,
        .device = self->device,
    };
    res = VULKAN_CHECK(vmaCreateAllocator(&allocator_create_info, &self->vma_allocator));
    return res;
}

VulkanResult vulkan_viewport_no_gp_create_instance(viewport_ptr self, VulkanViewportCreationProperties creation_properties, const VkDebugUtilsMessengerCreateInfoEXT* debug_create_info) {
    WwDArray(const char*) required_extensions = ww_darray_init(self->allocator, const char*);
    WwDArray(VkLayerProperties) available_layers = ww_darray_init(self->allocator, VkLayerProperties);
    VulkanResult res = {};
    if (WW_ARRAY_SIZE(validation_layers) > 0) {
        res = vulkan_enumerate_instance_layer_properties(&available_layers);
        if (res.failed) {
            goto failed;
        }

        WW_ARRAY_FOREACH(validation_layers, required_layer) {
            b8 layer_found = false;
            ww_darray_foreach_by_ref(&available_layers, VkLayerProperties, available_layer) {
                if (strcmp(*required_layer, available_layer->layerName) == 0) {
                    layer_found = true;
                    break;
                }
            }

            if (!layer_found) {
                WW_LOG_ERROR("[vulkan viewport_no_gp] Couldn't find %s vulkan layer\n", *required_layer);
                res = VULKAN_CHECK(VK_ERROR_LAYER_NOT_PRESENT);
                goto failed;
            }
        }
    }

    if (!ww_darray_append_many(&required_extensions, creation_properties.instance_extensions, creation_properties.instance_extension_count)) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        goto failed;
    }

    const char* debug_utils_extenion_name = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    if (debug_create_info != NULL && !ww_darray_append(&required_extensions, debug_utils_extenion_name)) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        goto failed;
    }

    VulkanInstanceCreateInfo create_info = {
        .enabled_extension_count = ww_darray_len(&required_extensions),
        .enabled_extensions = ww_darray_ptr(&required_extensions),
        .enabled_layer_count = WW_ARRAY_SIZE(validation_layers),
        .enabled_layers = validation_layers,
        .p_next = debug_create_info,
    };
    res = vulkan_create_instance(&create_info, &self->instance);
failed:
    ww_darray_deinit(&required_extensions);
    ww_darray_deinit(&available_layers);
    return res;
}

VulkanResult vulkan_viewport_no_gp_pick_physical_device(viewport_ptr self, u32 device_index) {
    WwDArray(VkExtensionProperties) available_extension_properties = ww_darray_init(self->allocator, VkExtensionProperties);

    VulkanResult res = vulkan_pick_physical_device(self->instance, device_index, &self->physical_device);
    if (res.failed) {
        goto failed;
    }

    VulkanQueueFamilyIndices queue_family_indices;
    res = vulkan_physical_device_get_queue_family_indices(self->physical_device, self->surface, self->allocator, &queue_family_indices);
    if (res.failed) {
        goto failed;
    }
    
    if (!queue_family_indices.present.found) {
        WW_LOG_ERROR("[vulkan viewport_no_gp] Couldn't find present queue family\n");
        res = VULKAN_CHECK(VK_ERROR_INITIALIZATION_FAILED);
        goto failed;
    } else {
        WW_LOG_INFO("[vulkan viewport_no_gp] present queue family index: %d\n", queue_family_indices.present.index);
        self->queue_family_indices.present = queue_family_indices.present.index;
    } 

    res = vulkan_enumerate_device_extension_properties(self->physical_device, NULL, &available_extension_properties);
    if (res.failed) {
        goto failed;
    }

    WW_ARRAY_FOREACH(device_extensions, required_extension) {
        b8 found_extension = false;
        ww_darray_foreach_by_ref(&available_extension_properties, VkExtensionProperties, available_extension) {
            if (!strcmp(*required_extension, available_extension->extensionName)) {
                found_extension = true;
                break;
            }
        }

        if (!found_extension) {
            WW_LOG_ERROR("[vulkan viewport_no_gp] Couldn't find device extension: %s\n", *required_extension);
            res = VULKAN_CHECK(VK_ERROR_EXTENSION_NOT_PRESENT);
            goto failed;
        }
    }

failed:
    ww_darray_deinit(&available_extension_properties);
    return res;
}

VulkanResult vulkan_viewport_no_gp_get_swap_chain_details(viewport_ptr self) {
    WwDArray(VkSurfaceFormatKHR) formats = ww_darray_init(self->allocator, VkSurfaceFormatKHR);
    WwDArray(VkPresentModeKHR) present_modes = ww_darray_init(self->allocator, VkSurfaceFormatKHR);

    VulkanResult res = VULKAN_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(self->physical_device, self->surface, &self->swapchain.details.capabilities));
    if (res.failed) {
        goto failed;
    }

    res = vulkan_get_physical_device_surface_formats(self->physical_device, self->surface, &formats);
    if (res.failed) {
        goto failed;
    } else if (ww_darray_len(&formats) == 0) {
        WW_LOG_ERROR("[vulkan viewport_no_gp] No device surface formats");
        res = VULKAN_CHECK(VK_ERROR_INITIALIZATION_FAILED);
        goto failed;
    }

    res = vulkan_get_physical_device_surface_present_modes(self->physical_device, self->surface, &present_modes);
    if (res.failed) {
        goto failed;
    } else if (ww_darray_len(&present_modes) == 0) {
        WW_LOG_ERROR("[vulkan viewport_no_gp] No device surface present modes");
        res = VULKAN_CHECK(VK_ERROR_INITIALIZATION_FAILED);
        goto failed;
    }

    b8 found_format = false;
    VkSurfaceFormatKHR wanted_format = {
        .format = VK_FORMAT_B8G8R8A8_SRGB,
        .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
    };
    ww_darray_foreach_by_ref(&formats, VkSurfaceFormatKHR, f) {
        if (f->format == wanted_format.format && f->colorSpace == wanted_format.colorSpace) {
            found_format = true;
            self->swapchain.details.format = wanted_format;
            break;
        }
    }

    if (!found_format) {
        WW_LOG_WARN("[vulkan viewport_no_gp] Couldn't find wanted surface fomat");
        self->swapchain.details.format = ww_darray_get(&formats, VkSurfaceFormatKHR, 0);
    }

    b8 found_mailbox_mode = false;
    b8 found_immediate_mode = false;
    ww_darray_foreach_by_ref(&present_modes, VkPresentModeKHR, pm) {
        if (*pm == VK_PRESENT_MODE_MAILBOX_KHR) {
            found_mailbox_mode = true;
        }

        if (*pm == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            found_immediate_mode = true;
        }
    }

    if (found_mailbox_mode) {
        self->swapchain.details.present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
    } else if (found_immediate_mode) {
        self->swapchain.details.present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    } else {
        WW_LOG_WARN("[vulkan viewport_no_gp] Couldn't find wanted surface present mode\n");
        self->swapchain.details.present_mode = VK_PRESENT_MODE_FIFO_KHR;
    }

failed:
    ww_darray_deinit(&formats);
    ww_darray_deinit(&present_modes);
    return res;
}

VulkanResult vulkan_viewport_no_gp_create_logical_device(viewport_ptr self) {
    f32 queue_priority = 1.0f;
    VkDeviceQueueCreateInfo present_queue_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = self->queue_family_indices.present,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    VulkanDeviceCreateInfo device_create_info = {
        .queue_create_info_count = 1,
        .queue_create_infos = &present_queue_create_info,
        .enabled_extension_count = WW_ARRAY_SIZE(device_extensions),
        .enabled_extensions = device_extensions,
        .enabled_layer_count = WW_ARRAY_SIZE(validation_layers),
        .enabled_layers = validation_layers,
    };
    VulkanResult res = vulkan_create_device(self->physical_device, &device_create_info, &self->device);
    if (res.failed) {
        return res;
    }

    vkGetDeviceQueue(self->device, self->queue_family_indices.present, 0, &self->present_queue);

    return res;
}

VulkanResult vulkan_viewport_no_gp_create_command_pool(viewport_ptr self) {
    VkCommandPoolCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = self->queue_family_indices.present,
    };
    return VULKAN_CHECK(vkCreateCommandPool(self->device, &create_info, NULL, &self->command_pool));
}

VulkanResult vulkan_viewport_no_gp_create_command_buffers(viewport_ptr self) {
    VulkanResult res = {};
    if (!ww_darray_ensure_total_capacity_precise(&self->command_buffers, self->frames_in_flight)) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        return res;
    }

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = self->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = self->frames_in_flight,
    };
    res = VULKAN_CHECK(vkAllocateCommandBuffers(self->device, &alloc_info, ww_darray_ptr(&self->command_buffers)));
    if (res.failed) {
        return res;
    }

    ww_darray_resize_assume_capacity(&self->command_buffers, self->frames_in_flight);
    return res;
}

VulkanResult vulkan_viewport_no_gp_create_sync_objects(viewport_ptr self) {
    VulkanResult res = {};
    if (!ww_darray_ensure_total_capacity_precise(&self->image_available_semaphores, self->frames_in_flight)
        || !ww_darray_ensure_total_capacity_precise(&self->render_finished_semaphores, self->frames_in_flight)
        || !ww_darray_ensure_total_capacity_precise(&self->in_flight_fences, self->frames_in_flight)) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY); 
        return res;
    }

    VkSemaphoreCreateInfo semaphore_create_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fence_create_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    for (usize i = 0; i < self->frames_in_flight; i++) {
        VkSemaphore semaphore;
        res = VULKAN_CHECK(vkCreateSemaphore(self->device, &semaphore_create_info, NULL, &semaphore));
        if (res.failed) {
            return res;
        }

        ww_darray_append_assume_capacity(&self->image_available_semaphores, semaphore);

        res = VULKAN_CHECK(vkCreateSemaphore(self->device, &semaphore_create_info, NULL, &semaphore));
        if (res.failed) {
            return res;
        }

        ww_darray_append_assume_capacity(&self->render_finished_semaphores, semaphore);

        VkFence fence;
        res = VULKAN_CHECK(vkCreateFence(self->device, &fence_create_info, NULL, &fence));
        if (res.failed) {
            return res;
        }

        ww_darray_append_assume_capacity(&self->in_flight_fences, fence);
    }

    return res;
}

VulkanResult vulkan_viewport_no_gp_create_swapchain(viewport_ptr self, u32 width, u32 height) {
    VkSurfaceCapabilitiesKHR capabilities = self->swapchain.details.capabilities;
    if (capabilities.currentExtent.width != WW_U32_MAX) {
        self->swapchain.details.extent = capabilities.currentExtent;
    } else {
        self->swapchain.details.extent = (VkExtent2D){
            .width = WW_CLAMP(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            .height = WW_CLAMP(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    u32 image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
        image_count = capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR create_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = self->surface,
        .minImageCount = image_count,
        .imageFormat = self->swapchain.details.format.format,
        .imageColorSpace = self->swapchain.details.format.colorSpace,
        .imageExtent = self->swapchain.details.extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .preTransform = self->swapchain.details.capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = self->swapchain.details.present_mode,
        .clipped = VK_TRUE,
    };

    VulkanResult res = VULKAN_CHECK(vkCreateSwapchainKHR(self->device, &create_info, NULL, &self->swapchain.swapchain));
    if (res.failed) {
        return res;
    }

    res = vulkan_get_swapchain_images(self->device, self->swapchain.swapchain, &self->swapchain.images);
    if (res.failed) {
        return res;
    }

    if (!ww_darray_ensure_total_capacity_precise(&self->swapchain.has_undefined_layout, ww_darray_len(&self->swapchain.images))) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        return res;
    }

    ww_darray_resize_assume_capacity(&self->swapchain.has_undefined_layout, ww_darray_len(&self->swapchain.images));
    ww_darray_foreach_by_ref(&self->swapchain.has_undefined_layout, b8, item_ref) {
        *item_ref = true;
    }

    return res;
}

static VulkanResult vulkan_viewport_no_gp_create_input_image(viewport_ptr self) {
    VkFormat input_image_format = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkImageCreateInfo img_create_info = { 
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .extent = {
            .width = self->swapchain.details.extent.width,
            .height = self->swapchain.details.extent.height,
            .depth = 1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .format = input_image_format,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .samples = VK_SAMPLE_COUNT_1_BIT,
    };
    self->use_cmd_blit = img_create_info.format != self->swapchain.details.format.format;

    VmaAllocationCreateInfo alloc_create_info = {
        .usage = VMA_MEMORY_USAGE_AUTO,
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        .priority = 1.0f,
    };

    VulkanResult res = VULKAN_CHECK(vmaCreateImage(
        self->vma_allocator,
        &img_create_info,
        &alloc_create_info,
        &self->input.image,
        &self->input.allocation,
        NULL
    ));
    if (res.failed) {
        return res;
    } else {
        self->input.extent = img_create_info.extent;
        self->input.has_undefined_layout = true;
    }
    VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = self->input.image, 
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = img_create_info.format,
        .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .subresourceRange.levelCount = 1,
        .subresourceRange.layerCount = 1,
    };
    res = VULKAN_CHECK(vkCreateImageView(self->device, &view_info, NULL, &self->input.view));
    if (res.failed) {
        return res;
    }

    usize pixel_size = vulkan_get_pixel_size(img_create_info.format);
    VkBufferCreateInfo buf_create_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = img_create_info.extent.width * img_create_info.extent.height * pixel_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    };
    alloc_create_info = (VmaAllocationCreateInfo){
        .usage = VMA_MEMORY_USAGE_AUTO,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
    };

    res = VULKAN_CHECK(vmaCreateBuffer(
        self->vma_allocator,
        &buf_create_info,
        &alloc_create_info,
        &self->input.staging_buffer,
        &self->input.staging_buffer_allocation,
        &self->input.staging_buffer_allocation_info
    ));

    return res;
}

VulkanResult vulkan_viewport_no_gp_record_command_buffer(viewport_ptr self, VkCommandBuffer command_buffer, u32 image_index) {
    VkCommandBufferBeginInfo begin_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VulkanResult res = VULKAN_CHECK(vkBeginCommandBuffer(command_buffer, &begin_info));
    if (res.failed) {
        return res;
    }

    TransitionImageLayoutInfo transition_image_layout_info = {
        .from_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .from_access = VK_ACCESS_TRANSFER_READ_BIT,
        .from_pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT,
        .to_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .to_access = VK_ACCESS_TRANSFER_WRITE_BIT,
        .to_pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT,
    };
    if (self->input.has_undefined_layout) {
        transition_image_layout_info.from_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        transition_image_layout_info.from_access = VK_ACCESS_NONE;
        transition_image_layout_info.from_pipeline_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        self->input.has_undefined_layout = false;
    }
    transition_image_layout(command_buffer, self->input.image, transition_image_layout_info);

    VkBufferImageCopy region = {
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .layerCount = 1,
        },
        .imageExtent = self->input.extent,
    };
    vkCmdCopyBufferToImage(command_buffer, self->input.staging_buffer, self->input.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    transition_image_layout_info = (TransitionImageLayoutInfo){
        .from_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .from_access = VK_ACCESS_TRANSFER_WRITE_BIT,
        .from_pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT,
        .to_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .to_access = VK_ACCESS_TRANSFER_READ_BIT,
        .to_pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT,
    };
    transition_image_layout(command_buffer, self->input.image, transition_image_layout_info); 

    VkImage present_image = ww_darray_get(&self->swapchain.images, VkImage, image_index);
    transition_image_layout_info = (TransitionImageLayoutInfo) {
        .from_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .from_pipeline_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        .to_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .to_access = VK_ACCESS_TRANSFER_WRITE_BIT,
        .to_pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT,
    };
    b8* present_image_has_undefined_layout = ww_darray_get_ref(&self->swapchain.has_undefined_layout, b8, image_index);
    if (*present_image_has_undefined_layout) {
        transition_image_layout_info.from_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        *present_image_has_undefined_layout = false;
    }
    transition_image_layout(command_buffer, present_image, transition_image_layout_info);

    if (self->use_cmd_blit) {
        VkImageBlit blit = {
            .srcOffsets[1] = {
                .x = self->swapchain.details.extent.width,
                .y = self->swapchain.details.extent.height,
                .z = 1,
            },
            .dstOffsets[1] = {
                .x = self->swapchain.details.extent.width,
                .y = self->swapchain.details.extent.height,
                .z = 1,
            },
            .srcSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
            .dstSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
        };
        vkCmdBlitImage(
            command_buffer,
            self->input.image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            present_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blit,
            VK_FILTER_NEAREST
        );
    } else {
        VkImageCopy copy = {
            .srcSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
            .dstSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
            .extent = self->input.extent,
        };
        vkCmdCopyImage(
            command_buffer,
            self->input.image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            present_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &copy
        );
    }
    transition_image_layout_info = (TransitionImageLayoutInfo) {
        .from_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .from_access = VK_ACCESS_TRANSFER_WRITE_BIT,
        .from_pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT,
        .to_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .to_access = VK_ACCESS_NONE,
        .to_pipeline_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
    };
    transition_image_layout(command_buffer, present_image, transition_image_layout_info);

    res = VULKAN_CHECK(vkEndCommandBuffer(command_buffer));
    return res;
}

ViewportResult vulkan_viewport_no_gp_render(viewport_ptr self) {
    assert(self);
    VkFence in_flight_fence = ww_darray_get(&self->in_flight_fences, VkFence, self->current_frame);
    VkSemaphore image_available_semaphore = ww_darray_get(&self->image_available_semaphores, VkSemaphore, self->current_frame);
    VkSemaphore render_finished_semaphore = ww_darray_get(&self->render_finished_semaphores, VkSemaphore, self->current_frame);
    VkCommandBuffer command_buffer = ww_darray_get(&self->command_buffers, VkCommandBuffer, self->current_frame);

    VulkanResult res = VULKAN_CHECK(vkWaitForFences(self->device, 1, &in_flight_fence, VK_TRUE, WW_U64_MAX));
    if (res.failed) {
        return to_viewport_result(res);
    }

    u32 image_index;
    res = VULKAN_CHECK(vkAcquireNextImageKHR(self->device, self->swapchain.swapchain, WW_U64_MAX, image_available_semaphore, VK_NULL_HANDLE, &image_index));
    if (res.code == VK_ERROR_OUT_OF_DATE_KHR) {
        return to_viewport_result(res);
    } else if (res.failed && res.code != VK_SUBOPTIMAL_KHR) {
        return to_viewport_result(res);
    }

    res = VULKAN_CHECK(vkResetFences(self->device, 1, &in_flight_fence)); 
    if (res.failed) {
        return to_viewport_result(res);
    }

    res = VULKAN_CHECK(vkResetCommandBuffer(command_buffer, 0));
    if (res.failed) {
        return to_viewport_result(res);
    }

    res = vulkan_viewport_no_gp_record_command_buffer(self, command_buffer, image_index);
    if (res.failed) {
        return to_viewport_result(res);
    }

    VkSemaphore signal_semaphores[] = { render_finished_semaphore };
    VkSemaphore wait_semaphores[] = { image_available_semaphore };
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSubmitInfo submit_info = { 
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        
        .waitSemaphoreCount = WW_ARRAY_SIZE(wait_semaphores),
        .pWaitSemaphores = wait_semaphores,
        .pWaitDstStageMask = wait_stages,

        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,

        .signalSemaphoreCount = WW_ARRAY_SIZE(signal_semaphores),
        .pSignalSemaphores = signal_semaphores
    };

    res = VULKAN_CHECK(vkQueueSubmit(self->present_queue, 1, &submit_info, in_flight_fence));
    if (res.failed) {
        return to_viewport_result(res);
    }

    VkSwapchainKHR swap_chains[] = { self->swapchain.swapchain };
    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = WW_ARRAY_SIZE(signal_semaphores),
        .pWaitSemaphores = signal_semaphores,
        .swapchainCount = WW_ARRAY_SIZE(swap_chains),
        .pSwapchains = swap_chains,
        .pImageIndices = &image_index,
    };

    res = VULKAN_CHECK(vkQueuePresentKHR(self->present_queue, &present_info));
    self->current_frame = (self->current_frame + 1) % self->frames_in_flight;
    return to_viewport_result(res);
}

void vulkan_viewport_no_gp_cleanup_swapchain_and_input_image(viewport_ptr self) {
    if (self->input.staging_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(self->vma_allocator, self->input.staging_buffer, self->input.staging_buffer_allocation);
        self->input.staging_buffer = VK_NULL_HANDLE;
    }

    if (self->input.view != VK_NULL_HANDLE) {
        vkDestroyImageView(self->device, self->input.view, NULL);
        self->input.view = VK_NULL_HANDLE;
    }

    if (self->input.image != VK_NULL_HANDLE) {
        vmaDestroyImage(self->vma_allocator, self->input.image, self->input.allocation);
        self->input.image = VK_NULL_HANDLE;
    }

    ww_darray_resize_assume_capacity(&self->swapchain.has_undefined_layout, 0);
    ww_darray_resize_assume_capacity(&self->swapchain.images, 0);

    if (self->swapchain.swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(self->device, self->swapchain.swapchain, NULL);
        self->swapchain.swapchain = VK_NULL_HANDLE;
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL vulkan_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT type,
                                              const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                                              void* user_data) {
    #define debug_callback_log_msg "[vulkan viewport_no_gp] validation layer%s: %s\n"
    const char* msg_type_str = "";
    switch (type) {
        case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
            break;
        case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
            msg_type_str = " (validation)";
            break;
        case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
            msg_type_str = " (performance)";
            break;
        default:
            msg_type_str = " (uknown type of debug msg)";
            break;
    }
    
    switch (severity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            WW_LOG_DEBUG(debug_callback_log_msg, msg_type_str, callback_data->pMessage); 
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            WW_LOG_INFO(debug_callback_log_msg, msg_type_str, callback_data->pMessage); 
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            WW_LOG_WARN(debug_callback_log_msg, msg_type_str, callback_data->pMessage); 
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            WW_LOG_ERROR(debug_callback_log_msg, msg_type_str, callback_data->pMessage); 
            break;
        default:
            WW_LOG_ERROR("[vulkan viewport_no_gp] validation layer%s and uknown msg severity: %s\n", msg_type_str, callback_data->pMessage); 
            break;
    }

    return VK_FALSE;
}

void transition_image_layout(VkCommandBuffer command_buffer, VkImage image, TransitionImageLayoutInfo info) {
    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = info.from_layout,
        .newLayout = info.to_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
        .srcAccessMask = info.from_access,
        .dstAccessMask = info.to_access,
    };

    vkCmdPipelineBarrier(
        command_buffer, 
        info.from_pipeline_stage, 
        info.to_pipeline_stage,
        0,
        0, NULL,
        0, NULL,
        1, &barrier
    );
}

ViewportResult to_viewport_result(VulkanResult vulkan_result) {
    ViewportResult res = { .failed = vulkan_result.failed };
    switch (vulkan_result.code) {
        case VK_ERROR_OUT_OF_DATE_KHR:
            res.code = VIEWPORT_ERROR_OUT_OF_DATE;
            break;
        case VK_SUBOPTIMAL_KHR:
            res.code = VIEWPORT_SUBOPTIMAL;
            break;
        default:
            res.code = VIEWPORT_ERROR_INTERNAL;
            break;
    }

    return res;
}
