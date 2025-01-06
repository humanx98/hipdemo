#include <vulkan/vulkan_core.h>
#if defined(_WIN64)
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>
#include <ww/vulkan/viewport.h>
#include <string.h>
#include <stdlib.h>
#include <ww/defines.h>
#include <ww/file.h>
#include <ww/log.h>
#include "vma.h"

#ifdef NDEBUG
static const char* validation_layers[] = {};
#else
static const char* validation_layers[] = {
    "VK_LAYER_KHRONOS_validation"
};
#endif

typedef struct QueueFamilyIndices {
    u32 graphics;
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
    b8 prefer_vsync;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkSurfaceKHR surface;
    QueueFamilyIndices queue_family_indices;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue graphics_queue;
    VkQueue present_queue;
    VkCommandPool command_pool;
    WwDArray(VkCommandBuffer) command_buffers;
    WwDArray(VkSemaphore) image_available_semaphores;
    WwDArray(VkSemaphore) render_finished_semaphores;
    WwDArray(VkFence) in_flight_fences;
    VmaAllocator vma_allocator;
    VkShaderModule vert_module;
    VkShaderModule frag_module;
    struct {
        VkDescriptorSetLayout set_layout;
        VkDescriptorPool pool;
        WwDArray(VkDescriptorSet) sets;
    } descriptor;
    struct {
        struct {
            VkSurfaceCapabilitiesKHR capabilities;
            VkSurfaceFormatKHR format;
            VkPresentModeKHR present_mode;
            VkExtent2D extent;
        } details;
        VkSwapchainKHR swapchain;
        WwDArray(VkImage) images;
        WwDArray(VkImageView) image_views;
        WwDArray(VkFramebuffer) framebuffers;
        b8 handle_out_of_date;
    } swapchain;
    VkRenderPass render_pass;
    VkPipelineLayout pipeline_layout;
    VkPipeline graphics_pipeline;
    struct {
        VkImage image;
        VkExtent3D extent;
        VmaAllocation allocation;
        VkImageView view;
        VkBuffer staging_buffer;
        VmaAllocation staging_buffer_allocation;
        VmaAllocationInfo staging_buffer_allocation_info;
        VkSampler sampler;
    } input;
    struct {
        b8 enabled;
        VmaPool pool;
        ViewportExternalHandle handle;
    } external_memory;
    struct {
        b8 enabled;
        VkSemaphore wait;
        ViewportExternalHandle wait_handle;
        VkSemaphore signal;
        ViewportExternalHandle signal_handle;
    } external_semaphores;
} viewport_ptr_impl;

static void vulkan_viewport_destroy(viewport_ptr self);
static ViewportResult __ww_must_check vulkan_viewport_render(viewport_ptr self);
static ViewportResult __ww_must_check vulkan_viewport_wait_idle(viewport_ptr self);
static void* __ww_must_check vulkan_viewport_get_mapped_input(viewport_ptr self);
static ViewportExternalHandle __ww_must_check vulkan_viewport_get_external_memory(viewport_ptr self);
static ViewportExternalSemaphores __ww_must_check vulkan_viewport_get_external_semaphores(viewport_ptr self);
static ViewportResult __ww_must_check vulkan_viewport_set_resolution(viewport_ptr self, u32 width, u32 height);
static void vulkan_viewport_get_resolution(viewport_ptr self, u32* width, u32* height);

static VulkanResult __ww_must_check vulkan_viewport_init_vulkan(viewport_ptr self, VulkanViewportCreationProperties creation_properties);
static VulkanResult __ww_must_check vulkan_viewport_create_instance(viewport_ptr self, VulkanViewportCreationProperties creation_properties, const VkDebugUtilsMessengerCreateInfoEXT* debug_create_info);
static VulkanResult __ww_must_check vulkan_viewport_pick_physical_device(viewport_ptr self, u32 device_index, const WwDArray(const char*)* device_extensions);
static VulkanResult __ww_must_check vulkan_viewport_get_swap_chain_details(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_create_logical_device(viewport_ptr self, const WwDArray(const char*)* device_extensions);
static VulkanResult __ww_must_check vulkan_viewport_create_command_pool(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_create_command_buffers(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_create_sync_objects(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_create_shader_modules(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_allocate_descriptor_sets(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_create_swapchain(viewport_ptr self, u32 width, u32 height);
static VulkanResult __ww_must_check vulkan_viewport_create_swapchain_image_views(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_create_render_pass(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_create_graphics_pipeline(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_create_swapchain_framebuffers(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_create_input_image(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_create_external_memory_pool(viewport_ptr self, const VkBufferCreateInfo* create_info, const VmaAllocationCreateInfo* alloc_create_info);
static void vulkan_viewport_update_descriptor_sets(viewport_ptr self);
static VulkanResult __ww_must_check vulkan_viewport_record_command_buffer(viewport_ptr self, VkCommandBuffer command_buffer, u32 image_index);
static void vulkan_viewport_cleanup_swapchain_pipeline_input_image(viewport_ptr self);
static VKAPI_ATTR VkBool32 VKAPI_CALL vulkan_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data); 
static void transition_image_layout(VkCommandBuffer command_buffer, VkImage image, TransitionImageLayoutInfo info);
static ViewportResult __ww_must_check to_viewport_result(VulkanResult res);
static VulkanResult __ww_must_check create_external_semaphore(VkDevice device, VkSemaphore* semaphore, ViewportExternalHandle* handle);

VulkanResult vulkan_viewport_create(VulkanViewportCreationProperties creation_properties, Viewport* viewport) {
    assert(viewport);

    ww_auto_type alloc_result = ww_allocator_alloc_type(creation_properties.allocator, viewport_ptr_impl);
    if (alloc_result.failed) {
        return VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
    }

    viewport_ptr self = alloc_result.ptr;
    VulkanResult res = vulkan_viewport_init_vulkan(self, creation_properties);
    if (res.failed) {
        vulkan_viewport_destroy(self);
        return res;
    }

    const static viewport_vtable vtable = {
        .render = vulkan_viewport_render,
        .wait_idle = vulkan_viewport_wait_idle,
        .get_mapped_input = vulkan_viewport_get_mapped_input,
        .get_external_memory = vulkan_viewport_get_external_memory,
        .get_external_semaphores = vulkan_viewport_get_external_semaphores,
        .set_resolution = vulkan_viewport_set_resolution,
        .get_resolution = vulkan_viewport_get_resolution,
        .destroy = vulkan_viewport_destroy,
    };

    *viewport = (Viewport) {
        .ptr = self,
        .vtable = &vtable,
    };

    return res;
}

VulkanResult vulkan_viewport_init_vulkan(viewport_ptr self, VulkanViewportCreationProperties creation_properties) {
    *self = (viewport_ptr_impl) {
        .allocator = creation_properties.allocator,
        .frames_in_flight = creation_properties.frames_in_flight,
        .prefer_vsync = creation_properties.prefer_vsync,
        .swapchain.images = ww_darray_init(creation_properties.allocator, VkImage),
        .swapchain.image_views = ww_darray_init(creation_properties.allocator, VkImageView),
        .swapchain.framebuffers = ww_darray_init(creation_properties.allocator, VkFramebuffer),
        .command_buffers = ww_darray_init(creation_properties.allocator, VkCommandBuffer),
        .image_available_semaphores = ww_darray_init(creation_properties.allocator, VkSemaphore),
        .render_finished_semaphores = ww_darray_init(creation_properties.allocator, VkSemaphore),
        .in_flight_fences = ww_darray_init(creation_properties.allocator, VkFence),
        .external_memory = {
            .enabled = creation_properties.external_memory,
        },
        .external_semaphores = {
            .enabled = creation_properties.external_semaphores,
        },
        .descriptor = {
            .sets = ww_darray_init(creation_properties.allocator, VkDescriptorSet),
        },
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
    VulkanResult res = vulkan_viewport_create_instance(self, creation_properties, p_debug_create_info);
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

    WwDArray(const char*) device_extensions = ww_darray_init(self->allocator, const char*);
    const char* device_default_extensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    const char* device_external_memory_extensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    #if defined(_WIN64)
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    #else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    #endif
    };

    const char* device_external_semaphore_extensions[] = {
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
    #if defined(_WIN64)
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME
    #else
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
    #endif
    };

    if (!ww_darray_append_many(&device_extensions, device_default_extensions, WW_ARRAY_SIZE(device_default_extensions))) {
        goto failed_device_creation;
    }
    
    if (self->external_memory.enabled && !ww_darray_append_many(&device_extensions, device_external_memory_extensions, WW_ARRAY_SIZE(device_external_memory_extensions))) {
        goto failed_device_creation;
    }

    if (self->external_semaphores.enabled && !ww_darray_append_many(&device_extensions, device_external_semaphore_extensions, WW_ARRAY_SIZE(device_external_semaphore_extensions))) {
        goto failed_device_creation;
    }

    res = vulkan_viewport_pick_physical_device(self, creation_properties.device_index, &device_extensions);
    if (res.failed) {
        goto failed_device_creation;
    }

    res = vulkan_viewport_create_logical_device(self, &device_extensions);
failed_device_creation:
    ww_darray_deinit(&device_extensions);
    if (res.failed) {
        return res;
    }

    res = vulkan_viewport_create_command_pool(self);
    if (res.failed) {
        return res;
    }

    res = vulkan_viewport_create_command_buffers(self);
    if (res.failed) {
        return res;
    }

    res = vulkan_viewport_create_sync_objects(self);
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
    if (res.failed) {
        return res;
    }

    res = vulkan_viewport_create_shader_modules(self);
    if (res.failed) {
        return res;
    }

    VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
    };
    res = VULKAN_CHECK(vkCreateSampler(self->device, &sampler_info, NULL, &self->input.sampler));
    if (res.failed) {
        return res;
    }

    res = vulkan_viewport_allocate_descriptor_sets(self);
    return res;
}

void vulkan_viewport_destroy(viewport_ptr self) {
    assert(self);

    vulkan_viewport_cleanup_swapchain_pipeline_input_image(self);
    ww_darray_deinit(&self->swapchain.framebuffers);
    ww_darray_deinit(&self->swapchain.image_views);
    ww_darray_deinit(&self->swapchain.images);

    ww_darray_deinit(&self->descriptor.sets);
    if (self->descriptor.pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(self->device, self->descriptor.pool, NULL);
    }

    if (self->descriptor.set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(self->device, self->descriptor.set_layout, NULL);
    }

    if (self->input.sampler != VK_NULL_HANDLE) {
        vkDestroySampler(self->device, self->input.sampler, NULL);
    }

    if (self->frag_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(self->device, self->frag_module, NULL);
    }

    if (self->vert_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(self->device, self->vert_module, NULL);
    }

    if (self->external_memory.pool) {
        vmaDestroyPool(self->vma_allocator, self->external_memory.pool);
    }

    if (self->vma_allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(self->vma_allocator);
    }

    if (self->external_semaphores.wait) {
        vkDestroySemaphore(self->device, self->external_semaphores.wait, NULL);
    }

    if (self->external_semaphores.signal) {
        vkDestroySemaphore(self->device, self->external_semaphores.signal, NULL);
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

ViewportResult vulkan_viewport_wait_idle(viewport_ptr self) {
    assert(self);
    return to_viewport_result(VULKAN_CHECK(vkQueueWaitIdle(self->present_queue)));
}

ViewportResult vulkan_viewport_set_resolution(viewport_ptr self, u32 width, u32 height) {
    assert(self);

    VulkanResult res;
    if (self->external_semaphores.enabled && self->swapchain.handle_out_of_date) {
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo submit_info = { 
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &self->external_semaphores.signal,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &self->external_semaphores.wait,
            .pWaitDstStageMask = &wait_stage,
        };

        res = VULKAN_CHECK(vkQueueSubmit(self->present_queue, 1, &submit_info, NULL));
        if (res.failed) {
            return to_viewport_result(res);
        }

        self->swapchain.handle_out_of_date = false;
        self->current_frame = (self->current_frame + 1) % self->frames_in_flight;
    }

    res = VULKAN_CHECK(vkQueueWaitIdle(self->present_queue));
    if (res.failed) {
        return to_viewport_result(res);
    }

    vulkan_viewport_cleanup_swapchain_pipeline_input_image(self);

    res = vulkan_viewport_get_swap_chain_details(self);
    if (res.failed) {
        return to_viewport_result(res);
    }

    res = vulkan_viewport_create_swapchain(self, width, height);
    if (res.failed) {
        return to_viewport_result(res);
    }

    res = vulkan_viewport_create_swapchain_image_views(self);
    if (res.failed) {
        return to_viewport_result(res);
    }

    res = vulkan_viewport_create_render_pass(self);
    if (res.failed) {
        return to_viewport_result(res);
    }

    res = vulkan_viewport_create_swapchain_framebuffers(self);
    if (res.failed) {
        return to_viewport_result(res);
    }

    res = vulkan_viewport_create_input_image(self);
    if (res.failed) {
        return to_viewport_result(res);
    }
    VkCommandBufferBeginInfo begin_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VkCommandBuffer command_buffer = ww_darray_get(&self->command_buffers, VkCommandBuffer, 0);
    res = VULKAN_CHECK(vkResetCommandBuffer(command_buffer, 0));
    if (res.failed) {
        return to_viewport_result(res);
    }

    res = VULKAN_CHECK(vkBeginCommandBuffer(command_buffer, &begin_info));
    if (res.failed) {
        return to_viewport_result(res);
    }

    TransitionImageLayoutInfo transition_image_layout_info = {
        .from_pipeline_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        .to_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .to_access = VK_ACCESS_SHADER_READ_BIT,
        .to_pipeline_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    };
    transition_image_layout(command_buffer, self->input.image, transition_image_layout_info);

    res = VULKAN_CHECK(vkEndCommandBuffer(command_buffer));
    if (res.failed) {
        return to_viewport_result(res);
    }

    VkSubmitInfo submit_info = { 
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };

    res = VULKAN_CHECK(vkQueueSubmit(self->present_queue, 1, &submit_info, NULL));
    if (res.failed) {
        return to_viewport_result(res);
    }

    res = VULKAN_CHECK(vkQueueWaitIdle(self->present_queue));
    if (res.failed) {
        return to_viewport_result(res);
    }

    res = vulkan_viewport_create_graphics_pipeline(self);
    if (res.failed) {
        return to_viewport_result(res);
    }

    vulkan_viewport_update_descriptor_sets(self);
    if (res.failed) {
        return to_viewport_result(res);
    }

    return to_viewport_result(res);
}

void vulkan_viewport_get_resolution(viewport_ptr self, u32* width, u32* height) {
    assert(self);
    assert(width);
    assert(height);

    *width = self->swapchain.details.extent.width;
    *height = self->swapchain.details.extent.height;
}

void* vulkan_viewport_get_mapped_input(viewport_ptr self) {
    assert(self);
    assert(!self->external_memory.enabled);
    return self->input.staging_buffer_allocation_info.pMappedData;
}

ViewportExternalHandle vulkan_viewport_get_external_memory(viewport_ptr self) {
    assert(self);
    assert(self->external_memory.enabled);
    return self->external_memory.handle;
}

ViewportExternalSemaphores vulkan_viewport_get_external_semaphores(viewport_ptr self) {
    assert(self);
    assert(self->external_semaphores.enabled);
    return (ViewportExternalSemaphores){
        .signal_external_memory_for_viewport = self->external_semaphores.wait_handle,
        .wait_for_signal_external_memory_from_viewport = self->external_semaphores.signal_handle,
    };
}

VulkanResult vulkan_viewport_create_instance(viewport_ptr self, VulkanViewportCreationProperties creation_properties, const VkDebugUtilsMessengerCreateInfoEXT* debug_create_info) {
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
                WW_LOG_ERROR("[vulkan viewport] Couldn't find %s vulkan layer\n", *required_layer);
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

    const char* external_memory_extensions[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
    };
    if (self->external_memory.enabled && !ww_darray_append_many(&required_extensions, external_memory_extensions, WW_ARRAY_SIZE(external_memory_extensions))) {
        goto failed;
    }

    const char* external_semaphore_capabilities_extension_name = VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME;
    if (self->external_semaphores.enabled && !ww_darray_append(&required_extensions, external_semaphore_capabilities_extension_name)) {
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

VulkanResult vulkan_viewport_pick_physical_device(viewport_ptr self, u32 device_index, const WwDArray(const char*)* device_extensions) {
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

    if (!queue_family_indices.graphics.found) {
        WW_LOG_ERROR("[vulkan viewport] Couldn't find graphics queue family\n");
        res = VULKAN_CHECK(VK_ERROR_INITIALIZATION_FAILED);
        goto failed;
    } else {
        WW_LOG_INFO("[vulkan viewport] graphics queue family index: %d\n", queue_family_indices.graphics.index);
        self->queue_family_indices.graphics = queue_family_indices.graphics.index;
    }    
    
    if (!queue_family_indices.present.found) {
        WW_LOG_ERROR("[vulkan viewport] Couldn't find present queue family\n");
        res = VULKAN_CHECK(VK_ERROR_INITIALIZATION_FAILED);
        goto failed;
    } else {
        WW_LOG_INFO("[vulkan viewport] present queue family index: %d\n", queue_family_indices.present.index);
        self->queue_family_indices.present = queue_family_indices.present.index;
    } 

    res = vulkan_enumerate_device_extension_properties(self->physical_device, NULL, &available_extension_properties);
    if (res.failed) {
        goto failed;
    }

    ww_darray_foreach_by_ref(device_extensions, const char*, required_extension) {
        b8 found_extension = false;
        ww_darray_foreach_by_ref(&available_extension_properties, VkExtensionProperties, available_extension) {
            if (!strcmp(*required_extension, available_extension->extensionName)) {
                found_extension = true;
                break;
            }
        }

        if (!found_extension) {
            WW_LOG_ERROR("[vulkan viewport] Couldn't find device extension: %s\n", *required_extension);
            res = VULKAN_CHECK(VK_ERROR_EXTENSION_NOT_PRESENT);
            goto failed;
        }
    }

failed:
    ww_darray_deinit(&available_extension_properties);
    return res;
}

VulkanResult vulkan_viewport_get_swap_chain_details(viewport_ptr self) {
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
        WW_LOG_ERROR("[vulkan viewport] No device surface formats");
        res = VULKAN_CHECK(VK_ERROR_INITIALIZATION_FAILED);
        goto failed;
    }

    res = vulkan_get_physical_device_surface_present_modes(self->physical_device, self->surface, &present_modes);
    if (res.failed) {
        goto failed;
    } else if (ww_darray_len(&present_modes) == 0) {
        WW_LOG_ERROR("[vulkan viewport] No device surface present modes");
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
        WW_LOG_WARN("[vulkan viewport] Couldn't find wanted surface fomat");
        self->swapchain.details.format = ww_darray_get(&formats, VkSurfaceFormatKHR, 0);
    }

    b8 found_fifo_mode = false;
    b8 found_mailbox_mode = false;
    b8 found_immediate_mode = false;
    ww_darray_foreach_by_ref(&present_modes, VkPresentModeKHR, pm) {
        if (*pm == VK_PRESENT_MODE_MAILBOX_KHR) {
            found_mailbox_mode = true;
        }

        if (*pm == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            found_immediate_mode = true;
        }

        if (*pm == VK_PRESENT_MODE_FIFO_KHR) {
            found_fifo_mode = true;
        }
    }

    if (self->prefer_vsync && found_fifo_mode) {
        self->swapchain.details.present_mode = VK_PRESENT_MODE_FIFO_KHR;
    } else if (found_mailbox_mode) {
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

VulkanResult vulkan_viewport_create_logical_device(viewport_ptr self, const WwDArray(const char*)* device_extensions) {
    WwDArray(VkDeviceQueueCreateInfo) queue_create_infos = ww_darray_init(self->allocator, VkDeviceQueueCreateInfo);
    VulkanResult res = {};
    if (!ww_darray_ensure_total_capacity_precise(&queue_create_infos, 2)) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        goto failed;
    }

    f32 queue_priority = 1.0f;
    VkDeviceQueueCreateInfo graphics_queue_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = self->queue_family_indices.graphics,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    ww_darray_append_assume_capacity(&queue_create_infos, graphics_queue_create_info);

    if (self->queue_family_indices.graphics != self->queue_family_indices.present) {
        VkDeviceQueueCreateInfo present_queue_create_info = graphics_queue_create_info;
        present_queue_create_info.queueFamilyIndex = self->queue_family_indices.present;
        ww_darray_append_assume_capacity(&queue_create_infos, present_queue_create_info);
    }

    VulkanDeviceCreateInfo device_create_info = {
        .queue_create_info_count = ww_darray_len(&queue_create_infos),
        .queue_create_infos = ww_darray_ptr(&queue_create_infos),
        .enabled_extension_count = ww_darray_len(device_extensions),
        .enabled_extensions = ww_darray_ptr(device_extensions),
        .enabled_layer_count = WW_ARRAY_SIZE(validation_layers),
        .enabled_layers = validation_layers,
    };
    res = vulkan_create_device(self->physical_device, &device_create_info, &self->device);
    if (res.failed) {
        goto failed;
    }

    vkGetDeviceQueue(self->device, self->queue_family_indices.graphics, 0, &self->graphics_queue);
    vkGetDeviceQueue(self->device, self->queue_family_indices.present, 0, &self->present_queue);

failed:
    ww_darray_deinit(&queue_create_infos);
    return res;
}

VulkanResult vulkan_viewport_create_command_pool(viewport_ptr self) {
    VkCommandPoolCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = self->queue_family_indices.graphics,
    };
    return VULKAN_CHECK(vkCreateCommandPool(self->device, &create_info, NULL, &self->command_pool));
}

VulkanResult vulkan_viewport_create_command_buffers(viewport_ptr self) {
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

VulkanResult vulkan_viewport_create_sync_objects(viewport_ptr self) {
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

    if (self->external_semaphores.enabled) {
        res = create_external_semaphore(self->device, &self->external_semaphores.signal, &self->external_semaphores.signal_handle);
        if (res.failed) {
            return res;
        } else {
            VkSubmitInfo submit_info = { 
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &self->external_semaphores.signal,
            };

            res = VULKAN_CHECK(vkQueueSubmit(self->present_queue, 1, &submit_info, NULL));
            if (res.failed) {
                return res;
            }

            res = VULKAN_CHECK(vkQueueWaitIdle(self->present_queue));
            if (res.failed) {
                return res;
            }
        }

        res = create_external_semaphore(self->device, &self->external_semaphores.wait, &self->external_semaphores.wait_handle);
        if (res.failed) {
            return res;
        }
    }

    return res;
}

VulkanResult vulkan_viewport_create_shader_modules(viewport_ptr self) {
    WwDArray(u8) vert_shader_code = ww_darray_init(self->allocator, u8);
    WwDArray(u8) frag_shader_code = ww_darray_init(self->allocator, u8);
    VulkanResult res = {};
    if (!ww_read_file_to_darray("hip_spv_bin/fullscreen_quad.vert.spv", &vert_shader_code)
        || !ww_read_file_to_darray("hip_spv_bin/fullscreen_quad.frag.spv", &frag_shader_code)) {
        res = VULKAN_CHECK(VK_ERROR_INITIALIZATION_FAILED);
        goto failed;
    }

    VkShaderModuleCreateInfo vert_shader_create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = ww_darray_len(&vert_shader_code) * ww_darray_elem_size(&vert_shader_code),
        .pCode = ww_darray_ptr(&vert_shader_code),
    };

    res = VULKAN_CHECK(vkCreateShaderModule(self->device, &vert_shader_create_info, NULL, &self->vert_module));
    if (res.failed) {
        goto failed;
    }

    VkShaderModuleCreateInfo frag_shader_create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = ww_darray_len(&frag_shader_code) * ww_darray_elem_size(&frag_shader_code),
        .pCode = ww_darray_ptr(&frag_shader_code),
    };

    res = VULKAN_CHECK(vkCreateShaderModule(self->device, &frag_shader_create_info, NULL, &self->frag_module));
    if (res.failed) {
        goto failed;
    }

failed:
    ww_darray_deinit(&vert_shader_code);
    ww_darray_deinit(&frag_shader_code);
    return res;
}

VulkanResult vulkan_viewport_allocate_descriptor_sets(viewport_ptr self) {
    VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };

    VkDescriptorSetLayoutCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };

    VulkanResult res = VULKAN_CHECK(vkCreateDescriptorSetLayout(self->device, &info, NULL, &self->descriptor.set_layout));
    if (res.failed) {
        return res;
    }

    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = self->frames_in_flight,
    };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
        .maxSets = self->frames_in_flight,
    };
    res = VULKAN_CHECK(vkCreateDescriptorPool(self->device, &pool_info, NULL, &self->descriptor.pool));
    if (res.failed) {
        return res;
    }
    
    if (!ww_darray_ensure_total_capacity_precise(&self->descriptor.sets, self->frames_in_flight)) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        return res;
    }

    VkDescriptorSetLayout layouts[self->frames_in_flight];
    for (usize i = 0; i < self->frames_in_flight; i++) {
        layouts[i] = self->descriptor.set_layout;
    }
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = self->descriptor.pool,
        .descriptorSetCount = self->frames_in_flight,
        .pSetLayouts = layouts,
    };
    res = VULKAN_CHECK(vkAllocateDescriptorSets(self->device, &alloc_info, ww_darray_ptr(&self->descriptor.sets)));
    if (res.failed) {
        return res;
    }

    ww_darray_resize_assume_capacity(&self->descriptor.sets, self->frames_in_flight);
    return res;
}

VulkanResult vulkan_viewport_create_swapchain(viewport_ptr self, u32 width, u32 height) {
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
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform = self->swapchain.details.capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = self->swapchain.details.present_mode,
        .clipped = VK_TRUE,
    };

    u32 queue_family_indices[] = {
        self->queue_family_indices.graphics,
        self->queue_family_indices.present,
    };
    if (self->queue_family_indices.graphics != self->queue_family_indices.present) {
        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = WW_ARRAY_SIZE(queue_family_indices);
        create_info.pQueueFamilyIndices = queue_family_indices; 
    }

    VulkanResult res = VULKAN_CHECK(vkCreateSwapchainKHR(self->device, &create_info, NULL, &self->swapchain.swapchain));
    if (res.failed) {
        return res;
    }

    res = vulkan_get_swapchain_images(self->device, self->swapchain.swapchain, &self->swapchain.images);
    return res;
}

VulkanResult vulkan_viewport_create_swapchain_image_views(viewport_ptr self) {
    VulkanResult res = {};

    if (!ww_darray_ensure_total_capacity_precise(&self->swapchain.image_views, ww_darray_len(&self->swapchain.images))) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        return res;
    }

    ww_darray_foreach_by_ref(&self->swapchain.images, VkImage, img) {
        VkImageViewCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = *img, 
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = self->swapchain.details.format.format,
            .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .subresourceRange.levelCount = 1,
            .subresourceRange.layerCount = 1,
        };
        VkImageView view;
        res = VULKAN_CHECK(vkCreateImageView(self->device, &create_info, NULL, &view));
        if (res.failed) {
            return res;
        }

        ww_darray_append_assume_capacity(&self->swapchain.image_views, view);
    }

    return res;
}

VulkanResult vulkan_viewport_create_render_pass(viewport_ptr self) {
    VkAttachmentDescription attachment = {
        .format = self->swapchain.details.format.format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    VkAttachmentReference attachment_ref = {
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment_ref,
    };

    VkSubpassDependency dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    VkRenderPassCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    return VULKAN_CHECK(vkCreateRenderPass(self->device, &create_info, NULL, &self->render_pass));
}

VulkanResult vulkan_viewport_create_graphics_pipeline(viewport_ptr self) {
    VkPipelineShaderStageCreateInfo shader_stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = self->vert_module,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = self->frag_module,
            .pName = "main",
        }
    };

    VkPipelineVertexInputStateCreateInfo vertex_input_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .lineWidth = 1.0f,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
    };

    VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable = VK_FALSE,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    VkPipelineColorBlendAttachmentState color_blend_attachment = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        .blendEnable = VK_FALSE,
    };
    VkPipelineColorBlendStateCreateInfo color_blending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
        .blendConstants[0] = 0.0f,
        .blendConstants[1] = 0.0f,
        .blendConstants[2] = 0.0f,
        .blendConstants[3] = 0.0f,
    };

    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamic_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = WW_ARRAY_SIZE(dynamic_states),
        .pDynamicStates = dynamic_states, 
    };

    VkPipelineLayoutCreateInfo layout_info = { 
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &self->descriptor.set_layout,
    };
    VulkanResult res = VULKAN_CHECK(vkCreatePipelineLayout(self->device, &layout_info, NULL, &self->pipeline_layout));
    if (res.failed) {
        return res;
    }

    VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = WW_ARRAY_SIZE(shader_stages),
        .pStages = shader_stages,
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_state,
        .layout = self->pipeline_layout,
        .renderPass = self->render_pass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
    };
    res = VULKAN_CHECK(vkCreateGraphicsPipelines(self->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &self->graphics_pipeline));
    return res;
}

VulkanResult vulkan_viewport_create_swapchain_framebuffers(viewport_ptr self) {
    VulkanResult res = {};
    if (!ww_darray_ensure_total_capacity_precise(&self->swapchain.framebuffers, ww_darray_len(&self->swapchain.image_views))) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        return res;
    }

    ww_darray_foreach_by_ref(&self->swapchain.image_views, VkImageView, view) {
        VkImageView attachments[] = { *view };
        VkFramebufferCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = self->render_pass,
            .attachmentCount = WW_ARRAY_SIZE(attachments),
            .pAttachments = attachments,
            .width = self->swapchain.details.extent.width,
            .height = self->swapchain.details.extent.height,
            .layers = 1,
        };
        VkFramebuffer framebuffer;
        res = VULKAN_CHECK(vkCreateFramebuffer(self->device, &create_info, NULL, &framebuffer));
        if (res.failed) {
            return res;
        }

        ww_darray_append_assume_capacity(&self->swapchain.framebuffers, framebuffer);
    }

    return res;
}

static VulkanResult vulkan_viewport_create_input_image(viewport_ptr self) {
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
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .samples = VK_SAMPLE_COUNT_1_BIT,
    };

    VmaAllocationCreateInfo alloc_create_info = {
        .usage = VMA_MEMORY_USAGE_AUTO,
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
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
    VkExternalMemoryBufferCreateInfoKHR external_create_info = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR,
#if defined(_WIN64)
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
#else
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
#endif
    };

    if (self->external_memory.enabled) {
        buf_create_info.pNext = &external_create_info;
        alloc_create_info = (VmaAllocationCreateInfo){
            .usage = VMA_MEMORY_USAGE_AUTO,
            .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        };

        if (!self->external_memory.pool) {
            res = vulkan_viewport_create_external_memory_pool(self, &buf_create_info, &alloc_create_info);
            if (res.failed) {
                return res;
            }
        }

        alloc_create_info.pool = self->external_memory.pool;

    } else {
        alloc_create_info = (VmaAllocationCreateInfo){
            .usage = VMA_MEMORY_USAGE_AUTO,
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
        };
    }

    res = VULKAN_CHECK(vmaCreateBuffer(
        self->vma_allocator,
        &buf_create_info,
        &alloc_create_info,
        &self->input.staging_buffer,
        &self->input.staging_buffer_allocation,
        &self->input.staging_buffer_allocation_info
    ));
    if (res.failed) {
        return res;
    }

    if (self->external_memory.enabled) {
#if defined(_WIN64)
        VkMemoryGetWin32HandleInfoKHR get_handle_info = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
            .memory = self->input.staging_buffer_allocation_info.deviceMemory,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
        };

        self->external_memory.handle.type = VIEWPORT_EXTERNAL_HANDLE_WIN32;
        res = VULKAN_CHECK(vkGetMemoryWin32HandleKHR(self->device, &get_handle_info, &self->external_memory.handle.handle.win32));
        if (res.failed) {
            return res;
        }
#else
        VkMemoryGetFdInfoKHR get_handle_info = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
            .memory = self->input.staging_buffer_allocation_info.deviceMemory,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
        };

        self->external_memory.handle.type = VIEWPORT_EXTERNAL_HANDLE_FD;
        res = VULKAN_CHECK(vkGetMemoryFdKHR(self->device, &get_handle_info, &self->external_memory.handle.handle.fd));
        if (res.failed) {
            return res;
        }
#endif
    }

    return res;
}

VulkanResult vulkan_viewport_create_external_memory_pool(viewport_ptr self, const VkBufferCreateInfo* create_info, const VmaAllocationCreateInfo* alloc_create_info) {
    u32 mem_type_index;
    VulkanResult res = VULKAN_CHECK(vmaFindMemoryTypeIndexForBufferInfo(self->vma_allocator, create_info, alloc_create_info, &mem_type_index));
    if (res.failed) {
        return res;
    }

    static VkExportMemoryAllocateInfoKHR export_mem_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
#if defined(_WIN64)
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
#else
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
#endif
    };
    VmaPoolCreateInfo pool_create_info = {
        .memoryTypeIndex = mem_type_index,
        .pMemoryAllocateNext = &export_mem_alloc_info,
    };
    return VULKAN_CHECK(vmaCreatePool(self->vma_allocator, &pool_create_info, &self->external_memory.pool));
}

void vulkan_viewport_update_descriptor_sets(viewport_ptr self) {
    VulkanResult res  = {};
    for (usize i = 0; i < self->frames_in_flight; i++) {
        VkDescriptorImageInfo image_info = {
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .imageView = self->input.view,
            .sampler = self->input.sampler,
        };

        VkWriteDescriptorSet write = {
           .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ww_darray_get(&self->descriptor.sets, VkDescriptorSet, i),
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .pImageInfo = &image_info,
        };
        vkUpdateDescriptorSets(self->device, 1, &write, 0, NULL);
    }
}

VulkanResult vulkan_viewport_record_command_buffer(viewport_ptr self, VkCommandBuffer command_buffer, u32 image_index) {
    VkCommandBufferBeginInfo begin_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VulkanResult res = VULKAN_CHECK(vkBeginCommandBuffer(command_buffer, &begin_info));
    if (res.failed) {
        return res;
    }

    {
        TransitionImageLayoutInfo transition_image_layout_info = {
            .from_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .from_access = VK_ACCESS_SHADER_READ_BIT,
            .from_pipeline_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .to_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .to_access = VK_ACCESS_TRANSFER_WRITE_BIT,
            .to_pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT,
        };
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
            .to_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .to_access = VK_ACCESS_SHADER_READ_BIT,
            .to_pipeline_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        };
        transition_image_layout(command_buffer, self->input.image, transition_image_layout_info); 
    }

    {
        VkClearValue clear_value = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        VkRenderPassBeginInfo render_pass_info = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = self->render_pass,
            .framebuffer = ww_darray_get(&self->swapchain.framebuffers, VkFramebuffer, image_index),
            .renderArea.offset = {0, 0},
            .renderArea.extent = self->swapchain.details.extent,
            .clearValueCount = 1,
            .pClearValues = &clear_value,
        };
        vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
        {
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self->graphics_pipeline);
            VkDescriptorSet descriptor_set = ww_darray_get(&self->descriptor.sets, VkDescriptorSet, self->current_frame);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self->pipeline_layout, 0, 1, &descriptor_set, 0, NULL);

            VkViewport viewport = {
                viewport.x = 0.0f,
                viewport.y = 0.0f,
                viewport.width = (f32)self->swapchain.details.extent.width,
                viewport.height = (f32)self->swapchain.details.extent.height,
                viewport.minDepth = 0.0f,
                viewport.maxDepth = 1.0f,
            };
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);

            VkRect2D scissor = { .offset = { 0, 0}, .extent = self->swapchain.details.extent };
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);
            
            vkCmdDraw(command_buffer, 3, 1, 0, 0);
        }
        vkCmdEndRenderPass(command_buffer);
    }

    res = VULKAN_CHECK(vkEndCommandBuffer(command_buffer));
    return res;
}

ViewportResult vulkan_viewport_render(viewport_ptr self) {
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
        self->swapchain.handle_out_of_date = true;
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

    res = vulkan_viewport_record_command_buffer(self, command_buffer, image_index);
    if (res.failed) {
        return to_viewport_result(res);
    }

    VkSemaphore wait_for_external_memory = NULL;
    VkSemaphore signal_external_memory = NULL;
    u32 wait_count = 1;
    u32 signal_count = 1;
    if (self->external_semaphores.enabled) {
        wait_for_external_memory = self->external_semaphores.wait;
        signal_external_memory = self->external_semaphores.signal;
        wait_count = 2;
        signal_count = 2;
    }

    VkSemaphore signal_semaphores[] = { render_finished_semaphore, signal_external_memory };
    VkSemaphore wait_semaphores[] = { image_available_semaphore, wait_for_external_memory };
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSubmitInfo submit_info = { 
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        
        .waitSemaphoreCount = wait_count,
        .pWaitSemaphores = wait_semaphores,
        .pWaitDstStageMask = wait_stages,

        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,

        .signalSemaphoreCount = signal_count,
        .pSignalSemaphores = signal_semaphores
    };

    res = VULKAN_CHECK(vkQueueSubmit(self->graphics_queue, 1, &submit_info, in_flight_fence));
    if (res.failed) {
        return to_viewport_result(res);
    }

    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &render_finished_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &self->swapchain.swapchain,
        .pImageIndices = &image_index,
    };

    res = VULKAN_CHECK(vkQueuePresentKHR(self->present_queue, &present_info));
    self->current_frame = (self->current_frame + 1) % self->frames_in_flight;
    return to_viewport_result(res);
}

void vulkan_viewport_cleanup_swapchain_pipeline_input_image(viewport_ptr self) {
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

    ww_darray_foreach_by_ref(&self->swapchain.framebuffers, VkFramebuffer, fb) {
        vkDestroyFramebuffer(self->device, *fb, NULL);
    }
    ww_darray_resize_assume_capacity(&self->swapchain.framebuffers, 0);

    if (self->pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(self->device, self->pipeline_layout, NULL);
        self->pipeline_layout = VK_NULL_HANDLE;
    }

    if (self->graphics_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(self->device, self->graphics_pipeline, NULL);
        self->graphics_pipeline = VK_NULL_HANDLE;
    }

    if (self->render_pass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(self->device, self->render_pass, NULL);
        self->render_pass = VK_NULL_HANDLE;
    }

    ww_darray_foreach_by_ref(&self->swapchain.image_views, VkImageView, v) {
        vkDestroyImageView(self->device, *v, NULL);
    }
    ww_darray_resize_assume_capacity(&self->swapchain.image_views, 0);
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
    #define debug_callback_log_msg "[vulkan viewport] validation layer%s: %s\n"
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
            WW_LOG_ERROR("[vulkan viewport] validation layer%s and uknown msg severity: %s\n", msg_type_str, callback_data->pMessage); 
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

VulkanResult create_external_semaphore(VkDevice device, VkSemaphore* semaphore, ViewportExternalHandle* handle) {
    VkSemaphoreCreateInfo semaphore_create_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkExportSemaphoreCreateInfoKHR export_create_info = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR,
        .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
    };
    semaphore_create_info.pNext = &export_create_info;

    VulkanResult res = VULKAN_CHECK(vkCreateSemaphore(device, &semaphore_create_info, NULL, semaphore));
    if (res.failed) {
        return res;
    }


#if defined(_WIN64)
    *handle = (ViewportExternalHandle){ .type = VIEWPORT_EXTERNAL_HANDLE_WIN32 };
    VkSemaphoreGetWin32HandleInfoKHR get_handle_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
        .semaphore = *semaphore,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
    };
    res = VULKAN_CHECK(vkGetSemaphoreWin32HandleKHR(device, &get_handle_info, &handle->handle.win32));
#else
    *handle = (ViewportExternalHandle){ .type = VIEWPORT_EXTERNAL_HANDLE_FD };
    VkSemaphoreGetFdInfoKHR get_handle_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
        .semaphore = *semaphore,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
    };
    res = VULKAN_CHECK(vkGetSemaphoreFdKHR(device, &get_handle_info, &handle->handle.fd));
#endif
    if (res.failed) {
        return res;
    }

    return res;
}