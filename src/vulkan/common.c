#include <ww/vulkan/common.h>
#include <ww/defines.h>
#include <string.h>
#include <stdlib.h>
#include <ww/log.h>
#include <ww/exit.h>

VulkanResult vulkan_check(const char * file, const i32 line, VkResult res, const char* expression) {
    switch (res) {
        case VK_SUCCESS:
            break;
        case VK_SUBOPTIMAL_KHR:
            WW_LOG_WARN("%s:%d: \"%s\" returned res %d\n", file, line, expression, res);
            break;
        default:
            WW_LOG_ERROR("%s:%d: \"%s\" returned error %d\n", file, line, expression, res);
            break;
    }

    return (VulkanResult){
        .failed = res != VK_SUCCESS ? true : false,
        .code = res,
    };
}

VulkanResult vulkan_create_debugger_messenger(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* debug_create_info, VkDebugUtilsMessengerEXT* debug_messenger) {
    assert(instance != VK_NULL_HANDLE);
    assert(debug_messenger != VK_NULL_HANDLE);

    if (debug_create_info == NULL) {
        return VULKAN_CHECK(VK_SUCCESS);
    }
   
    PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func == NULL) {
        return VULKAN_CHECK(VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    return VULKAN_CHECK(func(instance, debug_create_info, NULL, debug_messenger));
}

VulkanResult vulkan_enumerate_physical_devices(VkInstance instance, WwDArray(VkPhysicalDevice)* physical_devices) {
    assert(instance != VK_NULL_HANDLE);
    assert(physical_devices);

    u32 count;
    VulkanResult res = VULKAN_CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
    if (res.failed) {
        return res;
    }

    if (!ww_darray_ensure_total_capacity_precise(physical_devices, count)) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        return res;
    }

    res = VULKAN_CHECK(vkEnumeratePhysicalDevices(instance, &count, ww_darray_ptr(physical_devices)));
    if (res.failed) {
        return res;
    }

    ww_darray_resize_assume_capacity(physical_devices, count);
    return res; 
}

VulkanResult vulkan_enumerate_device_extension_properties(VkPhysicalDevice physical_device, const char* layer_name, WwDArray(VkExtensionProperties)* extension_properties) {
    assert(physical_device != VK_NULL_HANDLE);
    assert(extension_properties);

    u32 count;
    VulkanResult res = VULKAN_CHECK(vkEnumerateDeviceExtensionProperties(physical_device, layer_name, &count, NULL));
    if (res.failed) {
        return res;
    }

    if (!ww_darray_ensure_total_capacity_precise(extension_properties, count)) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        return res;
    }

    res = VULKAN_CHECK(vkEnumerateDeviceExtensionProperties(physical_device, layer_name, &count, ww_darray_ptr(extension_properties)));
    if (res.failed) {
        return res;
    }

    ww_darray_resize_assume_capacity(extension_properties, count);
    return res;
}

VulkanResult vulkan_get_physical_device_queue_family_properties(VkPhysicalDevice physical_device, WwDArray(VkQueueFamilyProperties)* queue_family_properties) {
    assert(physical_device != VK_NULL_HANDLE);
    assert(queue_family_properties);

    u32 count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, NULL);

    if (!ww_darray_resize(queue_family_properties, count)) {
        return VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
    }

    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, ww_darray_ptr(queue_family_properties));
    return VULKAN_CHECK(VK_SUCCESS);
}

VulkanResult vulkan_physical_device_get_queue_family_indices(VkPhysicalDevice physical_device, VkSurfaceKHR surface, WwAllocator allocator, VulkanQueueFamilyIndices* queue_family_indices) {
    assert(physical_device != VK_NULL_HANDLE);
    assert(surface != VK_NULL_HANDLE);
    assert(queue_family_indices);

    *queue_family_indices = (VulkanQueueFamilyIndices){};
    WwDArray(VkQueueFamilyProperties) queue_family_properties = ww_darray_init(allocator, VkQueueFamilyProperties);
    VulkanResult res = vulkan_get_physical_device_queue_family_properties(physical_device, &queue_family_properties);
    if (res.failed) {
        goto failed;
    }

    u32 i = 0;
    ww_darray_foreach_by_ref(&queue_family_properties, VkQueueFamilyProperties, props) {
        b8 support_graphics = (props->queueFlags & VK_QUEUE_GRAPHICS_BIT) == VK_QUEUE_GRAPHICS_BIT;
        b8 support_compute = (props->queueFlags & VK_QUEUE_COMPUTE_BIT) == VK_QUEUE_COMPUTE_BIT;
        b8 support_transfer = (props->queueFlags & VK_QUEUE_TRANSFER_BIT) == VK_QUEUE_TRANSFER_BIT;

        if (!queue_family_indices->graphics.found && support_graphics) {
            queue_family_indices->graphics.index = i;
            queue_family_indices->graphics.found = true;
        }

        if (!queue_family_indices->compute.found && support_compute) {
            queue_family_indices->compute.index = i;
            queue_family_indices->compute.found = true;
        }

        if (!queue_family_indices->compute_no_graphics.found && support_compute && !support_graphics) {
            queue_family_indices->compute_no_graphics.index = i;
            queue_family_indices->compute_no_graphics.found = true;
        }

        if (surface != VK_NULL_HANDLE && !queue_family_indices->present.found) {
            VkBool32 present_support = false;
            res = VULKAN_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, surface, &present_support));
            if (res.failed) {
                goto failed;
            }

            if (present_support) {
                queue_family_indices->present.index = i;
                queue_family_indices->present.found = true;
            }
        }

        if (!queue_family_indices->transfer.found && support_transfer) {
            queue_family_indices->transfer.index = i;
            queue_family_indices->transfer.found = true;
        }

        if (!queue_family_indices->dma_transfer.found && support_transfer && !support_graphics && !support_compute) {
            queue_family_indices->dma_transfer.index = i;
            queue_family_indices->dma_transfer.found = true;
        }

        i++;
    }

failed:
    ww_darray_deinit(&queue_family_properties);
    return res;
}

VulkanResult vulkan_enumerate_instance_layer_properties(WwDArray(VkLayerProperties)* layer_properties) {
    assert(layer_properties);

    u32 count;
    VulkanResult res = VULKAN_CHECK(vkEnumerateInstanceLayerProperties(&count, NULL));
    if (res.failed) {
        return res;
    }

    if (!ww_darray_ensure_total_capacity_precise(layer_properties, count)) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        return res;
    }

    res = VULKAN_CHECK(vkEnumerateInstanceLayerProperties(&count, ww_darray_ptr(layer_properties)));
    if (res.failed) {
        return res;
    }
    

    ww_darray_resize_assume_capacity(layer_properties, count);
    return res;
}

VulkanResult vulkan_get_physical_device_surface_formats(VkPhysicalDevice physical_device, VkSurfaceKHR surface, WwDArray(VkSurfaceFormatKHR)* formats) {
    assert(physical_device != VK_NULL_HANDLE);
    assert(surface != VK_NULL_HANDLE);
    assert(formats);

    u32 count;
    VulkanResult res = VULKAN_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &count, NULL));
    if (res.failed) {
        return res;
    }

    if (!ww_darray_ensure_total_capacity_precise(formats, count)) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        return res;
    }

    res = VULKAN_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &count, ww_darray_ptr(formats)));
    if (res.failed) {
        return res;
    }
    
    ww_darray_resize_assume_capacity(formats, count);
    return res;
}

VulkanResult vulkan_get_physical_device_surface_present_modes(VkPhysicalDevice physical_device, VkSurfaceKHR surface, WwDArray(VkPresentModeKHR)* present_modes) {
    assert(physical_device != VK_NULL_HANDLE);
    assert(surface != VK_NULL_HANDLE);
    assert(present_modes);

    u32 count;
    VulkanResult res = VULKAN_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &count, NULL));
    if (res.failed) {
        return res;
    }

    if (!ww_darray_ensure_total_capacity_precise(present_modes, count)) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        return res;
    }

    res = VULKAN_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &count, ww_darray_ptr(present_modes)));
    if (res.failed) {
        return res;
    }

    ww_darray_resize_assume_capacity(present_modes, count);
    return res;
}

VulkanResult vulkan_get_swapchain_images(VkDevice device, VkSwapchainKHR swapchain, WwDArray(VkImage)* images) {
    assert(device != VK_NULL_HANDLE);
    assert(swapchain != VK_NULL_HANDLE);
    assert(images);

    u32 count;
    VulkanResult res = VULKAN_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &count, NULL));
    if (res.failed) {
        return res;
    }

    if (!ww_darray_ensure_total_capacity_precise(images, count)) {
        res = VULKAN_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
        return res;
    }

    res = VULKAN_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &count, ww_darray_ptr(images)));
    if (res.failed) {
        return res;
    }

    ww_darray_resize_assume_capacity(images, count);
    return res;
}

usize vulkan_get_pixel_size(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_SRGB: return 4 * sizeof(u8);
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 4 * sizeof(f32);
        default: break;
    }

    WW_EXIT_WITH_MSG("vulkan_get_pixel_size is not implemented for format: %d", format);
}

VulkanResult vulkan_create_instance(const VulkanInstanceCreateInfo* create_info, VkInstance* instance) {
    assert(create_info);
    assert(instance);

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "VulkanViewport app name",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "VulkanViewport engine name",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };
    VkInstanceCreateInfo instance_create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = create_info->enabled_extension_count,
        .ppEnabledExtensionNames = create_info->enabled_extensions,
        .enabledLayerCount = create_info->enabled_layer_count,
        .ppEnabledLayerNames = create_info->enabled_layers,
        .pNext = create_info->p_next,
    };
    return VULKAN_CHECK(vkCreateInstance(&instance_create_info, NULL, instance));
}

VulkanResult vulkan_pick_physical_device(VkInstance instance, u32 device_index, VkPhysicalDevice* physical_device) {
    assert(instance != VK_NULL_HANDLE);
    assert(physical_device);

    u32 count;
    VulkanResult res = VULKAN_CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
    if (res.failed) {
        return res;
    }

    if (device_index >= count) {
        WW_LOG_ERROR("VkPhysicalDevice index is out of range\n");
        res = VULKAN_CHECK(VK_ERROR_INITIALIZATION_FAILED);
        return res;
    }
    
    VkPhysicalDevice physical_devices[count];
    res = VULKAN_CHECK(vkEnumeratePhysicalDevices(instance, &count, physical_devices));
    if (res.failed) {
        return res;
    }

    *physical_device = physical_devices[device_index];
    return res; 
}

VulkanResult vulkan_create_device(VkPhysicalDevice physical_device, const VulkanDeviceCreateInfo* create_info, VkDevice* device) {
    assert(physical_device != VK_NULL_HANDLE);
    assert(create_info);
    assert(device);

    VkPhysicalDeviceFeatures device_features = {};
    VkDeviceCreateInfo device_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = create_info->queue_create_info_count,
        .pQueueCreateInfos = create_info->queue_create_infos,
        .pEnabledFeatures = &device_features,
        .enabledExtensionCount = create_info->enabled_extension_count,
        .ppEnabledExtensionNames = create_info->enabled_extensions,
        .enabledLayerCount = create_info->enabled_layer_count,
        .ppEnabledLayerNames = create_info->enabled_layers,
    };
    return VULKAN_CHECK(vkCreateDevice(physical_device, &device_create_info, NULL, device));
}

static VulkanResult vulkan_create_info_instance(VkInstance* instance) {
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "print device app name",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "print device engine name",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };

    const char* extensions[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };
    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = WW_ARRAY_SIZE(extensions),
        .ppEnabledExtensionNames = extensions, 
    };

    return VULKAN_CHECK(vkCreateInstance(&create_info, NULL, instance));
}

VulkanResult vulkan_print_devices_and_get_count(WwAllocator allocator, u32* device_count) {
    assert(device_count);
    VkInstance instance = NULL;
    WwDArray(VkPhysicalDevice) physical_devices = ww_darray_init(allocator, VkPhysicalDevice);

    VulkanResult res = vulkan_create_info_instance(&instance);
    if (res.failed) {
        goto failed;
    }

    res = vulkan_enumerate_physical_devices(instance, &physical_devices);
    if (res.failed) {
        goto failed; 
    }

    usize i = 0;
    ww_darray_foreach_by_ref(&physical_devices, VkPhysicalDevice, d) {
        VkPhysicalDeviceIDProperties id_properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &id_properties,
        };

        vkGetPhysicalDeviceProperties2(*d, &properties);

        WW_LOG_INFO("(VK) %zu. %s\n", i, properties.properties.deviceName);
        // fix printing luid and uuid
        // WW_LOG_INFO("    luid: %s\n", id_properties.deviceLUID);
        // WW_LOG_INFO("    uuid: %s\n", id_properties.deviceUUID);
        WW_LOG_INFO("    type: ");
        switch (properties.properties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_OTHER:
                WW_LOG_WARN("other");
                break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                WW_LOG_INFO("igpu");
                break;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                WW_LOG_INFO("dgpu");
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                WW_LOG_INFO("vgpu");
                break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                WW_LOG_INFO("cpu");
                break;
            default:
                WW_LOG_WARN("unknown vulkan device type"); 
                break;
        }
        WW_LOG_INFO("\n");
        i++;
    }

    *device_count = ww_darray_len(&physical_devices);
failed:
    ww_darray_deinit(&physical_devices);
    vkDestroyInstance(instance, NULL);
    return res; 
}

VulkanResult __ww_must_check vulkan_get_device_uuid(WwAllocator allocator, u32 device_id, VulkanUUID* result) {
    assert(result);
    VkInstance instance = NULL;
    WwDArray(VkPhysicalDevice) physical_devices = ww_darray_init(allocator, VkPhysicalDevice);

    VulkanResult res = vulkan_create_info_instance(&instance);
    if (res.failed) {
        goto failed;
    }

    res = vulkan_enumerate_physical_devices(instance, &physical_devices);
    if (res.failed) {
        goto failed; 
    }

    usize i = 0;
    VkPhysicalDevice d = ww_darray_get(&physical_devices, VkPhysicalDevice, device_id);
    VkPhysicalDeviceIDProperties id_properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 properties = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &id_properties,
    };

    vkGetPhysicalDeviceProperties2(d, &properties);
    WW_STATIC_ASSERT_EXPR(sizeof(result->bytes) == VK_UUID_SIZE, "");
    memcpy(result->bytes, id_properties.deviceUUID, VK_UUID_SIZE);

failed:
    ww_darray_deinit(&physical_devices);
    vkDestroyInstance(instance, NULL);
    return res; 
}