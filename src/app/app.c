#include "app.h"
#include <ww/collections/darray.h>
#include <ww/math.h>
#include <ww/renderer.h>
#include <stdlib.h>
#include <time.h>
#include <vulkan/vulkan_core.h>
#include <ww/log.h>
#include <ww/defines.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <ww/hip/common.h>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/mesh.h>
#include <assimp/postprocess.h>

#define USE_HIPRT_RENDERER 1
#if USE_HIPRT_RENDERER
#include <ww/hiprt/renderer.h>
typedef HipRTCreationProperties HipCreationProperties;
#define hip_renderer_create hiprt_renderer_create
#else
#include <ww/hip/renderer.h>
#endif

#define USE_VK_VIEWPORT_NO_GP 1
#if USE_VK_VIEWPORT_NO_GP
#include <ww/vulkan/viewport_no_gp.h>
typedef VulkanViewportNoGPCreationProperties VulkanViewportCreationProperties;
#define vulkan_viewport_create vulkan_viewport_no_gp_create
#else
#include <ww/vulkan/viewport.h>
#endif

typedef struct App {
    WwAllocator allocator;
    b8 window_resized;
    GLFWwindow* window;
    Viewport viewport;
    Renderer renderer;
    Camera camera;
    Scene scene;
    struct {
        b8 w_pressed;
        b8 s_pressed;
        b8 a_pressed;
        b8 d_pressed;
    } keys;
    struct {
        vec3 look_from;
        vec3 look_at;
        vec3 up;
        f32 speed;
    } camera_params;
    WwDArray(TriangleMesh) triangle_meshes;
    WwDArray(ObjectInstance) object_instances;
} App;

static void framebuffer_resize_callback(GLFWwindow* window, int width, int height);
static VkResult __ww_must_check vulkan_create_surface(VkInstance instance, void* window, VkSurfaceKHR* surface);
static void app_init_window(App* self, u32 width, u32 height);
static b8 __ww_must_check app_init_viewport(App* self, VulkanViewportCreationProperties creation_properties);
static b8 __ww_must_check app_init_renderer(App* self, HipCreationProperties creation_properties);
static b8 __ww_must_check app_load_scene(App* self, const char* file);
static b8 __ww_must_check app_load_cornellplot(App* self);
static b8 __ww_must_check app_load_lucy(App* self);
static b8 __ww_must_check app_update(App* self);
static b8 __ww_must_check app_handle_resize(App* self);
static b8 __ww_must_check app_handle_keys(App* self, f32 delta_time_in_seconds);

AppResult app_create(AppCreationProperties creation_properties, App** app) {
    assert(app);

    if (!print_hip_devices()) {
        return APP_FAILED;
    }
    
    if (print_vulkan_devices(creation_properties.allocator).failed) {
        return APP_FAILED;
    }


    ww_auto_type alloc_result = ww_allocator_alloc_type(creation_properties.allocator, App);
    if (alloc_result.failed) {
        return APP_FAILED;
    }
    
    App* self = alloc_result.ptr;
    *self = (App){
        .allocator = creation_properties.allocator,
        .window_resized = true,
        .triangle_meshes = ww_darray_init(creation_properties.allocator, TriangleMesh),
        .object_instances = ww_darray_init(creation_properties.allocator, ObjectInstance),
    };

    app_init_window(self, creation_properties.width, creation_properties.height);

    u32 glfw_extension_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    VulkanViewportCreationProperties vulkan_viewport_creation_properties = {
        .allocator = creation_properties.allocator,
        .device_index = 0,
        .frames_in_flight = 2,
        .instance_extension_count = glfw_extension_count,
        .instance_extensions = glfw_extensions,
        .window = self->window,
        .vulkan_create_surface = vulkan_create_surface,
    };
    if (!app_init_viewport(self, vulkan_viewport_creation_properties)) {
        goto failed;
    }

    HipCreationProperties renderer_creation_properties = {
        .allocator = creation_properties.allocator,
        .device_index = vulkan_viewport_creation_properties.device_index,
    };
    if (!app_init_renderer(self, renderer_creation_properties)) {
        goto failed;
    }

    if (!app_load_lucy(self)) {
        goto failed;
    }

    *app = self;
    return APP_SUCCESS;

failed:
    app_destroy(self);
    return APP_FAILED;
}

void app_destroy(App* self) {
    assert(self);

    if (self->scene.ptr) {
        scene_destroy(self->scene);
    }

    if (self->camera.ptr) {
        camera_destroy(self->camera);
    }

    ww_darray_foreach_by_ref(&self->object_instances, ObjectInstance, oi)
        object_instance_destroy(*oi);
    ww_darray_foreach_by_ref(&self->triangle_meshes, TriangleMesh, tm)
        triangle_mesh_destroy(*tm);

    ww_darray_deinit(&self->object_instances);
    ww_darray_deinit(&self->triangle_meshes);

    if (self->renderer.ptr) {
        renderer_destroy(self->renderer);
    }

    if (self->viewport.ptr) {
        viewport_destroy(self->viewport);
    }

    glfwDestroyWindow(self->window);
    glfwTerminate();

    ww_allocator_free(self->allocator, self);
}

AppResult app_run(App* self) {
    assert(self);
    clock_t delta_time = 0;
    clock_t time = 0;
    clock_t begin_frame = clock();
    clock_t end_frame = clock();
    u32 frames = 0;
    const u32 render_iterations = 1;
    ViewportResult viewport_result = {};
    while (!glfwWindowShouldClose(self->window)) {
        delta_time = end_frame - begin_frame;
        time += delta_time;
        frames += render_iterations;
        f64 time_in_seconds = (time / (f64)CLOCKS_PER_SEC);
        if (time_in_seconds > 1.0) { // every second
            WW_LOG_INFO(
                "[App] Iterations per second = %f, Time per iteration = %fms\n",
                frames / time_in_seconds,
                time_in_seconds * 1000.0 / frames
            );
            frames = 0;
            time -= CLOCKS_PER_SEC;
        }
        begin_frame = clock();
        {
            glfwPollEvents();
            if (viewport_result.code == VIEWPORT_ERROR_OUT_OF_DATE || viewport_result.code == VIEWPORT_SUBOPTIMAL || self->window_resized) {
                self->window_resized = false;
                if (!app_handle_resize(self)) {
                    goto failed;
                }
            } else if (viewport_result.failed) {
                goto failed;
            }

            if (!app_handle_keys(self, (f32)(delta_time / (f64)CLOCKS_PER_SEC))) {
                goto failed;
            }

            if (renderer_render(self->renderer).failed) {
                goto failed;
            }

            if (renderer_copy_target_to(self->renderer, viewport_get_mapped_input(self->viewport)).failed) {
                goto failed;
            }

            viewport_result = viewport_render(self->viewport);
        }
        end_frame = clock();
    }

    if (viewport_wait_idle(self->viewport).failed) {
        goto failed;
    }

    return APP_SUCCESS;
failed:
    // ignore result
    ViewportResult wait_res = viewport_wait_idle(self->viewport);
    return APP_FAILED;
}

void app_init_window(App* self, u32 width, u32 height) {
    WW_RUN_ONCE glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
    self->window = glfwCreateWindow(width, height, "App name", NULL, NULL);
    glfwSetWindowUserPointer(self->window, self);
    glfwSetFramebufferSizeCallback(self->window, framebuffer_resize_callback);
}

b8 app_init_viewport(App* self, VulkanViewportCreationProperties creation_properties) {
    WW_LOG_INFO("Selected vk device: %d\n", creation_properties.device_index);
    if (vulkan_viewport_create(creation_properties, &self->viewport).failed) {
        return false;
    } 

    return true;
}

b8 app_init_renderer(App* self, HipCreationProperties creation_properties) {
    if (hip_renderer_create(creation_properties, &self->renderer).failed) {
        return false;
    }
    
    if (renderer_create_scene(self->renderer, &self->scene).failed) {
        return false;
    }

    if (renderer_set_scene(self->renderer, self->scene.ptr).failed) {
        return false;
    }

    if (renderer_create_camera(self->renderer, &self->camera).failed) {
        return false;
    }

    if (scene_set_camera(self->scene, self->camera.ptr).failed) {
        return false;
    }

    return true;
}

b8 app_load_scene(App* self, const char* file) {
    u32 ai_import_flags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType | aiProcess_GenSmoothNormals;
    const struct aiScene* ai_scene = aiImportFile(file, ai_import_flags);

    if (ai_scene == NULL) {
        WW_LOG_ERROR("Assimp failed to load %s\n", file);
        return false;
    }

    WwDArray(u32) indices = ww_darray_init(self->allocator, u32);
    usize max_triangle_count = 0;
    for (u32 i = 0; i < ai_scene->mNumMeshes; i++) {
        max_triangle_count = WW_MAX(max_triangle_count, ai_scene->mMeshes[i]->mNumFaces);
    }

    if (!ww_darray_ensure_total_capacity_precise(&indices, max_triangle_count * 3)) {
        goto failed_scene_import;
    }

    if (!ww_darray_ensure_total_capacity_precise(&self->triangle_meshes, ai_scene->mNumMeshes)) {
        goto failed_scene_import;
    }
    
    if (!ww_darray_ensure_total_capacity_precise(&self->object_instances, ai_scene->mNumMeshes)) {
        goto failed_scene_import;
    }

    for (u32 i = 0; i < ai_scene->mNumMeshes; i++) {
        const struct aiMesh* mesh = ai_scene->mMeshes[i];

        for (u32 j = 0; j < mesh->mNumFaces; j++) {
            ww_darray_append_assume_capacity(&indices, mesh->mFaces[j].mIndices[0]);
            assert(ai_import_flags & aiProcess_Triangulate);
            ww_darray_append_assume_capacity(&indices, mesh->mFaces[j].mIndices[1]);
            ww_darray_append_assume_capacity(&indices, mesh->mFaces[j].mIndices[2]);
        }

        WW_STATIC_ASSERT_EXPR(sizeof(mesh->mVertices[0]) == sizeof(vec3), "Check assimp vertices type");
        TriangleMeshCreationProperties triangle_mesh_creation_properties = {
            .triangle_count = mesh->mNumFaces,
            .triangle_indices = (u32*)ww_darray_ptr(&indices),
            .vertex_count = mesh->mNumVertices,
            .vertices = (vec3*)mesh->mVertices,
        };

        TriangleMesh triangle_mesh = {};
        if (renderer_create_triangle_mesh(self->renderer, triangle_mesh_creation_properties, &triangle_mesh).failed) {
            goto failed_scene_import;
        }

        ww_darray_append_assume_capacity(&self->triangle_meshes, triangle_mesh);
        ww_darray_resize_assume_capacity(&indices, 0);

        ObjectInstance object_instance = {};
        if (renderer_create_object_instance(self->renderer, triangle_mesh.ptr, &object_instance).failed) {
            goto failed_scene_import;
        }

        ww_darray_append_assume_capacity(&self->object_instances, object_instance);

        if (scene_attach_object_instance(self->scene, object_instance.ptr).failed) {
            goto failed_scene_import;
        }

        // if (object_instance_set_transform(object_instance, mat4_translate(0.0f, 3.0f, 0.0f)).failed) {
        //     goto failed_scene_import;
        // }
    }

    ww_darray_deinit(&indices);
    aiReleaseImport(ai_scene);
    return true;

failed_scene_import:
    ww_darray_deinit(&indices);
    aiReleaseImport(ai_scene);
    return false;
}

b8 app_load_cornellplot(App* self) {
    self->camera_params.speed = 20.0f;
    self->camera_params.look_from = make_vec3(0.0f, 2.5f, 20.0f);
    self->camera_params.look_at = make_vec3(0.0f, 2.5f, 0.0);
    self->camera_params.up = make_vec3(0.0f, 1.0f, 0.0f);
    if (camera_set_look_at(self->camera, self->camera_params.look_from, self->camera_params.look_at, self->camera_params.up).failed) {
        return false;
    }

    if (camera_set_focus_dist(self->camera, 10.0f).failed) {
        return false;
    }
    return app_load_scene(self, "meshes/cornellpot.obj");
}

b8 app_load_lucy(App* self) {
    self->camera_params.speed = 10000.0f;
    self->camera_params.look_from = make_vec3(0.0f, 1600.0f, 1500.0f);
    self->camera_params.look_at = make_vec3(0.0f, 450.0f, -300.0);
    self->camera_params.up = make_vec3(0.0f, 1.0f, 0.0f);
    if (camera_set_look_at(self->camera, self->camera_params.look_from, self->camera_params.look_at, self->camera_params.up).failed) {
        return false;
    }

    if (camera_set_focus_dist(self->camera, 10.0f).failed) {
        return false;
    }
    return app_load_scene(self, "meshes/lucy.obj");
}

b8 app_handle_resize(App* self) {
    i32 i32_width = 0;
    i32 i32_height = 0;
    glfwGetFramebufferSize(self->window, &i32_width, &i32_height);
    while (i32_width == 0 || i32_height == 0) {
        glfwGetFramebufferSize(self->window, &i32_width, &i32_height);
        glfwWaitEvents();
    }

    u32 width = (u32)i32_width;
    u32 height = (u32)i32_height;
    WW_LOG_DEBUG("[App] viewport resize (%d, %d)\n", width, height);
    if (viewport_set_resolution(self->viewport, width, height).failed) {
        return false;
    }

    viewport_get_resolution(self->viewport, &width, &height);
    WW_LOG_DEBUG("[App] renderer resize (%d, %d)\n", width, height);
    if (renderer_set_target_resolution(self->renderer, width, height).failed) {
        return false;
    }

    if (camera_set_aspect_ratio(self->camera, (f32)width / height).failed) {
        return false;
    }

    return true;
}

b8 app_handle_keys(App* self, f32 delta_time_in_seconds) {
    b8 w_pressed = glfwGetKey(self->window, GLFW_KEY_W) == GLFW_PRESS;
    b8 s_pressed = glfwGetKey(self->window, GLFW_KEY_S) == GLFW_PRESS;
    b8 a_pressed = glfwGetKey(self->window, GLFW_KEY_A) == GLFW_PRESS;
    b8 d_pressed = glfwGetKey(self->window, GLFW_KEY_D) == GLFW_PRESS;

    f32 camera_move_distance = self->camera_params.speed * delta_time_in_seconds;
    if (w_pressed || s_pressed || d_pressed || a_pressed) {
        vec3 target = self->camera_params.look_at;
        vec3 eye = self->camera_params.look_from;
        vec3 up = self->camera_params.up;
        vec3 forward = vec3_sub(target, eye);
        vec3 forward_norm = vec3_normalize(forward);
        f32 forward_mag = vec3_length(forward);

        if (w_pressed && forward_mag > camera_move_distance) {
            eye = vec3_add(eye, vec3_mul(forward_norm, camera_move_distance));
        }

        if (s_pressed) {
            eye = vec3_sub(eye, vec3_mul(forward_norm, camera_move_distance));
        }

        vec3 right = vec3_cross(forward_norm, up);
        // Redo radius calc in case the fowrard/backward is pressed.
        forward = vec3_sub(target, eye);
        forward_mag = vec3_length(forward);

        if (d_pressed) {
            // Rescale the distance between the target and eye so
            // that it doesn't change. The eye therefore still
            // lies on the circle made by the target and eye.
            eye = vec3_sub(target, vec3_mul(vec3_normalize(vec3_sub(forward, vec3_mul(right, camera_move_distance))), forward_mag));
        }

        if (a_pressed) {
            eye = vec3_sub(target, vec3_mul(vec3_normalize(vec3_add(forward, vec3_mul(right, camera_move_distance))), forward_mag));
        }

        self->camera_params.look_from = eye;
        self->camera_params.look_at = target;
        self->camera_params.up = up;
        if (camera_set_look_at(self->camera, eye, target, up).failed) {
            return false;
        }
    }

    return true;
}

void framebuffer_resize_callback(GLFWwindow* window, int width, int height) {
    App* self = glfwGetWindowUserPointer(window);
    self->window_resized = true;
}

VkResult vulkan_create_surface(VkInstance instance, void* window, VkSurfaceKHR* surface) {
    return glfwCreateWindowSurface(instance, (GLFWwindow*)window, NULL, surface);
}
