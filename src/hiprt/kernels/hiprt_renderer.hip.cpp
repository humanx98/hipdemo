#include <hip/hip_runtime.h>
#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_types.h>
#include "types/camera.h"

HIPRT_DEVICE bool intersectFunc(uint32_t geomType, uint32_t rayType, const hiprtFuncTableHeader& tableHeader, const hiprtRay& ray, void* payload, hiprtHit& hit) {
    const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
    const void* data = tableHeader.funcDataSets[index].intersectFuncData;
    switch (index) {
    default: 
        break;
    }

    return false;
}

HIPRT_DEVICE bool filterFunc(uint32_t geomType, uint32_t rayType, const hiprtFuncTableHeader& tableHeader, const hiprtRay& ray, void* payload, const hiprtHit& hit) {
    const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
    const void* data = tableHeader.funcDataSets[index].filterFuncData;
    switch (index) {

    default:
        break;
    }
    
    return false;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3(const float4& a) { return make_float3(a.x, a.y, a.z); }
HIPRT_HOST_DEVICE HIPRT_INLINE float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HIPRT_HOST_DEVICE HIPRT_INLINE float3 cross(const float3& a, const float3& b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
HIPRT_HOST_DEVICE HIPRT_INLINE float3 normalize(const float3& a) { return a / sqrtf(dot(a, a)); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 rotate(const float4& rotation, const float3& p) {
    float3 a = sinf(rotation.w / 2.0f) * normalize(make_float3(rotation));
    float  c = cosf(rotation.w / 2.0f);
    return 2.0f * dot(a, p) * a + (c * c - dot(a, a)) * p + 2.0f * c * cross(a, p);
}


HIPRT_HOST_DEVICE HIPRT_INLINE hiprtRay generateRay(float x, float y, int2 res) {
    float fov = 45.0f * hiprt::Pi / 180.f;
    float4 rotation = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
    float3 translation = make_float3(0.0f, 2.5f, 5.8f);

    const float offset = 0.5f;
    const float2 sensorSize = make_float2(0.024f * (res.x / static_cast<float>(res.y)), 0.024f);
    const float2 xy = make_float2((x + offset) / res.x, (y + offset) / res.y) - make_float2(0.5f, 0.5f);
    const float3 dir = make_float3(xy.x * sensorSize.x, xy.y * sensorSize.y, sensorSize.y / (2.0f * tan(fov / 2.0f)));

    const float3 holDir = rotate(rotation, make_float3(1.0f, 0.0f, 0.0f));
    const float3 upDir = rotate(rotation, make_float3(0.0f, -1.0f, 0.0f));
    const float3 viewDir = rotate(rotation, make_float3(0.0f, 0.0f, -1.0f));

    hiprtRay ray;
    ray.origin = translation;
    ray.direction = normalize(dir.x * holDir + dir.y * upDir + dir.z * viewDir);
    return ray;
}

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtRay generateRay(const device::Camera& camera, float s, float t) {
    hiprtRay ray;
    ray.origin = camera.origin;
    ray.direction = camera.lower_left_corner+ s * camera.horizontal + t * camera.vertical - camera.origin;
    return ray;
}

extern "C" __global__ void SceneIntersectionKernel(hiprtScene scene, device::Camera camera, float* pixels, int2 res, bool flip_y) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= res.x * res.y) {
        return;
    }

    const uint32_t y = index / res.x;
    const uint32_t x = index % res.x;

    hiprtRay ray = generateRay(camera, (float)x / (res.x - 1), (float)y / (res.y - 1));
    hiprtSceneTraversalClosest tr(scene, ray);
    hiprtHit hit = tr.getNextHit();

    float3 color = { 0.0f, 0.0f, 0.0f };
    if (hit.hasHit()) {
        float3 n = normalize(hit.normal);
        color.x	 = ((n.x + 1.0f) * 0.5f);
        color.y	 = ((n.y + 1.0f) * 0.5f);
        color.z	 = ((n.z + 1.0f) * 0.5f);
    }

    if (flip_y) {
        uint32_t flipped_y = res.y - y - 1;
        index = res.x * flipped_y + x; 
    }
    pixels[index * 4 + 0] = color.x;
    pixels[index * 4 + 1] = color.y;
    pixels[index * 4 + 2] = color.z;
    pixels[index * 4 + 3] = 1.0f;
}