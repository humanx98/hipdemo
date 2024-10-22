#include <hip/hip_runtime.h>
#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_types.h>

HIPRT_DEVICE bool intersectFunc(
    uint32_t					geomType,
    uint32_t					rayType,
    const hiprtFuncTableHeader& tableHeader,
    const hiprtRay&				ray,
    void*						payload,
    hiprtHit&					hit )
{
    const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
    const void*	   data	 = tableHeader.funcDataSets[index].intersectFuncData;
    switch ( index )
    {
    default: 
        break;
    }

    return false;
}

HIPRT_DEVICE bool filterFunc(
    uint32_t					geomType,
    uint32_t					rayType,
    const hiprtFuncTableHeader& tableHeader,
    const hiprtRay&				ray,
    void*						payload,
    const hiprtHit&				hit )
{
    const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
    const void*	   data	 = tableHeader.funcDataSets[index].filterFuncData;
    switch ( index )
    {

    default:
        break;
    }
    
    return false;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HIPRT_HOST_DEVICE HIPRT_INLINE float3 normalize(const float3& a) { return a / sqrtf(dot(a, a)); }

extern "C" __global__ void SceneIntersectionKernel(hiprtScene scene, float* pixels, int2 res)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = x + y * res.x;

    if (index >= res.x * res.y) {
        return;
    }

    float3 o = { x / static_cast<float>( res.x ) - 0.5f, y / static_cast<float>( res.y ) - 0.5f, -1.0f };
    float3 d = { 0.0f, 0.0f, 1.0f };

    hiprtRay ray;
    ray.origin = o;
    ray.direction = d;

    hiprtSceneTraversalClosest tr( scene, ray, 0xffffffff );
    hiprtHit hit = tr.getNextHit();

    // pixels[index * 4 + 0] = hit.hasHit() ? ( static_cast<float>( x ) / res.x ) : 0.0f;
    // pixels[index * 4 + 1] = hit.hasHit() ? ( static_cast<float>( y ) / res.y ) : 0.0f;
    // pixels[index * 4 + 2] = 0.0f;
    // pixels[index * 4 + 3] = 1.0f;
    
    float3 color = { 0.0f, 0.0f, 0.0f };
    if (hit.hasHit()) {
        float3 n = normalize( hit.normal );
        color.x	 = ((n.x + 1.0f) * 0.5f);
        color.y	 = ((n.y + 1.0f) * 0.5f);
        color.z	 = ((n.z + 1.0f) * 0.5f);
    }

    pixels[index * 4 + 0] = color.x;
    pixels[index * 4 + 1] = color.y;
    pixels[index * 4 + 2] = color.z;
    pixels[index * 4 + 3] = 1.0f;
}