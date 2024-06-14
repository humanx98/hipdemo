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

extern "C" __global__ void SceneIntersectionKernel(hiprtScene scene, float* pixels, int2 res)
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	float3 o = { x / static_cast<float>( res.x ) - 0.5f, y / static_cast<float>( res.y ) - 0.5f, -1.0f };
	float3 d = { 0.0f, 0.0f, 1.0f };

	hiprtRay ray;
	ray.origin	  = o;
	ray.direction = d;

	hiprtSceneTraversalClosest tr( scene, ray, 0xffffffff );
	hiprtHit				   hit = tr.getNextHit();

	pixels[index * 4 + 0] = hit.hasHit() ? ( static_cast<float>( x ) / res.x ) : 0.0f;
	pixels[index * 4 + 1] = hit.hasHit() ? ( static_cast<float>( y ) / res.y ) : 0.0f;
	pixels[index * 4 + 2] = 0.0f;
	pixels[index * 4 + 3] = 1.0f;
}