
#include <hip/hip_runtime.h>

extern "C" __global__ void ray_trace(float* pixels, int2 res)
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	if (index >= res.x * res.y) {
		return;
	}

	pixels[index * 4 + 0] = x / (float)res.x;
	pixels[index * 4 + 1] = y / (float)res.y;
	pixels[index * 4 + 2] = 0.0f;
	pixels[index * 4 + 3] = 1.0f;
}