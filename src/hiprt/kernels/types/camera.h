#pragma once

#if (defined(__CUDACC__) || defined(__HIPCC__))
#define __KERNELCC__
#endif

#ifdef __KERNELCC__
#include <hiprt/hiprt_types.h>
#else
extern "C" {
#include <ww/math.h>
}
#define hiprtFloat3 vec3
#endif

namespace device {

typedef struct Camera {
    hiprtFloat3 origin;
    hiprtFloat3 lower_left_corner;
    hiprtFloat3 horizontal;
    hiprtFloat3 vertical;
    hiprtFloat3 u;
    hiprtFloat3 v;
    hiprtFloat3 w;
    float lens_radius;
    float vfov;
    float focus_dist;
    float aspect_ratio;
    hiprtFloat3 look_from;
    hiprtFloat3 look_at;
    hiprtFloat3 vup;
} Camera;

}