#pragma once

#if (defined(__CUDACC__) || defined(__HIPCC__))
#define __KERNELCC__
#endif

#ifdef __KERNELCC__
#include <hiprt/hiprt_types.h>
#define vec3 hiprtFloat3
#else
extern "C" {
#include <ww/math.h>
}
#endif

namespace device {

typedef struct Camera {
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u;
    vec3 v;
    vec3 w;
    float lens_radius;
    float vfov;
    float focus_dist;
    float aspect_ratio;
    vec3 look_from;
    vec3 look_at;
    vec3 vup;
} Camera;

}