#pragma once

#include <ww/defines.h>
#include <ww/prim_types.h>
#include <hip/hip_runtime.h>
#include <ww/renderer/result.h>

b8 __ww_must_check hip_failed(const char * file, const i32 line, hipError_t err, const char* expression);
#define HIP_FAILED(err) hip_failed(__FILE__, __LINE__, err, #err)

RendererResult __ww_must_check hip_check(const char * file, const i32 line, hipError_t err, const char* expression);
#define HIP_CHECK(err) hip_check(__FILE__, __LINE__, err, #err)

b8 print_hip_devices();