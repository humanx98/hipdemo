#pragma

#include <stddef.h>
#include <stdbool.h>
#include <limits.h>

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef signed char i8;
typedef signed short i16;
typedef signed int i32;
typedef signed long long i64;
typedef size_t usize;
typedef float f32;
typedef double f64;
typedef _Bool b8;

#define WW_U32_MIN (u32)0
#define WW_U32_MAX UINT_MAX
#define WW_U64_MIN (u32)0
#define WW_U64_MAX UINT64_MAX

#define _MIN(a, b) ((a) < b ? (a) : (b))
#define _MAX(a, b) ((a) > b ? (a) : (b))

#define WW_MAX(a, b) \
    ({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b);    \
    _MAX(_a, _b); })
#define WW_MIN(a, b) \
    ({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b);    \
    _MIN(_a, _b); })

#define WW_CLAMP(value, a, b)               \
    ({ __typeof__ (value) _a = (a);         \
    __typeof__ (value) _b = (b);            \
    __typeof__ (value) _value = (value);\
    _MIN(_MAX(_value, _a), _b); })
