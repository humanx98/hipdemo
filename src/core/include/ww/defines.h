#pragma once

#include <assert.h>
#include <ww/prim_types.h>

#define WW_NAME2_HELPER(a, b)  a ## b
#define WW_NAME2(a, b) WW_NAME2_HELPER(a, b)
#define WW_UNIQUE_NAME(prefix) WW_NAME2(WW_NAME2(WW_NAME2(WW_NAME2(prefix, _), _), _), __LINE__)

#ifdef NDEBUG
#define WW_ASSERT_RUN_ONCE()  (void)0
#else
#define WW_ASSERT_RUN_ONCE()                                     \
    do {                                                      \
        static b8 WW_UNIQUE_NAME(asser_run_once_called) = false; \
        assert(!WW_UNIQUE_NAME(asser_run_once_called));          \
        WW_UNIQUE_NAME(asser_run_once_called) = true;            \
    } while (0)
#endif

#define WW_RUN_ONCE                          \
    static b8 WW_UNIQUE_NAME(run_once_called) = false; \
    if (WW_UNIQUE_NAME(run_once_called) || !(WW_UNIQUE_NAME(run_once_called) = true)) ; else

#define WW_IS_ARRAY(a)           \
    _Generic( &(a),           \
        typeof(*a) (*)[]: 1,  \
        default         : 0   \
    )

#define WW_STATIC_ASSERT_EXPR(expr, msg) (!!sizeof( struct { static_assert( (expr), msg ); char c; } ))

#define WW_ARRAY_SIZE(a) (       \
    sizeof(a) / sizeof(0[a])  \
    * WW_STATIC_ASSERT_EXPR( WW_IS_ARRAY(a), #a " must be an array" ))

#define WW_ARRAY_FOREACH(a, var) for ( typeof(0[a])* var = (a); var < (a) + WW_ARRAY_SIZE(a); ++var )

#define WW_DEFINE_HANDLE(object) typedef struct object##_impl* object

#define __ww_must_check __attribute__((warn_unused_result))

#ifdef __cplusplus
#define ww_auto_type auto
#else
#define ww_auto_type __auto_type
#endif
