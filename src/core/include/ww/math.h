#pragma once

#if (defined(__CUDACC__) || defined(__HIPCC__))
#define __KERNELCC__
#endif

#if !defined(__KERNELCC__)
#define HOST
#define DEVICE
#define HOST_DEVICE
#else
#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__
#endif

#if defined(__CUDACC__)
#define INLINE __forceinline__
#else
#define INLINE static inline
#endif

#include <math.h>
#include <ww/prim_types.h>

typedef union {
    struct {
        u32 x;
        u32 y;
    };
    u32 data[2];
} uvec2;

typedef union {
    struct {
        i32 x;
        i32 y;
    };
    i32 data[2];
} ivec2;

typedef union {
    struct {
        i32 x;
        i32 y;
        i32 z;
    };
    i32 data[3];
} ivec3;

typedef union {
    struct {
        i32 x;
        i32 y;
        i32 z;
        i32 w;
    };
    i32 data[4];
} ivec4;

typedef union {
    struct {
        f32 x;
        f32 y;
    };
    f32 data[2];
} vec2;

typedef union {
    struct {
        f32 x;
        f32 y;
        f32 z;
    };
    f32 data[3];
} vec3;

typedef union {
    struct {
        f32 x;
        f32 y;
        f32 z;
        f32 w;
    };
    f32 data[4];
} vec4;

HOST_DEVICE INLINE uvec2 make_uvec2(u32 x, u32 y) { return (uvec2){ x, y }; }
HOST_DEVICE INLINE ivec2 make_ivec2(i32 x, i32 y) { return (ivec2){ x, y }; }
HOST_DEVICE INLINE ivec3 make_ivec3(i32 x, i32 y, i32 z) { return (ivec3){ x, y, z }; }
HOST_DEVICE INLINE ivec4 make_ivec4(i32 x, i32 y, i32 z, i32 w) { return (ivec4){ x, y, z, w }; }

HOST_DEVICE INLINE vec2 make_vec2(f32 x, f32 y) { return (vec2){ x, y }; }
HOST_DEVICE INLINE vec3 make_vec3(f32 x, f32 y, f32 z) { return (vec3){ x, y, z }; }
HOST_DEVICE INLINE vec4 make_vec4(f32 x, f32 y, f32 z, f32 w) { return (vec4){ x, y, z, w }; }

typedef vec4 quat;

typedef struct mat4 {
    union {
        vec4 r[4];
        f32 e[4][4];
    };
} mat4;

#define VEC3_UNIT_X ((vec3){ 1.0f, 0.0f, 0.0f });
#define VEC3_UNIT_Y ((vec3){ 0.0f, 1.0f, 0.0f });
#define VEC3_UNIT_Z ((vec3){ 0.0f, 0.0f, 1.0f });
#define MAT4_IDENTITY ((mat4){ .e = { \
        { 1.0f, 0.0f, 0.0f, 0.0f },   \
        { 0.0f, 1.0f, 0.0f, 0.0f },   \
        { 0.0f, 0.0f, 1.0f, 0.0f },   \
        { 0.0f, 0.0f, 0.0f, 1.0f },   \
    } })                              \

HOST_DEVICE INLINE ivec2 ivec2_add(ivec2 a, ivec2 b) { return make_ivec2(a.x + b.x, a.y + b.y); }
HOST_DEVICE INLINE ivec2 ivec2_sub(ivec2 a, ivec2 b) { return make_ivec2(a.x - b.x, a.y - b.y); }
HOST_DEVICE INLINE ivec2 ivec2_mul(ivec2 a, i32 b) { return make_ivec2(a.x * b, a.y * b); }
HOST_DEVICE INLINE ivec2 ivec2_div(ivec2 a, i32 b) { return make_ivec2(a.x / b, a.y / b); }
HOST_DEVICE INLINE ivec2 ivec2_neg(ivec2 a) { return make_ivec2(-a.x, -a.y); }

HOST_DEVICE INLINE ivec3 ivec3_add(ivec3 a, ivec3 b) { return make_ivec3(a.x + b.x, a.y + b.y, a.z + b.z); }
HOST_DEVICE INLINE ivec3 ivec3_sub(ivec3 a, ivec3 b) { return make_ivec3(a.x - b.x, a.y - b.y, a.z - b.z); }
HOST_DEVICE INLINE ivec3 ivec3_mul(ivec3 a, i32 b) { return make_ivec3(a.x * b, a.y * b, a.z * b); }
HOST_DEVICE INLINE ivec3 ivec3_div(ivec3 a, i32 b) { return make_ivec3(a.x / b, a.y / b, a.z / b); }
HOST_DEVICE INLINE ivec3 ivec3_neg(ivec3 a) { return make_ivec3(-a.x, -a.y, -a.z); }

HOST_DEVICE INLINE ivec4 ivec4_add(ivec4 a, ivec4 b) { return make_ivec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
HOST_DEVICE INLINE ivec4 ivec4_sub(ivec4 a, ivec4 b) { return make_ivec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
HOST_DEVICE INLINE ivec4 ivec4_mul(ivec4 a, i32 b) { return make_ivec4(a.x * b, a.y * b, a.z * b, a.w * b); }
HOST_DEVICE INLINE ivec4 ivec4_div(ivec4 a, i32 b) { return make_ivec4(a.x / b, a.y / b, a.z / b, a.w / b); }
HOST_DEVICE INLINE ivec4 ivec4_neg(ivec4 a) { return make_ivec4(-a.x, -a.y, -a.z, -a.w); }

HOST_DEVICE INLINE vec2 vec2_add(vec2 a, vec2 b) { return make_vec2(a.x + b.x, a.y + b.y); }
HOST_DEVICE INLINE vec2 vec2_sub(vec2 a, vec2 b) { return make_vec2(a.x - b.x, a.y - b.y); }
HOST_DEVICE INLINE vec2 vec2_mul(vec2 a, f32 b) { return make_vec2(a.x * b, a.y * b); }
HOST_DEVICE INLINE vec2 vec2_div(vec2 a, f32 b) { return make_vec2(a.x / b, a.y / b); }
HOST_DEVICE INLINE vec2 vec2_neg(vec2 a) { return make_vec2(-a.x, -a.y); }

HOST_DEVICE INLINE vec3 vec3_add(vec3 a, vec3 b) { return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
HOST_DEVICE INLINE vec3 vec3_sub(vec3 a, vec3 b) { return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
HOST_DEVICE INLINE vec3 vec3_mul(vec3 a, f32 b) { return make_vec3(a.x * b, a.y * b, a.z * b); }
HOST_DEVICE INLINE vec3 vec3_div(vec3 a, f32 b) { return make_vec3(a.x / b, a.y / b, a.z / b); }
HOST_DEVICE INLINE vec3 vec3_neg(vec3 a) { return make_vec3(-a.x, -a.y, -a.z); }

HOST_DEVICE INLINE vec4 vec4_add(vec4 a, vec4 b) { return make_vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
HOST_DEVICE INLINE vec4 vec4_sub(vec4 a, vec4 b) { return make_vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
HOST_DEVICE INLINE vec4 vec4_mul(vec4 a, f32 b) { return make_vec4(a.x * b, a.y * b, a.z * b, a.w * b); }
HOST_DEVICE INLINE vec4 vec4_div(vec4 a, f32 b) { return make_vec4(a.x / b, a.y / b, a.z / b, a.w / b); }
HOST_DEVICE INLINE vec4 vec4_neg(vec4 a) { return make_vec4(-a.x, -a.y, -a.z, -a.w); }

HOST_DEVICE INLINE vec3 vec3_cross(vec3 a, vec3 b) { return make_vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
HOST_DEVICE INLINE f32 vec3_dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HOST_DEVICE INLINE f32 vec3_length_squared(vec3 a) { return vec3_dot(a, a); }
HOST_DEVICE INLINE f32 vec3_length(vec3 a) { return sqrtf(vec3_length_squared(a)); }
HOST_DEVICE INLINE vec3 vec3_normalize(vec3 a) { return vec3_div(a, vec3_length(a)); }

HOST_DEVICE INLINE f32 vec4_dot(vec4 a, vec4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
HOST_DEVICE INLINE f32 vec4_length_squared(vec4 a) { return vec4_dot(a, a); }
HOST_DEVICE INLINE f32 vec4_length(vec4 a) { return sqrtf(vec4_length_squared(a)); }
HOST_DEVICE INLINE vec4 vec4_normalize(vec4 a) { return vec4_div(a, vec4_length(a)); }

HOST_DEVICE INLINE vec4 mat4_mul_vec4(mat4 m, vec4 v)
{
    // ROW MAJOR
    return make_vec4(vec4_dot(m.r[0], v), vec4_dot(m.r[1], v), vec4_dot(m.r[2], v), vec4_dot(m.r[3], v));

    // ROW MAJOR
    // return make_vec4(
    // 	v.x * m.e[0][0] + v.y * m.e[0][1] + v.z * m.e[0][2] + v.w * m.e[0][3],
    // 	v.x * m.e[1][0] + v.y * m.e[1][1] + v.z * m.e[1][2] + v.w * m.e[1][3],
    // 	v.x * m.e[2][0] + v.y * m.e[2][1] + v.z * m.e[2][2] + v.w * m.e[2][3],
    // 	v.x * m.e[3][0] + v.y * m.e[3][1] + v.z * m.e[3][2] + v.w * m.e[3][3]
    // );

    // COLUMN MAJOR
    // return make_vec4(
    // 	v.x * m.e[0][0] + v.y * m.e[1][0] + v.z * m.e[2][0] + v.w * m.e[3][0],
    // 	v.x * m.e[0][1] + v.y * m.e[1][1] + v.z * m.e[2][1] + v.w * m.e[3][1],
    // 	v.x * m.e[0][2] + v.y * m.e[1][2] + v.z * m.e[2][2] + v.w * m.e[3][2],
    // 	v.x * m.e[0][3] + v.y * m.e[1][3] + v.z * m.e[2][3] + v.w * m.e[3][3]
    // );
}

HOST_DEVICE INLINE mat4 mat4_mul_mat4(mat4 a, mat4 b)
{
    mat4 m;
    for (i32 r = 0; r < 4; ++r) {
        for (i32 c = 0; c < 4; ++c) {
            m.e[r][c] = 0.0f;
            for (i32 k = 0; k < 4; ++k)
                m.e[r][c] += a.e[r][k] * b.e[k][c];
        }
    }

    return m;
}

HOST_DEVICE INLINE mat4 mat4_translate(f32 x, f32 y, f32 z)
{
    return (mat4){ .r = {
                 { 1.0f, 0.0f, 0.0f, x },
                 { 0.0f, 1.0f, 0.0f, y },
                 { 0.0f, 0.0f, 1.0f, z },
                 { 0.0f, 0.0f, 0.0f, 1.0f },
             } };
}

HOST_DEVICE INLINE mat4 mat4_scale(f32 x, f32 y, f32 z)
{
    return (mat4){ .e = {
                 { x, 0.0f, 0.0f, 0.0f },
                 { 0.0f, y, 0.0f, 0.0f },
                 { 0.0f, 0.0f, z, 0.0f },
                 { 0.0f, 0.0f, 0.0f, 1.0f },
             } };
}

HOST_DEVICE INLINE mat4 mat4_rotate(vec3 axis, f32 angle)
{
    f32 x = axis.x, y = axis.y, z = axis.z;
    f32 sa = sinf(angle); 
    f32 ca = cosf(angle);
    f32 xx = x * x, yy = y * y, zz = z * z;
    f32 xy = x * y, xz = x * z, yz = y * z;

    return (mat4){ .e = {
        { xx + ca * (1.0f - xx), xy - ca * xy - sa * z, xz - ca * xz + sa * y, 0.0f },
        { xy - ca * xy + sa * z, yy + ca * (1.0f - yy), yz - ca * yz - sa * x, 0.0f },
        { xz - ca * xz - sa * y, yz - ca * yz + sa * x, zz + ca * (1.0f - zz), 0.0f },
        { 0.0f, 0.0f, 0.0f, 1.0f },
    } };
}

HOST_DEVICE INLINE mat4 mat4_transpose(mat4 a)
{
    mat4 ret;
    ret.r[0] = make_vec4(a.e[0][0], a.e[1][0], a.e[2][0], a.e[3][0]);
    ret.r[1] = make_vec4(a.e[0][1], a.e[1][1], a.e[2][1], a.e[3][1]);
    ret.r[2] = make_vec4(a.e[0][2], a.e[1][2], a.e[2][2], a.e[3][2]);
    ret.r[3] = make_vec4(a.e[0][3], a.e[1][3], a.e[2][3], a.e[3][3]);
    return ret;
}

HOST_DEVICE INLINE mat4 mat4_inverse(mat4 matrix)
{
    f32 a = matrix.e[0][0], b = matrix.e[0][1], c = matrix.e[0][2], d = matrix.e[0][3];
    f32 e = matrix.e[1][0], f = matrix.e[1][1], g = matrix.e[1][2], h = matrix.e[1][3];
    f32 i = matrix.e[2][0], j = matrix.e[2][1], k = matrix.e[2][2], l = matrix.e[2][3];
    f32 m = matrix.e[3][0], n = matrix.e[3][1], o = matrix.e[3][2], p = matrix.e[3][3];

    f32 kp_lo = k * p - l * o;
    f32 jp_ln = j * p - l * n;
    f32 jo_kn = j * o - k * n;
    f32 ip_lm = i * p - l * m;
    f32 io_km = i * o - k * m;
    f32 in_jm = i * n - j * m;

    f32 a11 = +(f * kp_lo - g * jp_ln + h * jo_kn);
    f32 a12 = -(e * kp_lo - g * ip_lm + h * io_km);
    f32 a13 = +(e * jp_ln - f * ip_lm + h * in_jm);
    f32 a14 = -(e * jo_kn - f * io_km + g * in_jm);

    f32 det = a * a11 + b * a12 + c * a13 + d * a14;
    f32 invDet = 1.0f / det;

    mat4 result;
    result.e[0][0] = a11 * invDet;
    result.e[1][0] = a12 * invDet;
    result.e[2][0] = a13 * invDet;
    result.e[3][0] = a14 * invDet;

    result.e[0][1] = -(b * kp_lo - c * jp_ln + d * jo_kn) * invDet;
    result.e[1][1] = +(a * kp_lo - c * ip_lm + d * io_km) * invDet;
    result.e[2][1] = -(a * jp_ln - b * ip_lm + d * in_jm) * invDet;
    result.e[3][1] = +(a * jo_kn - b * io_km + c * in_jm) * invDet;

    f32 gp_ho = g * p - h * o;
    f32 fp_hn = f * p - h * n;
    f32 fo_gn = f * o - g * n;
    f32 ep_hm = e * p - h * m;
    f32 eo_gm = e * o - g * m;
    f32 en_fm = e * n - f * m;

    result.e[0][2] = +(b * gp_ho - c * fp_hn + d * fo_gn) * invDet;
    result.e[1][2] = -(a * gp_ho - c * ep_hm + d * eo_gm) * invDet;
    result.e[2][2] = +(a * fp_hn - b * ep_hm + d * en_fm) * invDet;
    result.e[3][2] = -(a * fo_gn - b * eo_gm + c * en_fm) * invDet;

    f32 gl_hk = g * l - h * k;
    f32 fl_hj = f * l - h * j;
    f32 fk_gj = f * k - g * j;
    f32 el_hi = e * l - h * i;
    f32 ek_gi = e * k - g * i;
    f32 ej_fi = e * j - f * i;

    result.e[0][3] = -(b * gl_hk - c * fl_hj + d * fk_gj) * invDet;
    result.e[1][3] = +(a * gl_hk - c * el_hi + d * ek_gi) * invDet;
    result.e[2][3] = -(a * fl_hj - b * el_hi + d * ej_fi) * invDet;
    result.e[3][3] = +(a * fk_gj - b * ek_gi + c * ej_fi) * invDet;
    
    return result;
}

HOST_DEVICE INLINE mat4 mat4_from_quat(quat q)
{
    mat4 result = MAT4_IDENTITY;
    f32 qxx = q.x * q.x;
    f32 qyy = q.y * q.y;
    f32 qzz = q.z * q.z;
    f32 qxz = q.x * q.z;
    f32 qxy = q.x * q.y;
    f32 qyz = q.y * q.z;
    f32 qwx = q.w * q.x;
    f32 qwy = q.w * q.y;
    f32 qwz = q.w * q.z;
    
    result.e[0][0] = 1.0f - 2.0f * (qyy +  qzz);
    result.e[1][0] = 2.0f * (qxy + qwz);
    result.e[2][0] = 2.0f * (qxz - qwy);

    result.e[0][1] = 2.0f * (qxy - qwz);
    result.e[1][1] = 1.0f - 2.0f * (qxx +  qzz);
    result.e[2][1] = 2.0f * (qyz + qwx);

    result.e[0][2] = 2.0f * (qxz + qwy);
    result.e[1][2] = 2.0f * (qyz - qwx);
    result.e[2][2] = 1.0f - 2.0f * (qxx +  qyy);
    return result;
}

HOST_DEVICE INLINE vec3 mat4_transform_point(mat4 transform, vec3 p)
{
    vec4 result = mat4_mul_vec4(transform, make_vec4(p.x, p.y, p.z, 1.0f));
    return (vec3){ result.x, result.y, result.z };
}

HOST_DEVICE INLINE vec3 mat4_transform_vector(mat4 transform, vec3 v)
{
    vec4 result = mat4_mul_vec4(transform, make_vec4(v.x, v.y, v.z, 0.0f));
    return (vec3){ result.x, result.y, result.z };
}

HOST_DEVICE INLINE vec3 mat4_transform_normal(mat4 transform, vec3 n)
{
    mat4 inv_transform = mat4_inverse(transform);
    vec4 result = mat4_mul_vec4(mat4_transpose(inv_transform), make_vec4(n.x, n.y, n.z, 0.0f));
    return (vec3){ result.x, result.y, result.z };
}

HOST_DEVICE INLINE vec3 mat4_transform_normal_with_inv_transform(mat4 inv_transform, vec3 n)
{
    vec4 result = mat4_mul_vec4(mat4_transpose(inv_transform), make_vec4(n.x, n.y, n.z, 0.0f));
    return (vec3){ result.x, result.y, result.z };
}
