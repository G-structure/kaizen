#ifndef MY_HHASH_H
#define MY_HHASH_H

/**extra 'h' before hhash can avoid some strange error by the compiler*/

#include <string.h>
#include <assert.h>

// Only include C++ specific headers when compiling as C++
#ifdef __cplusplus
#include "config_pc.hpp"
#if defined(USE_X86_INTRINSICS)
    #include <immintrin.h>
#endif
#endif

#include "flo-shani.h"

// Define USESHA3 only for x86 architectures
#if !defined(USE_ARM_ARCHITECTURE) && defined(USE_X86_INTRINSICS)
#define USESHA3
#endif

#ifdef USESHA3
extern "C" {
#include "../lib/libXKCP.a.headers/SimpleFIPS202.h"
}
#endif

#ifdef __cplusplus
// C++ specific implementation
#ifdef USE_X86_INTRINSICS
// x86 specific implementation
class __hhash_digest
{
public:
    __m128i h0, h1;
};

inline bool equals(const __hhash_digest &a, const __hhash_digest &b)
{
    __m128i v0 = _mm_xor_si128(a.h0, b.h0);
    __m128i v1 = _mm_xor_si128(a.h1, b.h1);
    return _mm_test_all_zeros(v0, v0) && _mm_test_all_zeros(v1, v1);
}
#else
// Generic implementation for ARM and other architectures
class __hhash_digest
{
public:
    unsigned char bytes[32]; // SHA256 produces a 32-byte digest
};

inline bool equals(const __hhash_digest &a, const __hhash_digest &b)
{
    return memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0;
}
#endif
#else
// C-compatible implementation
typedef struct __hhash_digest {
    unsigned char bytes[32]; // SHA256 produces a 32-byte digest
} __hhash_digest;

static inline int equals(const __hhash_digest *a, const __hhash_digest *b)
{
    return memcmp(a->bytes, b->bytes, sizeof(a->bytes)) == 0;
}
#endif

#include <string.h>
inline void my_hhash(const void* src, void* dst)
{
#ifdef USESHA3
    SHA3_256((unsigned char*)dst, (const unsigned char*)src, 64);
#else
    //memset(dst, 0, sizeof(__hhash_digest));
    sha256_update_shani((const unsigned char*)src, 64, (unsigned char*)dst);
#endif
}

#endif
