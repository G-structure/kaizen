//
// Created by 69029 on 6/25/2021.
//

// Platform specific configuration
#if defined(__x86_64__) || defined(_M_X64)
    // x86 specific configuration
    #define USE_X86_INTRINSICS
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    // ARM specific configuration
    #define USE_ARM_ARCHITECTURE
#else
    // Generic fallback
    #define USE_GENERIC_IMPLEMENTATION
#endif

//#include <mcl/bls12_381.hpp>
#include "fieldElement.hpp"

//using namespace mcl::bn;
using namespace std;

#define F   virgo::fieldElement
#define F_ONE   virgo::fieldElement::one()
#define F_ZERO  virgo::fieldElement::zero()


//#define F Fr
//#define F_ONE (Fr(1))
//#define F_ZERO (Fr(0))



typedef unsigned __int128 u128;
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

typedef __int128 i128;
typedef long long i64;
typedef int i32;
typedef char i8;
