#include <stdio.h>
#include "src/config_pc.hpp"
#include "src/fieldElement.hpp"

int main() {
    printf("Testing Kaizen on ARM architecture\n");
    
#if defined(USE_X86_INTRINSICS)
    printf("Using x86 intrinsics\n");
#elif defined(USE_ARM_ARCHITECTURE)
    printf("Using ARM architecture\n");
#else
    printf("Using generic implementation\n");
#endif

    // Initialize the field element system
    virgo::fieldElement::init();
    
    // Test basic field element operations
    virgo::fieldElement a(1234);
    virgo::fieldElement b(5678);
    
    virgo::fieldElement c = a + b;
    virgo::fieldElement d = a * b;
    
    printf("a = %llu\n", a.real);
    printf("b = %llu\n", b.real);
    printf("a + b = %llu\n", c.real);
    printf("a * b = %llu\n", d.real);
    
    return 0;
} 