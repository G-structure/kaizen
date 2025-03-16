/*
 * The MIT License (MIT)
 * Copyright (c) 2018 Armando Faz
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 * OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "flo-shani.h"
#include <string.h>

const ALIGN uint32_t CONST_K[64] = {
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

// SHA-256 helper functions
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

static void sha256_transform(uint32_t state[8], const uint8_t data[64]) {
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

    // Prepare message schedule
    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = ((uint32_t)data[j] << 24) | ((uint32_t)data[j + 1] << 16) |
               ((uint32_t)data[j + 2] << 8) | ((uint32_t)data[j + 3]);
    
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    // Initialize working variables
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    // Main loop
    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + CONST_K[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Update state
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

static void update_shani(uint32_t *state, const uint8_t *msg, uint32_t num_blocks) {
    while (num_blocks > 0) {
        sha256_transform(state, msg);
        msg += 64;
        num_blocks--;
    }
}

static void update_shani_2x(
    uint32_t *state0, const uint8_t *msg0,
    uint32_t *state1, const uint8_t *msg1,
    uint32_t num_blocks) {
    
    while (num_blocks > 0) {
        sha256_transform(state0, msg0);
        sha256_transform(state1, msg1);
        msg0 += 64;
        msg1 += 64;
        num_blocks--;
    }
}

static void update_shani_4x(
    uint32_t *state0, const uint8_t *msg0,
    uint32_t *state1, const uint8_t *msg1,
    uint32_t *state2, const uint8_t *msg2,
    uint32_t *state3, const uint8_t *msg3,
    uint32_t num_blocks) {
    
    while (num_blocks > 0) {
        sha256_transform(state0, msg0);
        sha256_transform(state1, msg1);
        sha256_transform(state2, msg2);
        sha256_transform(state3, msg3);
        msg0 += 64;
        msg1 += 64;
        msg2 += 64;
        msg3 += 64;
        num_blocks--;
    }
}

static void update_shani_8x(
    uint32_t *state0, const uint8_t *msg0,
    uint32_t *state1, const uint8_t *msg1,
    uint32_t *state2, const uint8_t *msg2,
    uint32_t *state3, const uint8_t *msg3,
    uint32_t *state4, const uint8_t *msg4,
    uint32_t *state5, const uint8_t *msg5,
    uint32_t *state6, const uint8_t *msg6,
    uint32_t *state7, const uint8_t *msg7,
    uint32_t num_blocks) {
    
    while (num_blocks > 0) {
        sha256_transform(state0, msg0);
        sha256_transform(state1, msg1);
        sha256_transform(state2, msg2);
        sha256_transform(state3, msg3);
        sha256_transform(state4, msg4);
        sha256_transform(state5, msg5);
        sha256_transform(state6, msg6);
        sha256_transform(state7, msg7);
        msg0 += 64;
        msg1 += 64;
        msg2 += 64;
        msg3 += 64;
        msg4 += 64;
        msg5 += 64;
        msg6 += 64;
        msg7 += 64;
        num_blocks--;
    }
}

unsigned char * sha256_update_shani(const unsigned char *message, long unsigned int message_length, unsigned char *digest) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    uint32_t num_blocks = message_length / 64;
    update_shani(state, message, num_blocks);
    
    // Copy the final state to the digest
    for (int i = 0; i < 8; i++) {
        digest[i*4] = (state[i] >> 24) & 0xFF;
        digest[i*4+1] = (state[i] >> 16) & 0xFF;
        digest[i*4+2] = (state[i] >> 8) & 0xFF;
        digest[i*4+3] = state[i] & 0xFF;
    }
    
    return digest;
}

void sha256_x2_update_shani_2x(
    unsigned char *message[2],
    long unsigned int message_length,
    unsigned char *digest[2]) {
    
    uint32_t state0[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    uint32_t state1[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    uint32_t num_blocks = message_length / 64;
    update_shani_2x(state0, message[0], state1, message[1], num_blocks);
    
    // Copy the final states to the digests
    for (int i = 0; i < 8; i++) {
        digest[0][i*4] = (state0[i] >> 24) & 0xFF;
        digest[0][i*4+1] = (state0[i] >> 16) & 0xFF;
        digest[0][i*4+2] = (state0[i] >> 8) & 0xFF;
        digest[0][i*4+3] = state0[i] & 0xFF;
        
        digest[1][i*4] = (state1[i] >> 24) & 0xFF;
        digest[1][i*4+1] = (state1[i] >> 16) & 0xFF;
        digest[1][i*4+2] = (state1[i] >> 8) & 0xFF;
        digest[1][i*4+3] = state1[i] & 0xFF;
    }
}

void sha256_x4_update_shani_4x(
    unsigned char *message[4],
    long unsigned int message_length,
    unsigned char *digest[4]) {
    
    uint32_t state0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t state1[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t state2[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t state3[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    
    uint32_t num_blocks = message_length / 64;
    update_shani_4x(state0, message[0], state1, message[1], state2, message[2], state3, message[3], num_blocks);
    
    // Copy the final states to the digests
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            uint32_t *state = (j == 0) ? state0 : (j == 1) ? state1 : (j == 2) ? state2 : state3;
            digest[j][i*4] = (state[i] >> 24) & 0xFF;
            digest[j][i*4+1] = (state[i] >> 16) & 0xFF;
            digest[j][i*4+2] = (state[i] >> 8) & 0xFF;
            digest[j][i*4+3] = state[i] & 0xFF;
        }
    }
}

void sha256_x8_update_shani_8x(
    unsigned char *message[8],
    long unsigned int message_length,
    unsigned char *digest[8]) {
    
    uint32_t state0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t state1[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t state2[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t state3[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t state4[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t state5[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t state6[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t state7[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    
    uint32_t num_blocks = message_length / 64;
    update_shani_8x(
        state0, message[0], state1, message[1], 
        state2, message[2], state3, message[3],
        state4, message[4], state5, message[5],
        state6, message[6], state7, message[7],
        num_blocks
    );
    
    // Copy the final states to the digests
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            uint32_t *state = NULL;
            switch (j) {
                case 0: state = state0; break;
                case 1: state = state1; break;
                case 2: state = state2; break;
                case 3: state = state3; break;
                case 4: state = state4; break;
                case 5: state = state5; break;
                case 6: state = state6; break;
                case 7: state = state7; break;
            }
            digest[j][i*4] = (state[i] >> 24) & 0xFF;
            digest[j][i*4+1] = (state[i] >> 16) & 0xFF;
            digest[j][i*4+2] = (state[i] >> 8) & 0xFF;
            digest[j][i*4+3] = state[i] & 0xFF;
        }
    }
}

// Implementations for the other functions
void sha256_vec_4256b(uint8_t *message[4], uint8_t *digest[4]) {
    sha256_x4_update_shani_4x(message, 256/8, digest);
}

void sha256_vec_8256b(uint8_t *message[8], uint8_t *digest[8]) {
    sha256_x8_update_shani_8x(message, 256/8, digest);
}

void sha256_4w(uint8_t *message[4], unsigned int message_length, uint8_t *digest[4]) {
    sha256_x4_update_shani_4x(message, message_length, digest);
}

void sha256_8w(uint8_t *message[8], unsigned int message_length, uint8_t *digest[8]) {
    sha256_x8_update_shani_8x(message, message_length, digest);
}

void sha256_16w(uint8_t *message[16], unsigned int message_length, uint8_t *digest[16]) {
    // For 16-wide, we just do two 8-wide operations
    sha256_x8_update_shani_8x(message, message_length, digest);
    sha256_x8_update_shani_8x(message + 8, message_length, digest + 8);
}
