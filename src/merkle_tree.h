#ifndef VIRGO_MERKLE_TREE_H
#define VIRGO_MERKLE_TREE_H

#include <vector>
#include "config_pc.hpp"
#if defined(USE_X86_INTRINSICS)
    #include <immintrin.h>
#endif
#include "mimc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(USE_X86_INTRINSICS)
    #include <wmmintrin.h>
#endif
#include "my_hhash.h"
#include "fieldElement.hpp"

namespace merkle_tree
{
extern int size_after_padding;

void get_int32(vector<unsigned int> &arr, __hhash_digest hash);

__hhash_digest hash_single_field_element(virgo::fieldElement x);

__hhash_digest hash_double_field_element_merkle_damgard(virgo::fieldElement x, virgo::fieldElement y, virgo::fieldElement z, virgo::fieldElement w, __hhash_digest prev_hash);

namespace merkle_tree_prover
{
    //Merkle tree functions used by the prover
    //void create_tree(void* data_source_array, int lengh, void* &target_tree_array, const int single_element_size = 256/8)
    void create_tree(int ele_num, vector<vector<__hhash_digest>> &dst_ptr, const int element_size, bool alloc_required);
    void create_tree_mimc( int ele_num, vector<vector<F>> &dst_ptr, int level, const int element_size, bool alloc_required );
    void create_tree_sha( int ele_num, vector<vector<__hhash_digest>> &dst_ptr, int level, const int element_size, bool alloc_required);
}

namespace merkle_tree_verifier
{
    //Merkle tree functions used by the verifier
    bool verify_claim(__hhash_digest root_hhash, const __hhash_digest* tree, __hhash_digest hhash_element, int pos_element, int N, bool *visited, long long &proof_size);
}
}

#endif