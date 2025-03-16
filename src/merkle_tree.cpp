#include "fieldElement.hpp"
#include "config_pc.hpp"
#include "merkle_tree.h"
#include <vector>
#include "mimc.h"
#include <cstring>
#include <cassert>
#include <iostream>

using namespace std;

namespace merkle_tree
{
int size_after_padding;

void get_int32(vector<unsigned int> &arr, __hhash_digest hash)
{
#ifdef USE_X86_INTRINSICS
    unsigned int* res_array = reinterpret_cast<unsigned int*>(&hash.h0);
    for (int i = 0; i < 4; i++)
        arr.push_back(res_array[i]);
    
    res_array = reinterpret_cast<unsigned int*>(&hash.h1);
    for (int i = 0; i < 4; i++)
        arr.push_back(res_array[i]);
#else
    // For ARM and other architectures, access the bytes array directly
    for (int i = 0; i < 8; i++) {
        unsigned int val = 0;
        for (int j = 0; j < 4; j++) {
            val = (val << 8) | hash.bytes[i*4 + j];
        }
        arr.push_back(val);
    }
#endif
}

__hhash_digest hash_single_field_element(virgo::fieldElement x)
{
    __hhash_digest data[2];
#ifdef USE_X86_INTRINSICS
    memcpy(&data[0].h0, &x, sizeof(data[0].h0));
    memset(&data[0].h1, 0, sizeof(data[0].h1));
    memset(&data[1], 0, sizeof(data[1]));
#else
    // For ARM and other architectures, access the bytes array directly
    memcpy(data[0].bytes, &x, sizeof(virgo::fieldElement));
    memset(data[0].bytes + sizeof(virgo::fieldElement), 0, sizeof(data[0].bytes) - sizeof(virgo::fieldElement));
    memset(data[1].bytes, 0, sizeof(data[1].bytes));
#endif
    
    __hhash_digest res;
    my_hhash(data, &res);
    return res;
}

__hhash_digest hash_double_field_element_merkle_damgard(virgo::fieldElement x, virgo::fieldElement y, virgo::fieldElement z, virgo::fieldElement w, __hhash_digest prev_hash)
{
    __hhash_digest data[2];
#ifdef USE_X86_INTRINSICS
    memcpy(&data[0].h0, &x, sizeof(x));
    memcpy(&data[0].h1, &y, sizeof(y));
    memcpy(&data[1].h0, &z, sizeof(z));
    memcpy(&data[1].h1, &w, sizeof(w));
#else
    // For ARM and other architectures, access the bytes array directly
    memcpy(data[0].bytes, &x, sizeof(x));
    memcpy(data[0].bytes + sizeof(x), &y, sizeof(y));
    memcpy(data[1].bytes, &z, sizeof(z));
    memcpy(data[1].bytes + sizeof(z), &w, sizeof(w));
#endif
    
    __hhash_digest res;
    my_hhash(data, &res);
    
    // XOR with previous hash
#ifdef USE_X86_INTRINSICS
    res.h0 = _mm_xor_si128(res.h0, prev_hash.h0);
    res.h1 = _mm_xor_si128(res.h1, prev_hash.h1);
#else
    // For ARM and other architectures, XOR byte by byte
    for (int i = 0; i < 32; i++) {
        res.bytes[i] ^= prev_hash.bytes[i];
    }
#endif
    
    return res;
}

void merkle_tree::merkle_tree_prover::create_tree_mimc(int ele_num, vector<vector<F>> &dst_ptr, int level, const int element_size = 256 / 8, bool alloc_required = false){
    int current_lvl_size = ele_num;
    //#pragma omp parallel for
    vector<F> aux(4);
    int curr_level = 1;
    current_lvl_size /= 2;
    while(current_lvl_size >= 1)
    {
        //printf("%d,%d\n",dst_ptr[curr_level].size(),current_lvl_size);
        //#pragma omp parallel for
        for(int i = 0; i < current_lvl_size; ++i)
        {
            //printf("%d,%d\n",start_ptr + current_lvl_size + i * 2,start_ptr + current_lvl_size + i * 2+1);
            F data[2];
            data[0] = dst_ptr[curr_level-1][i * 2];
            data[1] = dst_ptr[curr_level-1][i * 2 + 1];
            dst_ptr[curr_level-1][i] = my_mimc_hash(data[0],data[1],aux);
        }
        curr_level++;
        current_lvl_size /= 2;
    }
}


void merkle_tree::merkle_tree_prover::create_tree_sha(int ele_num, vector<vector<__hhash_digest>> &dst_ptr, int level, const int element_size = 256 / 8, bool alloc_required = false){
    int current_lvl_size = ele_num;
    //#pragma omp parallel for
   
    int curr_level = 1;
    current_lvl_size /= 2;
    while(curr_level != level)
    {
        //printf("%d,%d,%d\n",current_lvl_size,dst_ptr[curr_level].size(),curr_level);
        //#pragma omp parallel for
        for(int i = 0; i < current_lvl_size; ++i)
        {
            //printf("%d,%d\n",start_ptr + current_lvl_size + i * 2,start_ptr + current_lvl_size + i * 2+1);
            __hhash_digest data[2];
            data[0] = dst_ptr[curr_level-1][i * 2];
            data[1] = dst_ptr[curr_level-1][i * 2 + 1];
            my_hhash(data, &dst_ptr[curr_level][i]);
        }
        curr_level++;
        current_lvl_size /= 2;
    }
}


void merkle_tree::merkle_tree_prover::create_tree(int ele_num, vector<vector<__hhash_digest>> &dst_ptr, const int element_size = 256 / 8, bool alloc_required = false)
{
    //assert(element_size == sizeof(prime_field::field_element) * 2);
    
    int start_ptr = ele_num;
    int current_lvl_size = ele_num;
    
    int level = 1;
    current_lvl_size /= 2;
    start_ptr -= current_lvl_size;
    
    while(current_lvl_size >= 1)
    {
        
        //#pragma omp parallel for
        for(int i = 0; i < current_lvl_size; ++i)
        {
            //printf("%d,%d\n",start_ptr + current_lvl_size + i * 2,start_ptr + current_lvl_size + i * 2+1);
            __hhash_digest data[2];
            data[0] = dst_ptr[level-1][ i * 2];
            data[1] = dst_ptr[level-1][ i * 2 + 1];
            my_hhash(data, &dst_ptr[level][i]);
        }
        current_lvl_size /= 2;
        start_ptr -= current_lvl_size;
        level++;
    }
    printf("OK\n");
    //dst = dst_ptr.data();
}

bool merkle_tree::merkle_tree_verifier::verify_claim(__hhash_digest root_hhash, const __hhash_digest* tree, __hhash_digest leaf_hash, int pos_element_arr, int N, bool *visited, long long &proof_size)
{
    //check N is power of 2
    assert((N & (-N)) == N);

    int pos_element = pos_element_arr + N;
    __hhash_digest data[2];
    while(pos_element != 1)
    {
        data[pos_element & 1] = leaf_hash;
        data[(pos_element & 1) ^ 1] = tree[pos_element ^ 1];
        if(!visited[pos_element ^ 1])
        {
            visited[pos_element ^ 1] = true;
            proof_size += sizeof(__hhash_digest);
        }
        my_hhash(data, &leaf_hash);
        pos_element /= 2;
    }
    return equals(root_hhash, leaf_hash);
}
}