/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : guqiong96
 * @LastEditTime : 2025-08-12 07:43:41
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "moe.h"
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <pthread.h>
#include <vector>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif
#include <mutex>
static std::mutex print_mutex;

static void bind_callback_thread_once() {
    static thread_local bool bound = false;
    if (bound) {
        return;
    }
    const char* callback_cpu_env = std::getenv("LK_CALLBACK_CPU");
    if (callback_cpu_env == nullptr) {
        return;
    }
    char* end = nullptr;
    long callback_cpu = std::strtol(callback_cpu_env, &end, 10);
    if (end == callback_cpu_env || callback_cpu < 0) {
        return;
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(static_cast<int>(callback_cpu), &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) ==
        0) {
        bound = true;
    }
}

static void enable_decode_task_thread_steal_once() {
    static thread_local bool enabled = false;
    if (enabled) {
        return;
    }
    const char* env = std::getenv("LK_DECODE_TASK_STEAL");
    if (env != nullptr && std::string(env) == "0") {
        return;
    }
    Backend_NUMA::thread_local_id_ = 0;
#ifdef USE_NUMA
    Backend_NUMA::numa_node_ = 0;
#endif
    enabled = true;
}

static inline bool is_valid_expert_id(uint64_t expert_id, int expert_num) {
    return expert_id < static_cast<uint64_t>(expert_num);
}


MOE::MOE(MOEConfig config) {

    config_ = config;

    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    struct ggml_init_params pdata = {
            .mem_size   = 128LL * 1024 * 1024,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };

    ggml_init(pdata);

    if (config_.stride <= 0 || config_.stride % 32 != 0 ||
        config_.hidden_size % config_.stride != 0 ||
        config_.intermediate_size % config_.stride != 0) {
        std::cerr << "Invalid MOE stride " << config_.stride
                  << ", falling back to 32" << std::endl;
        config_.stride = 32;
    }
    const char* profile_env = std::getenv("LVLLM_LK_PROFILE");
    profile_enabled_ = profile_env != nullptr && std::string(profile_env) == "1";
    const char* profile_detailed_env =
        std::getenv("LVLLM_LK_PROFILE_DETAILED");
    profile_detailed_enabled_ =
        profile_detailed_env != nullptr &&
        std::string(profile_detailed_env) == "1";
    const char* fuse_decode_down_sum_env =
        std::getenv("LK_FUSE_DECODE_DOWN_SUM");
    fuse_decode_down_sum_ =
        fuse_decode_down_sum_env != nullptr &&
        std::string(fuse_decode_down_sum_env) == "1";
    if (fuse_decode_down_sum_) {
        static std::once_flag log_fuse_decode_down_sum_once;
        std::call_once(log_fuse_decode_down_sum_once, []() {
            std::cerr << "LK_MOE: enabled decode down+sum fusion"
                      << std::endl;
        });
    }
    use_fp32_buffer_ = false;
    #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    std::cout << "AMX enabled ...... " << std::endl;
    if(config_.gate_type == GGML_TYPE_F16){
        use_fp32_buffer_ = true;
        if(config_.up_type != GGML_TYPE_F16 || config_.down_type != GGML_TYPE_F16){
            throw std::runtime_error("GGML_TYPE_F16 not same with gate,up,down");
        }
    }
    #endif
    if(config_.up_type == GGML_TYPE_BF16 && config_.hidden_type == GGML_TYPE_BF16){
        #if defined(__AMX_INT8__) || defined(__AVX512VNNI__) || defined(__AVX512BF16__) || defined(__AVX512F__)

            std::cout << "amx is enabled for bf16 ...... " << std::endl;
        #else
            use_fp32_buffer_ = true;
        #endif
    }


    hidden_type_size = ggml_type_size(config_.hidden_type);
    hidden_blk_size = ggml_blck_size(config_.hidden_type);

    gate_vec_dot_type = ggml_internal_get_type_traits(config_.gate_type).vec_dot_type;
    gate_vec_dot_type_size = ggml_type_size(gate_vec_dot_type);
    gate_vec_dot_blk_size = ggml_blck_size(gate_vec_dot_type);
    gate_type_size = ggml_type_size(config_.gate_type);
    gate_blk_size = ggml_blck_size(config_.gate_type);

    up_vec_dot_type = ggml_internal_get_type_traits(config_.up_type).vec_dot_type;
    up_vec_dot_type_size = ggml_type_size(up_vec_dot_type);
    up_vec_dot_blk_size = ggml_blck_size(up_vec_dot_type);
    up_type_size = ggml_type_size(config_.up_type);
    up_blk_size = ggml_blck_size(config_.up_type);

    down_vec_dot_type = ggml_internal_get_type_traits(config_.down_type).vec_dot_type;
    down_vec_dot_type_size = ggml_type_size(down_vec_dot_type);
    down_vec_dot_blk_size = ggml_blck_size(down_vec_dot_type);
    down_type_size = ggml_type_size(config_.down_type);
    down_blk_size = ggml_blck_size(config_.down_type);

    // std::cout << "gate_vec_dot_type: " << gate_vec_dot_type << std::endl;
    // std::cout << "up_vec_dot_type: " << up_vec_dot_type << std::endl;
    // std::cout << "down_vec_dot_type: " << down_vec_dot_type << std::endl;

    // std::cout << "config_.stride : " << config_.stride << " down_blk_size :" << down_blk_size << " hidden_blk_size :" << hidden_blk_size << std::endl;
    // std::cout << "config_.hidden_type : " << ggml_internal_get_type_traits(config_.hidden_type).type_name << std::endl;
    // std::cout << "config_.gate_type : " << ggml_internal_get_type_traits(config_.gate_type).type_name << std::endl;
    // std::cout << "config_.up_type : " << ggml_internal_get_type_traits(config_.up_type).type_name << std::endl;
    // std::cout << "config_.down_type : " << ggml_internal_get_type_traits(config_.down_type).type_name << std::endl;
    // std::cout << "gate_type_size: " << gate_type_size << std::endl;
    // std::cout << "gate_blk_size: " << gate_blk_size << std::endl;
    // std::cout << "buff_gate_bytes: " << config_.hidden_size * gate_vec_dot_type_size / gate_vec_dot_blk_size << std::endl;
    // std::cout << "up_type_size: " << up_type_size << std::endl;
    // std::cout << "up_blk_size: " << up_blk_size << std::endl;
    // std::cout << "buff_up_bytes: " << config_.hidden_size * up_vec_dot_type_size / up_vec_dot_blk_size << std::endl;
    // std::cout << "down_type_size: " << down_type_size << std::endl;
    // std::cout << "down_blk_size: " << down_blk_size << std::endl;
    // std::cout << "buff_down_bytes: " << config_.intermediate_size * down_vec_dot_type_size / down_vec_dot_blk_size << std::endl;
    // std::cout << "config_.hidden_size: " << config_.hidden_size << std::endl;
    // std::cout << "config_.intermediate_size: " << config_.intermediate_size << std::endl;

    gate_numa_.resize(numa_nodes_);
    up_numa_.resize(numa_nodes_);
    down_numa_.resize(numa_nodes_);

    gate_numa_size_.resize(numa_nodes_);
    up_numa_size_.resize(numa_nodes_);
    down_numa_size_.resize(numa_nodes_);

    int nth = config_.intermediate_size / config_.stride;
    stride_gate_bytes_ = config_.stride * config_.hidden_size * gate_type_size / gate_blk_size;
    stride_up_bytes_ = config_.stride * config_.hidden_size * up_type_size / up_blk_size;

    #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    amx_stride_gate_bytes_ = get_amx_packed_size(config_.gate_type, config_.hidden_size, config_.stride);
    amx_stride_up_bytes_ = get_amx_packed_size(config_.up_type, config_.hidden_size, config_.stride);
    #endif
    int base = nth / numa_nodes_;
    int remain = nth % numa_nodes_;

    gate_up_blocks_.resize(numa_nodes_);
    int current_block = 0;
    for (int nid = 0; nid < numa_nodes_; nid++) {
        int n_blocks = (base + (nid < remain));
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        gate_numa_size_[nid] = config_.expert_num * n_blocks * amx_stride_gate_bytes_;
        up_numa_size_[nid] = config_.expert_num * n_blocks * amx_stride_up_bytes_;
        #else
        gate_numa_size_[nid] = config_.expert_num * n_blocks * stride_gate_bytes_;
        up_numa_size_[nid] = config_.expert_num * n_blocks * stride_up_bytes_;
        #endif
        gate_up_blocks_[nid] = NumaBlock{
            .node_id = nid,
            .start_block = current_block,
            .num_blocks = n_blocks
        };
        current_block += n_blocks;
    }
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, numa_nodes_, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;
        int start_block = gate_up_blocks_[nid].start_block;
        int num_blocks = gate_up_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;
        assert(nid == task_id);
        gate_numa_[nid] = allocate_aligned_numa(gate_numa_size_[nid], nid);
        up_numa_[nid] = allocate_aligned_numa(up_numa_size_[nid], nid);
    }, nullptr);

     Backend_NUMA::getInstance().do_k_work_stealing_job(config_.expert_num, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;
        int start_block = gate_up_blocks_[nid].start_block;
        int num_blocks = gate_up_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;

        int x = task_id - start_block * config_.expert_num;
        int expert_id = x / num_blocks;

        int offset = x % num_blocks;
        int ith = start_block + offset;

        void* gate_ptr = (uint8_t*)gate_proj_ + (expert_id * nth + ith) * stride_gate_bytes_;
        void* up_ptr = (uint8_t*)up_proj_ +  (expert_id * nth + ith) * stride_up_bytes_;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        uint8_t* local_gate_ptr = (uint8_t*)gate_numa_[nid] + (expert_id * num_blocks + offset) * amx_stride_gate_bytes_;
        uint8_t* local_up_ptr = (uint8_t*)up_numa_[nid] + (expert_id * num_blocks + offset) * amx_stride_up_bytes_;
        convert_weight_to_amx_format(
            local_gate_ptr,
            gate_ptr,
            config_.gate_type,
            config_.hidden_size,
            config_.stride
        );
        convert_weight_to_amx_format(
            local_up_ptr,
            up_ptr,
            config_.up_type,
            config_.hidden_size,
            config_.stride
        );
#else
        void* local_gate_ptr = (uint8_t*)gate_numa_[nid] + (expert_id * num_blocks + offset) * stride_gate_bytes_;
        void* local_up_ptr = (uint8_t*)up_numa_[nid] + (expert_id * num_blocks + offset) * stride_up_bytes_;
        memcpy(local_gate_ptr, gate_ptr, stride_gate_bytes_);
        memcpy(local_up_ptr, up_ptr, stride_up_bytes_);
#endif
    }, nullptr);

    nth = config_.hidden_size / config_.stride;
    stride_down_bytes_ = config_.stride * config_.intermediate_size * down_type_size / down_blk_size;

    #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    amx_stride_down_bytes_ = get_amx_packed_size(config_.down_type, config_.intermediate_size, config_.stride);
    #endif

    base = nth / numa_nodes_;
    remain = nth % numa_nodes_;
    down_blocks_.resize(numa_nodes_);
    current_block = 0;
    for (int nid = 0; nid < numa_nodes_; nid++) {
        int n_blocks = (base + (nid < remain));
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        down_numa_size_[nid] = config_.expert_num * n_blocks * amx_stride_down_bytes_;
        #else
        down_numa_size_[nid] = config_.expert_num * n_blocks * stride_down_bytes_;
        #endif
        down_blocks_[nid] = NumaBlock{
            .node_id = nid,
            .start_block = current_block,
            .num_blocks = n_blocks
        };

        current_block += n_blocks;
    }
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, numa_nodes_, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;
        int start_block = down_blocks_[nid].start_block;
        int num_blocks = down_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;
        assert(nid == task_id);
        down_numa_[nid] = allocate_aligned_numa(down_numa_size_[nid], nid);
    }, nullptr);



    Backend_NUMA::getInstance().do_k_work_stealing_job(config_.expert_num, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;
        int start_block = down_blocks_[nid].start_block;
        int num_blocks = down_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;

        int x = task_id - start_block * config_.expert_num;
        int expert_id = x / num_blocks;

        int offset = x % num_blocks;
        int ith = start_block + offset;


        void* down_ptr = (uint8_t*)down_proj_ + (expert_id * nth + ith) * stride_down_bytes_;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        uint8_t* local_down_ptr = (uint8_t*)down_numa_[nid] + (expert_id * num_blocks + offset) * amx_stride_down_bytes_;

        convert_weight_to_amx_format(
            local_down_ptr,
            down_ptr,
            config_.down_type,
            config_.intermediate_size,
            config_.stride
        );
#else
        void* local_down_ptr = (uint8_t*)down_numa_[nid] + (expert_id * num_blocks + offset) * stride_down_bytes_;
        memcpy(local_down_ptr, down_ptr, stride_down_bytes_);
#endif
    }, nullptr);

    s_input_fp32_ = (float*)allocate_aligned(sizeof(float) * config_.hidden_size);
    s_gate_output_ = (float*)allocate_aligned(config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    s_up_output_ = (float*)allocate_aligned(config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    s_down_output_ = (float*)allocate_aligned(config_.routed_expert_num * sizeof(float) * config_.hidden_size);
    if(!use_fp32_buffer_){
        s_gate_input_ = (uint8_t*)allocate_aligned(config_.hidden_size * gate_vec_dot_type_size / gate_vec_dot_blk_size);
        s_up_input_ = (uint8_t*)allocate_aligned(config_.hidden_size * up_vec_dot_type_size / up_vec_dot_blk_size);
        s_down_input_ = (uint8_t*)allocate_aligned(config_.routed_expert_num * config_.intermediate_size * down_vec_dot_type_size / down_vec_dot_blk_size);
    }


    input_fp32_ = (float*)allocate_aligned(config_.group_max_len * sizeof(float) * config_.hidden_size);
    gate_output_ = (float*)allocate_aligned(config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    up_output_ = (float*)allocate_aligned(config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    down_output_ = (float*)allocate_aligned(config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.hidden_size);
    output_fp32_ = (float*)allocate_aligned(config_.group_max_len * sizeof(float) * config_.hidden_size);
    if(!use_fp32_buffer_){
        gate_input_ = (uint8_t*)allocate_aligned(config_.group_max_len * config_.hidden_size * gate_vec_dot_type_size / gate_vec_dot_blk_size);
        up_input_ = (uint8_t*)allocate_aligned(config_.group_max_len * config_.hidden_size * up_vec_dot_type_size / up_vec_dot_blk_size);
        down_input_ = (uint8_t*)allocate_aligned(config_.group_max_len * config_.routed_expert_num * config_.intermediate_size * down_vec_dot_type_size / down_vec_dot_blk_size);
        m_gate_input_ = (uint8_t*)allocate_aligned( config_.group_max_len * config_.routed_expert_num * config_.hidden_size * gate_vec_dot_type_size / gate_vec_dot_blk_size);
        m_up_input_ = (uint8_t*)allocate_aligned( config_.group_max_len * config_.routed_expert_num * config_.hidden_size * up_vec_dot_type_size / up_vec_dot_blk_size);
    }else{
        m_gate_input_ = (float*)allocate_aligned( config_.group_max_len * config_.routed_expert_num * sizeof(float) *  config_.hidden_size);
    }

    forward_one_impl = &MOE::forward_one;
    forward_many_impl = &MOE::forward_many_m;
    decode_task_thread_ = std::thread(&MOE::cpu_decode_task_loop, this);

    config_.gate_proj = nullptr;
    config_.up_proj = nullptr;
    config_.down_proj = nullptr;


}

MOE::~MOE() {
    {
        std::lock_guard<std::mutex> lock(decode_task_mutex_);
        decode_task_thread_exit_ = true;
    }
    decode_task_cv_.notify_one();
    if (decode_task_thread_.joinable()) {
        decode_task_thread_.join();
    }
    for (int nid = 0; nid < numa_nodes_; nid++) {
        free_aligned_numa(gate_numa_[nid], gate_numa_size_[nid]);
        free_aligned_numa(up_numa_[nid], up_numa_size_[nid]);
        free_aligned_numa(down_numa_[nid], down_numa_size_[nid]);
    }
    free_aligned(s_input_fp32_, sizeof(float) * config_.hidden_size);
    free_aligned(s_gate_output_, config_.group_max_len * sizeof(float) * config_.intermediate_size);
    free_aligned(s_up_output_, config_.group_max_len * sizeof(float) * config_.intermediate_size);
    free_aligned(s_down_output_, config_.group_max_len * sizeof(float) * config_.hidden_size);
    if(!use_fp32_buffer_){
        free_aligned(s_gate_input_, config_.hidden_size * gate_vec_dot_type_size / gate_vec_dot_blk_size);
        free_aligned(s_up_input_, config_.hidden_size * up_vec_dot_type_size / up_vec_dot_blk_size);
        free_aligned(s_down_input_, config_.routed_expert_num * config_.intermediate_size * down_vec_dot_type_size / down_vec_dot_blk_size);
    }

    free_aligned(input_fp32_, config_.group_max_len * sizeof(float) * config_.hidden_size);
    free_aligned(gate_output_, config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    free_aligned(up_output_, config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    free_aligned(down_output_, config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.hidden_size);
    free_aligned(output_fp32_, config_.group_max_len * sizeof(float) * config_.hidden_size);
    if(!use_fp32_buffer_){
        free_aligned(gate_input_ , config_.group_max_len * config_.hidden_size * gate_vec_dot_type_size / gate_vec_dot_blk_size);
        free_aligned(up_input_ , config_.group_max_len * config_.hidden_size * up_vec_dot_type_size / up_vec_dot_blk_size);
        free_aligned(down_input_ , config_.group_max_len * config_.routed_expert_num * config_.intermediate_size * down_vec_dot_type_size / down_vec_dot_blk_size);
        free_aligned(m_gate_input_, config_.group_max_len * config_.routed_expert_num * config_.hidden_size * gate_vec_dot_type_size / gate_vec_dot_blk_size);
        free_aligned(m_up_input_ , config_.group_max_len * config_.routed_expert_num * config_.hidden_size * up_vec_dot_type_size / up_vec_dot_blk_size);
    }else{
        free_aligned(m_gate_input_, config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.hidden_size);
    }
    if (decode_expert_ids_host_ != nullptr) {
        cudaFreeHost(decode_expert_ids_host_);
    }
    if (decode_expert_ids_i32_host_ != nullptr) {
        cudaFreeHost(decode_expert_ids_i32_host_);
    }
    if (decode_weights_host_ != nullptr) {
        cudaFreeHost(decode_weights_host_);
    }
    if (decode_input_host_ != nullptr) {
        cudaFreeHost(decode_input_host_);
    }
    if (decode_output_host_ != nullptr) {
        cudaFreeHost(decode_output_host_);
    }
    if (decode_bsz_host_ != nullptr) {
        cudaFreeHost(decode_bsz_host_);
    }
    for (CpuDecodeParams* params : decode_params_) {
        if (params->expert_ids_host != nullptr) {
            cudaFreeHost(params->expert_ids_host);
        }
        if (params->expert_ids_i32_host != nullptr) {
            cudaFreeHost(params->expert_ids_i32_host);
        }
        if (params->weights_host != nullptr) {
            cudaFreeHost(params->weights_host);
        }
        if (params->input_host != nullptr) {
            cudaFreeHost(params->input_host);
        }
        if (params->output_host != nullptr) {
            cudaFreeHost(params->output_host);
        }
        if (params->bsz_host != nullptr) {
            cudaFreeHost(params->bsz_host);
        }
        delete params;
    }
}

void MOE::warm_up() {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * hidden_type_size / hidden_blk_size);
    std::vector<uint8_t> output(config_.hidden_size * hidden_type_size / hidden_blk_size);
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
    for (int i = 0; i < config_.expert_num; i++) {
        uint64_t expert_ids = i;
        float weights = 0;
        forward_one(1, &expert_ids, &weights, input.data(), output.data());
    }
}



static void act_fn(float* up, float* gate, int n, float swiglu_limit) {

#if defined(__AVX2__)
    constexpr int VEC_SIZE = 8;
    const __m256 v_log2e = _mm256_set1_ps(1.44269504089f);
    const __m256 v_ln2 = _mm256_set1_ps(0.69314718056f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_neg_inf = _mm256_set1_ps(-128.0f);
    const __m256 v_pos_inf = _mm256_set1_ps(127.0f);
    for (int i = 0; i < n; i += VEC_SIZE) {
        __m256 v_gate = _mm256_load_ps(gate + i);
        __m256 v_up = _mm256_load_ps(up + i);

        if (swiglu_limit > 0.0f) {
            const __m256 v_limit = _mm256_set1_ps(swiglu_limit);
            const __m256 v_neg_limit = _mm256_set1_ps(-swiglu_limit);
            v_gate = _mm256_min_ps(v_gate, v_limit);
            v_up = _mm256_min_ps(_mm256_max_ps(v_up, v_neg_limit), v_limit);
        }

        __m256 v_x = _mm256_mul_ps(v_gate, v_log2e);

        v_x = _mm256_max_ps(_mm256_min_ps(v_x, v_pos_inf), v_neg_inf);

        __m256i v_k = _mm256_cvtps_epi32(v_x);
        __m256 v_k_f = _mm256_cvtepi32_ps(v_k);
        __m256 v_r = _mm256_sub_ps(v_x, v_k_f);

        __m256i v_k_bias = _mm256_add_epi32(v_k, _mm256_set1_epi32(127));
        __m256i v_k_bits = _mm256_slli_epi32(v_k_bias, 23);
        __m256 v_two_k = _mm256_castsi256_ps(v_k_bits);

        __m256 v_t = _mm256_mul_ps(v_r, v_ln2);
        __m256 v_t2 = _mm256_mul_ps(v_t, v_t);
        __m256 v_t3 = _mm256_mul_ps(v_t2, v_t);
        __m256 v_t4 = _mm256_mul_ps(v_t3, v_t);
        __m256 v_two_r = _mm256_add_ps(v_one,
            _mm256_fmadd_ps(v_t, v_one,
                _mm256_fmadd_ps(v_t2, _mm256_set1_ps(1.0f/2.0f),
                    _mm256_fmadd_ps(v_t3, _mm256_set1_ps(1.0f/6.0f),
                        _mm256_mul_ps(v_t4, _mm256_set1_ps(1.0f/24.0f))))));

        __m256 v_two_x = _mm256_mul_ps(v_two_k, v_two_r);

        __m256 v_denom = _mm256_add_ps(v_one, v_two_x);
        __m256 v_sigmoid = _mm256_div_ps(v_two_x, v_denom);

        __m256 v_swish = _mm256_mul_ps(v_gate, v_sigmoid);
        __m256 v_out = _mm256_mul_ps(v_up, v_swish);

        _mm256_store_ps(up + i, v_out);
    }
#else
    for (int i = 0; i < n; ++i) {
        float gate_value = gate[i];
        float up_value = up[i];
        if (swiglu_limit > 0.0f) {
            gate_value = std::min(gate_value, swiglu_limit);
            up_value = std::min(std::max(up_value, -swiglu_limit), swiglu_limit);
        }
        up[i] = up_value * (gate_value / (1.0f + expf(-gate_value)));
    }
#endif
}

void MOE::forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output) {
    const auto profile_start = profile_enabled_
                                   ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};

    const void* gate_input_ptr;
    const void* up_input_ptr;
    size_t gate_input_em = config_.hidden_size / gate_blk_size;
    size_t up_input_em = config_.hidden_size / up_blk_size;
    size_t down_input_em = config_.intermediate_size / down_blk_size;

    if(use_fp32_buffer_){
        to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
        gate_input_ptr = up_input_ptr = s_input_fp32_;
        gate_input_em = up_input_em = config_.hidden_size / hidden_blk_size;
        down_input_em = config_.intermediate_size / hidden_blk_size;
    }else{
        if (config_.hidden_type == gate_vec_dot_type && config_.hidden_type == up_vec_dot_type) {
            gate_input_ptr = up_input_ptr = input;
        } else {
            to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
            if (gate_vec_dot_type == up_vec_dot_type) {
                from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, gate_vec_dot_type);
                gate_input_ptr = up_input_ptr = s_gate_input_;
            } else {
                if (config_.hidden_type != gate_vec_dot_type) {
                    from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, gate_vec_dot_type);
                    gate_input_ptr = s_gate_input_;
                } else {
                    gate_input_ptr = input;
                }
                if (config_.hidden_type != up_vec_dot_type) {
                    from_float(s_input_fp32_, s_up_input_, config_.hidden_size, up_vec_dot_type);
                    up_input_ptr = s_up_input_;
                } else {
                    up_input_ptr = input;
                }
            }
        }
    }

    size_t nth = config_.intermediate_size / config_.stride;
    const auto gate_up_start = profile_enabled_
                                   ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
    Backend_NUMA::getInstance().do_k_work_stealing_job(k, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;
        int start_block = gate_up_blocks_[nid].start_block;
        int num_blocks = gate_up_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;

        int x = task_id - start_block * k;
        int expert_idx = x / num_blocks;
        uint64_t expert_id = expert_ids[expert_idx];
        if (!is_valid_expert_id(expert_id, config_.expert_num)) {
            return;
        }
        int offset = x % num_blocks;
        int ith = start_block + offset;
        size_t n_stride = config_.stride;

        size_t offsets_i = expert_idx * config_.intermediate_size;

        float* gate_output_ptr = s_gate_output_ + offsets_i + ith * config_.stride;
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        uint8_t* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  (static_cast<int>(expert_id) * num_blocks + offset) * amx_stride_gate_bytes_;
        amx_gemm_compute(config_.gate_type, gate_proj_ptr, gate_input_ptr, gate_output_ptr, 1, n_stride, config_.hidden_size, n_stride);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  (static_cast<int>(expert_id) * num_blocks + offset) * stride_gate_bytes_;
        llamafile_sgemm(n_stride, 1, config_.hidden_size / gate_blk_size, gate_proj_ptr, config_.hidden_size / gate_blk_size, gate_input_ptr, gate_input_em, gate_output_ptr, n_stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, use_fp32_buffer_ ? GGML_TYPE_F32 : gate_vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        #endif

        float* up_output_ptr = s_up_output_ + offsets_i + ith * config_.stride;
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        uint8_t* up_proj_ptr = (uint8_t*)up_numa_[nid] +  (static_cast<int>(expert_id) * num_blocks  + offset) * amx_stride_up_bytes_;
        amx_gemm_compute(config_.up_type, up_proj_ptr, up_input_ptr, up_output_ptr, 1, n_stride, config_.hidden_size, n_stride);
        #else
        void* up_proj_ptr = (uint8_t*)up_numa_[nid] +  (static_cast<int>(expert_id) * num_blocks  + offset) * stride_up_bytes_;
        llamafile_sgemm(n_stride, 1, config_.hidden_size / up_blk_size, up_proj_ptr, config_.hidden_size / up_blk_size, up_input_ptr, up_input_em, up_output_ptr, n_stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, use_fp32_buffer_ ? GGML_TYPE_F32 : up_vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        #endif
        act_fn(up_output_ptr, gate_output_ptr, n_stride, config_.swiglu_limit);
        if (config_.stride % down_vec_dot_blk_size == 0 && !use_fp32_buffer_) {
            void* down_input_ptr = s_down_input_ + (offsets_i + ith * config_.stride) * down_vec_dot_type_size / down_vec_dot_blk_size;
            from_float(up_output_ptr, down_input_ptr, n_stride, down_vec_dot_type);
        }
    }, nullptr);
    const auto gate_up_end = profile_enabled_
                                 ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    if (config_.stride % down_vec_dot_blk_size != 0 && !use_fp32_buffer_) {
        Backend_NUMA::getInstance().do_k_work_stealing_job(1, k, nullptr, [&](int task_id) {
            int expert_idx = task_id;
            if (!is_valid_expert_id(expert_ids[expert_idx],
                                    config_.expert_num)) {
                return;
            }
            float* up_output_ptr = s_up_output_ + expert_idx * config_.intermediate_size;
            void* down_input_ptr = s_down_input_ + expert_idx * config_.intermediate_size * down_vec_dot_type_size / down_vec_dot_blk_size;
            from_float(up_output_ptr, down_input_ptr, config_.intermediate_size, down_vec_dot_type);
        }, nullptr);
    }
    nth = config_.hidden_size / config_.stride;
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    const bool fuse_down_sum = false;
#else
    const bool fuse_down_sum = fuse_decode_down_sum_;
#endif
    if (fuse_down_sum) {
        const auto down_sum_start = profile_enabled_
                                        ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
        Backend_NUMA::getInstance().do_k_work_stealing_job(1, nth, nullptr, [&](int task_id) {
            int nid = Backend_NUMA::numa_node_;
            int start_block = down_blocks_[nid].start_block;
            int num_blocks = down_blocks_[nid].num_blocks;

            if (num_blocks == 0) return;

            int offset = task_id - start_block;
            int ith = task_id;
            size_t n_stride = config_.stride;
            float* down_output_acc = s_down_output_ + ith * config_.stride;
            thread_local std::vector<float> down_output_tmp;
            if (down_output_tmp.size() < n_stride) {
                down_output_tmp.resize(n_stride);
            }

            bool has_valid_expert = false;
            for (int expert_idx = 0; expert_idx < k; ++expert_idx) {
                uint64_t expert_id = expert_ids[expert_idx];
                if (!is_valid_expert_id(expert_id, config_.expert_num)) {
                    continue;
                }
                void* down_input_ptr;
                if(use_fp32_buffer_){
                    down_input_ptr = s_up_output_ + expert_idx * config_.intermediate_size;
                }else{
                    down_input_ptr = s_down_input_ + expert_idx * config_.intermediate_size * down_vec_dot_type_size / down_vec_dot_blk_size;
                }
                float* down_output_ptr = down_output_tmp.data();
                void* down_proj_ptr = (uint8_t*)down_numa_[nid] + (static_cast<int>(expert_id) * num_blocks  + offset) * stride_down_bytes_;
                llamafile_sgemm(n_stride, 1, config_.intermediate_size / down_blk_size, down_proj_ptr, config_.intermediate_size / down_blk_size, down_input_ptr, down_input_em, down_output_ptr, n_stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, use_fp32_buffer_ ? GGML_TYPE_F32 : down_vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
                if (!has_valid_expert) {
                    for(int j=0; j<config_.stride; ++j){
                        down_output_acc[j] = down_output_ptr[j] *
                                             weights[expert_idx];
                    }
                    has_valid_expert = true;
                } else {
                    const float weight = weights[expert_idx];
                    for(int j=0; j<config_.stride; ++j){
                        down_output_acc[j] += down_output_ptr[j] * weight;
                    }
                }
            }
            if (!has_valid_expert) {
                for(int j=0; j<config_.stride; ++j){
                    down_output_acc[j] = 0.0f;
                }
            }
            void * output_ptr = (uint8_t*)output + ith * config_.stride * hidden_type_size / hidden_blk_size;
            from_float(down_output_acc, output_ptr, config_.stride, config_.hidden_type);
        }, nullptr);
        if (profile_enabled_) {
            const auto down_sum_end = std::chrono::steady_clock::now();
            profile_forward_gate_up_ms_ +=
                std::chrono::duration<double, std::milli>(
                    gate_up_end - gate_up_start)
                    .count();
            profile_forward_down_proj_ms_ +=
                std::chrono::duration<double, std::milli>(
                    down_sum_end - down_sum_start)
                    .count();
            profile_forward_sum_ms_ += 0.0;
            profile_forward_one_breakdown_calls_++;
            profile_decode_down_sum_ms_ +=
                std::chrono::duration<double, std::milli>(
                    down_sum_end - down_sum_start)
                    .count();
            profile_decode_down_sum_calls_++;
            if (profile_decode_down_sum_calls_ % 32 == 0) {
                std::cerr << "LK_MOE_PROFILE decode_down_sum_fused calls="
                          << profile_decode_down_sum_calls_
                          << " avg_ms="
                          << (profile_decode_down_sum_ms_ /
                              static_cast<double>(
                                  profile_decode_down_sum_calls_))
                          << " stride=" << config_.stride
                          << " k=" << k << std::endl;
            }
        }
    } else {
        const auto down_sum_start = profile_enabled_
                                        ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
        Backend_NUMA::getInstance().do_k_work_stealing_job(k, nth, nullptr, [&](int task_id) {
            int nid = Backend_NUMA::numa_node_;
            int start_block = down_blocks_[nid].start_block;
            int num_blocks = down_blocks_[nid].num_blocks;

            if (num_blocks == 0) return;

            int x = task_id - start_block * k;
            int expert_idx = x / num_blocks;
            uint64_t expert_id = expert_ids[expert_idx];
            if (!is_valid_expert_id(expert_id, config_.expert_num)) {
                return;
            }
            int offset = x % num_blocks;
            int ith = start_block + offset;
            size_t n_stride = config_.stride;
            void* down_input_ptr;
            if(use_fp32_buffer_){
                down_input_ptr = s_up_output_ + expert_idx * config_.intermediate_size;
            }else{
                down_input_ptr = s_down_input_ + expert_idx * config_.intermediate_size * down_vec_dot_type_size / down_vec_dot_blk_size;
            }
            float* down_output_ptr = s_down_output_ + expert_idx * config_.hidden_size + ith * config_.stride;
            #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
            uint8_t* down_proj_ptr = (uint8_t*)down_numa_[nid] + (static_cast<int>(expert_id) * num_blocks  + offset) * amx_stride_down_bytes_;
            amx_gemm_compute(config_.down_type, down_proj_ptr, down_input_ptr, down_output_ptr, 1, config_.stride, config_.intermediate_size, n_stride);
            #else
            void* down_proj_ptr = (uint8_t*)down_numa_[nid] + (static_cast<int>(expert_id) * num_blocks  + offset) * stride_down_bytes_;
            llamafile_sgemm(n_stride, 1, config_.intermediate_size / down_blk_size, down_proj_ptr, config_.intermediate_size / down_blk_size, down_input_ptr, down_input_em, down_output_ptr, n_stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, use_fp32_buffer_ ? GGML_TYPE_F32 : down_vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            #endif
        }, nullptr);
        const auto down_proj_end = profile_enabled_
                                       ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
        nth = config_.hidden_size / config_.stride;

        Backend_NUMA::getInstance().do_k_work_stealing_job(1, nth, nullptr, [&](int task_id) {
            int ith = task_id;
            float * down_output_ptr_0 = output_fp32_ + ith * config_.stride;
            for(int j=0; j<config_.stride; ++j){
                down_output_ptr_0[j] = 0.0f;
            }
            bool has_valid_expert = false;
            for(int i=0; i<k; ++i){
                if (!is_valid_expert_id(expert_ids[i], config_.expert_num)) {
                    continue;
                }
                float * down_output_ptr =
                    s_down_output_ + i * config_.hidden_size +
                    ith * config_.stride;
                const float weight = weights[i];
                for(int j=0; j<config_.stride; ++j){
                    down_output_ptr_0[j] += down_output_ptr[j] * weight;
                }
                has_valid_expert = true;
            }
            (void)has_valid_expert;
            void * output_ptr = (uint8_t*)output + ith * config_.stride * hidden_type_size / hidden_blk_size;
            from_float(down_output_ptr_0, output_ptr, config_.stride, config_.hidden_type);
        }, nullptr);
        if (profile_enabled_) {
            const auto down_sum_end = std::chrono::steady_clock::now();
            profile_forward_gate_up_ms_ +=
                std::chrono::duration<double, std::milli>(
                    gate_up_end - gate_up_start)
                    .count();
            profile_forward_down_proj_ms_ +=
                std::chrono::duration<double, std::milli>(
                    down_proj_end - down_sum_start)
                    .count();
            profile_forward_sum_ms_ +=
                std::chrono::duration<double, std::milli>(
                    down_sum_end - down_proj_end)
                    .count();
            profile_forward_one_breakdown_calls_++;
            profile_decode_down_sum_ms_ +=
                std::chrono::duration<double, std::milli>(
                    down_sum_end - down_sum_start)
                    .count();
            profile_decode_down_sum_calls_++;
            if (profile_decode_down_sum_calls_ % 32 == 0) {
                std::cerr << "LK_MOE_PROFILE decode_down_sum_unfused calls="
                          << profile_decode_down_sum_calls_
                          << " avg_ms="
                          << (profile_decode_down_sum_ms_ /
                              static_cast<double>(
                                  profile_decode_down_sum_calls_))
                      << " stride=" << config_.stride
                      << " k=" << k << std::endl;
            }
        }
    }
    if (profile_enabled_) {
        const auto profile_end = std::chrono::steady_clock::now();
        profile_forward_one_ms_ +=
            std::chrono::duration<double, std::milli>(profile_end - profile_start)
                .count();
        profile_forward_one_calls_++;
        if (profile_forward_one_calls_ % 32 == 0) {
            const double breakdown_calls =
                profile_forward_one_breakdown_calls_ > 0
                    ? static_cast<double>(profile_forward_one_breakdown_calls_)
                    : 1.0;
            std::cerr << "LK_MOE_PROFILE forward_one calls="
                      << profile_forward_one_calls_
                      << " avg_ms="
                      << (profile_forward_one_ms_ /
                          static_cast<double>(profile_forward_one_calls_))
                      << " avg_gate_up_ms="
                      << (profile_forward_gate_up_ms_ / breakdown_calls)
                      << " avg_down_proj_ms="
                      << (profile_forward_down_proj_ms_ / breakdown_calls)
                      << " avg_sum_ms="
                      << (profile_forward_sum_ms_ / breakdown_calls)
                      << " stride=" << config_.stride
                      << " k=" << k << std::endl;
        }
    }
}
void MOE::forward_many_m(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output) {
    const int token_expert_count = qlen * k;
    const uint64_t* expert_ids_local = expert_ids;

    size_t gate_input_em = config_.hidden_size / gate_blk_size;
    size_t up_input_em = config_.hidden_size / up_blk_size;
    size_t down_input_em = config_.intermediate_size / down_blk_size;
    if(use_fp32_buffer_){
        gate_input_em = up_input_em = config_.hidden_size / hidden_blk_size;
        down_input_em = config_.intermediate_size / hidden_blk_size;
    }

    forward_many_expert_reorder_offset_.assign(config_.expert_num, 0);
    forward_many_expert_selected_num_.assign(config_.expert_num, 0);
    forward_many_active_experts_.clear();
    forward_many_active_experts_.reserve(
        std::min(config_.expert_num, token_expert_count));
    forward_many_token_expert_pos_.resize(token_expert_count);
    int* expert_reorder_offset =
        forward_many_expert_reorder_offset_.data();
    int* expert_selected_num = forward_many_expert_selected_num_.data();
    std::vector<int>& active_experts = forward_many_active_experts_;
    int* token_expert_pos = forward_many_token_expert_pos_.data();


    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            const int mapping_idx = i * k + j;
            uint64_t expert_id = expert_ids_local[mapping_idx];
            if (!is_valid_expert_id(expert_id, config_.expert_num)) {
                token_expert_pos[mapping_idx] = -1;
                continue;
            }
            const int expert_id_i = static_cast<int>(expert_id);
            int pos = expert_selected_num[expert_id_i]++;
            if (pos == 0) {
                active_experts.push_back(expert_id_i);
            }
            token_expert_pos[mapping_idx] = pos;
        }
    }

    uint64_t reorder_offset = 0;
    for (int expert_id : active_experts) {
        expert_reorder_offset[expert_id] = reorder_offset;
        reorder_offset += expert_selected_num[expert_id];
    }


    int nth = config_.hidden_size / config_.stride;
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, qlen, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;
        int token_id = task_id;


        void* input_uint8_ptr = (uint8_t*)input + token_id * config_.hidden_size * hidden_type_size / hidden_blk_size;
        float* input_fp32_ptr = input_fp32_ + token_id * config_.hidden_size;
        void* gate_input_ptr = gate_input_ + token_id * config_.hidden_size * gate_vec_dot_type_size / gate_vec_dot_blk_size;
        void* up_input_ptr = up_input_ + token_id * config_.hidden_size * up_vec_dot_type_size / up_vec_dot_blk_size;
        if(use_fp32_buffer_){
            to_float(input_uint8_ptr, input_fp32_ptr, config_.hidden_size, config_.hidden_type);
        }else{
            if (config_.hidden_type == gate_vec_dot_type && config_.hidden_type == up_vec_dot_type) {
                memcpy(gate_input_ptr, input_uint8_ptr, config_.hidden_size * hidden_type_size / hidden_blk_size);
                // up not need to copy
            } else {
                to_float(input_uint8_ptr, input_fp32_ptr, config_.hidden_size, config_.hidden_type);
                if (gate_vec_dot_type == up_vec_dot_type) {
                    from_float(input_fp32_ptr, gate_input_ptr, config_.hidden_size, gate_vec_dot_type);
                    // up not need to copy
                } else {
                    if (config_.hidden_type != gate_vec_dot_type) {
                        from_float(input_fp32_ptr, gate_input_ptr, config_.hidden_size, gate_vec_dot_type);
                    } else {
                        memcpy(gate_input_ptr, input_uint8_ptr, config_.hidden_size * hidden_type_size / hidden_blk_size);
                    }
                    if (config_.hidden_type != up_vec_dot_type) {
                        from_float(input_fp32_ptr, up_input_ptr, config_.hidden_size, up_vec_dot_type);
                    } else {
                        memcpy(up_input_ptr, input_uint8_ptr, config_.hidden_size * hidden_type_size / hidden_blk_size);
                    }
                }
            }
        }
    }, nullptr);

    Backend_NUMA::getInstance().do_k_work_stealing_job(1, qlen*k, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;
        int token_id = task_id / k;
        int expert_idx =  task_id % k;
        const int mapping_idx = token_id * k + expert_idx;
        uint64_t expert_id = expert_ids_local[mapping_idx];
        if (!is_valid_expert_id(expert_id, config_.expert_num)) {
            return;
        }

        uint64_t expert_id_from_mapping = expert_ids_local[mapping_idx];

        if (expert_id_from_mapping != expert_id) {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cerr << "MOE::forward_many_m --------------------" << std::endl
                    << "  expert_id : " << expert_id << std::endl
                    << "  expert_id_from_mapping : " << expert_id_from_mapping << std::endl
                    << "  task_id : " << task_id << std::endl
                    << "  token_id : " << token_id << std::endl
                    << "  expert_idx : " << expert_idx << std::endl
                    << "  qlen : " << qlen << std::endl
                    << "  k : " << k << std::endl
                    << "  token_expert_mapping.size : " << qlen << std::endl;

            if (token_id < qlen) {
                std::cerr << "  token_expert_mapping : " << std::endl;
                for (int j = 0; j < k; j++) {
                    const int dump_idx = token_id * k + j;
                    std::cerr << "    [" << j << "] : ("
                            << expert_ids_local[dump_idx]
                            << ", "
                            << token_expert_pos[dump_idx]
                            << ")"  << std::endl;
                }
            }
            std::abort();
        }

        assert(expert_id_from_mapping == expert_id);


        const int expert_id_i = static_cast<int>(expert_id);
        int base = expert_reorder_offset[expert_id_i];
        int pos = token_expert_pos[mapping_idx];
        void* gate_input_ptr;
        void* up_input_ptr;
        void* m_gate_input_ptr;
        void* m_up_input_ptr;
        if(use_fp32_buffer_){
            gate_input_ptr = input_fp32_ + token_id * config_.hidden_size;
            m_gate_input_ptr = (float*)m_gate_input_ + (base+pos) * config_.hidden_size;
            memcpy(m_gate_input_ptr, gate_input_ptr, config_.hidden_size*sizeof(float));
        }else{
            gate_input_ptr = (uint8_t*)gate_input_ + token_id * config_.hidden_size * gate_vec_dot_type_size / gate_vec_dot_blk_size;
            up_input_ptr = (uint8_t*)up_input_ + token_id * config_.hidden_size * up_vec_dot_type_size / up_vec_dot_blk_size;
            m_gate_input_ptr = (uint8_t*)m_gate_input_ + (base+pos) * config_.hidden_size * gate_vec_dot_type_size / gate_vec_dot_blk_size;
            m_up_input_ptr = (uint8_t*)m_up_input_ + (base+pos)  * config_.hidden_size * up_vec_dot_type_size / up_vec_dot_blk_size;
            memcpy(m_gate_input_ptr, gate_input_ptr, config_.hidden_size * gate_vec_dot_type_size / gate_vec_dot_blk_size);
            if(gate_vec_dot_type != up_vec_dot_type){
                memcpy(m_up_input_ptr, up_input_ptr, config_.hidden_size * up_vec_dot_type_size / up_vec_dot_blk_size);
            }
        }
    }, nullptr);


    nth = config_.intermediate_size / config_.stride;
    const int active_expert_num = static_cast<int>(active_experts.size());
    if (active_expert_num == 0) {
        const int hidden_type_bytes =
            config_.hidden_size * hidden_type_size / hidden_blk_size;
        Backend_NUMA::getInstance().do_k_work_stealing_job(
            1, qlen, nullptr,
            [&](int task_id) {
                void* output_ptr =
                    static_cast<uint8_t*>(output) +
                    task_id * hidden_type_bytes;
                std::memset(output_ptr, 0, hidden_type_bytes);
            },
            nullptr);
        return;
    }
    Backend_NUMA::getInstance().do_k_work_stealing_job(active_expert_num, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;
        int start_block = gate_up_blocks_[nid].start_block;
        int num_blocks = gate_up_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;

        int x = task_id - start_block * active_expert_num;
        int active_expert_idx = x / num_blocks;
        int expert_id = active_experts[active_expert_idx];

        int offset = x % num_blocks;
        int ith = start_block + offset;

        int expert_offsets = expert_reorder_offset[expert_id];
        int n = expert_selected_num[expert_id];
        size_t n_stride = config_.stride;
        void* gate_input_ptr;
        if(use_fp32_buffer_){
            gate_input_ptr = (uint8_t*)m_gate_input_ + expert_offsets * config_.hidden_size * sizeof(float);
        }else{
            gate_input_ptr = (uint8_t*)m_gate_input_ + expert_offsets * config_.hidden_size * gate_vec_dot_type_size / gate_vec_dot_blk_size;
        }


        float* gate_output_ptr = gate_output_ + expert_offsets * config_.intermediate_size + ith * config_.stride;
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        void* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  (expert_id * num_blocks + offset) * amx_stride_gate_bytes_;
        amx_gemm_compute(config_.gate_type, gate_proj_ptr, gate_input_ptr, gate_output_ptr, n, config_.stride, config_.hidden_size, config_.intermediate_size);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  (expert_id * num_blocks + offset) * stride_gate_bytes_;
        bool sgemm_ok = llamafile_sgemm(n_stride, n, config_.hidden_size / gate_blk_size, gate_proj_ptr, config_.hidden_size / gate_blk_size, gate_input_ptr, gate_input_em, gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, use_fp32_buffer_ ? GGML_TYPE_F32 : gate_vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        #endif
        void* up_input_ptr;
        if(use_fp32_buffer_){
            up_input_ptr = gate_input_ptr;
        }else{
            up_input_ptr = (gate_vec_dot_type == up_vec_dot_type)
                    ? gate_input_ptr
                    : (uint8_t*)m_up_input_ + expert_offsets * config_.hidden_size * up_vec_dot_type_size / up_vec_dot_blk_size;
        }

        float* up_output_ptr = up_output_ + expert_offsets * config_.intermediate_size + ith * config_.stride;
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        void* up_proj_ptr = (uint8_t*)up_numa_[nid] +  (expert_id * num_blocks + offset) * amx_stride_up_bytes_;
        amx_gemm_compute(config_.up_type, up_proj_ptr, up_input_ptr, up_output_ptr, n, config_.stride, config_.hidden_size, config_.intermediate_size);
        #else
        void* up_proj_ptr = (uint8_t*)up_numa_[nid] +  (expert_id * num_blocks + offset) * stride_up_bytes_;
        llamafile_sgemm(n_stride, n, config_.hidden_size / up_blk_size, up_proj_ptr, config_.hidden_size / up_blk_size, up_input_ptr, up_input_em, up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, use_fp32_buffer_ ? GGML_TYPE_F32 : up_vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        #endif

        for(int i=0; i<n; i++){
            act_fn(up_output_ptr + i * config_.intermediate_size,
                   gate_output_ptr + i * config_.intermediate_size,
                   n_stride, config_.swiglu_limit);
            if(!use_fp32_buffer_){
                if (config_.stride % down_vec_dot_blk_size == 0) {
                    void* down_input_ptr = down_input_ + ((expert_offsets + i) * config_.intermediate_size + ith * config_.stride) * down_vec_dot_type_size / down_vec_dot_blk_size;
                    from_float(up_output_ptr + i * config_.intermediate_size, down_input_ptr, n_stride, down_vec_dot_type);
                }
            }
        }


    }, nullptr);
    if(!use_fp32_buffer_){
        if (config_.stride % down_vec_dot_blk_size != 0) {
            Backend_NUMA::getInstance().do_k_work_stealing_job(1, active_expert_num, nullptr, [&](int task_id) {
                int nid = Backend_NUMA::numa_node_;
                int expert_id = active_experts[task_id];
                int expert_offsets = expert_reorder_offset[expert_id];
                int n = expert_selected_num[expert_id];
                float* up_output_ptr_ = up_output_ + expert_offsets * config_.intermediate_size;
                void* down_input_ptr = down_input_ + (expert_offsets * config_.intermediate_size) * down_vec_dot_type_size / down_vec_dot_blk_size;
                from_float(up_output_ptr_, down_input_ptr, n * config_.intermediate_size, down_vec_dot_type);
            }, nullptr);
        }
    }


    nth = config_.hidden_size / config_.stride;
    Backend_NUMA::getInstance().do_k_work_stealing_job(active_expert_num, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;
        int start_block = down_blocks_[nid].start_block;
        int num_blocks = down_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;

        int x = task_id - start_block * active_expert_num;
        int active_expert_idx = x / num_blocks;
        int expert_id = active_experts[active_expert_idx];

        int offset = x % num_blocks;
        int ith = start_block + offset;

        int expert_offsets = expert_reorder_offset[expert_id];
        int n = expert_selected_num[expert_id];
        size_t n_stride = config_.stride;
        void* down_input_ptr;
        if(use_fp32_buffer_){
            down_input_ptr = up_output_ + expert_offsets * config_.intermediate_size;
        }else{
            down_input_ptr = down_input_ + expert_offsets * config_.intermediate_size * down_vec_dot_type_size / down_vec_dot_blk_size;
        }
        float* down_output_ptr = down_output_  + expert_offsets * config_.hidden_size + ith * config_.stride;
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        uint8_t* down_proj_ptr = (uint8_t*)down_numa_[nid] + (expert_id * num_blocks + offset) * amx_stride_down_bytes_;
        amx_gemm_compute(config_.down_type, down_proj_ptr, down_input_ptr, down_output_ptr, n, n_stride, config_.intermediate_size, config_.hidden_size);
        #else
        void* down_proj_ptr = (uint8_t*)down_numa_[nid] + (expert_id * num_blocks + offset) * stride_down_bytes_;
        llamafile_sgemm(n_stride, n, config_.intermediate_size / down_blk_size, down_proj_ptr, config_.intermediate_size / down_blk_size, down_input_ptr, down_input_em, down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, use_fp32_buffer_ ? GGML_TYPE_F32 : down_vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        #endif
    }, nullptr);

      Backend_NUMA::getInstance().do_k_work_stealing_job(qlen, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;
        int start_block = down_blocks_[nid].start_block;
        int num_blocks = down_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;

        int x = task_id - start_block * qlen;
        int token_id = x / num_blocks;
        int offset = x % num_blocks;

        int ith = start_block + offset;
        size_t n_stride = config_.stride;

        float* down_output_acc = output_fp32_ + token_id * config_.hidden_size +
                                 ith * config_.stride;
        for(int j=0; j<n_stride; ++j){
            down_output_acc[j] = 0.0f;
        }
        for(int i=0; i<k; i++){
            const int mapping_idx = token_id * k + i;
            uint64_t expert_id = expert_ids_local[mapping_idx];
            if (!is_valid_expert_id(expert_id, config_.expert_num)) {
                continue;
            }
            int pos = token_expert_pos[mapping_idx];
            if (pos < 0) {
                continue;
            }
            const int expert_id_i = static_cast<int>(expert_id);
            int base = expert_reorder_offset[expert_id_i];
            float* down_output_ptr = down_output_  +  (base + pos) *
                                     config_.hidden_size +
                                     ith * config_.stride;
            const float weight = weights[mapping_idx];
            for(int j=0; j<n_stride; ++j){
                    down_output_acc[j] += down_output_ptr[j] * weight;
            }
        }

        void* output_ptr = (uint8_t*)output + (token_id * config_.hidden_size + ith * config_.stride) * hidden_type_size / hidden_blk_size;
        from_float(down_output_acc, output_ptr, n_stride, config_.hidden_type);
    }, nullptr);


}



void MOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, int* bsz_tensor) {
    int batch_size = bsz_tensor[0];
    uint64_t local_forward_one_tokens = 0;
    uint64_t local_forward_many_calls = 0;
    uint64_t local_forward_many_tokens = 0;
    int processed = 0;
    while (processed < batch_size) {
        int remaining = batch_size - processed;

        if (remaining < config_.group_min_len) {
            local_forward_one_tokens += remaining;
            for (int i = 0; i < remaining; i++) {
                int current_pos = processed + i;
                (this->*forward_one_impl)(
                    k,
                    expert_ids + current_pos * k,
                    weights + current_pos * k,
                    (uint8_t*)input + current_pos * config_.hidden_size * hidden_type_size / hidden_blk_size,
                    (uint8_t*)output + current_pos * config_.hidden_size * hidden_type_size / hidden_blk_size
                );
            }
            break;
        }

        int forward_len = std::min(config_.group_max_len, remaining);
        local_forward_many_calls++;
        local_forward_many_tokens += forward_len;
        (this->*forward_many_impl)(
            forward_len,
            k,
            expert_ids + processed * k,
            weights + processed * k,
            (uint8_t*)input + processed * config_.hidden_size * hidden_type_size / hidden_blk_size,
            (uint8_t*)output + processed * config_.hidden_size * hidden_type_size / hidden_blk_size
        );

        processed += forward_len;
    }
    if (profile_enabled_) {
        profile_forward_calls_++;
        profile_forward_one_tokens_ += local_forward_one_tokens;
        profile_forward_many_calls_ += local_forward_many_calls;
        profile_forward_many_tokens_ += local_forward_many_tokens;
        if (profile_forward_calls_ % 32 == 0) {
            std::cerr << "LK_MOE_PROFILE forward calls="
                      << profile_forward_calls_
                      << " batch_size=" << batch_size
                      << " group_min_len=" << config_.group_min_len
                      << " group_max_len=" << config_.group_max_len
                      << " one_tokens=" << profile_forward_one_tokens_
                      << " many_calls=" << profile_forward_many_calls_
                      << " many_tokens=" << profile_forward_many_tokens_
                      << std::endl;
        }
    }
    sync_flag.store(true, std::memory_order_seq_cst);


}


static void forward_wrapper(void* args) {
    ForwardParams* params = (ForwardParams*)args;
        params->moe_ptr->forward(
            params->qlen,
            params->k,
            params->expert_ids,
            params->weights,
            params->input,
            params->output,
            params->bsz_tensor
        );
        // delete params;  !don't delete params here
}

void MOE::ensure_decode_buffers(int qlen, int k, bool expert_ids_i32) {
    const size_t expert_ids_bytes =
        static_cast<size_t>(qlen) * k * sizeof(uint64_t);
    const size_t expert_ids_i32_bytes =
        static_cast<size_t>(qlen) * k * sizeof(int32_t);
    const size_t weights_bytes = static_cast<size_t>(qlen) * k * sizeof(float);
    const size_t hidden_bytes = static_cast<size_t>(qlen) * config_.hidden_size *
                                hidden_type_size / hidden_blk_size;

    auto resize_host = [](void** ptr, size_t* capacity, size_t needed) {
        if (*capacity >= needed) {
            return;
        }
        if (*ptr != nullptr) {
            cudaFreeHost(*ptr);
            *ptr = nullptr;
            *capacity = 0;
        }
        cudaError_t err = cudaHostAlloc(ptr, needed, cudaHostAllocPortable);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaHostAlloc failed: ") +
                                     cudaGetErrorString(err));
        }
        *capacity = needed;
    };

    resize_host(reinterpret_cast<void**>(&decode_expert_ids_host_),
                &decode_expert_ids_capacity_bytes_, expert_ids_bytes);
    if (expert_ids_i32) {
        resize_host(reinterpret_cast<void**>(&decode_expert_ids_i32_host_),
                    &decode_expert_ids_i32_capacity_bytes_,
                    expert_ids_i32_bytes);
    }
    resize_host(reinterpret_cast<void**>(&decode_weights_host_),
                &decode_weights_capacity_bytes_, weights_bytes);
    resize_host(&decode_input_host_, &decode_input_capacity_bytes_, hidden_bytes);
    resize_host(&decode_output_host_, &decode_output_capacity_bytes_, hidden_bytes);
    if (decode_bsz_host_ == nullptr) {
        cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&decode_bsz_host_),
                                        sizeof(int), cudaHostAllocPortable);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaHostAlloc failed: ") +
                                     cudaGetErrorString(err));
        }
    }
}

void MOE::ensure_decode_param_buffers(CpuDecodeParams* params) {
    const int qlen = params->qlen;
    const int k = params->k;
    const bool expert_ids_i32 = params->expert_ids_i32;
    const size_t expert_ids_bytes =
        static_cast<size_t>(qlen) * k * sizeof(uint64_t);
    const size_t expert_ids_i32_bytes =
        static_cast<size_t>(qlen) * k * sizeof(int32_t);
    const size_t weights_bytes = static_cast<size_t>(qlen) * k * sizeof(float);
    const size_t hidden_bytes = static_cast<size_t>(qlen) * config_.hidden_size *
                                hidden_type_size / hidden_blk_size;

    auto resize_host = [](void** ptr, size_t* capacity, size_t needed) {
        if (*capacity >= needed) {
            return;
        }
        if (*ptr != nullptr) {
            cudaFreeHost(*ptr);
            *ptr = nullptr;
            *capacity = 0;
        }
        cudaError_t err = cudaHostAlloc(ptr, needed, cudaHostAllocPortable);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaHostAlloc failed: ") +
                                     cudaGetErrorString(err));
        }
        *capacity = needed;
    };

    resize_host(reinterpret_cast<void**>(&params->expert_ids_host),
                &params->expert_ids_capacity_bytes, expert_ids_bytes);
    if (expert_ids_i32) {
        resize_host(reinterpret_cast<void**>(&params->expert_ids_i32_host),
                    &params->expert_ids_i32_capacity_bytes,
                    expert_ids_i32_bytes);
    }
    resize_host(reinterpret_cast<void**>(&params->weights_host),
                &params->weights_capacity_bytes, weights_bytes);
    resize_host(&params->input_host, &params->input_capacity_bytes,
                hidden_bytes);
    resize_host(&params->output_host, &params->output_capacity_bytes,
                hidden_bytes);
    if (params->bsz_host == nullptr) {
        cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&params->bsz_host),
                                        sizeof(int), cudaHostAllocPortable);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaHostAlloc failed: ") +
                                     cudaGetErrorString(err));
        }
    }
}

CpuDecodeParams* MOE::get_decode_params(int qlen, int k,
                                         bool expert_ids_i32) {
    for (CpuDecodeParams* params : decode_params_) {
        if (params->qlen == qlen && params->k == k &&
            params->expert_ids_i32 == expert_ids_i32) {
            return params;
        }
    }
    CpuDecodeParams* params =
        new CpuDecodeParams{this, qlen, k, expert_ids_i32};
    decode_params_.push_back(params);
    return params;
}

void MOE::enqueue_decode_task(CpuDecodeParams* params) {
    auto task = std::make_shared<CpuDecodeTask>();
    task->params = params;
    params->task = task;
    {
        std::lock_guard<std::mutex> lock(decode_task_mutex_);
        decode_tasks_.push(task);
    }
    decode_task_cv_.notify_one();
}

void MOE::wait_decode_task(CpuDecodeParams* params) {
    std::shared_ptr<CpuDecodeTask> task = params->task;
    if (!task) {
        throw std::runtime_error("cpu decode task missing");
    }
    std::unique_lock<std::mutex> lock(task->mutex);
    task->cv.wait(lock, [&] { return task->done; });
    if (task->exception) {
        std::rethrow_exception(task->exception);
    }
    params->task.reset();
}

void MOE::cpu_decode_task_loop() {
    bind_callback_thread_once();
    enable_decode_task_thread_steal_once();
    while (true) {
        std::shared_ptr<CpuDecodeTask> task;
        {
            std::unique_lock<std::mutex> lock(decode_task_mutex_);
            decode_task_cv_.wait(lock, [&] {
                return decode_task_thread_exit_ || !decode_tasks_.empty();
            });
            if (decode_task_thread_exit_ && decode_tasks_.empty()) {
                return;
            }
            task = decode_tasks_.front();
            decode_tasks_.pop();
        }
        run_cpu_decode_task(task.get());
    }
}

void MOE::run_cpu_decode_task(CpuDecodeTask* task) {
    CpuDecodeParams* params = task->params;
    try {
        const bool profile_enabled = profile_enabled_;
        const auto profile_start = profile_enabled
                                       ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
        if (profile_detailed_enabled_) {
            params->profile_cpu_start = std::chrono::steady_clock::now();
        }
        if (params->expert_ids_i32) {
            const int count = params->qlen * params->k;
            for (int i = 0; i < count; ++i) {
                params->expert_ids_host[i] =
                    static_cast<uint64_t>(params->expert_ids_i32_host[i]);
            }
        }
        params->bsz_host[0] = params->qlen;
        forward(params->qlen, params->k, params->expert_ids_host,
                params->weights_host, params->input_host, params->output_host,
                params->bsz_host);
        if (profile_detailed_enabled_) {
            params->profile_cpu_end = std::chrono::steady_clock::now();
        }
        if (profile_enabled) {
            const auto profile_end = std::chrono::steady_clock::now();
            profile_cpu_decode_forward_ms_ +=
                std::chrono::duration<double, std::milli>(
                    profile_end - profile_start)
                    .count();
            profile_cpu_decode_forward_calls_++;
            if (profile_cpu_decode_forward_calls_ % 32 == 0) {
                std::cerr << "LK_MOE_PROFILE cpu_decode_forward calls="
                          << profile_cpu_decode_forward_calls_
                          << " avg_ms="
                          << (profile_cpu_decode_forward_ms_ /
                              static_cast<double>(
                                  profile_cpu_decode_forward_calls_))
                          << " qlen=" << params->qlen
                          << " k=" << params->k
                          << " ids="
                          << (params->expert_ids_i32 ? "i32" : "i64")
                          << std::endl;
            }
        }
    } catch (...) {
        task->exception = std::current_exception();
    }
    {
        std::lock_guard<std::mutex> lock(task->mutex);
        task->done = true;
    }
    task->cv.notify_one();
}

void cpu_decode_profile_start_wrapper(void* args) {
    CpuDecodeParams* params = static_cast<CpuDecodeParams*>(args);
    const auto now = std::chrono::steady_clock::now();
    params->profile_start = now;
    params->profile_cpu_start = now;
    params->profile_cpu_end = now;
}

void cpu_decode_profile_after_d2h_wrapper(void* args) {
    CpuDecodeParams* params = static_cast<CpuDecodeParams*>(args);
    params->profile_cpu_start = std::chrono::steady_clock::now();
}

void cpu_decode_enqueue_wrapper(void* args) {
    bind_callback_thread_once();
    CpuDecodeParams* params = static_cast<CpuDecodeParams*>(args);
    params->moe_ptr->enqueue_decode_task(params);
}

void cpu_decode_forward_wrapper(void* args) {
    cpu_decode_enqueue_wrapper(args);
    cpu_decode_wait_wrapper(args);
}

void cpu_decode_wait_wrapper(void* args) {
    bind_callback_thread_once();
    CpuDecodeParams* params = static_cast<CpuDecodeParams*>(args);
    params->moe_ptr->wait_decode_task(params);
}

void cpu_decode_profile_done_wrapper(void* args) {
    CpuDecodeParams* params = static_cast<CpuDecodeParams*>(args);
    MOE* moe = params->moe_ptr;
    const auto end = std::chrono::steady_clock::now();
    auto ms = [](const auto& a, const auto& b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    moe->profile_decode_total_ms_ += ms(params->profile_start, end);
    moe->profile_decode_pre_cpu_ms_ +=
        ms(params->profile_start, params->profile_cpu_start);
    moe->profile_decode_cpu_ms_ +=
        ms(params->profile_cpu_start, params->profile_cpu_end);
    moe->profile_decode_post_cpu_ms_ += ms(params->profile_cpu_end, end);
    moe->profile_cpu_decode_completed_calls_++;
    if (moe->profile_cpu_decode_completed_calls_ % 32 == 0) {
        const double calls =
            static_cast<double>(moe->profile_cpu_decode_completed_calls_);
        std::cerr << "LK_MOE_PROFILE_DETAIL cpu_decode_completed calls="
                  << moe->profile_cpu_decode_completed_calls_
                  << " avg_total_ms=" << (moe->profile_decode_total_ms_ / calls)
                  << " avg_pre_cpu_ms="
                  << (moe->profile_decode_pre_cpu_ms_ / calls)
                  << " avg_cpu_ms=" << (moe->profile_decode_cpu_ms_ / calls)
                  << " avg_post_cpu_ms="
                  << (moe->profile_decode_post_cpu_ms_ / calls)
                  << " qlen=" << params->qlen << " k=" << params->k
                  << " ids=" << (params->expert_ids_i32 ? "i32" : "i64")
                  << " caller_steal="
                  << (Backend_NUMA::getInstance().caller_steal_enabled_ ? 1 : 0)
                  << std::endl;
    }
}

void MOE::cpu_decode(intptr_t user_cuda_stream, int qlen, int k,
                     const uint64_t* expert_ids_dev, const float* weights_dev,
                     const void* input_dev, void* output_dev) {
    cpu_decode_impl(user_cuda_stream, qlen, k, expert_ids_dev,
                    static_cast<size_t>(qlen) * k * sizeof(uint64_t), false,
                    weights_dev, input_dev, output_dev);
}

void MOE::cpu_decode_i32(intptr_t user_cuda_stream, int qlen, int k,
                         const int32_t* expert_ids_dev,
                         const float* weights_dev, const void* input_dev,
                         void* output_dev) {
    cpu_decode_impl(user_cuda_stream, qlen, k, expert_ids_dev,
                    static_cast<size_t>(qlen) * k * sizeof(int32_t), true,
                    weights_dev, input_dev, output_dev);
}

void MOE::cpu_decode_nowait(intptr_t user_cuda_stream, int qlen, int k,
                            const uint64_t* expert_ids_dev,
                            const float* weights_dev, const void* input_dev,
                            void* output_dev) {
    cpu_decode_nowait_impl(user_cuda_stream, qlen, k, expert_ids_dev,
                           static_cast<size_t>(qlen) * k * sizeof(uint64_t),
                           false, weights_dev, input_dev, output_dev);
}

void MOE::cpu_decode_nowait_i32(intptr_t user_cuda_stream, int qlen, int k,
                                const int32_t* expert_ids_dev,
                                const float* weights_dev,
                                const void* input_dev, void* output_dev) {
    cpu_decode_nowait_impl(user_cuda_stream, qlen, k, expert_ids_dev,
                           static_cast<size_t>(qlen) * k * sizeof(int32_t),
                           true, weights_dev, input_dev, output_dev);
}

void MOE::cpu_decode_wait(intptr_t user_cuda_stream, int qlen, int k,
                          bool expert_ids_i32, void* output_dev) {
    CpuDecodeParams* params = get_decode_params(qlen, k, expert_ids_i32);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(user_cuda_stream);
    cudaError_t err = cudaLaunchHostFunc(stream, &cpu_decode_wait_wrapper,
                                         params);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaLaunchHostFunc wait failed: ") +
                                 cudaGetErrorString(err));
    }
    const size_t hidden_bytes = static_cast<size_t>(qlen) * config_.hidden_size *
                                hidden_type_size / hidden_blk_size;
    err = cudaMemcpyAsync(output_dev, params->output_host, hidden_bytes,
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync output failed: ") +
                                 cudaGetErrorString(err));
    }
    if (profile_detailed_enabled_) {
        err = cudaLaunchHostFunc(stream, &cpu_decode_profile_done_wrapper,
                                 params);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaLaunchHostFunc profile done failed: ") +
                cudaGetErrorString(err));
        }
    }
}

void MOE::cpu_decode_sync(intptr_t user_cuda_stream, int qlen, int k,
                          const uint64_t* expert_ids_dev,
                          const float* weights_dev, const void* input_dev,
                          void* output_dev) {
    cpu_decode_sync_impl(user_cuda_stream, qlen, k, expert_ids_dev,
                         static_cast<size_t>(qlen) * k * sizeof(uint64_t),
                         false, weights_dev, input_dev, output_dev);
}

void MOE::cpu_decode_sync_i32(intptr_t user_cuda_stream, int qlen, int k,
                              const int32_t* expert_ids_dev,
                              const float* weights_dev, const void* input_dev,
                              void* output_dev) {
    cpu_decode_sync_impl(user_cuda_stream, qlen, k, expert_ids_dev,
                         static_cast<size_t>(qlen) * k * sizeof(int32_t),
                         true, weights_dev, input_dev, output_dev);
}

void MOE::cpu_decode_sync_impl(intptr_t user_cuda_stream, int qlen, int k,
                               const void* expert_ids_dev,
                               size_t expert_ids_bytes, bool expert_ids_i32,
                               const float* weights_dev, const void* input_dev,
                               void* output_dev) {
    const bool profile_detail = profile_detailed_enabled_;
    const auto sync_profile_start =
        profile_detail ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point{};
    ensure_decode_buffers(qlen, k, expert_ids_i32);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(user_cuda_stream);
    const size_t weights_bytes = static_cast<size_t>(qlen) * k * sizeof(float);
    const size_t hidden_bytes = static_cast<size_t>(qlen) * config_.hidden_size *
                                hidden_type_size / hidden_blk_size;

    void* expert_ids_host = expert_ids_i32
                                ? static_cast<void*>(decode_expert_ids_i32_host_)
                                : static_cast<void*>(decode_expert_ids_host_);
    cudaError_t err = cudaMemcpyAsync(expert_ids_host, expert_ids_dev,
                                      expert_ids_bytes, cudaMemcpyDeviceToHost,
                                      stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync expert ids failed: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaMemcpyAsync(decode_weights_host_, weights_dev, weights_bytes,
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync weights failed: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaMemcpyAsync(decode_input_host_, input_dev, hidden_bytes,
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync input failed: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") +
                                 cudaGetErrorString(err));
    }
    const auto sync_profile_cpu_start =
        profile_detail ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point{};

    if (expert_ids_i32) {
        const int count = qlen * k;
        for (int i = 0; i < count; ++i) {
            decode_expert_ids_host_[i] =
                static_cast<uint64_t>(decode_expert_ids_i32_host_[i]);
        }
    }
    decode_bsz_host_[0] = qlen;
    forward(qlen, k, decode_expert_ids_host_, decode_weights_host_,
            decode_input_host_, decode_output_host_, decode_bsz_host_);
    const auto sync_profile_cpu_end =
        profile_detail ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point{};

    err = cudaMemcpyAsync(output_dev, decode_output_host_, hidden_bytes,
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync output failed: ") +
                                 cudaGetErrorString(err));
    }
    if (profile_detail) {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaStreamSynchronize output failed: ") +
                cudaGetErrorString(err));
        }
        const auto sync_profile_end = std::chrono::steady_clock::now();
        auto ms = [](const auto& a, const auto& b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        profile_sync_decode_total_ms_ +=
            ms(sync_profile_start, sync_profile_end);
        profile_sync_decode_pre_cpu_ms_ +=
            ms(sync_profile_start, sync_profile_cpu_start);
        profile_sync_decode_cpu_ms_ +=
            ms(sync_profile_cpu_start, sync_profile_cpu_end);
        profile_sync_decode_post_cpu_ms_ +=
            ms(sync_profile_cpu_end, sync_profile_end);
        profile_cpu_decode_sync_completed_calls_++;
        if (profile_cpu_decode_sync_completed_calls_ % 32 == 0) {
            const double calls =
                static_cast<double>(profile_cpu_decode_sync_completed_calls_);
            std::cerr << "LK_MOE_PROFILE_DETAIL cpu_decode_sync_completed calls="
                      << profile_cpu_decode_sync_completed_calls_
                      << " avg_total_ms="
                      << (profile_sync_decode_total_ms_ / calls)
                      << " avg_pre_cpu_ms="
                      << (profile_sync_decode_pre_cpu_ms_ / calls)
                      << " avg_cpu_ms="
                      << (profile_sync_decode_cpu_ms_ / calls)
                      << " avg_post_cpu_ms="
                      << (profile_sync_decode_post_cpu_ms_ / calls)
                      << " qlen=" << qlen << " k=" << k << " ids="
                      << (expert_ids_i32 ? "i32" : "i64")
                      << " caller_steal="
                      << (Backend_NUMA::getInstance().caller_steal_enabled_ ? 1
                                                                            : 0)
                      << std::endl;
        }
    }
}

void MOE::cpu_decode_impl(intptr_t user_cuda_stream, int qlen, int k,
                          const void* expert_ids_dev,
                          size_t expert_ids_bytes, bool expert_ids_i32,
                          const float* weights_dev, const void* input_dev,
                          void* output_dev) {
    cpu_decode_nowait_impl(user_cuda_stream, qlen, k, expert_ids_dev,
                           expert_ids_bytes, expert_ids_i32, weights_dev,
                           input_dev, output_dev);
    cpu_decode_wait(user_cuda_stream, qlen, k, expert_ids_i32, output_dev);
}

void MOE::cpu_decode_nowait_impl(intptr_t user_cuda_stream, int qlen, int k,
                                 const void* expert_ids_dev,
                                 size_t expert_ids_bytes, bool expert_ids_i32,
                                 const float* weights_dev,
                                 const void* input_dev, void* output_dev) {
    const auto profile_start = profile_enabled_
                                   ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
    CpuDecodeParams* params = get_decode_params(qlen, k, expert_ids_i32);
    ensure_decode_param_buffers(params);
    params->bsz_host[0] = qlen;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(user_cuda_stream);
    const size_t weights_bytes = static_cast<size_t>(qlen) * k * sizeof(float);
    const size_t hidden_bytes = static_cast<size_t>(qlen) * config_.hidden_size *
                                hidden_type_size / hidden_blk_size;

    cudaError_t err;
    if (profile_detailed_enabled_) {
        err = cudaLaunchHostFunc(stream, &cpu_decode_profile_start_wrapper,
                                 params);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaLaunchHostFunc profile start failed: ") +
                cudaGetErrorString(err));
        }
    }

    void* expert_ids_host = expert_ids_i32
                                ? static_cast<void*>(params->expert_ids_i32_host)
                                : static_cast<void*>(params->expert_ids_host);
    err = cudaMemcpyAsync(expert_ids_host, expert_ids_dev,
                          expert_ids_bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync expert ids failed: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaMemcpyAsync(params->weights_host, weights_dev, weights_bytes,
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync weights failed: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaMemcpyAsync(params->input_host, input_dev, hidden_bytes,
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync input failed: ") +
                                 cudaGetErrorString(err));
    }

    err = cudaLaunchHostFunc(stream, &cpu_decode_enqueue_wrapper, params);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaLaunchHostFunc enqueue failed: ") +
                                 cudaGetErrorString(err));
    }

    if (profile_enabled_) {
        const auto profile_end = std::chrono::steady_clock::now();
        profile_cpu_decode_enqueue_ms_ +=
            std::chrono::duration<double, std::milli>(profile_end - profile_start)
                .count();
        profile_cpu_decode_calls_++;
        if (profile_cpu_decode_calls_ % 32 == 0) {
            std::cerr << "LK_MOE_PROFILE cpu_decode_enqueued calls="
                      << profile_cpu_decode_calls_
                      << " avg_enqueue_ms="
                      << (profile_cpu_decode_enqueue_ms_ /
                          static_cast<double>(profile_cpu_decode_calls_))
                      << " qlen=" << qlen << " k=" << k << std::endl;
        }
    }
}

void MOE::submit_with_cuda_stream(intptr_t user_cuda_stream, int qlen, int k, const uint64_t* expert_ids,
                                 const float* weights, const void* input, void* output, int* bsz_tensor) {
    sync_flag.store(false, std::memory_order_seq_cst);
    ForwardParams* params = new ForwardParams();
    params->moe_ptr = this;
    params->qlen = qlen;
    params->k = k;
    params->expert_ids = expert_ids;
    params->weights = weights;
    params->input = input;
    params->output = output;
    params->bsz_tensor = bsz_tensor;

    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)forward_wrapper, params);
}

void MOE::sync() {
    while (!sync_flag.load(std::memory_order_seq_cst))
        ;
}
static void sync_(void * moe_ptr) {
    MOE* moe = (MOE*)moe_ptr;
    moe->sync();
}

void MOE::sync_with_cuda_stream(intptr_t user_cuda_stream) {
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)this);
}
