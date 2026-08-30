/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : guqiong96
 * @LastEditTime : 2025-08-12 10:35:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_MOE_H
#define CPUINFER_OPERATOR_MOE_H

#include <cuda_runtime.h>

#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <chrono>
#include <thread>
#include <vector>

#include "backend_numa.h"
#include "conversion.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"
#include "llamafile/sgemm.h"
// #define __AMX_INT8__ 1
// #define __AVX512VNNI__ 1
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    #include "amx_gemm.hpp"
    #pragma message("AMX support enabled in compilation")
#else
    #pragma message("AMX support disabled in compilation")
#endif
#include <queue>

class MOE;
void cpu_decode_profile_start_wrapper(void* args);
void cpu_decode_enqueue_wrapper(void* args);
void cpu_decode_forward_wrapper(void* args);
void cpu_decode_wait_wrapper(void* args);
void cpu_decode_profile_done_wrapper(void* args);
struct CpuDecodeTask;
struct CpuDecodeParams {
    MOE* moe_ptr;
    int qlen;
    int k;
    bool expert_ids_i32;
    std::shared_ptr<CpuDecodeTask> task;
    uint64_t* expert_ids_host = nullptr;
    int32_t* expert_ids_i32_host = nullptr;
    float* weights_host = nullptr;
    void* input_host = nullptr;
    void* output_host = nullptr;
    int* bsz_host = nullptr;
    size_t expert_ids_capacity_bytes = 0;
    size_t expert_ids_i32_capacity_bytes = 0;
    size_t weights_capacity_bytes = 0;
    size_t input_capacity_bytes = 0;
    size_t output_capacity_bytes = 0;
    std::chrono::steady_clock::time_point profile_start;
    std::chrono::steady_clock::time_point profile_cpu_start;
    std::chrono::steady_clock::time_point profile_cpu_end;
};
struct CpuDecodeTask {
    CpuDecodeParams* params;
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    std::exception_ptr exception;
};
struct ForwardParams {
    MOE* moe_ptr;
    int qlen;
    int k;
    const uint64_t* expert_ids;
    const float* weights;
    const void* input;
    void* output;
    int* bsz_tensor;
    cudaEvent_t* forward_done_event;
};

struct MOEConfig {
    int expert_num;
    int routed_expert_num;
    int hidden_size;
    int intermediate_size;
    int stride;
    int group_min_len;
    int group_max_len;
    void* gate_proj;
    void* up_proj;
    void* down_proj;
    ggml_type gate_type;
    ggml_type up_type;
    ggml_type down_type;
    ggml_type hidden_type;
    float swiglu_limit;

    MOEConfig() {}

    MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int stride, int group_min_len, int group_max_len, void* gate_proj, void* up_proj, void* down_proj, ggml_type gate_type, ggml_type up_type, ggml_type down_type, ggml_type hidden_type, float swiglu_limit = 0.0f)
        : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size), intermediate_size(intermediate_size), stride(stride), group_min_len(group_min_len), group_max_len(group_max_len), gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), gate_type(gate_type), up_type(up_type), down_type(down_type), hidden_type(hidden_type), swiglu_limit(swiglu_limit) {}
};

class MOE {
   public:
    MOE(MOEConfig);
    ~MOE();
    void warm_up();
    void forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, int* bsz_tensor);
    void cpu_decode(intptr_t user_cuda_stream, int qlen, int k,
                    const uint64_t* expert_ids_dev, const float* weights_dev,
                    const void* input_dev, void* output_dev);
    void cpu_decode_i32(intptr_t user_cuda_stream, int qlen, int k,
                        const int32_t* expert_ids_dev, const float* weights_dev,
                        const void* input_dev, void* output_dev);
    void cpu_decode_nowait(intptr_t user_cuda_stream, int qlen, int k,
                           const uint64_t* expert_ids_dev,
                           const float* weights_dev, const void* input_dev,
                           void* output_dev);
    void cpu_decode_nowait_i32(intptr_t user_cuda_stream, int qlen, int k,
                               const int32_t* expert_ids_dev,
                               const float* weights_dev, const void* input_dev,
                               void* output_dev);
    void cpu_decode_wait(intptr_t user_cuda_stream, int qlen, int k,
                         bool expert_ids_i32, void* output_dev);
    void cpu_decode_sync(intptr_t user_cuda_stream, int qlen, int k,
                         const uint64_t* expert_ids_dev,
                         const float* weights_dev, const void* input_dev,
                         void* output_dev);
    void cpu_decode_sync_i32(intptr_t user_cuda_stream, int qlen, int k,
                             const int32_t* expert_ids_dev,
                             const float* weights_dev, const void* input_dev,
                             void* output_dev);
    void submit_with_cuda_stream(intptr_t user_cuda_stream, int qlen, int k, const uint64_t* expert_ids,
                             const float* weights, const void* input, void* output, int* bsz_tensor);
    void sync_with_cuda_stream(intptr_t user_cuda_stream);
    void sync();
   private:
    void forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output);
    void forward_many_m(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output);
    using ForwardOneImpl = void (MOE::*)(int, const uint64_t*, const float*, const void*, void*);
    using ForwardManyImpl = void (MOE::*)(int, int, const uint64_t*, const float*, const void*, void*);
    ForwardOneImpl forward_one_impl;
    ForwardManyImpl forward_many_impl;
    void ensure_decode_buffers(int qlen, int k, bool expert_ids_i32);
    void ensure_decode_param_buffers(CpuDecodeParams* params);
    CpuDecodeParams* get_decode_params(int qlen, int k, bool expert_ids_i32);
    void enqueue_decode_task(CpuDecodeParams* params);
    void wait_decode_task(CpuDecodeParams* params);
    void cpu_decode_task_loop();
    void run_cpu_decode_task(CpuDecodeTask* task);
    void cpu_decode_impl(intptr_t user_cuda_stream, int qlen, int k,
                         const void* expert_ids_dev, size_t expert_ids_bytes,
                         bool expert_ids_i32, const float* weights_dev,
                         const void* input_dev, void* output_dev);
    void cpu_decode_nowait_impl(intptr_t user_cuda_stream, int qlen, int k,
                                const void* expert_ids_dev,
                                size_t expert_ids_bytes, bool expert_ids_i32,
                                const float* weights_dev,
                                const void* input_dev, void* output_dev);
    void cpu_decode_sync_impl(intptr_t user_cuda_stream, int qlen, int k,
                              const void* expert_ids_dev,
                              size_t expert_ids_bytes, bool expert_ids_i32,
                              const float* weights_dev, const void* input_dev,
                              void* output_dev);
    MOEConfig config_;
    void* gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]

    std::vector<void*> gate_numa_;  // [numa_num, nth * stride * expert_num * hidden_size ( /32 if quantized)]
    std::vector<void*> up_numa_;    // [numa_num, nth * stride * expert_num * hidden_size ( /32 if quantized)]
    std::vector<void*> down_numa_;  // [numa_num, nth * stride * expert_num *  intermediate_size ( /32 if quantized)]
    std::vector<size_t> gate_numa_size_;
    std::vector<size_t> up_numa_size_;
    std::vector<size_t> down_numa_size_;
    size_t stride_gate_bytes_;
    size_t stride_up_bytes_;
    size_t stride_down_bytes_;
    size_t amx_stride_gate_bytes_;
    size_t amx_stride_up_bytes_;
    size_t amx_stride_down_bytes_;
    struct NumaBlock {
        int node_id;
        int start_block;
        int num_blocks;
    };
    std::vector<NumaBlock> gate_up_blocks_;
    std::vector<NumaBlock> down_blocks_;

    float* s_input_fp32_;                      // [hidden_size]
    float* s_gate_output_;        // [routed_expert_num, intermediate_size]
    float* s_up_output_;          // [routed_expert_num, intermediate_size]
    float* s_down_output_;        // [routed_expert_num, hidden_size]
    uint8_t* s_gate_input_;                    // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* s_up_input_;                      // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    uint8_t* s_down_input_;       // [routed_expert_num, intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]

    size_t hidden_type_size;
    size_t hidden_blk_size;

    ggml_type gate_vec_dot_type;
    size_t gate_type_size;
    size_t gate_blk_size;
    size_t gate_vec_dot_type_size;
    size_t gate_vec_dot_blk_size;

    ggml_type up_vec_dot_type;
    size_t up_type_size;
    size_t up_blk_size;
    size_t up_vec_dot_type_size;
    size_t up_vec_dot_blk_size;

    ggml_type down_vec_dot_type;
    size_t down_type_size;
    size_t down_blk_size;
    size_t down_vec_dot_type_size;
    size_t down_vec_dot_blk_size;

    float* input_fp32_;        //[ group_max_len * hidden_size]
    uint8_t* gate_input_;      //[ group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* up_input_;        //[ group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    float* gate_output_;       //[ group_max_len * routed_expert_num * intermediate_size]
    float* up_output_;         //[ group_max_len * routed_expert_num * intermediate_size]
    uint8_t* down_input_;      //[ group_max_len * routed_expert_num * intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    float* down_output_;       //[ group_max_len * routed_expert_num * hidden_size]
    float* output_fp32_;       //[ group_max_len * hidden_size]

    void* m_gate_input_;      //[ group_max_len * routed_expert_num * hidden_size //* ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    void* m_up_input_;        //[ group_max_len * routed_expert_num * hidden_size //* ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    bool use_fp32_buffer_;
    bool fuse_decode_down_sum_ = false;
    std::vector<int> forward_many_expert_reorder_offset_;
    std::vector<int> forward_many_expert_selected_num_;
    std::vector<int> forward_many_active_experts_;
    std::vector<int> forward_many_token_expert_pos_;

    std::atomic<bool> sync_flag{false};

    uint64_t* decode_expert_ids_host_ = nullptr;
    int32_t* decode_expert_ids_i32_host_ = nullptr;
    float* decode_weights_host_ = nullptr;
    void* decode_input_host_ = nullptr;
    void* decode_output_host_ = nullptr;
    int* decode_bsz_host_ = nullptr;
    size_t decode_expert_ids_capacity_bytes_ = 0;
    size_t decode_expert_ids_i32_capacity_bytes_ = 0;
    size_t decode_weights_capacity_bytes_ = 0;
    size_t decode_input_capacity_bytes_ = 0;
    size_t decode_output_capacity_bytes_ = 0;
    bool profile_enabled_ = false;
    bool profile_detailed_enabled_ = false;
    uint64_t profile_forward_one_calls_ = 0;
    uint64_t profile_cpu_decode_calls_ = 0;
    uint64_t profile_cpu_decode_forward_calls_ = 0;
    uint64_t profile_cpu_decode_completed_calls_ = 0;
    uint64_t profile_cpu_decode_sync_completed_calls_ = 0;
    uint64_t profile_decode_down_sum_calls_ = 0;
    uint64_t profile_forward_one_breakdown_calls_ = 0;
    uint64_t profile_forward_calls_ = 0;
    uint64_t profile_forward_one_tokens_ = 0;
    uint64_t profile_forward_many_calls_ = 0;
    uint64_t profile_forward_many_tokens_ = 0;
    double profile_forward_one_ms_ = 0.0;
    double profile_cpu_decode_enqueue_ms_ = 0.0;
    double profile_cpu_decode_forward_ms_ = 0.0;
    double profile_decode_down_sum_ms_ = 0.0;
    double profile_forward_gate_up_ms_ = 0.0;
    double profile_forward_down_proj_ms_ = 0.0;
    double profile_forward_sum_ms_ = 0.0;
    double profile_decode_total_ms_ = 0.0;
    double profile_decode_pre_cpu_ms_ = 0.0;
    double profile_decode_cpu_ms_ = 0.0;
    double profile_decode_post_cpu_ms_ = 0.0;
    double profile_sync_decode_total_ms_ = 0.0;
    double profile_sync_decode_pre_cpu_ms_ = 0.0;
    double profile_sync_decode_cpu_ms_ = 0.0;
    double profile_sync_decode_post_cpu_ms_ = 0.0;
    std::vector<CpuDecodeParams*> decode_params_;
    friend void cpu_decode_profile_start_wrapper(void* args);
    friend void cpu_decode_enqueue_wrapper(void* args);
    friend void cpu_decode_forward_wrapper(void* args);
    friend void cpu_decode_wait_wrapper(void* args);
    friend void cpu_decode_profile_done_wrapper(void* args);
    std::mutex decode_task_mutex_;
    std::condition_variable decode_task_cv_;
    std::queue<std::shared_ptr<CpuDecodeTask>> decode_tasks_;
    std::thread decode_task_thread_;
    bool decode_task_thread_exit_ = false;


};




#endif
