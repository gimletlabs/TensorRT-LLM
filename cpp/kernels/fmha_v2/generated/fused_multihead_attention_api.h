/*
* SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
* AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include <cuda.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_cross_attention.h>
#include <tuple>

using Params_v1         = bert::Fused_multihead_attention_params_v1;
using Params_v2         = bert::Fused_multihead_attention_params_v2;
using Params_mhca       = bert::Fused_multihead_attention_params_mhca;
using Launch_params     = bert::Fused_multihead_attention_launch_params;

void run_fmha_v2_fp16_64_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_64_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_64_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_128_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_128_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_128_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_256_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_256_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_256_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_64_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_64_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_64_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_128_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_128_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_128_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_256_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_256_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_256_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_384_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_384_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_384_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_512_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_512_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_512_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_384_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_384_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_384_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_512_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_512_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_512_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_bf16_64_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_64_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_64_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_bf16_128_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_128_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_128_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_bf16_256_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_256_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_256_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_bf16_64_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_64_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_64_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_bf16_128_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_128_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_128_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_bf16_256_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_256_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_256_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_bf16_384_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_384_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_384_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_bf16_512_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_512_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_512_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_bf16_384_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_384_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_384_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_bf16_512_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_512_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_bf16_512_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_fp32_64_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_fp32_128_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_fp32_256_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_fp32_64_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_fp32_128_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_fp32_256_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_fp32_384_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_384_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_384_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_fp32_512_32_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_512_32_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_512_32_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_fp32_384_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_384_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_384_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_fp16_fp32_512_64_ldgsts_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_512_64_ldgsts_sm90_nl(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_fp16_fp32_512_64_ldgsts_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_256_S_qkv_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_qkv_40_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_qkv_48_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_qkv_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_160_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_192_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_256_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_256_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_q_kv_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_q_kv_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_40_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_48_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_160_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_192_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_256_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_256_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_q_kv_32_softmax_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_q_kv_64_softmax_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_softmax_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_softmax_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_qkv_32_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_qkv_40_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_qkv_48_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_qkv_64_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_160_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_192_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_256_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_32_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_40_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_48_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_64_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_160_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_192_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_256_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_qkv_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_qkv_40_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_qkv_48_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_qkv_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_160_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_192_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_256_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_256_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_q_kv_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_q_kv_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_40_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_48_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_160_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_192_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_256_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_256_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_q_kv_32_softmax_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_q_kv_64_softmax_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_softmax_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_softmax_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_qkv_32_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_qkv_40_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_qkv_48_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_qkv_64_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_160_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_192_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_256_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_32_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_40_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_48_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_64_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_160_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_192_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_256_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_40_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_48_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_80_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_96_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_104_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_160_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_192_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_256_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_128_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_256_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_kv_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_kv_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_kv_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_40_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_48_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_80_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_96_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_104_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_160_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_192_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_256_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_128_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_256_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_32_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_40_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_48_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_64_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_80_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_96_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_104_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_128_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_160_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_192_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_256_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_32_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_40_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_48_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_64_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_80_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_96_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_104_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_128_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_160_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_192_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_256_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_80_sage_64_64_256_output_bf16_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_128_sage_64_64_256_output_bf16_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_40_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_48_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_160_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_192_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_256_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_256_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_kv_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_kv_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_32_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_40_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_48_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_64_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_160_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_192_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_256_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_256_softcapping_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_kv_32_softmax_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_kv_64_softmax_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_softmax_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_softmax_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_32_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_40_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_48_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_64_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_160_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_192_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_256_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_32_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_40_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_48_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_64_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_160_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_192_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_256_alibi_tma_ws_sm90(Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_32_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_64_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_kv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_kv_32_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_kv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_64_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_72_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_128_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_16_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_16_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_16_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_32_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_40_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_40_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_40_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_48_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_48_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_48_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_64_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_160_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_160_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_160_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_192_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_192_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_192_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_16_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_16_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_16_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_32_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_40_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_40_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_40_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_48_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_48_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_48_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_64_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_72_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_80_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_80_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_80_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_96_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_96_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_96_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_104_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_104_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_104_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_160_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_160_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_160_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_192_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_192_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_192_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_softcapping_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_softcapping_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_softcapping_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_32_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_64_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_kv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_kv_32_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_kv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_64_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_72_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_128_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_16_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_16_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_16_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_32_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_40_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_40_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_40_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_48_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_48_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_48_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_64_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_160_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_160_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_160_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_16_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_16_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_16_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_40_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_40_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_40_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_48_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_48_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_48_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_72_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_80_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_80_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_80_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_96_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_96_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_96_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_104_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_104_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_104_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_160_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_160_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_160_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_192_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_192_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_192_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_softcapping_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_softcapping_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_softcapping_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_32_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_64_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_kv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_kv_32_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_kv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_64_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_72_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_128_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_16_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_32_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_40_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_48_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_48_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_48_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_64_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_160_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_192_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_192_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_192_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_16_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_32_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_40_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_48_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_48_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_48_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_64_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_72_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_80_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_96_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_96_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_96_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_104_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_104_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_104_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_160_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_192_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_192_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_192_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_softcapping_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_softcapping_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_softcapping_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_32_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_64_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_kv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_kv_32_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_kv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_64_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_72_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_128_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_16_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_32_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_40_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_48_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_48_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_48_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_64_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_160_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_16_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_40_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_48_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_48_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_48_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_72_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_80_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_96_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_96_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_96_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_104_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_104_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_104_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_160_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_192_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_192_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_192_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_softcapping_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_softcapping_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_softcapping_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_32_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_64_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_kv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_kv_32_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_kv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_64_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_72_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_128_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_16_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_16_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_16_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_32_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_40_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_40_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_40_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_48_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_48_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_48_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_64_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_160_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_160_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_160_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_192_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_192_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_192_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_16_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_16_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_16_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_32_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_40_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_40_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_40_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_48_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_48_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_48_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_64_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_72_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_80_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_80_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_80_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_96_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_96_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_96_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_104_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_104_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_104_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_160_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_160_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_160_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_192_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_192_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_192_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_softcapping_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_softcapping_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_softcapping_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_32_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_64_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_kv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_kv_32_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_kv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_64_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_72_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_128_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_16_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_16_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_16_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_32_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_40_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_40_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_40_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_48_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_48_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_48_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_64_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_160_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_160_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_160_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_16_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_16_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_16_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_40_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_40_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_40_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_48_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_48_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_48_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_72_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_80_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_80_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_80_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_96_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_96_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_96_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_104_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_104_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_104_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_160_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_160_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_160_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_192_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_192_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_192_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_softcapping_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_softcapping_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_softcapping_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm100(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm100_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm100_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm100(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm100_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm100_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm100(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm100_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm100_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_32_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_64_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_kv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_kv_32_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_kv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_64_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_72_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_128_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_16_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_32_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_40_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_48_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_48_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_48_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_64_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_160_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_192_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_192_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_192_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_16_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_16_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_16_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_32_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_32_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_32_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_40_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_40_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_40_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_48_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_48_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_48_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_64_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_64_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_64_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_72_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_72_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_72_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_80_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_80_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_80_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_96_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_96_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_96_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_104_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_104_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_104_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_160_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_160_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_160_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_192_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_192_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_192_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_softcapping_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_softcapping_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_softcapping_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_softcapping_sm80_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_softcapping_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_32_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_64_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_kv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_kv_32_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_kv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_64_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_72_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_128_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_16_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_16_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_16_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_32_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_40_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_40_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_40_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_48_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_48_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_48_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_64_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_160_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_160_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_160_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_192_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_192_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_192_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_16_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_16_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_16_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_32_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_32_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_32_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_40_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_40_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_40_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_48_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_48_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_48_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_64_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_64_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_64_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_72_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_72_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_72_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_80_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_80_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_80_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_96_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_96_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_96_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_104_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_104_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_104_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_160_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_160_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_160_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_192_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_192_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_192_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_softcapping_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_softcapping_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_softcapping_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_softcapping_sm86_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_softcapping_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_32_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_64_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_kv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_kv_32_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_kv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_64_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_72_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_128_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_16_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_16_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_16_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_32_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_40_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_40_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_40_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_48_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_48_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_48_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_64_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_160_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_160_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_160_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_192_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_192_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_192_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_16_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_16_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_16_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_32_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_32_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_32_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_40_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_40_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_40_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_48_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_48_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_48_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_64_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_64_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_64_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_72_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_72_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_72_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_80_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_80_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_80_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_96_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_96_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_96_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_104_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_104_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_104_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_160_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_160_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_160_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_192_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_192_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_192_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_softcapping_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_softcapping_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_softcapping_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_softcapping_sm89_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_softcapping_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm90_nl(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm90_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm80(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm80_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm80_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm86(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm86_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm86_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm89(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm89_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm89_get_max_heads_per_wave(int*);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm90(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm90_nl_tiled(const Params_v2 &params, const Launch_params &launch_params, cudaStream_t stream);
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm90_get_max_heads_per_wave(int*);

inline void run_fmha_v1(Params_v1 &params,
                        const Launch_params &launch_params,
                        Data_type data_type,
                        Data_type output_data_type,
                        int sm,
                        cudaStream_t stream=0){
const size_t s                 = params.s;
const size_t b                 = params.b;
const size_t d                 = params.d;
const bool force_unroll        = launch_params.force_unroll;
const bool ignore_b1opt        = launch_params.ignore_b1opt;

const bool use_flash_attention = false;

if( false ) {}
else {
    assert(false && "Unsupported config.");
}

}

// Note: transitioning to moving kernel launch parameters into launch_params to reduce the
// occurrences the interface needs to be modified
inline void run_fmha_v2(Params_v2 &params,
                        const Launch_params &launch_params,
                        Data_type data_type,
                        Data_type output_data_type,
                        int sm,
                        cudaStream_t stream=0) {

const size_t s = params.s;
const size_t b = params.b;
const size_t h = params.h;
const size_t d = params.d;
const size_t dv = params.dv;
const size_t sage_block_size_q = params.sage.q.block_size;
const size_t sage_block_size_k = params.sage.k.block_size;
const size_t sage_block_size_v = params.sage.v.block_size;

const bool interleaved                       = launch_params.interleaved;
const bool force_unroll                      = launch_params.force_unroll;
const bool ignore_b1opt                      = launch_params.ignore_b1opt;
const bool force_fp32_acc                    = launch_params.force_fp32_acc;
const bool warp_specialization               = launch_params.warp_specialization;
const bool use_tma                           = launch_params.use_tma;
const bool use_flash_attention               = launch_params.flash_attention;
const bool enable_attn_logit_softcapping     = launch_params.enable_attn_logit_softcapping;
const int  attention_input_layout            = static_cast<int>(launch_params.attention_input_layout);
// tiled variant uses ldgsts
const bool  use_tiled            = launch_params.use_granular_tiling;

if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 64 && d == 32 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_64_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_64_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 128 && d == 32 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_128_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_128_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 256 && d == 32 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_256_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_256_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 64 && d == 64 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_64_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_64_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 128 && d == 64 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_128_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_128_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 256 && d == 64 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_256_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_256_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 384 && d == 32 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_384_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_384_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 512 && d == 32 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_512_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_512_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 384 && d == 64 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_384_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_384_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 512 && d == 64 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_512_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_512_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && s == 64 && d == 32 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_bf16_64_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_bf16_64_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && s == 128 && d == 32 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_bf16_128_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_bf16_128_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && s == 256 && d == 32 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_bf16_256_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_bf16_256_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && s == 64 && d == 64 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_bf16_64_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_bf16_64_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && s == 128 && d == 64 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_bf16_128_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_bf16_128_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && s == 256 && d == 64 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_bf16_256_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_bf16_256_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && s == 384 && d == 32 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_bf16_384_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_bf16_384_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && s == 512 && d == 32 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_bf16_512_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_bf16_512_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && s == 384 && d == 64 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_bf16_384_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_bf16_384_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && s == 512 && d == 64 && sm == 90
    && !use_flash_attention && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_bf16_512_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_bf16_512_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 64 && d == 32 && sm == 90
    && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_fp32_64_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 128 && d == 32 && sm == 90
    && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_fp32_128_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 256 && d == 32 && sm == 90
    && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_fp32_256_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 64 && d == 64 && sm == 90
    && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_fp32_64_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 128 && d == 64 && sm == 90
    && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_fp32_128_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 256 && d == 64 && sm == 90
    && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_fp32_256_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 384 && d == 32 && sm == 90
    && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_fp32_384_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_fp32_384_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 512 && d == 32 && sm == 90
    && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_fp32_512_32_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_fp32_512_32_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 384 && d == 64 && sm == 90
    && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_fp32_384_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_fp32_384_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && s == 512 && d == 64 && sm == 90
    && attention_input_layout == 0 && !interleaved && !use_tma && !warp_specialization && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
        run_fmha_v2_fp16_fp32_512_64_ldgsts_sm90(params, launch_params, stream);
    } else {
        run_fmha_v2_fp16_fp32_512_64_ldgsts_sm90_nl(params, launch_params, stream);
    }

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_qkv_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_qkv_40_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_qkv_48_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_qkv_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_160_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_192_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_256_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_256_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_q_kv_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_q_kv_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_40_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_48_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_160_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_192_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_256_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_256_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr != nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_q_kv_32_softmax_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr != nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_q_kv_64_softmax_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr != nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_softmax_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr != nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_softmax_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_qkv_32_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_qkv_40_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_qkv_48_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_qkv_64_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_160_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_192_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_256_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_32_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_40_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_48_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_256_S_q_paged_kv_64_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_160_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_192_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && !force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_256_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_qkv_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_qkv_40_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_qkv_48_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_qkv_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_160_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_192_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_256_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_256_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_q_kv_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_q_kv_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_40_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_48_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_160_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_192_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_256_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_256_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr != nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_q_kv_32_softmax_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr != nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_q_kv_64_softmax_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr != nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_softmax_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr != nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_softmax_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_qkv_32_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_qkv_40_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_qkv_48_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_qkv_64_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_160_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_192_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_256_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_32_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_40_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_48_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_256_S_q_paged_kv_64_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_160_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_192_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_256_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 32 && dv == 32 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 40 && dv == 40 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_40_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 48 && dv == 48 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_48_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 64 && dv == 64 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 80 && dv == 80 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_80_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 96 && dv == 96 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_96_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 104 && dv == 104 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_104_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 160 && dv == 160 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_160_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 192 && dv == 192 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_192_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_256_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_128_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_256_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 32 && dv == 32 && sm == 90
    && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_kv_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 64 && dv == 64 && sm == 90
    && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_kv_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_kv_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 32 && dv == 32 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 40 && dv == 40 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_40_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 48 && dv == 48 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_48_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 64 && dv == 64 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 80 && dv == 80 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_80_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 96 && dv == 96 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_96_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 104 && dv == 104 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_104_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 160 && dv == 160 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_160_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 192 && dv == 192 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_192_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_256_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_128_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_256_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 32 && dv == 32 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_32_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 40 && dv == 40 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_40_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 48 && dv == 48 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_48_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 64 && dv == 64 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_64_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 80 && dv == 80 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_80_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 96 && dv == 96 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_96_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 104 && dv == 104 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_104_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_128_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 160 && dv == 160 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_160_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 192 && dv == 192 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_192_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_qkv_256_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 32 && dv == 32 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_32_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 40 && dv == 40 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_40_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 48 && dv == 48 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_48_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 64 && dv == 64 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_64_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 80 && dv == 80 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_80_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 96 && dv == 96 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_96_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 104 && dv == 104 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_104_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_q_paged_kv_128_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 160 && dv == 160 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_160_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 192 && dv == 192 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_192_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_E4M3 && output_data_type == DATA_TYPE_E4M3 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_128_S_q_paged_kv_256_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 64 && sage_block_size_k == 64 && sage_block_size_v == 256 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_80_sage_64_64_256_output_bf16_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 64 && sage_block_size_k == 64 && sage_block_size_v == 256 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_e4m3_64_256_S_qkv_128_sage_64_64_256_output_bf16_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_40_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_48_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_160_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_192_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_256_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_256_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_kv_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_kv_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr == nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_32_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_40_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_48_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_64_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_160_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_192_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_256_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_256_softcapping_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr != nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_kv_32_softmax_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr != nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_kv_64_softmax_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr != nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_softmax_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 1 && !interleaved && use_tma && warp_specialization && force_fp32_acc && !params.has_alibi && params.softmax_stats_ptr != nullptr && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_softmax_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_32_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_40_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_48_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_qkv_64_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_160_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_192_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 0 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_256_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_32_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_40_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_48_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_256_S_q_paged_kv_64_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_160_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_192_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 2 && !interleaved && use_tma && warp_specialization && force_fp32_acc && params.has_alibi && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_256_alibi_tma_ws_sm90(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 89
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_32_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 89
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_64_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 89
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_kv_32_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_64_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_72_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_128_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_16_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_32_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_40_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_48_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_64_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_160_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_192_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_16_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_32_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_40_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_48_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_64_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_72_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_80_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_96_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_104_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_160_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_192_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_softcapping_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_softcapping_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_softcapping_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 89
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_32_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 89
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_64_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 89
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 89
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_kv_32_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_64_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_72_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_128_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_16_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_32_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_40_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_48_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_64_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_160_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_16_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_40_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_48_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_72_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_80_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_96_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_104_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_160_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_192_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_softcapping_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_softcapping_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 89
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_softcapping_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 80
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_32_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 80
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_64_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 80
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_kv_32_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_64_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_72_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_128_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_16_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_32_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_40_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_48_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_64_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_160_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_192_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_16_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_32_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_40_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_48_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_64_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_72_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_80_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_96_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_104_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_160_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_192_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_softcapping_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_softcapping_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_softcapping_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 80
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_32_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 80
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_64_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 80
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 80
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_kv_32_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_64_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_72_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_128_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_16_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_32_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_40_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_48_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_64_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_160_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_16_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_40_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_48_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_72_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_80_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_96_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_104_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_160_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_192_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_softcapping_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_softcapping_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 80
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_softcapping_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 86
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_32_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 86
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_kv_64_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 86
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_72_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_kv_32_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_64_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_72_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_kv_128_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_16_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_32_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_40_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_48_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_q_paged_kv_64_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_72_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_80_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_96_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_104_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_160_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_192_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_16_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_q_paged_kv_32_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_40_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_48_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_64_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_72_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_80_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_96_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_104_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_160_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_192_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_softcapping_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_256_softcapping_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_q_paged_kv_128_softcapping_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_q_paged_kv_256_softcapping_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 86
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_32_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 86
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_kv_64_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 86
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_72_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 86
    && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_kv_128_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_kv_32_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_64_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_72_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_kv_128_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_16_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_32_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_40_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_48_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_q_paged_kv_64_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_72_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_80_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_96_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_104_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_160_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_16_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_40_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_48_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_72_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_80_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_96_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_104_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_160_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_192_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_softcapping_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_256_softcapping_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_128_softcapping_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 86
    && !use_tiled && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_q_paged_kv_256_softcapping_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_32_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_40_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_48_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_128_128_S_qkv_64_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_72_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_80_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_96_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_104_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_160_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_192_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_16_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_64_S_qkv_32_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_40_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_48_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_64_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_72_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_80_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_96_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_104_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_160_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_192_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_128_softcapping_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_64_128_S_qkv_256_softcapping_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_32_S_qkv_128_softcapping_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && !force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_64_16_S_qkv_256_softcapping_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_16_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_32_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_40_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_48_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_128_128_S_qkv_64_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_72_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_80_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_96_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_160_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 16 && dv == 16 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_16_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 32 && dv == 32 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_64_S_qkv_32_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 40 && dv == 40 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_40_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 48 && dv == 48 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_48_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 64 && dv == 64 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_64_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 72 && dv == 72 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_72_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 80 && dv == 80 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_80_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 96 && dv == 96 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_96_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 104 && dv == 104 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_104_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 160 && dv == 160 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_160_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 192 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_192_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_128_softcapping_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 90
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_256_softcapping_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 128 && dv == 128 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_32_S_qkv_128_softcapping_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 256 && dv == 256 && sm == 90
    && !use_tiled && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_bf16_64_16_S_qkv_256_softcapping_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 128 && sm == 100
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm100_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 128 && sm == 100
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm100_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 576 && dv == 512 && sm == 100
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm100_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 80
    && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_32_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 80
    && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_64_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 80
    && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 80
    && !use_tiled && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_kv_32_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 80
    && !use_tiled && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_64_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 80
    && !use_tiled && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_72_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_128_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_16_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_32_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_40_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_48_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_64_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_160_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_192_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_16_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_32_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_40_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_48_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_64_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_72_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_80_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_96_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_104_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_160_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_192_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_softcapping_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_softcapping_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 80
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_softcapping_sm80_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 86
    && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_32_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 86
    && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_64_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 86
    && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 86
    && !use_tiled && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_kv_32_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 86
    && !use_tiled && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_64_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 86
    && !use_tiled && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_72_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_128_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_16_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_32_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_40_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_48_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_64_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_160_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_192_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_16_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_32_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_40_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_48_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_64_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_72_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_80_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_96_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_104_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_160_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_192_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_softcapping_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_softcapping_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 86
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_softcapping_sm86_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 89
    && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_32_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 89
    && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_kv_64_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 89
    && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_72_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_kv_128_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 89
    && !use_tiled && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_kv_32_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 89
    && !use_tiled && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_64_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 89
    && !use_tiled && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_72_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && attention_input_layout == 1 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_kv_128_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_16_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_32_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_40_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_48_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_q_paged_kv_64_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_72_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_80_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_96_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_104_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_160_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_192_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_16_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_q_paged_kv_32_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_40_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_48_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_64_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_72_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_80_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_96_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_104_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_160_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_192_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_128_softcapping_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_q_paged_kv_256_softcapping_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_q_paged_kv_128_softcapping_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 89
    && !use_tiled && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_q_paged_kv_256_softcapping_sm89_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_16_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_32_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_40_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_48_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_128_128_S_qkv_64_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_72_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_80_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_96_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_104_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_160_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_192_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 16 && dv == 16 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_16_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 32 && dv == 32 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_64_S_qkv_32_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 40 && dv == 40 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_40_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 48 && dv == 48 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_48_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 64 && dv == 64 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_64_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 72 && dv == 72 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_72_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 80 && dv == 80 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_80_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 96 && dv == 96 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_96_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 104 && dv == 104 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_104_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 160 && dv == 160 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_160_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 192 && dv == 192 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_192_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_128_softcapping_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_fp16_fp32_64_128_S_qkv_256_softcapping_sm90_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 128 && dv == 128 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_32_S_qkv_128_softcapping_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_FP16 && output_data_type == DATA_TYPE_FP16 && d == 256 && dv == 256 && sm == 90
    && !use_tiled && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max ) {

    run_fmha_v2_flash_attention_fp16_fp32_64_16_S_qkv_256_softcapping_sm90_nl(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 128 && sm == 80
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 128 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 576 && dv == 512 && sm == 80
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm80_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 128 && sm == 86
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 128 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 576 && dv == 512 && sm == 86
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm86_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 128 && sm == 89
    && use_flash_attention && attention_input_layout == 0 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_qkv_192x128_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 192 && dv == 128 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_192x128_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 576 && dv == 512 && sm == 89
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm89_nl_tiled(params, launch_params, stream);

} else if( data_type == DATA_TYPE_BF16 && output_data_type == DATA_TYPE_BF16 && d == 576 && dv == 512 && sm == 90
    && use_flash_attention && attention_input_layout == 2 && !interleaved && !warp_specialization && !use_tma && force_fp32_acc && !enable_attn_logit_softcapping && sage_block_size_q == 0 && sage_block_size_k == 0 && sage_block_size_v == 0 && !params.use_int8_scale_max  && use_tiled) {

    run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_576x512_sm90_nl_tiled(params, launch_params, stream);

} 
else {
    assert(false && "Unsupported config.");
}

}

#if false // fmhca api header

inline void run_fmhca(Params_mhca &params,
                      const Launch_params &launch_params,
                      Data_type data_type,
                      int sm,
                      cudaStream_t stream=0) {

const size_t s_kv   = params.s;
const size_t b      = params.b;
const size_t d      = params.d_padded;

const bool interleaved  = launch_params.interleaved;
const bool force_unroll = launch_params.force_unroll;
const bool ignore_b1opt = launch_params.ignore_b1opt;

if( false ) {}
else {
    assert(false && "Unsupported config");
}

}

#endif // fmhca api header

inline std::tuple<size_t, size_t, size_t> get_warps(Launch_params& launch_params,
                                                    int sm,
                                                    Data_type data_type,
                                                    size_t s,
                                                    size_t b,
                                                    size_t d,
                                                    int version) {
    size_t warps_m, warps_n, warps_k = 1;
    const bool interleaved           = launch_params.interleaved;
    const bool use_tma               = launch_params.use_tma;
    const bool force_unroll          = launch_params.force_unroll;
    const bool ignore_b1opt          = launch_params.ignore_b1opt;
    const bool use_flash_attention   = launch_params.flash_attention;
    // tiled variant uses ldgsts
    const bool use_tiled             = launch_params.use_granular_tiling;
    const bool warp_specialization   = launch_params.warp_specialization;

if( data_type == DATA_TYPE_FP16 && s == 64 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 128 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 256 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 64 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 128 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 256 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 384 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 512 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 384 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 512 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_BF16 && s == 64 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_BF16 && s == 128 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_BF16 && s == 256 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_BF16 && s == 64 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_BF16 && s == 128 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_BF16 && s == 256 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_BF16 && s == 384 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_BF16 && s == 512 && d == 32 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_BF16 && s == 384 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_BF16 && s == 512 && d == 64 && sm == 90 && !use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 64 && d == 32 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 128 && d == 32 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 256 && d == 32 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 64 && d == 64 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 128 && d == 64 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 256 && d == 64 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 1;
    } else {
      warps_m = 4;
      warps_n = 1;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 384 && d == 32 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 512 && d == 32 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 384 && d == 64 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_FP16 && s == 512 && d == 64 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    if (!1 || (!force_unroll && (ignore_b1opt || b > 1))) {
      warps_m = 4;
      warps_n = 2;
    } else {
      warps_m = 4;
      warps_n = 2;
    }
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 90 && use_flash_attention && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 32 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 40 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 48 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 64 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 80 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 96 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 104 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 160 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 192 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 256 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 256 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 32 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 64 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 32 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 40 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 48 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 64 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 80 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 96 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 104 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 160 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 192 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 256 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 256 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 32 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 40 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 48 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 64 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 80 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 96 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 104 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 160 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 192 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 256 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 32 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 40 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 48 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 64 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 80 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 96 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 104 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 160 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 192 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_E4M3 && d == 256 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_tma && warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 16 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 32 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 40 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 48 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 64 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 72 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 80 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 96 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 104 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 160 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 128 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 256 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 100 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 100 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 576 && sm == 100 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 80 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 86 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 89 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 16 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 32 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 40 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 48 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 64 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 72 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 80 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 96 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 104 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 160 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 192 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 128 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_FP16 && d == 256 && sm == 90 && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 576 && sm == 80 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 576 && sm == 86 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 192 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 576 && sm == 89 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else if( data_type == DATA_TYPE_BF16 && d == 576 && sm == 90 && use_flash_attention && !use_tma && !warp_specialization 
    && version == 2 ) {
    warps_m = 4;
    warps_n = 1;
} else {
	assert(false && "Unsupported config");
}

    return std::make_tuple(warps_m, warps_n, warps_k);
}

// The constant is defined in "setup.py".
constexpr int MAX_STGS_PER_LOOP = 4;

// The number of CTAs and threads per CTA to launch the kernel.
inline void get_grid_size(int &heads_per_wave,
                          int &ctas_per_head,
                          int sm,
                          Data_type data_type,
                          size_t b,
                          size_t s,
                          size_t h,
                          size_t d,
                          bool use_multi_ctas,
                          int version) {

    // Determine the number of CTAs per head (kernel constant).
    int max_heads_per_wave = 0;
    ctas_per_head = 1;
    heads_per_wave = b*h;


    // Adjust the number of heads per wave.
    if( heads_per_wave > max_heads_per_wave ) {
        heads_per_wave = max_heads_per_wave;
    }
}

