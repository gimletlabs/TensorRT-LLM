/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// Anchors kernel entry points into libtensorrt_llm.so.
//
// kernels_src is linked into the shared library without WHOLE_ARCHIVE, so the linker keeps only
// the objects something in the library already references. These kernels are referenced solely by
// the torch ops in libth_common.so, so they end up exported from that library alone -- which makes
// calling them mean linking against libtorch. Embedders that drive the kernels directly from raw
// device pointers (Gimlet's TensorRT plugins) need them in libtensorrt_llm.so instead.
//
// Taking each instantiation's address is enough to pull its object out of the archive. Keep this
// list minimal: every entry grows the shared library.

#include "tensorrt_llm/common/attentionOp.h"
#include "tensorrt_llm/kernels/causalConv1d/causalConv1d.h"
#include "tensorrt_llm/kernels/mlaChunkedPrefill.cuh"
#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/moe/routing/customMoeRoutingKernels.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include <cuda_fp16.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace tensorrt_llm::kernels
{
namespace
{

template <typename InputT, typename OutputT>
void const* customMoeRoutingAnchor()
{
    return reinterpret_cast<void const*>(&invokeCustomMoeRouting<InputT, OutputT, int32_t, true>);
}

template <typename InputT, typename OutputT>
void const* customMoeRoutingAnchorNoSoftmax()
{
    return reinterpret_cast<void const*>(&invokeCustomMoeRouting<InputT, OutputT, int32_t, false>);
}

template <typename T>
void const* mxFp8QuantizationAnchor()
{
    return reinterpret_cast<void const*>(&invokeMxFP8Quantization<T>);
}

void const* moeAllToAllAnchors()
{
    // The MoE all-to-all launches are referenced only by the torch ops, so they otherwise ship
    // solely in libth_common.so.
    return reinterpret_cast<void const*>(&moe_comm::moe_a2a_dispatch_launch);
}

} // namespace

// Referenced by nothing; exists so the symbols above survive static linking.
void const* gimletKernelExportAnchors[] = {
    moeAllToAllAnchors(),
    reinterpret_cast<void const*>(&moe_comm::moe_a2a_combine_launch),
    reinterpret_cast<void const*>(&moe_comm::moe_a2a_prepare_dispatch_launch),
    reinterpret_cast<void const*>(&moe_comm::moe_a2a_prepare_combine_launch),
    reinterpret_cast<void const*>(&moe_comm::moe_a2a_sanitize_expert_ids_launch),
    reinterpret_cast<void const*>(&moe_comm::moeA2AGetTimeoutCycles),
    customMoeRoutingAnchor<float, float>(),
    customMoeRoutingAnchorNoSoftmax<float, float>(),
    mxFp8QuantizationAnchor<half>(),
#ifdef ENABLE_BF16
    reinterpret_cast<void const*>(&causal_conv1d::causal_conv1d_fwd_cuda<__nv_bfloat16, __nv_bfloat16>),
    reinterpret_cast<void const*>(&causal_conv1d::causal_conv1d_update_cuda<__nv_bfloat16, __nv_bfloat16>),
    customMoeRoutingAnchor<float, __nv_bfloat16>(),
    customMoeRoutingAnchor<__nv_bfloat16, float>(),
    customMoeRoutingAnchor<__nv_bfloat16, __nv_bfloat16>(),
    customMoeRoutingAnchorNoSoftmax<float, __nv_bfloat16>(),
    customMoeRoutingAnchorNoSoftmax<__nv_bfloat16, float>(),
    customMoeRoutingAnchorNoSoftmax<__nv_bfloat16, __nv_bfloat16>(),
    mxFp8QuantizationAnchor<__nv_bfloat16>(),
    reinterpret_cast<void const*>(&invokeMLALoadChunkedKV<__nv_bfloat16, __nv_bfloat16>),
    reinterpret_cast<void const*>(&invokeMergeAttnWithSoftmax<__nv_bfloat16>),
#endif
};

} // namespace tensorrt_llm::kernels

extern "C" void tllmResetCudaGraphGenerationUnsafeFlag()
{
    tensorrt_llm::common::op::resetCudaGraphGenerationUnsafeFlag();
}

extern "C" bool tllmCudaGraphGenerationUnsafeFlag()
{
    return tensorrt_llm::common::op::cudaGraphGenerationUnsafeFlag();
}
