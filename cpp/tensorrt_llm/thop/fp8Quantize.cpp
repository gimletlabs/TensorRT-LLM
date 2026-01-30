/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

using Fp8BlockScaleGemmRunnerPtr
    = std::unique_ptr<tensorrt_llm::kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunnerInterface>;

void fp8_quantize_1x128_out(
    at::Tensor const& self, at::Tensor& quantized_tensor, at::Tensor& scale_tensor, bool use_ue8m0)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);
    CHECK_TH_CUDA(quantized_tensor);
    CHECK_CONTIGUOUS(quantized_tensor);
    CHECK_TH_CUDA(scale_tensor);
    CHECK_CONTIGUOUS(scale_tensor);

    TORCH_CHECK(self.scalar_type() == at::ScalarType::BFloat16, "Input matrix dtype must be BF16.");
    TORCH_CHECK(self.dim() == 2, "input must be a 2D matrix");

    auto const m = self.sizes()[0];
    auto const n = self.sizes()[1];

    TORCH_CHECK(quantized_tensor.dim() == 2, "quantized_tensor must be a 2D matrix");
    TORCH_CHECK(scale_tensor.dim() == 1, "scale_tensor must be a 1D tensor");

    TORCH_CHECK(m <= std::numeric_limits<int32_t>::max(), "M must be within int32");
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "N must be within int32");

    // required by the sm90 fp8_block_scaling gemm kernel
    TORCH_CHECK(n % 16 == 0, "self.sizes()[1] must be a multiple of 16, but got ", n);

    auto mGemmRunner = tensorrt_llm::kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16,
        __nv_fp8_e4m3, __nv_bfloat16>();

    auto const m_padded = (m + 4 - 1) / 4 * 4;

    // Use preallocated tensors - they should have the correct size
    TORCH_CHECK(quantized_tensor.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "quantized_tensor dtype must be Float8_e4m3fn");
    TORCH_CHECK(quantized_tensor.sizes()[0] >= m_padded && quantized_tensor.sizes()[1] >= n,
        "quantized_tensor size mismatch. Expected at least [", m_padded, ", ", n, "], got ", quantized_tensor.sizes());

    int64_t scaleSizeInBytes = mGemmRunner.getActScaleSize(m, n); // 128-byte aligned
    int64_t elementSize = scaleSizeInBytes / torch::elementSize(FP8_BLOCK_SCALING_SF_DTYPE);

    // For sm100, we need a larger scale tensor to accommodate the reshaping
    int64_t required_scale_size = elementSize;
    if (tensorrt_llm::common::isSM100Family())
    {
        auto const num_n_blocks = (n + 127) / 128;
        auto const act_scal_elesize = num_n_blocks * m_padded;
        required_scale_size = act_scal_elesize;
    }

    TORCH_CHECK(scale_tensor.scalar_type() == FP8_BLOCK_SCALING_SF_DTYPE, "scale_tensor dtype mismatch");
    TORCH_CHECK(scale_tensor.numel() >= required_scale_size, "scale_tensor size mismatch. Expected at least ",
        required_scale_size, " elements, got ", scale_tensor.numel());

    __nv_fp8_e4m3* act_buffer = reinterpret_cast<__nv_fp8_e4m3*>(quantized_tensor.data_ptr());
    float* act_scale_buffer = reinterpret_cast<float*>(scale_tensor.data_ptr());

    auto stream = at::cuda::getCurrentCUDAStream(self.get_device());

    mGemmRunner.fp8CS1x128(
        act_buffer, act_scale_buffer, reinterpret_cast<__nv_bfloat16 const*>(self.data_ptr()), n, m, stream, use_ue8m0);

    // Post-process the scale tensor for sm100 gemm/moe kernel
    // Note: The caller should provide a scale_tensor large enough for the computation.
    // For sm100, the scale tensor will be reshaped by the caller if needed.
    // The data is written into the preallocated tensor, and the caller can create views as needed.
}

std::tuple<at::Tensor, at::Tensor> fp8_quantize_1x128(at::Tensor const& self, bool use_ue8m0 = false)
{
    CHECK_TH_CUDA(self);

    TORCH_CHECK(self.scalar_type() == at::ScalarType::BFloat16, "Input matrix dtype must be BF16.");
    TORCH_CHECK(self.dim() == 2, "input must be a 2D matrix");

    auto const m = self.sizes()[0];
    auto const n = self.sizes()[1];

    auto mGemmRunner = tensorrt_llm::kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16,
        __nv_fp8_e4m3, __nv_bfloat16>();

    auto const m_padded = (m + 4 - 1) / 4 * 4;

    // Allocate quantized tensor: row major, add padding required by the sm90 fp8_block_scaling gemm kernel
    at::Tensor quantized_tensor = at::detail::empty_cuda(
        {m_padded, n}, at::ScalarType::Float8_e4m3fn, self.device(), /* stride */ std::nullopt);

    // Allocate scale tensor
    int64_t scaleSizeInBytes = mGemmRunner.getActScaleSize(m, n); // 128-byte aligned
    int64_t elementSize = scaleSizeInBytes / torch::elementSize(FP8_BLOCK_SCALING_SF_DTYPE);

    // For sm100, we need a larger scale tensor to accommodate the reshaping
    int64_t scale_tensor_size = elementSize;
    if (tensorrt_llm::common::isSM100Family())
    {
        auto const num_n_blocks = (n + 127) / 128;
        auto const act_scal_elesize = num_n_blocks * m_padded;
        scale_tensor_size = act_scal_elesize;
    }

    at::Tensor scale_tensor = at::detail::empty_cuda(
        {scale_tensor_size}, FP8_BLOCK_SCALING_SF_DTYPE, self.device(), /* stride */ std::nullopt); // 1D tensor

    // Call the _out version
    fp8_quantize_1x128_out(self, quantized_tensor, scale_tensor, use_ue8m0);

    // Post-process the scale tensor for sm100 gemm/moe kernel
    at::Tensor final_scale_tensor = scale_tensor;
    if (tensorrt_llm::common::isSM100Family())
    {
        auto const num_n_blocks = (n + 127) / 128;
        auto const act_scal_elesize = num_n_blocks * m_padded;
        // scale_tensor = scale_tensor[0:num_n_blocks, 0:m] // no 4-element alignment in blackwell
        // TODO: This is a hack to use sm90 quantize kernel for sm100; ideally we should have a separate quantize kernel
        // for sm100.
        final_scale_tensor
            = scale_tensor.slice(0, 0, act_scal_elesize).view({num_n_blocks, m_padded}).slice(1, 0, m).contiguous();
    }
    else
    {
        // For non-sm100, slice to the right size if needed
        if (scale_tensor.numel() > elementSize)
        {
            final_scale_tensor = scale_tensor.slice(0, 0, elementSize);
        }
    }

    // Return the appropriately sliced tensors
    return {quantized_tensor.slice(0, 0, m), final_scale_tensor};
}

void fp8_batched_quantize_1x128_permute102_out(
    at::Tensor const& self, at::Tensor& quantized_tensor, at::Tensor& scale_tensor)
{
    CHECK_TH_CUDA(self);
    CHECK_TH_CUDA(quantized_tensor);
    CHECK_TH_CUDA(scale_tensor);

    TORCH_CHECK(self.scalar_type() == at::ScalarType::BFloat16, "Input matrix dtype must be BF16.");
    TORCH_CHECK(self.dim() == 3, "input must be a 3D tensor");

    // [seq, num_heads, qk_nope_head_dim]
    // [m, b, n]
    auto const m = self.sizes()[0];
    auto const b = self.sizes()[1];
    auto const n = self.sizes()[2];

    auto const lda = self.strides()[1];
    TORCH_CHECK(self.strides()[2] == 1, "Last stride of self must be 1, but got ", self.strides()[2]);
    TORCH_CHECK(self.strides()[0] == lda * b, "First stride of self is expected to be ", lda * b, ", but got ",
        self.strides()[0]);

    TORCH_CHECK(b <= std::numeric_limits<int32_t>::max(), "B must be within int32");
    TORCH_CHECK(m <= std::numeric_limits<int32_t>::max(), "M must be within int32");
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "N must be within int32");
    // required by the sm90 fp8_block_scaling gemm/bmm kernel
    TORCH_CHECK(n % 16 == 0, "self.sizes()[2] must be a multiple of 16, but got ", n);

    auto mGemmRunner = tensorrt_llm::kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16,
        __nv_fp8_e4m3, __nv_bfloat16>();

    auto const m_padded = (m + 4 - 1) / 4 * 4;

    // Check preallocated tensor sizes
    TORCH_CHECK(quantized_tensor.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "quantized_tensor dtype must be Float8_e4m3fn");
    TORCH_CHECK(quantized_tensor.numel() >= b * m_padded * n, "quantized_tensor size mismatch. Expected at least ",
        b * m_padded * n, " elements, got ", quantized_tensor.numel());

    int64_t scaleSizeInBytes = mGemmRunner.getActScaleSize(m, b * n);
    int64_t elementSize = scaleSizeInBytes / torch::elementSize(FP8_BLOCK_SCALING_SF_DTYPE);
    int m_4_align = (m + 3) / 4 * 4;
    int64_t expected_scale_size = b * m_4_align * (elementSize / b / m_4_align);

    TORCH_CHECK(scale_tensor.scalar_type() == FP8_BLOCK_SCALING_SF_DTYPE, "scale_tensor dtype mismatch");
    TORCH_CHECK(scale_tensor.numel() >= expected_scale_size, "scale_tensor size mismatch. Expected at least ",
        expected_scale_size, " elements, got ", scale_tensor.numel());

    // Reshape quantized_tensor to the expected shape for computation
    at::Tensor valueE4M3 = quantized_tensor.slice(0, 0, b * m_padded * n);
    at::Tensor scaleFP8SF
        = scale_tensor.slice(0, 0, expected_scale_size).view({b, m_4_align, elementSize / b / m_4_align});

    __nv_fp8_e4m3* act_buffer = reinterpret_cast<__nv_fp8_e4m3*>(valueE4M3.data_ptr());
    float* act_scale_buffer = reinterpret_cast<float*>(scaleFP8SF.data_ptr());

    auto stream = at::cuda::getCurrentCUDAStream(self.get_device());

    auto* output_buffer = reinterpret_cast<__nv_bfloat16 const*>(self.data_ptr());
    mGemmRunner.fp8CS1x128Reshape(act_buffer, act_scale_buffer, output_buffer, n, b, m, lda, stream);

    // Data is written into scale_tensor via scaleFP8SF (which is a view of scale_tensor)
    // The caller should provide scale_tensor with shape {b, m_4_align, elementSize / b / m_4_align}
}

std::tuple<at::Tensor, at::Tensor> fp8_batched_quantize_1x128_permute102(at::Tensor const& self)
{
    CHECK_TH_CUDA(self);

    TORCH_CHECK(self.scalar_type() == at::ScalarType::BFloat16, "Input matrix dtype must be BF16.");
    TORCH_CHECK(self.dim() == 3, "input must be a 3D tensor");

    // [seq, num_heads, qk_nope_head_dim]
    // [m, b, n]
    auto const m = self.sizes()[0];
    auto const b = self.sizes()[1];
    auto const n = self.sizes()[2];

    TORCH_CHECK(b <= std::numeric_limits<int32_t>::max(), "B must be within int32");
    TORCH_CHECK(m <= std::numeric_limits<int32_t>::max(), "M must be within int32");
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "N must be within int32");
    // required by the sm90 fp8_block_scaling gemm/bmm kernel
    TORCH_CHECK(n % 16 == 0, "self.sizes()[2] must be a multiple of 16, but got ", n);

    auto mGemmRunner = tensorrt_llm::kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16,
        __nv_fp8_e4m3, __nv_bfloat16>();

    auto const m_padded = (m + 4 - 1) / 4 * 4;

    // Allocate quantized tensor
    // input: [b, m, n]
    // apply 102 permute
    at::Tensor quantized_tensor = at::detail::empty_cuda(
        {b * m_padded * n}, at::ScalarType::Float8_e4m3fn, self.device(), /* stride */ std::nullopt);

    // Allocate scale tensor
    int64_t scaleSizeInBytes = mGemmRunner.getActScaleSize(m, b * n);
    int64_t elementSize = scaleSizeInBytes / torch::elementSize(FP8_BLOCK_SCALING_SF_DTYPE);
    int m_4_align = (m + 3) / 4 * 4;
    at::Tensor scale_tensor = at::detail::empty_cuda({b, m_4_align, elementSize / b / m_4_align},
        FP8_BLOCK_SCALING_SF_DTYPE, self.device(), /* stride */ std::nullopt);

    // Call the _out version
    fp8_batched_quantize_1x128_permute102_out(self, quantized_tensor, scale_tensor);

    // Return the appropriately sliced tensors
    return {quantized_tensor.slice(0, 0, b * m * n).view({b, m, n}), scale_tensor};
}
} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fp8_quantize_1x128(Tensor input, bool use_ue8m0=False) -> (Tensor, Tensor)");
    m.def("fp8_batched_quantize_1x128_permute102(Tensor input) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp8_quantize_1x128", &tensorrt_llm::torch_ext::fp8_quantize_1x128);
    m.impl("fp8_batched_quantize_1x128_permute102", &tensorrt_llm::torch_ext::fp8_batched_quantize_1x128_permute102);
}
