#pragma once

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include <cuda_fp16.h>
#include <string>
#include <vector>

namespace tensorrt_llm
{
namespace common
{

class TensorDebugUtils
{
public:
    /// @brief Safely test if a device pointer is accessible by reading a few elements
    /// @param ptr Device pointer to test
    /// @param numElements Number of elements to try reading (default 4)
    /// @param elementSize Size of each element in bytes
    /// @param name Name of the pointer for logging
    /// @param stream CUDA stream to use
    /// @return true if accessible, false if not
    static bool testDevicePointerAccessibility(
        void const* ptr, size_t numElements, size_t elementSize, std::string const& name, cudaStream_t stream = 0);

    /// @brief Test a half-precision tensor pointer
    /// @param ptr Device pointer to half tensor
    /// @param numElements Number of elements to test (will be clamped to max 16)
    /// @param name Name of the tensor for logging
    /// @param stream CUDA stream to use
    /// @return true if accessible, false if not
    static bool testHalfTensorPtr(
        void const* ptr, size_t numElements, std::string const& name, cudaStream_t stream = 0);

    /// @brief Test all pointers in MHARunnerParams for accessibility
    /// @param params The MHARunnerParams to test
    /// @return true if all non-null pointers are accessible, false otherwise
    static bool validateMHARunnerParams(kernels::MHARunnerParams const& params);

private:
    static constexpr size_t MAX_TEST_ELEMENTS = 16;
};

} // namespace common
} // namespace tensorrt_llm


