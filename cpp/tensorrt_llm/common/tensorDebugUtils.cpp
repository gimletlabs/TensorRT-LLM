#include "tensorDebugUtils.h"
#include <algorithm>

namespace tensorrt_llm
{
namespace common
{

bool TensorDebugUtils::testDevicePointerAccessibility(
    void const* ptr, size_t numElements, size_t elementSize, std::string const& name, cudaStream_t stream)
{
    if (ptr == nullptr)
    {
        TLLM_LOG_INFO("Pointer %s is NULL - skipping test", name.c_str());
        return true; // NULL pointers are "safe" in the sense they won't cause illegal access
    }

    if (numElements == 0)
    {
        TLLM_LOG_INFO("Pointer %s has 0 elements - skipping test", name.c_str());
        return true;
    }

    // Limit the number of elements to test to avoid excessive output and reduce risk
    size_t elementsToTest = std::min(numElements, MAX_TEST_ELEMENTS);
    size_t bytesToRead = elementsToTest * elementSize;

    // Allocate host memory for testing
    std::vector<uint8_t> hostBuffer(bytesToRead);

    TLLM_LOG_INFO(
        "Testing pointer %s (%p) - reading %zu elements (%zu bytes)", name.c_str(), ptr, elementsToTest, bytesToRead);

    // Ensure stream is synchronized before attempting memory copy
    cudaError_t syncResult = cudaStreamSynchronize(stream);
    if (syncResult != cudaSuccess)
    {
        TLLM_LOG_ERROR("Stream sync failed before testing %s: %s", name.c_str(), cudaGetErrorString(syncResult));
        return false;
    }

    // Attempt to copy data from device to host
    cudaError_t copyResult = cudaMemcpy(hostBuffer.data(), ptr, bytesToRead, cudaMemcpyDeviceToHost);

    if (copyResult != cudaSuccess)
    {
        TLLM_LOG_ERROR("Failed to read from pointer %s (%p): %s", name.c_str(), ptr, cudaGetErrorString(copyResult));
        return false;
    }

    TLLM_LOG_INFO("Successfully read from pointer %s (%p) - %zu bytes accessible", name.c_str(), ptr, bytesToRead);

    return true;
}

bool TensorDebugUtils::testHalfTensorPtr(
    void const* ptr, size_t numElements, std::string const& name, cudaStream_t stream)
{
    if (ptr == nullptr)
    {
        TLLM_LOG_INFO("Half tensor %s is NULL - skipping test", name.c_str());
        return true;
    }

    if (numElements == 0)
    {
        TLLM_LOG_INFO("Half tensor %s has 0 elements - skipping test", name.c_str());
        return true;
    }

    // Limit elements to test
    size_t elementsToTest = std::min(numElements, MAX_TEST_ELEMENTS);

    if (!testDevicePointerAccessibility(ptr, elementsToTest, sizeof(__half), name, stream))
    {
        return false;
    }

    // If accessible, try to read and print some values for debugging
    std::vector<__half> hostData(elementsToTest);
    cudaError_t copyResult = cudaMemcpy(hostData.data(), ptr, elementsToTest * sizeof(__half), cudaMemcpyDeviceToHost);

    if (copyResult == cudaSuccess)
    {
        std::string valuesStr = "";
        for (size_t i = 0; i < std::min(elementsToTest, size_t(8)); ++i)
        {
            valuesStr += std::to_string(static_cast<float>(hostData[i]));
            if (i < std::min(elementsToTest, size_t(8)) - 1)
            {
                valuesStr += ", ";
            }
        }
        if (elementsToTest > 8)
        {
            valuesStr += "...";
        }

        TLLM_LOG_INFO("Half tensor %s values: [%s]", name.c_str(), valuesStr.c_str());
    }

    return true;
}

bool TensorDebugUtils::validateMHARunnerParams(kernels::MHARunnerParams const& params)
{
    TLLM_LOG_INFO("=== Starting MHARunnerParams pointer validation ===");

    bool allValid = true;

    // Test main tensor pointers - these are the most likely to cause issues
    // We'll estimate reasonable sizes based on typical tensor dimensions

    // For most pointers, we can't easily determine the exact size, so we'll test a small amount
    const size_t SMALL_TEST_SIZE = 64; // Test first 64 elements

    // Test qkvPtr - usually the largest tensor
    if (!testDevicePointerAccessibility(params.qkvPtr, SMALL_TEST_SIZE, sizeof(__half), "qkvPtr", params.stream))
    {
        allValid = false;
    }

    // Test qPtr
    if (!testDevicePointerAccessibility(params.qPtr, SMALL_TEST_SIZE, sizeof(__half), "qPtr", params.stream))
    {
        allValid = false;
    }

    // Test kvPtr
    if (!testDevicePointerAccessibility(params.kvPtr, SMALL_TEST_SIZE, sizeof(__half), "kvPtr", params.stream))
    {
        allValid = false;
    }

    // Test outputPtr
    if (!testDevicePointerAccessibility(params.outputPtr, SMALL_TEST_SIZE, sizeof(__half), "outputPtr", params.stream))
    {
        allValid = false;
    }

    // Test outputSfPtr
    if (!testDevicePointerAccessibility(
            params.outputSfPtr, SMALL_TEST_SIZE, sizeof(float), "outputSfPtr", params.stream))
    {
        allValid = false;
    }

    // Test softmaxStatsPtr
    if (!testDevicePointerAccessibility(
            params.softmaxStatsPtr, SMALL_TEST_SIZE, sizeof(float2), "softmaxStatsPtr", params.stream))
    {
        allValid = false;
    }

    // Test integer arrays - these are typically smaller
    const size_t INT_ARRAY_TEST_SIZE = std::max(params.b + 1, 16); // At least batch size + 1

    // Test cuQSeqLenPtr - cumulative sequence lengths
    if (!testDevicePointerAccessibility(
            params.cuQSeqLenPtr, INT_ARRAY_TEST_SIZE, sizeof(int), "cuQSeqLenPtr", params.stream))
    {
        allValid = false;
    }

    // Test kvSeqLenPtr - kv sequence lengths
    if (!testDevicePointerAccessibility(params.kvSeqLenPtr, params.b, sizeof(int), "kvSeqLenPtr", params.stream))
    {
        allValid = false;
    }

    // Test cuKvSeqLenPtr - cumulative kv sequence lengths
    if (!testDevicePointerAccessibility(
            params.cuKvSeqLenPtr, INT_ARRAY_TEST_SIZE, sizeof(int), "cuKvSeqLenPtr", params.stream))
    {
        allValid = false;
    }

    // Test cuMaskRowsPtr
    if (!testDevicePointerAccessibility(
            params.cuMaskRowsPtr, INT_ARRAY_TEST_SIZE, sizeof(int), "cuMaskRowsPtr", params.stream))
    {
        allValid = false;
    }

    // Test single element pointers
    if (!testDevicePointerAccessibility(params.tileCounterPtr, 1, sizeof(int), "tileCounterPtr", params.stream))
    {
        allValid = false;
    }

    // Test scale pointers
    if (!testDevicePointerAccessibility(params.scaleBmm1Ptr, 1, sizeof(float), "scaleBmm1Ptr", params.stream))
    {
        allValid = false;
    }

    if (!testDevicePointerAccessibility(params.scaleBmm2Ptr, 1, sizeof(float), "scaleBmm2Ptr", params.stream))
    {
        allValid = false;
    }

    if (!testDevicePointerAccessibility(params.oSfScalePtr, 1, sizeof(float), "oSfScalePtr", params.stream))
    {
        allValid = false;
    }

    if (!testDevicePointerAccessibility(params.qScalePtr, 1, sizeof(float), "qScalePtr", params.stream))
    {
        allValid = false;
    }

    if (!testDevicePointerAccessibility(params.kScalePtr, 1, sizeof(float), "kScalePtr", params.stream))
    {
        allValid = false;
    }

    if (!testDevicePointerAccessibility(params.vScalePtr, 1, sizeof(float), "vScalePtr", params.stream))
    {
        allValid = false;
    }

    // Test packedMaskPtr - this could be variable size, test small amount
    if (!testDevicePointerAccessibility(
            params.packedMaskPtr, SMALL_TEST_SIZE, sizeof(uint32_t), "packedMaskPtr", params.stream))
    {
        allValid = false;
    }

    TLLM_LOG_INFO("=== MHARunnerParams validation complete. All valid: %s ===", allValid ? "true" : "false");

    return allValid;
}

} // namespace common
} // namespace tensorrt_llm


