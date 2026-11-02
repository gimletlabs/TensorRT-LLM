/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
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

#include "gemmPlugin.h"

#include "gemmPluginProfiler.h"
#include "plugin.h"
#include "pluginUtils.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"

#include <NvInferRuntime.h>

#include <cassert>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::GemmDims;
using tensorrt_llm::plugins::CublasLtGemmPluginProfiler;
using tensorrt_llm::plugins::CublasGemmWrapperPtr;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;

void getProblemParams(cublasOperation_t& transa, cublasOperation_t& transb, int& m, int& n, int& k, int& lda, int& ldb,
    int& ldc, bool transA, bool transB, int M, int N, int K, int padLda, int padLdb, int padLdc)
{
    transa = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    transb = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    m = N;
    n = M;
    k = K;
    lda = transB ? K + padLdb : N + padLdb;
    ldb = transA ? M + padLda : K + padLda;
    ldc = N + padLdc;
}

void runGemm(int const M, int const N, int const K, bool const transA, bool const transB, int const padLda,
    int const padLdb, int const padLdc, nvinfer1::DataType const type, CublasGemmWrapperPtr const& cublasWrapperPtr,
    void const* act, void const* weight, float const alpha, void* output,
    std::optional<cublasLtMatmulHeuristicResult_t> const& heuristic, void* workspace, cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0)
        return;

    cublasWrapperPtr->setStream(stream);
    cublasWrapperPtr->setWorkspace(workspace);

    cublasOperation_t transa, transb;
    int m, n, k;
    int lda, ldb, ldc;
    getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, transA, transB, M, N, K, padLda, padLdb, padLdc);

    cublasWrapperPtr->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
    cublasWrapperPtr->Gemm(transa, transb, m, n, k, weight, lda, act, ldb, output, ldc, alpha, 0.0f, heuristic);
    cublasWrapperPtr->destroyDescriptors();
}

void CublasLtGemmPluginProfiler::runTactic(
    int m, int n, int k, CublasLtGemmPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream)
{
    size_t dataSize = sizeof(half);
    if (mType == nvinfer1::DataType::kFLOAT)
    {
        dataSize = sizeof(float);
    }

    void* actPtr = reinterpret_cast<void*>(workspace);
    void* weightPtr = reinterpret_cast<void*>(
        nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t*>(actPtr), m * k * dataSize, ALIGNMENT));
    void* outputPtr = reinterpret_cast<void*>(
        nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t*>(weightPtr), n * k * dataSize, ALIGNMENT));
    char* workspacePtr = reinterpret_cast<char*>(
        nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t*>(outputPtr), m * (n + mPadLdc) * dataSize, ALIGNMENT));
    runGemm(m, n, k, mTransA, mTransB, mPadLda, mPadLdb, mPadLdc, mType, mRunner, actPtr, weightPtr, 1.0f, outputPtr,
        {tactic}, workspacePtr, stream);
}

bool CublasLtGemmPluginProfiler::checkTactic(int m, int n, int k, Config const& tactic) const
{
    cublasOperation_t transa, transb;
    int M = m, N = n, K = k;
    int lda, ldb, ldc;
    getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, mTransA, mTransB, M, N, K, mPadLda, mPadLdb, mPadLdc);

    mRunner->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);

    auto const checkResult = mRunner->checkTactic(transa, transb, m, n, k, lda, ldb, ldc, tactic.algo);

    mRunner->destroyDescriptors();

    return checkResult;
}

void CublasLtGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    size_t dataSize = getDTypeSize(static_cast<tensorrt_llm::DataType>(mType));
    size_t outputDataSize = getDTypeSize(static_cast<tensorrt_llm::DataType>(mOutputType));

    std::vector<size_t> workspaces = {
        maxM * k * dataSize,                   // A
        n * k * dataSize,                      // B
        maxM * (n + mPadLdc) * outputDataSize, // C
        CUBLAS_WORKSPACE_SIZE                  // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size(), ALIGNMENT);
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<CublasLtGemmPluginProfiler::Config> CublasLtGemmPluginProfiler::getTactics(int M, int N, int K) const
{
    cublasOperation_t transa, transb;
    int m, n, k;
    int lda, ldb, ldc;
    getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, mTransA, mTransB, M, N, K, mPadLda, mPadLdb, mPadLdc);

    mRunner->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
    auto const heruistics = mRunner->getTactics(transa, transb, m, n, k, lda, ldb, ldc);
    mRunner->destroyDescriptors();

    return heruistics;
}
