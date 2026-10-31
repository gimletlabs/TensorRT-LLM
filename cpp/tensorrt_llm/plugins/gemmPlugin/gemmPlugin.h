/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef TRT_GEMM_PLUGIN_H
#define TRT_GEMM_PLUGIN_H

#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include "tensorrt_llm/plugins/common/plugin.h"

#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{

using CublasGemmWrapper = tensorrt_llm::common::CublasMMWrapper;
using CublasGemmWrapperPtr = std::shared_ptr<CublasGemmWrapper>;

class CublasLtGemmPluginProfiler
    : public GemmPluginProfiler<cublasLtMatmulHeuristicResult_t, CublasGemmWrapperPtr, GemmIdCublas, GemmIdCublasHash>
{
public:
    using Config = cublasLtMatmulHeuristicResult_t;

    void setTranspose(bool transposeA, bool transposeB)
    {
        mTransA = transposeA;
        mTransB = transposeB;
    }

    void setPadLd(int padLda, int padLdb, int padLdc)
    {
        mPadLda = padLda;
        mPadLdb = padLdb;
        mPadLdc = padLdc;
    }

    void setOutputType(nvinfer1::DataType type)
    {
        mOutputType = type;
    }

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(size_t maxM, size_t n, size_t k) override;

    bool checkTactic(int m, int n, int k, Config const& tactic) const override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    bool mTransA;
    bool mTransB;
    int mPadLda;
    int mPadLdb;
    int mPadLdc;
    nvinfer1::DataType mOutputType;

    static constexpr size_t ALIGNMENT = 256;
};

} // namespace tensorrt_llm::plugins

#endif // TRT_GEMM_PLUGIN_H
