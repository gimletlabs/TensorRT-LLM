/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "DevKernel.h"
#include "RoutingKernel.h"
#include "runner.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/trtllm/gen/SfLayoutDecl.h"
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <iostream>
#include <tensorrt_llm/common/assert.h>
#include <tensorrt_llm/common/envUtils.h>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace trtllmGenFp8BlockScaleMoe
{

namespace btg = batchedGemm::trtllm::gen;

namespace Routing
{
namespace
{
inline int32_t computeLog2(int32_t val, std::string const& name = "")
{
    int32_t n = val;
    int32_t out = 0;
    while (n >>= 1)
    {
        ++out;
    }
    if ((1 << out) != val)
    {
        out = -1;
    }
    return out;
}
} // namespace

Runner::Runner() {}

Runner::Runner(int32_t tileTokensDim)
    : mTileTokensDim(tileTokensDim)
{
}

void Runner::run(void* routingLogits, void* routingBias, int32_t numTokens, int32_t numExperts, int32_t topK,
    int32_t nGroup, int32_t topkGroup, int32_t localExpertOffset, int32_t localNumExperts, float routedScalingFactor,
    int32_t* routingExpertIndexes, int32_t* expertCountHistogram, int32_t* permutedIdxSize,
    int32_t* expandedIdxToPermutedIdx, int32_t* permutedIdxToExpandedIdx, int32_t* permutedIdxToTokenIdx,
    void* expertWeights, int32_t* expertIds, int32_t* numTokensPerExpert, int32_t* ctaIdxXyToBatchIdx,
    int32_t* ctaIdxXyToMnLimit, int32_t* numNonExitingCtas, btg::Dtype dtypeElt, bool useRoutingScalesOnInput,
    bool useDeepSeekFp8, RoutingMethodType routingMethodType, cudaStream_t stream)
{
    if (routingMethodType == RoutingMethodType::DeepSeekV3)
    {
        TLLM_CHECK_WITH_INFO(topK <= 22, "For DeepSeek routing method, must have topK <= 22");
        TLLM_CHECK_WITH_INFO(topkGroup <= 4, "For DeepSeek routing method, must have topkGroup <= 4");
        moe::dev::routing::routingDeepSeek::Data routingData;
        routingData.mDtypeExpW = btg::Dtype::Bfloat16;
        routingData.mUsePdl = true;

        // output:
        routingData.mPtrTopKPacked = routingExpertIndexes;
        routingData.mPtrExpertCounts = expertCountHistogram;
        routingData.mPtrPermutedIdxSize = permutedIdxSize;
        routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
        routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
        routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
        routingData.mPtrTopKWeights = expertWeights;

        routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
        routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
        routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

        // input:
        routingData.mPtrRoutingBias = routingBias;
        // Pass-through raw pointer; kernels will cast to the proper InputT based on routing method
        routingData.mPtrScores = expertIds == nullptr ? routingLogits : nullptr;
        routingData.mPtrTopKIds = expertIds;
        routingData.mNumTokens = numTokens;
        routingData.mNumExperts = numExperts;
        routingData.mNumExpertGroups = nGroup;
        routingData.mNumLimitedGroups = topkGroup;
        routingData.mTopK = topK;
        routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
        routingData.mTileTokensDim = mTileTokensDim;
        routingData.mLocalExpertsStartIdx = localExpertOffset;
        routingData.mLocalExpertsStrideLog2 = 0;
        routingData.mNumLocalExperts = localNumExperts;
        routingData.mRouteScale = routedScalingFactor;
        routingData.mUseRoutingSoftmax = false;
        moe::dev::routing::routingDeepSeek::run(routingData, stream);
    }
    else if (routingMethodType == RoutingMethodType::Llama4)
    {
        TLLM_CHECK_WITH_INFO(topK == 1, "For Llama routing method, must have topK == 1");
        if (nGroup > 0 || topkGroup > 0)
        {
            TLLM_LOG_WARNING("For Llama routing method, nGroup/topkGroup is ignored, got %d/%d.", nGroup, topkGroup);
        }
        moe::dev::routing::routingLlama4::Data routingData;
        routingData.mDtypeExpW = btg::Dtype::Bfloat16;
        routingData.mUsePdl = true;

        // output:
        routingData.mPtrTopKPacked = routingExpertIndexes;
        routingData.mPtrExpertCounts = expertCountHistogram;
        routingData.mPtrPermutedIdxSize = permutedIdxSize;
        routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
        routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
        routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
        routingData.mPtrTopKWeights = expertWeights;

        routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
        routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
        routingData.mPtrNumNonExitingCtas = numNonExitingCtas;
        // routingData.mAllToAllRouteAct = false;

        // input:
        // routingData.mPtrRoutingWeights = args.mRoutingWeights;  // routing weights (don't need if not using gemm)
        // routingData.mPtrRoutingBias = routingBias;

        // Pass-through raw pointer; kernels will cast to the proper InputT based on routing method
        routingData.mPtrScores = expertIds == nullptr ? routingLogits : nullptr;
        routingData.mPtrTopKIds = expertIds;
        // routingData.mPtrIn = args.mInputActs;
        routingData.mNumTokens = numTokens;
        // routingData.mHiddenDim = args.mHiddenDim;
        routingData.mNumExperts = numExperts;
        // routingData.mNumExpertGroups = nGroup;
        // routingData.mNumLimitedGroups =topkGroup;
        routingData.mTopK = topK;
        routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
        routingData.mTileTokensDim = mTileTokensDim;
        routingData.mLocalExpertsStartIdx = localExpertOffset;
        routingData.mLocalExpertsStrideLog2 = 0;
        routingData.mNumLocalExperts = localNumExperts;
        // routingData.mRouteScale = routed_scaling_factor;
        // routingData.mUseRoutingSoftmax = false;
        moe::dev::routing::routingLlama4::run(routingData, stream);
    }
    else if (routingMethodType == RoutingMethodType::Renormalize /* default */
        || routingMethodType == RoutingMethodType::RenormalizeNaive /* Softmax -> TopK */)
    {
        moe::dev::routing::routingRenormalize::Data routingData;

        //
        // Config
        //

        routingData.mDtypeExpW = btg::Dtype::Bfloat16;
        // routingData.mDtypeElt = dtypeElt; // no-op for now as hidden_state is not input
        routingData.mUsePdl = tensorrt_llm::common::getEnvEnableTrtllmgenMoeRoutingRenormPDL();
        routingData.mDoSoftmaxBeforeTopK = routingMethodType == RoutingMethodType::RenormalizeNaive;
        routingData.mNormTopkProb = routingMethodType == RoutingMethodType::RenormalizeNaive;

        // Pass-through raw pointer; kernels will cast to the proper InputT based on routing method
        routingData.mPtrScores = expertIds == nullptr ? routingLogits : nullptr;
        //
        // Outputs
        //
        routingData.mPtrTopKPacked = routingExpertIndexes;
        routingData.mPtrExpertCounts = expertCountHistogram;
        routingData.mPtrPermutedIdxSize = permutedIdxSize;
        routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
        routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
        routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
        routingData.mPtrTopKWeights = expertWeights;
        routingData.mPtrTopKIds = expertIds;
        //
        // Grouped Gemm Launch Config Buffers
        //
        routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
        routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
        routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

        //
        // Inputs
        //
        routingData.mNumTokens = numTokens;
        routingData.mNumExperts = numExperts;
        routingData.mTopK = topK;
        routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
        routingData.mTileTokensDim = mTileTokensDim;
        routingData.mLocalExpertsStartIdx = localExpertOffset;
        routingData.mLocalExpertsStrideLog2 = 0;
        routingData.mNumLocalExperts = localNumExperts;

        moe::dev::routing::routingRenormalize::run(routingData, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unimplemented routing method %s of enum %d",
            serializeMoeRoutingMethodType(routingMethodType).c_str(), (int) routingMethodType);
    }
}
} // namespace Routing

namespace PermuteGemm1
{

tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions getOptions(btg::Dtype dtypeAct, btg::Dtype dtypeWeights,
    btg::Dtype dtypeOut, int32_t tileTokensDim, bool useDeepSeekFp8, ActType actType, bool forceNonFusedActivation)
{
    bool is_gated_activation = actType == ActType::SwiGlu;
    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options;

    if (is_gated_activation)
    {

        options = {// Swap A and B dtypes because transposeMmaOutput is hardcoded to true
            .dtypeA = dtypeWeights,
            .dtypeB = dtypeAct,
            .dtypeC = dtypeOut,
            .actType = actType,
            .deepSeekFp8 = useDeepSeekFp8,
            .fusedAct = !(useDeepSeekFp8 || forceNonFusedActivation),
            .routeAct = true,
            .staticBatch = false,
            .transposeMmaOutput = true,
            .tileSize = tileTokensDim,
            .epilogueTileM = useDeepSeekFp8 ? 64 : 128};
    }
    else
    {
        EltwiseActType eltwiseActType = EltwiseActType::None;
        switch (actType)
        {
        default:
        case ActType::Relu2: eltwiseActType = EltwiseActType::Relu2; break;
        case ActType::Silu: eltwiseActType = EltwiseActType::Silu; break;
        }
        options = {
            .dtypeA = dtypeWeights,
            .dtypeB = dtypeAct,
            .dtypeC = dtypeOut,
            .eltwiseActType = eltwiseActType,
            .deepSeekFp8 = useDeepSeekFp8,
            .fusedAct = false,
            .routeAct = true,
            .staticBatch = false,
            .transposeMmaOutput = true,
            .tileSize = tileTokensDim,
            .epilogueTileM = 128,
        };
    }
    return options;
}

Runner::Runner(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOut, bool useDeepSeekFp8,
    int tileTokensDim, ActType actType, bool forceNonFusedActivation)
    : mDtypeAct(dtypeAct)
    , mDtypeWeights(dtypeWeights)
    , mTileTokensDim(tileTokensDim)
    , mRunner(tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner(
          getOptions(mDtypeAct, mDtypeWeights, dtypeOut, mTileTokensDim, useDeepSeekFp8, actType,
              forceNonFusedActivation)))
    , mActType(actType)
{
}

void Runner::run(void* hiddenState, void* hiddenStateScale, void* weights, void* weightsScale, void* expertWeights,
    float* outputScalesScalar, float* outputScalesGateScalar, float* ptrBias, float* ptrAlpha, float* ptrBeta,
    float* ptrClampLimit, void* output, void* outputScale, int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
    int32_t numExperts, int32_t numTokens, int32_t* permutedIdxToTokenIdx, int32_t* ptrNumNonExitingCtas,
    int32_t* ptrTotalNumPaddedTokens, int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit,
    void* bmm1Workspace, bool useRoutingScalesOnInput, int device, cudaStream_t stream, int32_t configIndex,
    int32_t validHiddenSize, int32_t validIntermediateSize)
{
    if (mDtypeWeights == btg::Dtype::MxE2m1 && mDtypeAct == btg::Dtype::MxE4m3)
    {
        // The multiple is no less than 128 as TMA requires it for CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B types
        // FIXME: enforce valid hidden dim to be multiple of 512 due to unhandled OOB read in routeAct. Please keep this
        // in sync with
        // tensorrt_llm/_torch/modules/fused_moe/quantization.py:MXFP4WeightTRTLLMGenFusedMoEMethod.input_hidden_alignment
        validHiddenSize = tensorrt_llm::common::roundUp(validHiddenSize, 512);
    }
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    bool is_gated_activation = mActType == ActType::SwiGlu;
    int32_t intermediateSizeFactor = (is_gated_activation ? 2 : 1);
    mRunner.run(numTokens, intermediateSizeFactor * intermediateSize, hiddenSize, numTokens,
        intermediateSizeFactor * validIntermediateSize, validHiddenSize, {}, numTokens, numExperts,
        maxNumCtasInBatchDim, hiddenState, hiddenStateScale, weights, weightsScale,
        useRoutingScalesOnInput ? expertWeights : nullptr, /* perTokensSfB */ nullptr, outputScalesScalar,
        outputScalesGateScalar, ptrBias, ptrAlpha, ptrBeta, ptrClampLimit, output, outputScale, permutedIdxToTokenIdx,
        ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx, ptrCtaIdxXyToMnLimit, ptrNumNonExitingCtas, bmm1Workspace,
        stream, device, configIndex);
}

size_t Runner::getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts,
    int32_t numTokens, int32_t configIndex) const
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    int32_t const intermediateSizeFactor = mActType == ActType::SwiGlu ? 2 : 1;

    return mRunner.getWorkspaceSizeInBytes(numTokens, intermediateSizeFactor * intermediateSize, hiddenSize, {},
        numTokens, numExperts, maxNumCtasInBatchDim, configIndex);
}

int32_t Runner::getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
    int32_t numExperts, int32_t numTokens, int32_t validHiddenSize, int32_t validIntermediateSize) const
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    bool is_gated_activation = mActType == ActType::SwiGlu;
    return mRunner.getDefaultValidConfigIndex(numTokens, is_gated_activation ? 2 * intermediateSize : intermediateSize,
        hiddenSize, {}, numTokens, numExperts, maxNumCtasInBatchDim, numTokens, 2 * validIntermediateSize,
        validHiddenSize);
}

bool Runner::isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
    int32_t numExperts, int32_t numTokens, int32_t validHiddenSize, int32_t validIntermediateSize) const
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    bool is_gated_activation = mActType == ActType::SwiGlu;
    auto const isValid = mRunner.isValidConfigIndex(configIndex, numTokens,
        is_gated_activation ? 2 * intermediateSize : intermediateSize, hiddenSize, {}, numTokens, numExperts,
        maxNumCtasInBatchDim, numTokens, 2 * validIntermediateSize, validHiddenSize);

    return isValid;
}

std::vector<int64_t> Runner::getPassingConfigIndices() const
{
    return mRunner.getPassingConfigIndices();
}

std::string Runner::getKernelNameFromConfigIndex(int32_t configIndex) const
{
    return mRunner.getKernelNameFromConfigIndex(configIndex);
}

} // namespace PermuteGemm1

namespace Gemm2
{
tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions getOptions(
    btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOut, int32_t tileTokensDim, bool useDeepSeekFp8)
{
    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options
        = {// Swap A and B dtypes because transposeMmaOutput is hardcoded to true
            .dtypeA = dtypeWeights,
            .dtypeB = dtypeAct,
            .dtypeC = dtypeOut,
            .eltwiseActType = EltwiseActType::None,
            .deepSeekFp8 = useDeepSeekFp8,
            .fusedAct = false,
            .routeAct = false,
            .staticBatch = false,
            .transposeMmaOutput = true,
            .tileSize = tileTokensDim,
            .epilogueTileM = useDeepSeekFp8 ? 64 : 128};
    return options;
}

Runner::Runner(
    btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOut, bool useDeepSeekFp8, int tileTokensDim)
    : mDtypeAct(dtypeAct)
    , mDtypeWeights(dtypeWeights)
    , mDtypeOut(dtypeOut)
    , mTileTokensDim(tileTokensDim)
    , mRunner(tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner(
          getOptions(dtypeAct, dtypeWeights, dtypeOut, tileTokensDim, useDeepSeekFp8)))
{
}

void Runner::run(void* permutedHiddenState, void* permutedHiddenStateScale, void* weights, void* weightsScale,
    float* outputScalesScalar, float* ptrBias, void* output, void* outputScale, int32_t topK, int32_t hiddenSize,
    int32_t intermediateSize, int32_t numExperts, int32_t numTokens, int32_t* ptrNumNonExitingCtas,
    int32_t* ptrTotalNumPaddedTokens, int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit,
    void* bmm2Workspace, int device, cudaStream_t stream, int32_t configIndex, int32_t validHiddenSize,
    int32_t validIntermediateSize)
{
    if (mDtypeWeights == btg::Dtype::MxE2m1 && mDtypeAct == btg::Dtype::MxE4m3)
    {
        // The multiple is no less than 128 as TMA requires it for CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B types
        validIntermediateSize = tensorrt_llm::common::roundUp(validIntermediateSize, 128);
    }
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    mRunner.run(numTokens, hiddenSize, intermediateSize, numTokens, validHiddenSize, validIntermediateSize, {},
        numTokens, numExperts, maxNumCtasInBatchDim, permutedHiddenState, permutedHiddenStateScale, weights,
        weightsScale, /* perTokensSfA */ nullptr,
        /* perTokensSfB */ nullptr, outputScalesScalar, /* outputScalesGateScalar */ nullptr, ptrBias,
        /* ptrAlpha */ nullptr, /* ptrBeta */ nullptr, /* clampLimit */ nullptr, output, outputScale,
        /* permutedIdxToTokenIdx */ nullptr, ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx, ptrCtaIdxXyToMnLimit,
        ptrNumNonExitingCtas, bmm2Workspace, stream, device, configIndex);
}

size_t Runner::getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts,
    int32_t numTokens, int32_t configIndex) const
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    return mRunner.getWorkspaceSizeInBytes(
        numTokens, hiddenSize, intermediateSize, {}, numTokens, numExperts, maxNumCtasInBatchDim, configIndex);
}

int32_t Runner::getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
    int32_t numExperts, int32_t numTokens, int32_t validHiddenSize, int32_t validIntermediateSize) const
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    return mRunner.getDefaultValidConfigIndex(numTokens, hiddenSize, intermediateSize, {}, numTokens, numExperts,
        maxNumCtasInBatchDim, numTokens, validHiddenSize, validIntermediateSize);
}

bool Runner::isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
    int32_t numExperts, int32_t numTokens, int32_t validHiddenSize, int32_t validIntermediateSize) const
{

    auto const maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);

    auto const isValid = mRunner.isValidConfigIndex(configIndex, numTokens, hiddenSize, intermediateSize, {}, numTokens,
        numExperts, maxNumCtasInBatchDim, numTokens, validHiddenSize, validIntermediateSize);

    return isValid;
}

std::vector<int64_t> Runner::getPassingConfigIndices() const
{
    return mRunner.getPassingConfigIndices();
}

std::string Runner::getKernelNameFromConfigIndex(int32_t configIndex) const
{
    return mRunner.getKernelNameFromConfigIndex(configIndex);
}

} // namespace Gemm2

namespace MoE
{
namespace
{
btg::Dtype resolveDtype(btg::Dtype dtype, btg::Dtype defaultDtype)
{
    return dtype == btg::Dtype::Void ? defaultDtype : dtype;
}

bool hasActivationScale(MoERunnerArgs const& args)
{
    return args.activation_input_scale != nullptr || args.activation_output_scale != nullptr;
}
} // namespace

Runner::Runner(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, bool useDeepSeekFp8, int32_t tileTokensDim,
    ActType actType, btg::Dtype dtypeGemm1Out)
    : mDtypeGemm1Out(resolveDtype(dtypeGemm1Out, dtypeAct))
    , mDtypeWeights(dtypeWeights)
    , mUseStandaloneActivation(mDtypeGemm1Out != dtypeAct)
    , mPermuteGemm1(PermuteGemm1::Runner(
          dtypeAct, dtypeWeights, mDtypeGemm1Out, useDeepSeekFp8, tileTokensDim, actType, mUseStandaloneActivation))
    , mPermuteGemm1NonFused(
          PermuteGemm1::Runner(dtypeAct, dtypeWeights, mDtypeGemm1Out, useDeepSeekFp8, tileTokensDim, actType, true))
    , mGemm2(Gemm2::Runner(mDtypeGemm1Out, dtypeWeights, btg::Dtype::Bfloat16, useDeepSeekFp8, tileTokensDim))
    , mActType(actType)
{
    auto const& gemm1PassingIndices
        = (mUseStandaloneActivation ? mPermuteGemm1NonFused : mPermuteGemm1).getPassingConfigIndices();
    auto const& gemm2PassingIndices = mGemm2.getPassingConfigIndices();

    auto const totalPassingIndices = gemm1PassingIndices.size() * gemm2PassingIndices.size();
    mPassingConfigs.reserve(totalPassingIndices);

    for (auto const& indexGemm1 : gemm1PassingIndices)
    {
        for (auto const& indexGemm2 : gemm2PassingIndices)
        {
            mPassingConfigs.push_back(MoEConfig{indexGemm1, indexGemm2});
        }
    }

    TLLM_CHECK_WITH_INFO(!mPassingConfigs.empty(), "No compatible configs found for the fp8 block scale MoE runner.");
}

Runner::Runner(btg::Dtype dtypeElt, bool useDeepSeekFp8, int32_t tileTokensDim)
    : Runner(dtypeElt, dtypeElt, useDeepSeekFp8, tileTokensDim, ActType::SwiGlu)
{
}

void Runner::setOpsData(MoERunnerArgs const& args, MoEWorkspace const& workspace,
    moe::dev::convertsf::Data& convertSfData, moe::dev::activation::Data& activationData,
    moe::dev::finalize::Data& finalizeData)
{
    // Setup sf conversion data if needed
    convertSfData.inSfPtr = args.hidden_states_scale;
    convertSfData.outSfPtr = workspace.hidden_states_scale_linear;
    convertSfData.hiddenDimSf = args.hidden_size / 16;
    convertSfData.numTokens = args.num_tokens;
    convertSfData.sfLayoutSrc = btg::SfLayout::R128c4;
    convertSfData.sfLayoutDst = btg::SfLayout::Linear;
    convertSfData.mUsePdl = true;

    // Setup activation data
    activationData.mDtypeElt = resolveDtype(args.mDtypeGemm1Out, mDtypeGemm1Out);
    activationData.mUsePdl = true;
    activationData.mUseDeepSeekFp8 = args.mUseDeepSeekFp8;
    activationData.inPtr = workspace.gemm1_output;
    activationData.outPtr = workspace.activation_output;
    activationData.inDqSfsPtr = workspace.gemm1_output_scale;
    activationData.outDqSfsPtr = workspace.activation_output_scale;
    activationData.inScalePtr = args.activation_input_scale;
    activationData.outScalePtr = args.activation_output_scale;
    activationData.innerDim = args.intermediate_size * (mActType == ActType::SwiGlu ? 2 : 1);
    activationData.topK = args.top_k;
    activationData.numTokens = args.num_tokens;
    activationData.expandedIdxToPermutedIdx = workspace.expanded_idx_to_permuted_idx;
    activationData.expertIndexes = workspace.routing_expert_indexes;

    activationData.totalNumPaddedTokens = workspace.total_num_padded_tokens;

    if (args.do_finalize)
    {
        // Setup finalize data
        finalizeData.mDtypeElt = args.mDtypeOut;
        finalizeData.mDtypeExpW = args.mDtypeExpW;
        finalizeData.mUsePdl = true;
        finalizeData.mUseDeepSeekFp8 = false;
        finalizeData.inPtr = workspace.gemm2_output;
        finalizeData.outPtr = args.output;
        finalizeData.inDqSfsPtr = workspace.gemm2_output_scale;
        finalizeData.outDqSfsPtr = args.output_scale;
        finalizeData.inScalePtr = args.finalize_input_scale;
        if (args.mUseRoutingScalesOnInput)
        {
            finalizeData.expertWeightsPtr = nullptr;
        }
        else
        {
            finalizeData.expertWeightsPtr = workspace.expert_weights;
        }
        finalizeData.expandedIdxToPermutedIdx = workspace.expanded_idx_to_permuted_idx;
        finalizeData.expertIndexes = workspace.routing_expert_indexes;
        finalizeData.numTokens = args.num_tokens;
        finalizeData.numExperts = args.num_experts;
        finalizeData.topK = args.top_k;
        // We want to fuse unpadding into the finalize kernel, so we need to use the output hidden size.
        finalizeData.hiddenDim = args.valid_hidden_size.value_or(args.hidden_size);
        finalizeData.hiddenDimPadded = args.output_hidden_size.value_or(args.hidden_size);
        finalizeData.totalNumPaddedTokens = workspace.total_num_padded_tokens;
    }
}

std::tuple<int32_t, int32_t> Runner::getWorkspaceSizeInBytes(MoERunnerArgs const& args, int64_t configIndex) const
{
    auto const& config = mPassingConfigs[configIndex];
    bool const useStandaloneActivation = mUseStandaloneActivation || hasActivationScale(args) || args.mUseDeepSeekFp8;
    auto const& gemm1Runner = useStandaloneActivation ? mPermuteGemm1NonFused : mPermuteGemm1;
    int32_t gemm1Config = config.gemm1Config;
    if (useStandaloneActivation && !mUseStandaloneActivation)
    {
        gemm1Config = mPermuteGemm1NonFused.getDefaultValidConfigIndex(args.top_k, args.hidden_size,
            args.intermediate_size, args.local_num_experts, args.num_tokens,
            args.valid_hidden_size.value_or(args.hidden_size),
            args.valid_intermediate_size.value_or(args.intermediate_size));
    }

    auto workspace_size_fc1 = static_cast<int32_t>(gemm1Runner.getWorkspaceSizeInBytes(args.top_k, args.hidden_size,
        args.intermediate_size, args.local_num_experts, args.num_tokens, gemm1Config));
    auto workspace_size_fc2 = static_cast<int32_t>(mGemm2.getWorkspaceSizeInBytes(args.top_k, args.hidden_size,
        args.intermediate_size, args.local_num_experts, args.num_tokens, config.gemm2Config));
    return std::make_tuple(workspace_size_fc1, workspace_size_fc2);
}

std::vector<int64_t> Runner::getValidConfigIndices(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
    int32_t numLocalExperts, int32_t numTokens, int32_t validIntermediateSize, int32_t validHiddenSize) const
{
    std::vector<int64_t> validIndices;

    for (int i = 0; i < mPassingConfigs.size(); ++i)
    {
        auto const& config = mPassingConfigs[i];

        auto const& gemm1Runner = mUseStandaloneActivation ? mPermuteGemm1NonFused : mPermuteGemm1;
        if (gemm1Runner.isValidConfigIndex(config.gemm1Config, topK, hiddenSize, intermediateSize, numLocalExperts,
                numTokens, validHiddenSize, validIntermediateSize)
            && mGemm2.isValidConfigIndex(config.gemm2Config, topK, hiddenSize, intermediateSize, numLocalExperts,
                numTokens, validHiddenSize, validIntermediateSize))
        {
            validIndices.push_back(i);
            auto envVarVal = std::getenv("TLLM_BATCHED_GEMM_PRINT_CONFIGS");
            if (envVarVal && std::atoi(envVarVal) == 1)
            {
                auto kernel1 = gemm1Runner.getKernelNameFromConfigIndex(config.gemm1Config);
                auto kernel2 = mGemm2.getKernelNameFromConfigIndex(config.gemm2Config);
                printf("Valid config index: %d, Gemm1 %s, Gemm2 %s\n", i, kernel1.c_str(), kernel2.c_str());
            }
        }
    }

    return validIndices;
}

int64_t Runner::getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
    int32_t numLocalExperts, int32_t numTokens, int32_t validHiddenSize, int32_t validIntermediateSize) const
{

    auto const& gemm1Runner = mUseStandaloneActivation ? mPermuteGemm1NonFused : mPermuteGemm1;
    int32_t indexGemm1 = gemm1Runner.getDefaultValidConfigIndex(
        topK, hiddenSize, intermediateSize, numLocalExperts, numTokens, validHiddenSize, validIntermediateSize);
    int32_t indexGemm2 = mGemm2.getDefaultValidConfigIndex(
        topK, hiddenSize, intermediateSize, numLocalExperts, numTokens, validHiddenSize, validIntermediateSize);

    auto it = std::find_if(mPassingConfigs.begin(), mPassingConfigs.end(),
        [indexGemm1, indexGemm2](MoEConfig cfg)
        { return (cfg.gemm1Config == indexGemm1 && cfg.gemm2Config == indexGemm2); });
    TLLM_CHECK_WITH_INFO(it != mPassingConfigs.end(), "No compatible configs found for the block scale MoE runner.");
    return std::distance(mPassingConfigs.begin(), it);
}

namespace
{

// Decode a 4-bit E2m1 nibble (s ee m, exponent bias = 1) to float.
// Packing convention: lo nibble (bits 3:0) = even-indexed element,
//                     hi nibble (bits 7:4) = odd-indexed element.
inline float e2m1NibbleToFloat(uint8_t nibble)
{
    uint8_t const s = (nibble >> 3) & 0x1u;
    uint8_t const e = (nibble >> 1) & 0x3u;
    uint8_t const m = nibble & 0x1u;
    float val;
    if (e == 0)
    {
        // subnormal: (-1)^s * 2^(1-bias) * (m / 2)  where bias=1  ->  (-1)^s * 0.5 * m
        val = 0.5f * static_cast<float>(m);
    }
    else
    {
        // normal: (-1)^s * 2^(e-bias) * (1 + m * 0.5)  where bias=1
        val = std::ldexp(1.0f + 0.5f * static_cast<float>(m), static_cast<int>(e) - 1);
    }
    return s ? -val : val;
}

constexpr int kFp4SfBlockSize = 16;
constexpr int kMxSfBlockSize = 32;
constexpr int kFp8SfBlockSize = 128;

enum class DebugScaleType
{
    None,
    E4m3Linear,
    E4m3R128c4,
    E8m0Linear,
    E8m0R128c4,
    Fp32ColumnMajor
};

struct DebugScaleInfo
{
    void const* ptr{nullptr};
    DebugScaleType type{DebugScaleType::None};
    int32_t hiddenDim{0};
    int32_t blockSize{0};
    int32_t rowStride{0};
};

DebugScaleInfo makeDebugScaleInfo(
    void const* ptr, DebugScaleType type, int32_t hiddenDim, int32_t blockSize, int32_t rowStride = 0)
{
    if (ptr == nullptr || type == DebugScaleType::None)
    {
        return {};
    }
    return DebugScaleInfo{ptr, type, hiddenDim, blockSize, rowStride};
}

DebugScaleInfo getInputDebugScaleInfo(
    void const* scalePtr, btg::Dtype dtype, bool useDeepSeekFp8, int32_t hiddenDim, int32_t numTokens)
{
    if (dtype == btg::Dtype::E2m1)
    {
        return makeDebugScaleInfo(scalePtr, DebugScaleType::E4m3Linear, hiddenDim, kFp4SfBlockSize);
    }
    if (dtype == btg::Dtype::MxE2m1)
    {
        return makeDebugScaleInfo(scalePtr, DebugScaleType::E8m0Linear, hiddenDim, kMxSfBlockSize);
    }
    if (dtype == btg::Dtype::MxE4m3)
    {
        return makeDebugScaleInfo(scalePtr, DebugScaleType::E8m0Linear, hiddenDim, kMxSfBlockSize);
    }
    if (useDeepSeekFp8 && dtype == btg::Dtype::E4m3)
    {
        return makeDebugScaleInfo(scalePtr, DebugScaleType::Fp32ColumnMajor, hiddenDim, kFp8SfBlockSize, numTokens);
    }
    return {};
}

DebugScaleInfo getOutputDebugScaleInfo(void const* scalePtr, btg::Dtype dtype, int32_t hiddenDim, int32_t rowStride)
{
    if (dtype == btg::Dtype::E2m1)
    {
        return makeDebugScaleInfo(scalePtr, DebugScaleType::E4m3R128c4, hiddenDim, kFp4SfBlockSize);
    }
    if (dtype == btg::Dtype::MxE2m1)
    {
        return makeDebugScaleInfo(scalePtr, DebugScaleType::E8m0R128c4, hiddenDim, kMxSfBlockSize);
    }
    if (dtype == btg::Dtype::MxE4m3)
    {
        return makeDebugScaleInfo(scalePtr, DebugScaleType::E8m0R128c4, hiddenDim, kMxSfBlockSize);
    }
    if (dtype == btg::Dtype::E4m3)
    {
        return makeDebugScaleInfo(scalePtr, DebugScaleType::Fp32ColumnMajor, hiddenDim, kFp8SfBlockSize, rowStride);
    }
    return {};
}

DebugScaleInfo getWeightDebugScaleInfo(void const* scalePtr, btg::Dtype dtype, bool useDeepSeekFp8, int32_t hiddenDim)
{
    if (dtype == btg::Dtype::E2m1 || dtype == btg::Dtype::MxE2m1)
    {
        return makeDebugScaleInfo(scalePtr, DebugScaleType::E8m0Linear, hiddenDim, kMxSfBlockSize);
    }
    if (dtype == btg::Dtype::MxE4m3)
    {
        return makeDebugScaleInfo(scalePtr, DebugScaleType::E8m0Linear, hiddenDim, kMxSfBlockSize);
    }
    if (useDeepSeekFp8 && dtype == btg::Dtype::E4m3)
    {
        return makeDebugScaleInfo(scalePtr, DebugScaleType::Fp32ColumnMajor, hiddenDim, kFp8SfBlockSize, hiddenDim);
    }
    return {};
}

inline float e8m0ToFloat(uint8_t scale)
{
    return scale == 0 ? 0.0F : std::ldexp(1.0F, static_cast<int>(scale) - 127);
}

inline float e4m3ToFloat(uint8_t scale)
{
    __nv_fp8_e4m3 scaleVal;
    memcpy(&scaleVal, &scale, 1);
    return static_cast<float>(scaleVal);
}

// Host-side equivalent of convertsf::dev::getSfOffset for the R128c4 scale layout.
inline int64_t getR128c4SfOffset(int32_t dataRowIdx, int32_t dataBlkColIdx, int32_t numDataBlksPerRow)
{
    constexpr int32_t kNumRowsPerSfBlock = 128;
    constexpr int32_t kNumColsPerSfBlock = 4;
    constexpr int32_t kNumBytesPerSfBlock = kNumRowsPerSfBlock * kNumColsPerSfBlock;

    int const sfBlkRowIdx = dataRowIdx / kNumRowsPerSfBlock;
    int const sfBlkColIdx = dataBlkColIdx / kNumColsPerSfBlock;
    int const sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / kNumColsPerSfBlock + sfBlkColIdx;

    int const sfRowIdx = (dataRowIdx % 32) * 4 + (dataRowIdx % kNumRowsPerSfBlock) / 32;
    int const sfColIdx = dataBlkColIdx % kNumColsPerSfBlock;

    return sfBlkIdx * kNumBytesPerSfBlock + sfRowIdx * kNumColsPerSfBlock + sfColIdx;
}

// Print the first `count` elements of a device buffer, converting to float for display.
// Supported dtypes: E2m1, MxE2m1, E4m3, MxE4m3, E5m2, Bfloat16, Fp16, Fp32.
// Falls back to raw uint8 for other types.
void printDeviceTensor(char const* label, void const* devPtr, int count, btg::Dtype dtype, cudaStream_t stream,
    DebugScaleInfo scaleInfo = {})
{
    if (!devPtr)
    {
        printf("[MoE DBG] %s: nullptr\n", label);
        return;
    }
    if (count <= 0)
    {
        printf("[MoE DBG] %s (first 0 elems, dtype=%s):\n", label, btg::dtypeToString(dtype).c_str());
        return;
    }

    // For 4-bit packed types, 2 elements share 1 byte.
    bool const isFp4 = (dtype == btg::Dtype::E2m1 || dtype == btg::Dtype::MxE2m1);
    size_t bytes;
    if (isFp4)
    {
        bytes = static_cast<size_t>((count + 1) / 2);
    }
    else
    {
        int bytesPerElem = 1;
        switch (dtype)
        {
        case btg::Dtype::E4m3:
        case btg::Dtype::MxE4m3:
        case btg::Dtype::E5m2:
        case btg::Dtype::Int8:
        case btg::Dtype::UInt8: bytesPerElem = 1; break;
        case btg::Dtype::Bfloat16:
        case btg::Dtype::Fp16: bytesPerElem = 2; break;
        case btg::Dtype::Fp32:
        case btg::Dtype::Int32: bytesPerElem = 4; break;
        default: bytesPerElem = 1; break;
        }
        bytes = static_cast<size_t>(count) * bytesPerElem;
    }

    std::vector<uint8_t> host(bytes);
    cudaStreamSynchronize(stream);
    cudaMemcpy(host.data(), devPtr, bytes, cudaMemcpyDeviceToHost);

    bool const hasScale = scaleInfo.ptr != nullptr && scaleInfo.type != DebugScaleType::None;
    std::vector<uint8_t> scaleHost;
    if (hasScale)
    {
        TLLM_CHECK_WITH_INFO(scaleInfo.hiddenDim > 0, "Debug scale printing requires a positive hidden dimension.");
        TLLM_CHECK_WITH_INFO(scaleInfo.blockSize > 0, "Debug scale printing requires a positive scale block size.");
        TLLM_CHECK_WITH_INFO(scaleInfo.hiddenDim % scaleInfo.blockSize == 0,
            "Debug scale printing requires hiddenDim to be divisible by the scale block size.");

        int32_t const numSfPerRow = scaleInfo.hiddenDim / scaleInfo.blockSize;
        int64_t maxSfOffset = 0;
        for (int i = 0; i < count; ++i)
        {
            int32_t const dataRowIdx = i / scaleInfo.hiddenDim;
            int32_t const dataBlkColIdx = (i % scaleInfo.hiddenDim) / scaleInfo.blockSize;
            int64_t sfOffset = dataRowIdx * numSfPerRow + dataBlkColIdx;
            if (scaleInfo.type == DebugScaleType::E4m3R128c4 || scaleInfo.type == DebugScaleType::E8m0R128c4)
            {
                sfOffset = getR128c4SfOffset(dataRowIdx, dataBlkColIdx, numSfPerRow);
            }
            else if (scaleInfo.type == DebugScaleType::Fp32ColumnMajor)
            {
                TLLM_CHECK_WITH_INFO(
                    scaleInfo.rowStride > 0, "FP32 column-major scale printing requires a positive row stride.");
                sfOffset = dataRowIdx + scaleInfo.rowStride * dataBlkColIdx;
            }
            maxSfOffset = std::max(maxSfOffset, sfOffset);
        }

        size_t const scaleElementSize
            = scaleInfo.type == DebugScaleType::Fp32ColumnMajor ? sizeof(float) : sizeof(uint8_t);
        scaleHost.resize((static_cast<size_t>(maxSfOffset) + 1) * scaleElementSize);
        cudaMemcpy(scaleHost.data(), scaleInfo.ptr, scaleHost.size(), cudaMemcpyDeviceToHost);
    }

    if (hasScale)
    {
        printf("[MoE DBG] %s (first %d elems, dtype=%s, scaled):", label, count, btg::dtypeToString(dtype).c_str());
    }
    else
    {
        printf("[MoE DBG] %s (first %d elems, dtype=%s):", label, count, btg::dtypeToString(dtype).c_str());
    }
    for (int i = 0; i < count; ++i)
    {
        float val = 0.f;
        switch (dtype)
        {
        case btg::Dtype::E2m1:
        case btg::Dtype::MxE2m1:
        {
            // Two elements packed per byte: even index → lo nibble, odd index → hi nibble.
            uint8_t const byte = host[i / 2];
            uint8_t const nibble = (i % 2 == 0) ? (byte & 0x0fu) : (byte >> 4);
            val = e2m1NibbleToFloat(nibble);
            break;
        }
        case btg::Dtype::E4m3:
        case btg::Dtype::MxE4m3:
        {
            // MxE4m3 raw element bits are identical to E4m3; block-scale is a separate tensor.
            __nv_fp8_e4m3 fp8val;
            memcpy(&fp8val, host.data() + i, 1);
            val = static_cast<float>(fp8val);
            break;
        }
        case btg::Dtype::E5m2:
        {
            __nv_fp8_e5m2 fp8val;
            memcpy(&fp8val, host.data() + i, 1);
            val = static_cast<float>(fp8val);
            break;
        }
        case btg::Dtype::Bfloat16:
        {
            __nv_bfloat16 bf16val;
            memcpy(&bf16val, host.data() + i * 2, 2);
            val = __bfloat162float(bf16val);
            break;
        }
        case btg::Dtype::Fp16:
        {
            __half hval;
            memcpy(&hval, host.data() + i * 2, 2);
            val = __half2float(hval);
            break;
        }
        case btg::Dtype::Fp32: memcpy(&val, host.data() + i * 4, 4); break;
        default: val = static_cast<float>(host[i]); break;
        }
        if (hasScale)
        {
            int32_t const numSfPerRow = scaleInfo.hiddenDim / scaleInfo.blockSize;
            int32_t const dataRowIdx = i / scaleInfo.hiddenDim;
            int32_t const dataBlkColIdx = (i % scaleInfo.hiddenDim) / scaleInfo.blockSize;
            int64_t sfOffset = dataRowIdx * numSfPerRow + dataBlkColIdx;
            if (scaleInfo.type == DebugScaleType::E4m3R128c4 || scaleInfo.type == DebugScaleType::E8m0R128c4)
            {
                sfOffset = getR128c4SfOffset(dataRowIdx, dataBlkColIdx, numSfPerRow);
            }
            else if (scaleInfo.type == DebugScaleType::Fp32ColumnMajor)
            {
                sfOffset = dataRowIdx + scaleInfo.rowStride * dataBlkColIdx;
            }

            if (scaleInfo.type == DebugScaleType::Fp32ColumnMajor)
            {
                float scale = 1.0F;
                memcpy(&scale, scaleHost.data() + sfOffset * static_cast<int64_t>(sizeof(float)), sizeof(float));
                val *= scale;
            }
            else if (scaleInfo.type == DebugScaleType::E4m3Linear || scaleInfo.type == DebugScaleType::E4m3R128c4)
            {
                val *= e4m3ToFloat(scaleHost[static_cast<size_t>(sfOffset)]);
            }
            else
            {
                val *= e8m0ToFloat(scaleHost[static_cast<size_t>(sfOffset)]);
            }
        }
        printf(" %.4f", val);
    }
    printf("\n");
}

} // namespace

void Runner::run(
    MoERunnerArgs const& args, MoEWorkspace const& workspace, int device, cudaStream_t stream, int64_t configIndex)
{
    // Setup all operation data
    moe::dev::activation::Data activationData;
    moe::dev::finalize::Data finalizeData;
    moe::dev::convertsf::Data convertSfData;
    sync_check_cuda_error(stream);
    setOpsData(args, workspace, convertSfData, activationData, finalizeData);

    void* hidden_states_scale_linear{args.hidden_states_scale};

    auto const& config = mPassingConfigs[configIndex];
    bool const useActivationScale = hasActivationScale(args);
    bool const useStandaloneActivation = mUseStandaloneActivation || useActivationScale || args.mUseDeepSeekFp8;
    auto const activationDtype = resolveDtype(args.mDtypeGemm1Out, mDtypeGemm1Out);
    TLLM_CHECK_WITH_INFO(!useStandaloneActivation || mActType == ActType::SwiGlu,
        "Standalone activation is only supported for SwiGLU.");
    TLLM_CHECK_WITH_INFO(!useStandaloneActivation || activationDtype != btg::Dtype::E2m1,
        "Standalone activation is not supported for E2m1 activations.");
    TLLM_CHECK_WITH_INFO(!useStandaloneActivation || workspace.activation_output != nullptr,
        "Standalone activation requires activation output workspace.");
    TLLM_CHECK_WITH_INFO(!useActivationScale || workspace.routing_expert_indexes != nullptr,
        "Activation input/output scale factors require routing expert indexes.");
    TLLM_CHECK_WITH_INFO(!args.mUseDeepSeekFp8 || activationDtype == btg::Dtype::E4m3,
        "DeepSeek FP8 standalone activation is only supported for E4m3 activations.");
    TLLM_CHECK_WITH_INFO(!useActivationScale || activationDtype == btg::Dtype::E4m3
            || activationDtype == btg::Dtype::MxE4m3 || activationDtype == btg::Dtype::Bfloat16,
        "Activation input/output scale factors are only supported for E4m3, MxE4m3, and Bfloat16 activations.");
    TLLM_CHECK_WITH_INFO(
        !args.mUseDeepSeekFp8 || workspace.gemm1_output_scale != nullptr, "DeepSeek FP8 requires GEMM1 output scales.");
    TLLM_CHECK_WITH_INFO(!args.mUseDeepSeekFp8 || workspace.activation_output_scale != nullptr,
        "DeepSeek FP8 requires activation output scales.");
    TLLM_CHECK_WITH_INFO(!useActivationScale || activationDtype != btg::Dtype::MxE4m3
            || workspace.gemm1_output_scale != nullptr,
        "MxE4m3 activation input/output scale factors require GEMM1 output scales.");
    TLLM_CHECK_WITH_INFO(!useActivationScale || activationDtype != btg::Dtype::MxE4m3
            || workspace.activation_output_scale != nullptr,
        "MxE4m3 activation input/output scale factors require activation output scales.");
    auto& gemm1Runner = useStandaloneActivation ? mPermuteGemm1NonFused : mPermuteGemm1;
    int32_t gemm1Config = config.gemm1Config;
    if (useStandaloneActivation && !mUseStandaloneActivation)
    {
        gemm1Config = mPermuteGemm1NonFused.getDefaultValidConfigIndex(args.top_k, args.hidden_size,
            args.intermediate_size, args.local_num_experts, args.num_tokens,
            args.valid_hidden_size.value_or(args.hidden_size),
            args.valid_intermediate_size.value_or(args.intermediate_size));
    }

    // bool const dbg = (std::getenv("TLLM_MOE_DBG_TENSORS") != nullptr);
    const bool dbg = true;
    constexpr int kDbgElemCount = 16;
    int32_t dbgTotalNumPaddedTokens = workspace.total_max_padded_tokens;
    if (dbg && workspace.total_num_padded_tokens != nullptr)
    {
        cudaMemcpyAsync(&dbgTotalNumPaddedTokens, workspace.total_num_padded_tokens, sizeof(int32_t),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    if (dbg)
    {
        auto const getDbgCount = [&](int64_t elemCount)
        { return static_cast<int>(std::min<int64_t>(kDbgElemCount, std::max<int64_t>(elemCount, 0))); };
        int32_t const gemm1OutputHiddenDim = args.intermediate_size * (mActType == ActType::SwiGlu ? 2 : 1);
        int32_t const maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(
            args.num_tokens, args.top_k, args.local_num_experts, workspace.ProjUpTileN);
        int64_t const hiddenStatesScaleCount = args.mUseDeepSeekFp8
            ? static_cast<int64_t>(args.hidden_size / kFp8SfBlockSize) * args.num_tokens
            : static_cast<int64_t>(args.hidden_size / kMxSfBlockSize) * args.num_tokens;
        int64_t const gemm1WeightsScaleCount = args.mUseDeepSeekFp8
            ? static_cast<int64_t>(args.local_num_experts) * (gemm1OutputHiddenDim / kFp8SfBlockSize)
                * (args.hidden_size / kFp8SfBlockSize)
            : static_cast<int64_t>(args.local_num_experts) * gemm1OutputHiddenDim * (args.hidden_size / kMxSfBlockSize);

        printDeviceTensor("hidden_states (input)", args.hidden_states,
            getDbgCount(static_cast<int64_t>(args.num_tokens) * args.hidden_size), args.mDtypeElt, stream,
            getInputDebugScaleInfo(
                hidden_states_scale_linear, args.mDtypeElt, args.mUseDeepSeekFp8, args.hidden_size, args.num_tokens));
        printDeviceTensor("hidden_states_scale_linear (gemm1 input)", hidden_states_scale_linear,
            getDbgCount(hiddenStatesScaleCount), args.mUseDeepSeekFp8 ? btg::Dtype::Fp32 : btg::Dtype::UInt8, stream);
        printDeviceTensor("gemm1_weights (input)", args.gemm1_weights,
            getDbgCount(static_cast<int64_t>(args.local_num_experts) * gemm1OutputHiddenDim * args.hidden_size),
            mDtypeWeights, stream,
            getWeightDebugScaleInfo(args.gemm1_weights_scale, mDtypeWeights, args.mUseDeepSeekFp8, args.hidden_size));
        printDeviceTensor("gemm1_weights_scale (input)", args.gemm1_weights_scale, getDbgCount(gemm1WeightsScaleCount),
            args.mUseDeepSeekFp8 ? btg::Dtype::Fp32 : btg::Dtype::UInt8, stream);
        printDeviceTensor("expert_weights (gemm1 input)", workspace.expert_weights,
            getDbgCount(static_cast<int64_t>(args.num_tokens) * args.top_k), args.mDtypeExpW, stream);
        printDeviceTensor("output1_scales_scalar (gemm1 input)", args.output1_scales_scalar,
            getDbgCount(args.local_num_experts), btg::Dtype::Fp32, stream);
        printDeviceTensor("output1_scales_gate_scalar (gemm1 input)", args.output1_scales_gate_scalar,
            getDbgCount(args.local_num_experts), btg::Dtype::Fp32, stream);
        printDeviceTensor("gemm1_bias (input)", args.gemm1_bias,
            getDbgCount(static_cast<int64_t>(args.local_num_experts) * gemm1OutputHiddenDim), btg::Dtype::Fp32, stream);
        printDeviceTensor(
            "gemm1_alpha (input)", args.gemm1_alpha, getDbgCount(args.local_num_experts), btg::Dtype::Fp32, stream);
        printDeviceTensor(
            "gemm1_beta (input)", args.gemm1_beta, getDbgCount(args.local_num_experts), btg::Dtype::Fp32, stream);
        printDeviceTensor("gemm1_clamp_limit (input)", args.gemm1_clamp_limit, getDbgCount(args.local_num_experts),
            btg::Dtype::Fp32, stream);
        printDeviceTensor("permuted_idx_to_token_idx (gemm1 input)", workspace.permuted_idx_to_token_idx,
            getDbgCount(dbgTotalNumPaddedTokens), btg::Dtype::Int32, stream);
        printDeviceTensor("cta_idx_xy_to_batch_idx (gemm1 input)", workspace.cta_idx_xy_to_batch_idx,
            getDbgCount(maxNumCtasInBatchDim), btg::Dtype::Int32, stream);
        printDeviceTensor("cta_idx_xy_to_mn_limit (gemm1 input)", workspace.cta_idx_xy_to_mn_limit,
            getDbgCount(2LL * maxNumCtasInBatchDim), btg::Dtype::Int32, stream);
        printDeviceTensor(
            "num_non_exiting_ctas (gemm1 input)", workspace.num_non_exiting_ctas, 1, btg::Dtype::Int32, stream);
        printDeviceTensor(
            "total_num_padded_tokens (gemm1 input)", workspace.total_num_padded_tokens, 1, btg::Dtype::Int32, stream);
    }

    gemm1Runner.run(args.hidden_states, hidden_states_scale_linear, args.gemm1_weights, args.gemm1_weights_scale,
        workspace.expert_weights, args.output1_scales_scalar, args.output1_scales_gate_scalar, args.gemm1_bias,
        args.gemm1_alpha, args.gemm1_beta, args.gemm1_clamp_limit, workspace.gemm1_output, workspace.gemm1_output_scale,
        args.top_k, args.hidden_size, args.intermediate_size, args.local_num_experts, args.num_tokens,
        workspace.permuted_idx_to_token_idx, workspace.num_non_exiting_ctas, workspace.total_num_padded_tokens,
        workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit, workspace.bmm1_workspace,
        args.mUseRoutingScalesOnInput, device, stream, gemm1Config, args.valid_hidden_size.value_or(args.hidden_size),
        args.valid_intermediate_size.value_or(args.intermediate_size));

    if (dbg)
    {
        int32_t const gemm1OutputHiddenDim = args.intermediate_size * (mActType == ActType::SwiGlu ? 2 : 1);
        auto const gemm1OutputDtype = activationData.mDtypeElt;
        DebugScaleInfo const gemm1OutputScaleInfo = getOutputDebugScaleInfo(
            workspace.gemm1_output_scale, gemm1OutputDtype, gemm1OutputHiddenDim, dbgTotalNumPaddedTokens);
        printDeviceTensor("gemm1_output (post-PermuteGemm1)", workspace.gemm1_output, kDbgElemCount, gemm1OutputDtype,
            stream, gemm1OutputScaleInfo);
    }

    // Some dtype combinations require a standalone activation after FC1.
    void* gemm2_input = workspace.gemm1_output;
    void* gemm2_input_scale = workspace.gemm1_output_scale;
    if (useStandaloneActivation)
    {
        // Run activation
        moe::dev::activation::run(activationData, stream);
        gemm2_input = workspace.activation_output;
        gemm2_input_scale = workspace.activation_output_scale;

        if (dbg)
        {
            auto const activationOutputDtype = activationData.mDtypeElt;
            DebugScaleInfo const activationOutputScaleInfo = getOutputDebugScaleInfo(
                workspace.activation_output_scale, activationOutputDtype, args.intermediate_size, dbgTotalNumPaddedTokens);
            printDeviceTensor("activation_output (post-activation)", workspace.activation_output, kDbgElemCount,
                activationOutputDtype, stream, activationOutputScaleInfo);
        }
    }

    // Run gemm2
    mGemm2.run(gemm2_input, gemm2_input_scale, args.gemm2_weights, args.gemm2_weights_scale, args.output2_scales_scalar,
        args.gemm2_bias, workspace.gemm2_output, workspace.gemm2_output_scale, args.top_k,
        args.output_hidden_size.value_or(args.hidden_size), args.intermediate_size, args.local_num_experts,
        args.num_tokens, workspace.num_non_exiting_ctas, workspace.total_num_padded_tokens,
        workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit, workspace.bmm2_workspace, device, stream,
        config.gemm2Config, args.valid_hidden_size.value_or(args.hidden_size),
        args.valid_intermediate_size.value_or(args.intermediate_size));

    if (dbg)
    {
        DebugScaleInfo const gemm2OutputScaleInfo
            = getOutputDebugScaleInfo(workspace.gemm2_output_scale, btg::Dtype::Bfloat16,
                args.output_hidden_size.value_or(args.hidden_size), dbgTotalNumPaddedTokens);
        printDeviceTensor("gemm2_output (post-Gemm2)", workspace.gemm2_output, kDbgElemCount, btg::Dtype::Bfloat16,
            stream, gemm2OutputScaleInfo);
    }

    // Run finalize
    if (args.do_finalize)
    {
        // Run finalize
        TLLM_CHECK_WITH_INFO(args.finalize_input_scale == nullptr || workspace.routing_expert_indexes != nullptr,
            "Finalize input scale factors require routing expert indexes.");
        moe::dev::finalize::run(finalizeData, stream);
        sync_check_cuda_error(stream);

        if (dbg)
        {
            DebugScaleInfo const outputScaleInfo = getOutputDebugScaleInfo(args.output_scale, args.mDtypeOut,
                args.output_hidden_size.value_or(args.hidden_size), args.num_tokens);
            printDeviceTensor("output (post-finalize)", args.output, kDbgElemCount, args.mDtypeOut, stream,
                outputScaleInfo);
        }
    }
}
} // namespace MoE

} // namespace trtllmGenFp8BlockScaleMoe
} // namespace kernels

TRTLLM_NAMESPACE_END
