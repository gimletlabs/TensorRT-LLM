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

#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#ifdef BUILD_GPT_ATTENTION_PLUGIN
#include "tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h"
#endif
#ifdef BUILD_MOE_PLUGIN
#include "tensorrt_llm/plugins/cudaStreamPlugin/cudaStreamPlugin.h"
#include "tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#endif

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

extern "C" bool initTrtLlmPlugins(void* logger, char const* libNamespace)
{
    // Retain the legacy API signature; AttentionOp uses the TensorRT-LLM logger.
    (void) logger;
    if (libNamespace == nullptr)
    {
        return false;
    }
    try
    {
        static std::mutex mutex;
        using Creators = std::vector<std::unique_ptr<nvinfer1::IPluginCreator>>;
        static std::map<std::string, Creators> creators;
        std::lock_guard<std::mutex> lock(mutex);
        if (creators.contains(libNamespace))
        {
            return true;
        }
        auto* registry = getPluginRegistry();
        if (registry == nullptr)
        {
            return false;
        }
        Creators pending;
#ifdef BUILD_GPT_ATTENTION_PLUGIN
        pending.push_back(std::make_unique<tensorrt_llm::plugins::GPTAttentionPluginCreator>());
#endif
#ifdef BUILD_MOE_PLUGIN
        pending.push_back(std::make_unique<tensorrt_llm::plugins::MixtureOfExpertsPluginCreator>());
        pending.push_back(std::make_unique<tensorrt_llm::plugins::CudaStreamPluginCreator>());
#endif
        auto const entry = creators.emplace(libNamespace, std::move(pending)).first;
        size_t registered = 0;
        for (auto& creator : entry->second)
        {
            creator->setPluginNamespace(libNamespace);
            if (!registry->registerCreator(*creator, libNamespace))
            {
                for (size_t i = 0; i < registered; ++i)
                {
                    registry->deregisterCreator(*entry->second[i]);
                }
                creators.erase(entry);
                return false;
            }
            ++registered;
        }
        return true;
    }
    catch (std::exception const&)
    {
        return false;
    }
}
