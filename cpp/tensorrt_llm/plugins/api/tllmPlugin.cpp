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
#include "tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>

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
        static std::map<std::string, std::unique_ptr<tensorrt_llm::plugins::GPTAttentionPluginCreator>> creators;
        std::lock_guard<std::mutex> lock(mutex);
        if (creators.contains(libNamespace))
        {
            return true;
        }
        auto creator = std::make_unique<tensorrt_llm::plugins::GPTAttentionPluginCreator>();
        creator->setPluginNamespace(libNamespace);
        auto* registry = getPluginRegistry();
        if (registry == nullptr)
        {
            return false;
        }
        auto const entry = creators.emplace(libNamespace, std::move(creator)).first;
        if (!registry->registerCreator(*entry->second, libNamespace))
        {
            creators.erase(entry);
            return false;
        }
        return true;
    }
    catch (std::exception const&)
    {
        return false;
    }
}
