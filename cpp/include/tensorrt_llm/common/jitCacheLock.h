/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/config.h"

#include <mutex>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

//! \brief Scoped lock for one family of JIT-compiled kernels.
//!
//! Caches of JIT-compiled kernels are shared by every thread in the process, including threads that drive different
//! devices, and are not thread-safe by themselves. Hold this lock while looking up, compiling into or inserting into
//! such a cache, and for as long as you use anything the cache returned that it may still move or free.
//!
//! \tparam Tag A type that names the kernel family. Each family gets its own process-wide mutex, so families do not
//! wait on each other. The tag must have external linkage (not in an anonymous namespace); otherwise every translation
//! unit gets a different mutex.
template <typename Tag>
class JitCacheLock
{
public:
    JitCacheLock()
        : mLock(getMutex())
    {
    }

private:
    static std::mutex& getMutex()
    {
        static std::mutex sMutex;
        return sMutex;
    }

    std::lock_guard<std::mutex> mLock;
};

} // namespace common

TRTLLM_NAMESPACE_END
