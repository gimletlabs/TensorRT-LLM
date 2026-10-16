# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
target_sources(
  ${SHARED_TARGET}
  PRIVATE api/tllmPlugin.cpp
          common/checkMacrosPlugin.cpp
          common/plugin.cpp
          gptAttentionCommon/gptAttentionCommon.cpp
          gptAttentionPlugin/gptAttentionPlugin.cpp)
target_include_directories(${SHARED_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_compile_definitions(${SHARED_TARGET} PRIVATE BUILD_GPT_ATTENTION_PLUGIN)

find_library(NVINFER_LIBRARY NAMES nvinfer REQUIRED)
target_link_libraries(${SHARED_TARGET} PRIVATE ${NVINFER_LIBRARY})
