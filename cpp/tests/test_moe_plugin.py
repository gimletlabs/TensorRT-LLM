# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone MoE registration checks; requires TensorRT, NumPy, and a CUDA GPU.

Run with TRTLLM_PLUGIN_LIBRARY pointing at a library built with BUILD_MOE_PLUGIN=ON.
"""

import ctypes
import os
import unittest

import numpy as np
import tensorrt as trt


class MixtureOfExpertsPluginTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = trt.Logger(trt.Logger.WARNING)
        cls.library = ctypes.CDLL(os.environ["TRTLLM_PLUGIN_LIBRARY"], mode=ctypes.RTLD_GLOBAL)
        cls.register = cls.library.initTrtLlmPlugins
        cls.register.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        cls.register.restype = ctypes.c_bool

    def test_registration_and_creation(self):
        self.assertFalse(self.register(None, None))
        for namespace in ("tensorrt_llm", "gml", ""):
            with self.subTest(namespace=namespace):
                self.assertTrue(self.register(None, namespace.encode()))
                self.assertTrue(self.register(None, namespace.encode()))
                creator = trt.get_plugin_registry().get_plugin_creator(
                    "MixtureOfExperts", "1", namespace
                )
                self.assertIsNotNone(creator)
                names = [field.name for field in creator.field_names]
                self.assertEqual(len(names), len(set(names)))
                self.assertIn("force_determinism", names)
                for dtype in (trt.float16, trt.bfloat16):
                    values = dict.fromkeys(names, 0)
                    values.update(
                        remove_input_padding=1,
                        number_of_experts=4,
                        experts_per_token=2,
                        expert_hidden_size=128,
                        expert_inter_size=128,
                        activation_type=5,
                        group_size=-1,
                        type_id=int(dtype),
                        weight_type_id=int(dtype),
                        output_type_id=int(dtype),
                        use_final_scales=1,
                        tp_size=1,
                        ep_size=1,
                    )
                    arrays = {
                        key: np.array([value], dtype=np.int32) for key, value in values.items()
                    }
                    fields = trt.PluginFieldCollection(
                        [
                            trt.PluginField(key, value, trt.PluginFieldType.INT32)
                            for key, value in arrays.items()
                        ]
                    )
                    plugin = creator.create_plugin("moe", fields)
                    self.assertIsNotNone(plugin)
                    self.assertEqual(plugin.plugin_namespace, namespace)
                    self.assertEqual(plugin.num_outputs, 1)
                    clone = plugin.clone()
                    self.assertEqual(clone.plugin_namespace, namespace)
                    self.assertEqual(clone.plugin_type, "MixtureOfExperts")


if __name__ == "__main__":
    unittest.main()
