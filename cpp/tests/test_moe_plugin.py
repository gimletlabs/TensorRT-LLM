# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone MoE plugin checks; requires TensorRT, NumPy, PyTorch, and a CUDA GPU.

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

    def create_plugin(self, creator, dtype, **overrides):
        values = dict.fromkeys((field.name for field in creator.field_names), 0)
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
        values.update(overrides)
        arrays = {key: np.array([value], dtype=np.int32) for key, value in values.items()}
        fields = trt.PluginFieldCollection(
            [
                trt.PluginField(key, value, trt.PluginFieldType.INT32)
                for key, value in arrays.items()
            ]
        )
        return creator.create_plugin("moe", fields)

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
                    plugin = self.create_plugin(creator, dtype)
                    self.assertIsNotNone(plugin)
                    self.assertEqual(plugin.plugin_namespace, namespace)
                    self.assertEqual(plugin.num_outputs, 1)
                    clone = plugin.clone()
                    self.assertEqual(clone.plugin_namespace, namespace)
                    self.assertEqual(clone.plugin_type, "MixtureOfExperts")

    def test_side_stream_outputs_and_capture(self):
        import torch

        self.assertTrue(self.register(None, b"tensorrt_llm"))
        creator = trt.get_plugin_registry().get_plugin_creator(
            "MixtureOfExperts", "1", "tensorrt_llm"
        )
        for side_stream_id, output_width in ((0, 128), (1, 128)):
            with self.subTest(side_stream_id=side_stream_id):
                plugin = self.create_plugin(
                    creator,
                    trt.float16,
                    side_stream_id=side_stream_id,
                    expert_unpadded_hidden_size=output_width,
                )
                tokens, hidden, experts = 8, 128, 4
                x = torch.full((tokens, hidden), 0.5, dtype=torch.float16, device="cuda")
                fc1 = torch.zeros((experts, 2 * hidden, hidden), dtype=torch.float16, device="cuda")
                fc2 = torch.zeros((experts, hidden, hidden), dtype=torch.float16, device="cuda")
                diagonal = torch.arange(hidden, device="cuda")
                fc1[:, diagonal, diagonal] = 0.5
                fc1[:, hidden + diagonal, diagonal] = 0.25
                fc2[:, diagonal, diagonal] = 2.0
                selected = torch.tensor([[0, 1]] * tokens, dtype=torch.int32, device="cuda")
                scales = torch.tensor([[0.25, 0.75]] * tokens, dtype=torch.float32, device="cuda")
                values = [
                    ("x", x, trt.float16),
                    ("fc1", fc1, trt.float16),
                    ("fc2", fc2, trt.float16),
                    ("selected", selected, trt.int32),
                    ("scales", scales, trt.float32),
                ]
                if side_stream_id:
                    dummy = torch.arange(
                        tokens * hidden, dtype=torch.float16, device="cuda"
                    ).reshape(tokens, hidden)
                    values.append(("dummy", dummy, trt.float16))

                builder = trt.Builder(self.logger)
                network = builder.create_network()
                config = builder.create_builder_config()
                config.set_flag(trt.BuilderFlag.FP16)
                inputs = [
                    network.add_input(name, dtype, tuple(value.shape))
                    for name, value, dtype in values
                ]
                layer = network.add_plugin_v2(inputs, plugin)
                self.assertEqual(tuple(layer.get_output(0).shape), (tokens, output_width))
                if side_stream_id:
                    dummy_out = layer.get_output(1)
                    self.assertEqual(tuple(dummy_out.shape), tuple(dummy.shape))
                    dummy_out.name = "dummy_out"
                    network.mark_output(dummy_out)
                    dummy_out.dtype = trt.float16
                moe_output = layer.get_output(0)
                if side_stream_id:
                    join_creator = trt.get_plugin_registry().get_plugin_creator(
                        "CudaStream", "1", "tensorrt_llm"
                    )
                    self.assertIsNotNone(join_creator)
                    join_plugin = self.create_plugin(
                        join_creator, trt.float16, side_stream_id=side_stream_id, num_inputs=2
                    )
                    join = network.add_plugin_v2([moe_output, dummy_out], join_plugin)
                    moe_output = join.get_output(0)
                # CudaStream joins the expert work before downstream consumers run.
                one = network.add_constant((1, 1), np.ones((1, 1), dtype=np.float16))
                result = network.add_elementwise(
                    moe_output, one.get_output(0), trt.ElementWiseOperation.SUM
                ).get_output(0)
                result.name = "result"
                network.mark_output(result)
                result.dtype = trt.float16
                previous = os.environ.get("IS_BUILDING")
                os.environ["IS_BUILDING"] = "1"
                try:
                    plan = builder.build_serialized_network(network, config)
                finally:
                    if previous is None:
                        os.environ.pop("IS_BUILDING")
                    else:
                        os.environ["IS_BUILDING"] = previous
                self.assertIsNotNone(plan)
                runtime = trt.Runtime(self.logger)
                engine = runtime.deserialize_cuda_engine(plan)
                self.assertIsNotNone(engine)
                self.assertEqual(engine.get_tensor_dtype("result"), trt.float16)
                context = engine.create_execution_context()
                output = torch.empty((tokens, output_width), dtype=torch.float16, device="cuda")
                for name, value, _ in values:
                    self.assertTrue(context.set_tensor_address(name, value.data_ptr()))
                self.assertTrue(context.set_tensor_address("result", output.data_ptr()))
                if side_stream_id:
                    dummy_output = torch.empty_like(dummy)
                    self.assertTrue(
                        context.set_tensor_address("dummy_out", dummy_output.data_ptr())
                    )
                caller = torch.cuda.Stream()
                caller.wait_stream(torch.cuda.current_stream())
                for input_value in (0.5, -0.5):
                    with torch.cuda.stream(caller):
                        x.fill_(input_value)
                        output.fill_(float("nan"))
                        self.assertTrue(context.execute_async_v3(caller.cuda_stream))
                    caller.synchronize()
                    expected = 1 + 2 * (input_value * 0.5) * (input_value * 0.25) / (
                        1 + np.exp(-input_value * 0.25)
                    )
                    torch.testing.assert_close(
                        output, torch.full_like(output, expected), atol=2e-3, rtol=2e-3
                    )
                    if side_stream_id:
                        torch.testing.assert_close(dummy_output, dummy, atol=0, rtol=0)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=caller):
                    self.assertTrue(context.execute_async_v3(caller.cuda_stream))
                with torch.cuda.stream(caller):
                    output.fill_(float("nan"))
                    graph.replay()
                caller.synchronize()
                torch.testing.assert_close(
                    output, torch.full_like(output, expected), atol=2e-3, rtol=2e-3
                )
                if side_stream_id:
                    torch.testing.assert_close(dummy_output, dummy, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
