# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local standalone plugin checks; requires TensorRT, NumPy, PyTorch, and a CUDA GPU.

Run with TRTLLM_PLUGIN_LIBRARY pointing at libtensorrt_llm.so built with
BUILD_GPT_ATTENTION_PLUGIN=ON. Does not import the TensorRT-LLM Python package.
"""

import ctypes
import os
import unittest

import numpy as np
import tensorrt as trt


class GPTAttentionPluginTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = trt.Logger(trt.Logger.WARNING)
        cls.library = ctypes.CDLL(os.environ["TRTLLM_PLUGIN_LIBRARY"], mode=ctypes.RTLD_GLOBAL)
        cls.register = cls.library.initTrtLlmPlugins
        cls.register.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        cls.register.restype = ctypes.c_bool

    def test_registration(self):
        self.assertFalse(self.register(None, None))
        for namespace in (b"tensorrt_llm", b"gml", b""):
            self.assertTrue(self.register(None, namespace))
            self.assertTrue(self.register(None, namespace))
            creator = trt.get_plugin_registry().get_plugin_creator(
                "GPTAttention", "1", namespace.decode()
            )
            self.assertIsNotNone(creator)
            self.assertEqual(creator.plugin_namespace, namespace.decode())

    def create_plugin(self, use_sinks):
        self.assertTrue(self.register(None, b"gml"))
        creator = trt.get_plugin_registry().get_plugin_creator("GPTAttention", "1", "gml")
        integers = dict.fromkeys(INT_FIELDS, 0)
        integers.update(
            num_heads=8,
            num_kv_heads=2,
            num_kv_heads_origin=2,
            head_size=64,
            unidirectional=1,
            tp_size=1,
            cp_size=1,
            tokens_per_block=64,
            type_id=int(trt.float16),
            max_context_length=128,
            remove_input_padding=1,
            context_fmha_type=1,
            mask_type=1,
            use_attention_sinks=use_sinks,
        )
        floats = dict.fromkeys(FLOAT_FIELDS, 1.0)
        floats.update(attn_logit_softcapping_scale=0.0, rotary_embedding_base=10000.0)
        arrays = {name: np.array([value], dtype=np.int32) for name, value in integers.items()}
        arrays.update({name: np.array([value], dtype=np.float32) for name, value in floats.items()})
        arrays["cp_group"] = np.array([0], dtype=np.int32)
        fields = [
            trt.PluginField(
                name,
                value,
                trt.PluginFieldType.FLOAT32
                if value.dtype == np.float32
                else trt.PluginFieldType.INT32,
            )
            for name, value in arrays.items()
        ]
        plugin = creator.create_plugin("attention", trt.PluginFieldCollection(fields))
        self.assertIsNotNone(plugin)
        return plugin

    def test_serialization_with_and_without_attention_sinks(self):
        serialized = []
        for use_sinks in (0, 1):
            plugin = self.create_plugin(use_sinks)
            self.assertEqual(plugin.plugin_namespace, "gml")
            data = bytes(plugin.serialize())
            creator = trt.get_plugin_registry().get_plugin_creator("GPTAttention", "1", "gml")
            restored = creator.deserialize_plugin("restored", data)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.plugin_namespace, "gml")
            self.assertEqual(bytes(restored.serialize()), data)
            serialized.append(data)
        self.assertNotEqual(*serialized)

    def test_prefill_matches_reference(self):
        import torch

        torch.manual_seed(7)
        for use_sinks in (0, 1):
            with self.subTest(attention_sinks=use_sinks):
                plugin = self.create_plugin(use_sinks)
                builder = trt.Builder(self.logger)
                network = builder.create_network()
                config = builder.create_builder_config()
                config.set_flag(trt.BuilderFlag.FP16)
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
                # Packed QKV: eight query heads, two key/value heads, head size 64.
                qkv = torch.randn((16, 768), dtype=torch.float16, device="cuda")
                values = [
                    ("qkv", qkv, trt.float16),
                    ("window", torch.tensor([128], dtype=torch.int32), trt.int32),
                    ("sink_length", torch.tensor([0], dtype=torch.int32), trt.int32),
                    ("lengths", torch.tensor([16], dtype=torch.int32, device="cuda"), trt.int32),
                    ("request_types", torch.tensor([0], dtype=torch.int32), trt.int32),
                ]
                if use_sinks:
                    values.append(("sinks", torch.linspace(-1, 1, 8, device="cuda"), trt.float32))
                values.extend(
                    [
                        ("host_lengths", torch.tensor([16], dtype=torch.int32), trt.int32),
                        ("perf_knobs", torch.zeros(16, dtype=torch.int64), trt.int64),
                        ("progress", torch.zeros(1, dtype=torch.int64), trt.int64),
                    ]
                )
                inputs = []
                for name, value, dtype in values:
                    tensor = network.add_input(name, dtype, tuple(value.shape))
                    tensor.location = (
                        trt.TensorLocation.DEVICE if value.is_cuda else trt.TensorLocation.HOST
                    )
                    inputs.append(tensor)
                layer = network.add_plugin_v2(inputs, plugin)
                output = layer.get_output(0)
                output.name = "output"
                output.dtype = trt.float16
                network.mark_output(output)
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
                self.assertEqual(engine.get_tensor_dtype("output"), trt.float16)
                context = engine.create_execution_context()
                result = torch.empty((16, 512), dtype=torch.float16, device="cuda")
                for name, value, _ in values:
                    self.assertTrue(context.set_tensor_address(name, value.data_ptr()))
                self.assertTrue(context.set_tensor_address("output", result.data_ptr()))
                self.assertTrue(context.execute_async_v3(torch.cuda.current_stream().cuda_stream))
                torch.cuda.synchronize()
                q, k, v = qkv.float().split([512, 128, 128], dim=-1)
                q = q.reshape(16, 8, 64).transpose(0, 1)
                k = k.reshape(16, 2, 64).transpose(0, 1).repeat_interleave(4, dim=0)
                v = v.reshape(16, 2, 64).transpose(0, 1).repeat_interleave(4, dim=0)
                scores = q @ k.transpose(-1, -2) / 8
                scores.masked_fill_(
                    torch.ones((16, 16), device="cuda", dtype=torch.bool).triu(1), -float("inf")
                )
                if use_sinks:
                    sinks = dict((name, value) for name, value, _ in values)["sinks"]
                    scores = torch.cat([scores, sinks[:, None, None].expand(8, 16, 1)], dim=-1)
                probabilities = scores.softmax(dim=-1)[..., :16]
                expected = (probabilities @ v).transpose(0, 1).reshape(16, 512)
                torch.testing.assert_close(result.float(), expected, atol=3e-3, rtol=3e-3)


INT_FIELDS = [
    "layer_idx",
    "num_heads",
    "vision_start",
    "vision_length",
    "num_kv_heads",
    "num_kv_heads_origin",
    "head_size",
    "unidirectional",
    "position_embedding_type",
    "rotary_embedding_dim",
    "rotary_embedding_scale_type",
    "rotary_embedding_max_positions",
    "rotary_embedding_original_max_positions",
    "tp_size",
    "tp_rank",
    "unfuse_qkv_gemm",
    "use_logn_scaling",
    "use_attention_sinks",
    "context_fmha_type",
    "kv_cache_quant_mode",
    "remove_input_padding",
    "mask_type",
    "block_sparse_block_size",
    "block_sparse_homo_head_pattern",
    "block_sparse_num_local_blocks",
    "block_sparse_vertical_stride",
    "paged_kv_cache",
    "tokens_per_block",
    "type_id",
    "max_context_length",
    "qkv_bias_enabled",
    "do_cross_attention",
    "max_distance",
    "pos_shift_enabled",
    "dense_context_fmha",
    "use_paged_context_fmha",
    "use_fp8_context_fmha",
    "has_full_attention_mask",
    "use_cache",
    "is_spec_decoding_enabled",
    "spec_decoding_is_generation_length_variable",
    "spec_decoding_max_generation_length",
    "is_mla_enabled",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "fuse_fp4_quant",
    "skip_attn",
    "cp_size",
    "cp_rank",
]
FLOAT_FIELDS = [
    "q_scaling",
    "attn_logit_softcapping_scale",
    "rotary_embedding_base",
    "rotary_embedding_scale",
    "rotary_embedding_short_m_scale",
    "rotary_embedding_long_m_scale",
]


if __name__ == "__main__":
    unittest.main()
