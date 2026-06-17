import unittest

from sglang.srt.speculative.k_hw_roofline import (
    TargetModelShape,
    achieved_tflops,
    draft_lens_from_roofline,
    estimate_dense_weight_elements,
    estimate_k_roof,
    estimate_target_verify_flops,
    target_model_shape_from_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKhwRoofline(CustomTestCase):
    def test_shape_from_config_defaults_kv_heads_to_attention_heads(self):
        shape = target_model_shape_from_config(
            {
                "num_hidden_layers": 2,
                "hidden_size": 16,
                "intermediate_size": 64,
                "num_attention_heads": 4,
            }
        )

        self.assertEqual(shape.num_key_value_heads, 4)
        self.assertEqual(shape.head_dim, 4)
        self.assertEqual(shape.kv_hidden_size, 16)

    def test_dense_weight_elements_match_qkv_o_mlp_components(self):
        shape = TargetModelShape(
            num_hidden_layers=2,
            hidden_size=16,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        qkv = 16 * (16 + 2 * 8)
        output = 16 * 16
        mlp = 3 * 16 * 64
        self.assertEqual(estimate_dense_weight_elements(shape), 2 * (qkv + output + mlp))

    def test_k_roof_uses_dense_arithmetic_intensity_condition(self):
        self.assertEqual(
            estimate_k_roof(
                peak_tflops=100,
                memory_bw_gbps=1000,
                bytes_per_weight=2,
            ),
            100.0,
        )

    def test_draft_lens_from_roofline_are_unique_and_clamped(self):
        draft_lens = draft_lens_from_roofline(
            batch_size=32,
            k_roof=128,
            multipliers=[0.5, 1.0, 1.0, 4.0],
            max_draft_len=8,
        )

        self.assertEqual(draft_lens, [2, 4, 8])

    def test_estimated_verify_flops_scale_with_k(self):
        shape = TargetModelShape(
            num_hidden_layers=2,
            hidden_size=16,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        small = estimate_target_verify_flops(
            shape=shape,
            batch_size=2,
            draft_token_num=2,
            seq_lens_sum=16,
        )
        large = estimate_target_verify_flops(
            shape=shape,
            batch_size=2,
            draft_token_num=4,
            seq_lens_sum=16,
        )

        self.assertEqual(large["dense_flops"], 2 * small["dense_flops"])
        self.assertEqual(large["attention_flops"], 2 * small["attention_flops"])
        self.assertEqual(small["avg_seq_len"], 8)

    def test_achieved_tflops_returns_none_for_zero_latency(self):
        self.assertIsNone(achieved_tflops(1e12, 0))
        self.assertEqual(achieved_tflops(1e12, 1000), 1.0)


if __name__ == "__main__":
    unittest.main()
