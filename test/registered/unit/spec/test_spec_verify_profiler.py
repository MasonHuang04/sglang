import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.spec_verify_profiler import run_with_spec_verify_profile
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSpecVerifyProfiler(CustomTestCase):
    def test_disabled_profiler_does_not_write_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.jsonl"

            with (
                envs.SGLANG_DEBUG_SPEC_VERIFY_PROFILE.override(False),
                envs.SGLANG_DEBUG_SPEC_VERIFY_PROFILE_PATH.override(profile_path),
            ):
                result = run_with_spec_verify_profile(
                    lambda: "ok",
                    algorithm="DFLASH",
                    batch_size=2,
                    draft_token_num=4,
                    can_run_cuda_graph=False,
                    device="cpu",
                )

            self.assertEqual(result, "ok")
            self.assertFalse(profile_path.exists())

    def test_enabled_profiler_writes_expected_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile_{tp_rank}_{dp_rank}.jsonl"
            seq_lens = torch.tensor([7, 9], dtype=torch.int32)

            with (
                envs.SGLANG_DEBUG_SPEC_VERIFY_PROFILE.override(True),
                envs.SGLANG_DEBUG_SPEC_VERIFY_PROFILE_PATH.override(profile_path),
            ):
                result = run_with_spec_verify_profile(
                    lambda: "ok",
                    algorithm="DFLASH",
                    batch_size=2,
                    draft_token_num=4,
                    can_run_cuda_graph=True,
                    device="cpu",
                    seq_lens=seq_lens,
                    seq_lens_sum=16,
                    tp_rank=3,
                    dp_rank=5,
                    attention_backend="UnitTestBackend",
                )

            self.assertEqual(result, "ok")
            resolved_path = Path(tmpdir) / "profile_3_5.jsonl"
            records = resolved_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 1)

            record = json.loads(records[0])
            self.assertEqual(record["algorithm"], "DFLASH")
            self.assertEqual(record["batch_size"], 2)
            self.assertEqual(record["draft_token_num"], 4)
            self.assertEqual(record["num_verify_tokens"], 8)
            self.assertEqual(record["num_proposed_drafts"], 6)
            self.assertEqual(record["seq_lens_sum"], 16)
            self.assertEqual(record["seq_lens_max"], 9)
            self.assertEqual(record["tp_rank"], 3)
            self.assertEqual(record["dp_rank"], 5)
            self.assertEqual(record["device"], "cpu")
            self.assertEqual(record["attention_backend"], "UnitTestBackend")
            self.assertTrue(record["can_run_cuda_graph"])
            self.assertGreaterEqual(record["latency_ms"], 0)


if __name__ == "__main__":
    unittest.main()
