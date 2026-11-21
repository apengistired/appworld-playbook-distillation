import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from remap_lora_for_vllm import convert_adapter, remap_tensor_keys

ORIGINAL = "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight"
MAPPED = "base_model.model.language_model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight"
LM_HEAD = "base_model.model.lm_head.lora_A.weight"


class RemapTests(unittest.TestCase):
    def test_remaps_causal_layers_and_preserves_other_keys(self):
        self.assertEqual(remap_tensor_keys([ORIGINAL, LM_HEAD]), {ORIGINAL: MAPPED, LM_HEAD: LM_HEAD})

    def test_mapping_is_idempotent(self):
        first = remap_tensor_keys([ORIGINAL, LM_HEAD])
        self.assertEqual(remap_tensor_keys(first.values()), {MAPPED: MAPPED, LM_HEAD: LM_HEAD})

    def test_collision_is_detected(self):
        with self.assertRaisesRegex(ValueError, "collision"):
            remap_tensor_keys([ORIGINAL, MAPPED])

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "safetensors is not installed")
    def test_file_round_trip_preserves_bf16_metadata_config_and_source(self):
        from safetensors import deserialize

        payload = bytes.fromhex("803f004000008040")
        header = {
            "__metadata__": {"format": "pt", "test": "metadata preserved"},
            ORIGINAL: {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
            LM_HEAD: {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        }
        metadata = json.dumps(header).encode()
        metadata += b" " * (-len(metadata) % 8)
        original = struct.pack("<Q", len(metadata)) + metadata + payload
        with tempfile.TemporaryDirectory() as tmp:
            src, dst, twice = (Path(tmp) / name for name in ("src", "dst", "twice"))
            src.mkdir()
            (src / "adapter_model.safetensors").write_bytes(original)
            (src / "adapter_config.json").write_text('{"peft_type":"LORA"}')
            (src / "train_log.json").write_text('{"sequences":105}')
            (src / "notes.log").write_text("checkpoint complete\n")

            self.assertEqual(convert_adapter(src, dst), (1, 1))
            converted = (dst / "adapter_model.safetensors").read_bytes()
            self.assertEqual(dict(deserialize(converted)), {
                MAPPED: dict(deserialize(original))[ORIGINAL],
                LM_HEAD: dict(deserialize(original))[LM_HEAD],
            })
            header_size, = struct.unpack_from("<Q", converted)
            self.assertEqual(json.loads(converted[8:8 + header_size])["__metadata__"], header["__metadata__"])
            self.assertEqual(converted[8 + header_size:], payload)
            for name in ("adapter_config.json", "train_log.json", "notes.log"):
                self.assertEqual((src / name).read_bytes(), (dst / name).read_bytes())
            self.assertEqual((src / "adapter_model.safetensors").read_bytes(), original)
            self.assertEqual(convert_adapter(dst, twice), (0, 2))
            self.assertEqual((twice / "adapter_model.safetensors").read_bytes(), converted)
            with self.assertRaises(FileExistsError):
                convert_adapter(src, dst)


if __name__ == "__main__":
    unittest.main()
