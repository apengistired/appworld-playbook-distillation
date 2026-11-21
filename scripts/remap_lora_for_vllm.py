"""Adapt a Qwen3.5 causal-LM LoRA checkpoint for vLLM's conditional-generation model."""
from __future__ import annotations

import argparse
import json
import shutil
import struct
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory

SOURCE_PREFIX = "base_model.model.model."
VLLM_PREFIX = "base_model.model.language_model.model."
WEIGHTS = "adapter_model.safetensors"
SUPPORTING_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml", ".jinja", ".model"}
HEADER_SIZE = struct.Struct("<Q")


def remap_tensor_keys(keys: Iterable[str]) -> dict[str, str]:
    """Return the original-to-serving key map, rejecting any destination collision."""
    remapped = {}
    destinations = set()
    for key in keys:
        target = VLLM_PREFIX + key[len(SOURCE_PREFIX):] if key.startswith(SOURCE_PREFIX) else key
        if target in destinations:
            raise ValueError(f"Tensor key collision after remapping: {target}")
        remapped[key] = target
        destinations.add(target)
    return remapped


def convert_adapter(src: Path, dst: Path) -> tuple[int, int]:
    """Write a separate copy of a validated safetensors adapter and its supporting files."""
    from safetensors import SafetensorError, deserialize

    src, dst = src.resolve(), dst.resolve()
    if src == dst or src in dst.parents:
        raise ValueError("Destination must be outside the source adapter directory")
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    config = src / "adapter_config.json"
    if not config.is_file():
        raise FileNotFoundError(config)

    data = (src / WEIGHTS).read_bytes()
    try:
        tensors = deserialize(data)
    except SafetensorError as error:
        raise ValueError(f"Invalid safetensors file {src / WEIGHTS}: {error}") from error
    key_map = remap_tensor_keys(key for key, _ in tensors)
    del tensors
    payload_start = HEADER_SIZE.size + HEADER_SIZE.unpack_from(data)[0]
    header = json.loads(data[HEADER_SIZE.size:payload_start])
    renamed_header = {key_map.get(key, key): value for key, value in header.items()}
    # Offsets are relative to the data buffer. Keep that buffer byte-for-byte, including BF16.
    encoded = json.dumps(renamed_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".lora-remap-", dir=dst.parent) as work:
        staging = Path(work) / "adapter"
        staging.mkdir()
        for file in src.iterdir():
            if file.is_file() and not file.name.startswith(".") and file.suffix in SUPPORTING_SUFFIXES:
                shutil.copy2(file, staging / file.name)
        with (staging / WEIGHTS).open("wb") as out:
            out.write(HEADER_SIZE.pack(len(encoded)))
            out.write(encoded)
            out.write(memoryview(data)[payload_start:])
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
        staging.rename(dst)

    converted = sum(source != target for source, target in key_map.items())
    return converted, len(key_map) - converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Existing PEFT adapter directory")
    parser.add_argument("--dst", type=Path, required=True, help="New directory for the vLLM adapter")
    args = parser.parse_args()
    try:
        converted, retained = convert_adapter(args.src, args.dst)
    except ImportError:
        parser.error("Install safetensors to convert adapter weights")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Converted {converted} tensors, retained {retained} tensors. Wrote {args.dst}")


if __name__ == "__main__":
    main()
