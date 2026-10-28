"""SCAN and merged short-Dyck parsing with fixed-shape JAX batches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from .serial_model_jax import SCAN_ACTIONS, SCAN_EOS_BASIS
except ImportError:
    from serial_model_jax import SCAN_ACTIONS, SCAN_EOS_BASIS


PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
PAD_ID, BOS_ID, EOS_ID, UNK_ID = range(4)

_SPLIT_FILES = {
    "train": "tasks_train_length.txt",
    "dev": "tasks_dev_length.txt",
    "test": "tasks_test_length.txt",
}

DYCK_TOKENS = (")", "]", "}", ">")
DYCK_SPLITS = ("train", "dev", "test", "gen_length", "gen_depth")


def _dyck_root(data_root):
    """Dyck uses an explicit merged-set directory, never a k-specific view."""
    root = Path(data_root)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return None
    info = json.loads(manifest.read_text(encoding="utf-8"))
    if info.get("name") != "DyckLanguage" or info.get("is_view") or info.get("k") is not None:
        raise ValueError("Pass the merged DyckLanguage directory, not a k view.")
    return root


def readout_metadata(target_vocab):
    tokens = target_vocab.tokens[4:]
    if tokens not in (SCAN_ACTIONS, DYCK_TOKENS):
        raise ValueError("Unsupported direct-readout vocabulary.")
    return {
        "task": "dyck_merged" if tokens == DYCK_TOKENS else "scan",
        "output_tokens": list(tokens),
        "eos_basis": SCAN_EOS_BASIS,
        "invalid_bases": [i for i in range(8) if i >= len(tokens) and i != SCAN_EOS_BASIS],
    }


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]

    def __post_init__(self):
        if self.tokens[:4] != SPECIAL_TOKENS:
            raise ValueError("Vocabulary must begin with PAD/BOS/EOS/UNK.")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary contains duplicate tokens.")

    @property
    def token_to_id(self):
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, tokens: Sequence[str], allow_unk=False):
        lookup = self.token_to_id
        result = []
        for token in tokens:
            if token not in lookup:
                if not allow_unk:
                    raise ValueError(f"Unexpected out-of-vocabulary token {token!r}.")
                result.append(UNK_ID)
            else:
                result.append(lookup[token])
        return tuple(result)

    def decode(self, ids: Sequence[int]):
        result = []
        for index in ids:
            index = int(index)
            if index < 0 or index >= len(self.tokens):
                raise ValueError(f"Vocabulary ID {index} is out of range.")
            result.append(self.tokens[index])
        return tuple(result)

    def __len__(self):
        return len(self.tokens)


@dataclass(frozen=True)
class ScanExample:
    uid: str
    src_tokens: tuple[str, ...]
    tgt_tokens: tuple[str, ...]


@dataclass(frozen=True)
class EncodedScanExample:
    uid: str
    src_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    target_bases: tuple[int, ...]


def _read_vocab(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if any(token in SPECIAL_TOKENS for token in raw):
        raise ValueError(f"{path} must not repeat reserved special tokens.")
    return Vocabulary(SPECIAL_TOKENS + raw)


def load_vocabularies(data_root):
    dyck = _dyck_root(data_root)
    if dyck is not None:
        raw = json.loads((dyck / "vocab.json").read_text(encoding="utf-8"))
        specials = {"<pad>", "<unk>", "<bos>", "<eos>"}
        source = tuple(token for token, _ in sorted(raw["source"].items(), key=lambda pair: pair[1]) if token not in specials)
        target = set(raw["target"]) - specials
        if set(source) != set("()[]{}<>") or target != set(DYCK_TOKENS):
            raise ValueError("Merged Dyck requires eight source brackets and four closing brackets.")
        # Rebuild host IDs rather than importing the dataset's different special IDs.
        return Vocabulary(SPECIAL_TOKENS + source), Vocabulary(SPECIAL_TOKENS + DYCK_TOKENS)
    root = Path(data_root) / "SCAN-length"
    source = _read_vocab(root / "vocab.src.txt")
    target = _read_vocab(root / "vocab.tgt.txt")
    if tuple(target.tokens[4:]) != SCAN_ACTIONS:
        raise ValueError(
            "SCAN target vocabulary order must equal the six direct-readout actions."
        )
    return source, target


def parse_scan_line(line: str, uid: str):
    line = line.strip()
    if not line.startswith("IN: ") or " OUT: " not in line:
        raise ValueError(f"{uid}: expected 'IN: ... OUT: ...'.")
    source_text, target_text = line[4:].split(" OUT: ", 1)
    source = tuple(source_text.split())
    target = tuple(target_text.split())
    if not source or not target:
        raise ValueError(f"{uid}: source and target must be non-empty.")
    return ScanExample(uid, source, target)


def load_split(data_root, split: str):
    dyck = _dyck_root(data_root)
    if dyck is not None:
        if split not in DYCK_SPLITS:
            raise ValueError(f"Unknown Dyck split {split!r}.")
        manifest = json.loads((dyck / "manifest.json").read_text(encoding="utf-8"))
        path = dyck / manifest["splits"][split]
        examples = []
        seen = set()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            src, tgt = tuple(row["input"].split()), tuple(row["output"].split())
            uid = f"DyckLanguage:{split}:{number}"
            if not 1 <= len(src) <= 5 or not tgt or src in seen:
                raise ValueError(f"{uid}: invalid length, empty target or duplicate input.")
            stack = []
            closes = dict(zip("([{<", ")]}>"))
            for token in src:
                if token in closes:
                    stack.append(closes[token])
                elif not stack or stack.pop() != token:
                    raise ValueError(f"{uid}: invalid bracket prefix.")
            if tuple(reversed(stack)) != tgt:
                raise ValueError(f"{uid}: incorrect bracket completion.")
            seen.add(src)
            examples.append(ScanExample(uid, src, tgt))
        if not examples:
            raise ValueError(f"{path} contains no examples.")
        return tuple(examples)
    if split not in _SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}.")
    path = Path(data_root) / "SCAN-length" / _SPLIT_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(path)
    examples = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        examples.append(
            parse_scan_line(line, f"SCAN-length:{split}:{line_number}")
        )
    if not examples:
        raise ValueError(f"{path} contains no examples.")
    return tuple(examples)


def encode_examples(
    examples: Iterable[ScanExample],
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
):
    readout_metadata(target_vocab)
    action_to_basis = {token: index for index, token in enumerate(target_vocab.tokens[4:])}
    encoded = []
    for example in examples:
        unknown_actions = [
            token for token in example.tgt_tokens if token not in action_to_basis
        ]
        if unknown_actions:
            raise ValueError(
                f"{example.uid}: unsupported target tokens {unknown_actions}."
            )
        encoded.append(
            EncodedScanExample(
                example.uid,
                source_vocab.encode(example.src_tokens),
                target_vocab.encode(example.tgt_tokens),
                tuple(action_to_basis[token] for token in example.tgt_tokens),
            )
        )
    return tuple(encoded)


def required_shapes(*collections: Sequence[EncodedScanExample]):
    examples = tuple(item for collection in collections for item in collection)
    if not examples:
        raise ValueError("At least one encoded example is required.")
    max_source = max(len(example.src_ids) for example in examples)
    max_target_positions = max(len(example.target_ids) + 1 for example in examples)
    return max_source, max_target_positions


def make_batch(
    examples: Sequence[EncodedScanExample],
    max_source: int,
    max_target_positions: int,
    batch_size: int | None = None,
):
    """Pad examples; PAD is never passed into attention as a valid context."""
    if batch_size is None:
        batch_size = len(examples)
    if not 0 < len(examples) <= batch_size:
        raise ValueError("Batch must contain 1..batch_size real examples.")
    src_ids = np.zeros((batch_size, max_source), dtype=np.int32)
    src_mask = np.zeros((batch_size, max_source), dtype=np.bool_)
    target_ids = np.zeros((batch_size, max_target_positions), dtype=np.int32)
    target_mask = np.zeros((batch_size, max_target_positions), dtype=np.bool_)
    labels = np.zeros((batch_size, max_target_positions), dtype=np.int32)
    label_mask = np.zeros((batch_size, max_target_positions), dtype=np.bool_)
    example_mask = np.zeros((batch_size,), dtype=np.bool_)
    uids = []

    for row, example in enumerate(examples):
        source_length = len(example.src_ids)
        target_length = len(example.target_ids)
        if source_length > max_source:
            raise ValueError(f"{example.uid}: source exceeds max_source.")
        if target_length + 1 > max_target_positions:
            raise ValueError(f"{example.uid}: target including EOS exceeds shape.")
        src_ids[row, :source_length] = example.src_ids
        src_mask[row, :source_length] = True
        target_ids[row, :target_length] = example.target_ids
        target_mask[row, :target_length] = True
        labels[row, :target_length] = example.target_bases
        labels[row, target_length] = SCAN_EOS_BASIS
        label_mask[row, : target_length + 1] = True
        example_mask[row] = True
        uids.append(example.uid)

    # Keep dummy rows numerically defined; example_mask removes their metrics.
    for row in range(len(examples), batch_size):
        src_mask[row, 0] = True
        uids.append("<PAD-EXAMPLE>")

    return {
        "src_ids": src_ids,
        "src_mask": src_mask,
        "target_ids": target_ids,
        "target_mask": target_mask,
        "labels": labels,
        "label_mask": label_mask,
        "example_mask": example_mask,
        "uids": tuple(uids),
    }


def iter_batches(
    examples: Sequence[EncodedScanExample],
    order: Sequence[int],
    batch_size: int,
    max_source: int,
    max_target_positions: int,
    start_batch: int = 0,
):
    for batch_index, start in enumerate(
        range(0, len(order), batch_size)
    ):
        if batch_index < start_batch:
            continue
        chosen = [examples[int(index)] for index in order[start : start + batch_size]]
        yield batch_index, make_batch(
            chosen,
            max_source,
            max_target_positions,
            batch_size=batch_size,
        )


def split_statistics(examples: Sequence[ScanExample]):
    source_lengths = np.asarray([len(item.src_tokens) for item in examples])
    target_lengths = np.asarray([len(item.tgt_tokens) for item in examples])
    return {
        "examples": len(examples),
        "source_tokens": int(source_lengths.sum()),
        "target_action_tokens": int(target_lengths.sum()),
        "target_tokens_with_eos": int((target_lengths + 1).sum()),
        "source_min": int(source_lengths.min()),
        "source_max": int(source_lengths.max()),
        "target_action_min": int(target_lengths.min()),
        "target_action_max": int(target_lengths.max()),
        "target_with_eos_max": int(target_lengths.max() + 1),
    }


def dataset_sha256(data_root):
    dyck = _dyck_root(data_root)
    if dyck is not None:
        manifest = json.loads((dyck / "manifest.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(b"dyck_merged_readout_v1:),],},>:eos=6:invalid=4,5,7")
        for name in ("manifest.json", "vocab.json", *(manifest["splits"][s] for s in DYCK_SPLITS)):
            digest.update(name.encode("utf-8"))
            digest.update((dyck / name).read_bytes())
        return digest.hexdigest()
    root = Path(data_root) / "SCAN-length"
    digest = hashlib.sha256()
    for name in (
        "vocab.src.txt",
        "vocab.tgt.txt",
        *_SPLIT_FILES.values(),
    ):
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


__all__ = [
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
    "UNK_ID",
    "DYCK_TOKENS",
    "DYCK_SPLITS",
    "readout_metadata",
    "Vocabulary",
    "ScanExample",
    "EncodedScanExample",
    "load_vocabularies",
    "parse_scan_line",
    "load_split",
    "encode_examples",
    "required_shapes",
    "make_batch",
    "iter_batches",
    "split_statistics",
    "dataset_sha256",
]
