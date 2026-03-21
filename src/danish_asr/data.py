"""CoRal Danish ASR dataset and data module."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from danish_asr.utils import configure_project_cache_environment, get_project_hf_cache_dir, resolve_project_path


class CoRalDataset(Dataset):
    """CoRal Danish speech dataset wrapper.

    Wraps HuggingFace datasets for CoRal read-aloud Danish speech data.
    Handles audio loading, resampling to target sample rate, and text extraction.
    """

    def __init__(
        self,
        hf_dataset,
        processor=None,
        tokenizer=None,
        text_normalizer: Callable[[str], str] | None = None,
        target_sample_rate: int = 16000,
        max_duration: float = 30.0,
    ):
        self.dataset = hf_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.text_normalizer = text_normalizer
        self.target_sample_rate = target_sample_rate
        self.max_samples = int(max_duration * target_sample_rate)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        audio = item["audio"]

        # Get audio array and sample rate
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]

        # Resample if needed
        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sample_rate)

        # Truncate to max duration
        if waveform.shape[-1] > self.max_samples:
            waveform = waveform[..., : self.max_samples]

        text = item.get("text", item.get("sentence", ""))
        normalized_text = self.text_normalizer(text) if self.text_normalizer is not None else text

        result = {"audio": waveform, "text": text}
        if self.text_normalizer is not None:
            result["normalized_text"] = normalized_text

        # If processor is available, create model inputs
        if self.processor is not None:
            inputs = self.processor(
                waveform.numpy(),
                sampling_rate=self.target_sample_rate,
                return_tensors="pt",
                padding=False,
            )
            if hasattr(inputs, "input_features") and inputs.input_features is not None:
                result["input_features"] = inputs.input_features.squeeze(0)
            elif hasattr(inputs, "input_values") and inputs.input_values is not None:
                result["input_values"] = inputs.input_values.squeeze(0)
            if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                result["attention_mask"] = inputs.attention_mask.squeeze(0)

        # If tokenizer is available, tokenize labels
        if self.tokenizer is not None:
            labels = self.tokenizer(normalized_text, return_tensors="pt", padding=False)
            result["labels"] = labels.input_ids.squeeze(0)

        return result


class PreprocessedCoRalDataset(Dataset):
    """CoRal dataset from preprocessed Parquet (no on-the-fly resampling).

    Reads universal-format Parquet files produced by ``danish_asr.preprocessing``.
    Audio is stored as FLAC bytes and decoded to float32 on access.
    """

    def __init__(
        self,
        parquet_dir: str | Path,
        processor=None,
        tokenizer=None,
        text_normalizer: Callable[[str], str] | None = None,
        max_duration: float = 30.0,
        sample_rate: int = 16000,
    ):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "PreprocessedCoRalDataset requires the 'pyarrow' package. "
                "Install the 'omni' dependency group or add pyarrow explicitly, e.g.:\n"
                "  uv sync --group omni\n"
                "or:\n"
                "  uv add pyarrow"
            ) from exc

        parquet_dir = Path(parquet_dir)
        part_files = sorted(parquet_dir.glob("part-*.parquet"))
        if not part_files:
            raise FileNotFoundError(f"No part-*.parquet files found in {parquet_dir}")

        # Open files lazily and build a (file_idx, row_group_idx, row_idx) index
        # by reading only the lightweight duration_s column — avoids loading all
        # audio bytes into memory (~710 h of FLAC would OOM most machines).
        self._files: list[pq.ParquetFile] = [pq.ParquetFile(p) for p in part_files]
        self._indices: list[tuple[int, int, int]] = []
        for file_idx, pf in enumerate(self._files):
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx, columns=["duration_s"])
                durations = rg.column("duration_s").to_pylist()
                for row_idx, d in enumerate(durations):
                    if d is not None and d <= max_duration:
                        self._indices.append((file_idx, rg_idx, row_idx))

        self.processor = processor
        self.tokenizer = tokenizer
        self.text_normalizer = text_normalizer
        self.sample_rate = sample_rate
        # Cache for the most-recently-used row group: (file_idx, rg_idx, table)
        # Avoids redundant IO when consecutive samples share the same row group.
        self._rg_cache: tuple[int, int, Any] | None = None

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        file_idx, rg_idx, row_idx = self._indices[idx]
        pf = self._files[file_idx]
        if self._rg_cache is not None:
            cached_file_idx, cached_rg_idx, cached_table = self._rg_cache
            if cached_file_idx == file_idx and cached_rg_idx == rg_idx:
                table = cached_table
            else:
                table = pf.read_row_group(rg_idx)
                self._rg_cache = (file_idx, rg_idx, table)
        else:
            table = pf.read_row_group(rg_idx)
            self._rg_cache = (file_idx, rg_idx, table)
        row = {col: table.column(col)[row_idx].as_py() for col in table.column_names}

        # Decode FLAC bytes to float32 waveform
        audio_bytes = row["audio"]
        decoded, decoded_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        waveform = torch.from_numpy(decoded)

        if decoded_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, decoded_sr, self.sample_rate)

        text = row["text"]
        normalized_text = self.text_normalizer(text) if self.text_normalizer is not None else text

        result = {"audio": waveform, "text": text}
        if self.text_normalizer is not None:
            result["normalized_text"] = normalized_text

        if self.processor is not None:
            inputs = self.processor(
                waveform.numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=False,
            )
            if hasattr(inputs, "input_features") and inputs.input_features is not None:
                result["input_features"] = inputs.input_features.squeeze(0)
            elif hasattr(inputs, "input_values") and inputs.input_values is not None:
                result["input_values"] = inputs.input_values.squeeze(0)
            if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                result["attention_mask"] = inputs.attention_mask.squeeze(0)

        if self.tokenizer is not None:
            labels = self.tokenizer(normalized_text, return_tensors="pt", padding=False)
            result["labels"] = labels.input_ids.squeeze(0)

        return result


def _pad_1d_tensors(tensors: list[torch.Tensor], pad_value: float = 0.0) -> tuple[torch.Tensor, list[int]]:
    """Pad variable-length 1D tensors to uniform length.

    Returns (stacked_tensor, original_lengths).
    """
    max_len = max(t.shape[-1] for t in tensors)
    lengths = []
    padded = []
    for t in tensors:
        length = t.shape[-1]
        lengths.append(length)
        if length < max_len:
            t = torch.cat([t, torch.full((max_len - length,), pad_value, dtype=t.dtype)])
        padded.append(t)
    return torch.stack(padded), lengths


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for variable-length audio."""
    if not batch:
        return {}

    audios = [item["audio"] for item in batch]
    stacked_audio, audio_lengths = _pad_1d_tensors(audios)

    result = {
        "audio": stacked_audio,
        "audio_lengths": torch.tensor(audio_lengths),
        "text": [item["text"] for item in batch],
    }
    if "normalized_text" in batch[0]:
        result["normalized_text"] = [item["normalized_text"] for item in batch]

    if "input_values" in batch[0]:
        input_values = [item["input_values"] for item in batch]
        stacked_iv, _ = _pad_1d_tensors(input_values)
        result["input_values"] = stacked_iv

    if "input_features" in batch[0]:
        result["input_features"] = torch.stack([item["input_features"] for item in batch])

    if "labels" in batch[0]:
        label_tensors = [item["labels"] for item in batch]
        stacked_labels, _ = _pad_1d_tensors(label_tensors, pad_value=-100)
        result["labels"] = stacked_labels

    return result


class CoRalDataModule(pl.LightningDataModule):
    """Lightning DataModule for CoRal Danish ASR dataset."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        configure_project_cache_environment()
        self.cfg = cfg
        self.batch_size = cfg.get("batch_size", 8)
        self.num_workers = cfg.get("num_workers", 4)
        self.target_sample_rate = cfg.get("sample_rate", 16000)
        self.max_duration = cfg.get("max_duration", 30.0)
        self.subset = cfg.get("subset", "read_aloud")
        self.dataset_revision = cfg.get("dataset_revision", None)
        self.dataset_name = cfg.get("dataset_name", "CoRal-project/coral-v3")
        hf_cache_dir = cfg.get("hf_cache_dir")
        self.hf_cache_dir = str(resolve_project_path(hf_cache_dir)) if hf_cache_dir else str(get_project_hf_cache_dir())
        self.processor = None
        self.tokenizer = None
        self.text_normalizer: Callable[[str], str] | None = None
        self.train_dataset: CoRalDataset | None = None
        self.val_dataset: CoRalDataset | None = None
        self.test_dataset: CoRalDataset | None = None

    def set_processor_and_tokenizer(self, processor, tokenizer=None, text_normalizer=None) -> None:
        """Set the feature processor and optional tokenizer.

        Must be called before setup() for the processor to be used.
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.text_normalizer = text_normalizer

    def setup(self, stage: str | None = None) -> None:
        """Load and prepare CoRal dataset splits."""
        use_preprocessed = self.cfg.get("use_preprocessed", False)

        if use_preprocessed:
            self._setup_preprocessed(stage)
        else:
            self._setup_hf(stage)

        logger.info(
            f"Dataset loaded: train={len(self.train_dataset) if self.train_dataset else 0}, "
            f"val={len(self.val_dataset) if self.val_dataset else 0}, "
            f"test={len(self.test_dataset) if self.test_dataset else 0}"
        )

    def _setup_preprocessed(self, stage: str | None) -> None:
        """Load from preprocessed Parquet files."""
        preprocessed_dir = resolve_project_path(self.cfg.get("preprocessed_dir", "data/preprocessed"))
        subset_dir = preprocessed_dir / self.subset
        logger.info(f"Loading preprocessed CoRal dataset from {subset_dir}...")

        def _make_ds(split: str) -> PreprocessedCoRalDataset:
            return PreprocessedCoRalDataset(
                parquet_dir=subset_dir / split,
                processor=self.processor,
                tokenizer=self.tokenizer,
                text_normalizer=self.text_normalizer,
                max_duration=self.max_duration,
                sample_rate=self.target_sample_rate,
            )

        if stage == "fit" or stage is None:
            self.train_dataset = _make_ds("train")
            self.val_dataset = _make_ds("validation")

        if stage == "test" or stage is None:
            self.test_dataset = _make_ds("test")

    def _setup_hf(self, stage: str | None) -> None:
        """Load from HuggingFace (original on-the-fly resampling)."""
        from datasets import load_dataset

        logger.info(f"Loading CoRal dataset (subset={self.subset})...")
        logger.info(f"Using Hugging Face cache dir: {self.hf_cache_dir}")

        # Build kwargs for load_dataset
        kwargs: dict[str, object] = {}
        if self.dataset_revision is not None:
            kwargs["revision"] = self.dataset_revision
        kwargs["cache_dir"] = self.hf_cache_dir

        dataset = load_dataset(  # nosec B615
            self.dataset_name,
            self.subset,
            **kwargs,
        )

        if stage == "fit" or stage is None:
            self.train_dataset = CoRalDataset(
                dataset["train"],
                processor=self.processor,
                tokenizer=self.tokenizer,
                text_normalizer=self.text_normalizer,
                target_sample_rate=self.target_sample_rate,
                max_duration=self.max_duration,
            )
            self.val_dataset = CoRalDataset(
                dataset["val"],
                processor=self.processor,
                tokenizer=self.tokenizer,
                text_normalizer=self.text_normalizer,
                target_sample_rate=self.target_sample_rate,
                max_duration=self.max_duration,
            )

        if stage == "test" or stage is None:
            self.test_dataset = CoRalDataset(
                dataset["test"],
                processor=self.processor,
                tokenizer=self.tokenizer,
                text_normalizer=self.text_normalizer,
                target_sample_rate=self.target_sample_rate,
                max_duration=self.max_duration,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
