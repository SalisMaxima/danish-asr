"""CoRal Danish ASR dataset and data module."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torchaudio
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class CoRalDataset(Dataset):
    """CoRal Danish speech dataset wrapper.

    Wraps HuggingFace datasets for CoRal read-aloud Danish speech data.
    Handles audio loading, resampling to target sample rate, and text extraction.
    """

    def __init__(
        self,
        hf_dataset,
        processor=None,
        target_sample_rate: int = 16000,
        max_duration: float = 30.0,
    ):
        self.dataset = hf_dataset
        self.processor = processor
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

        result = {"audio": waveform, "text": text}

        # If processor is available, create model inputs
        if self.processor is not None:
            inputs = self.processor(
                waveform.numpy(),
                sampling_rate=self.target_sample_rate,
                return_tensors="pt",
                padding=False,
            )
            result["input_values"] = inputs.input_values.squeeze(0)
            if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                result["attention_mask"] = inputs.attention_mask.squeeze(0)

        return result


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for variable-length audio."""
    # Pad audio to max length in batch
    audios = [item["audio"] for item in batch]
    max_len = max(a.shape[-1] for a in audios)

    padded_audios = []
    audio_lengths = []
    for audio in audios:
        length = audio.shape[-1]
        audio_lengths.append(length)
        if length < max_len:
            pad = torch.zeros(max_len - length)
            audio = torch.cat([audio, pad])
        padded_audios.append(audio)

    result = {
        "audio": torch.stack(padded_audios),
        "audio_lengths": torch.tensor(audio_lengths),
        "text": [item["text"] for item in batch],
    }

    # Handle processor outputs if present
    if "input_values" in batch[0]:
        input_values = [item["input_values"] for item in batch]
        max_iv_len = max(iv.shape[-1] for iv in input_values)
        padded_iv = []
        for iv in input_values:
            if iv.shape[-1] < max_iv_len:
                pad = torch.zeros(max_iv_len - iv.shape[-1])
                iv = torch.cat([iv, pad])
            padded_iv.append(iv)
        result["input_values"] = torch.stack(padded_iv)

    return result


class CoRalDataModule(pl.LightningDataModule):
    """Lightning DataModule for CoRal Danish ASR dataset."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.get("batch_size", 8)
        self.num_workers = cfg.get("num_workers", 4)
        self.target_sample_rate = cfg.get("sample_rate", 16000)
        self.max_duration = cfg.get("max_duration", 30.0)
        self.subset = cfg.get("subset", "read_aloud")
        self.dataset_revision = cfg.get("dataset_revision", "main")
        self.processor = None
        self.train_dataset: CoRalDataset | None = None
        self.val_dataset: CoRalDataset | None = None
        self.test_dataset: CoRalDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load and prepare CoRal dataset splits."""
        from datasets import load_dataset

        logger.info(f"Loading CoRal dataset (subset={self.subset})...")

        dataset = load_dataset(  # nosec B615 - revision pinned via config
            "alexandrainst/coral",
            self.subset,
            trust_remote_code=True,
            revision=self.dataset_revision,
        )

        if stage == "fit" or stage is None:
            self.train_dataset = CoRalDataset(
                dataset["train"],
                processor=self.processor,
                target_sample_rate=self.target_sample_rate,
                max_duration=self.max_duration,
            )
            self.val_dataset = CoRalDataset(
                dataset["val"],
                processor=self.processor,
                target_sample_rate=self.target_sample_rate,
                max_duration=self.max_duration,
            )

        if stage == "test" or stage is None:
            self.test_dataset = CoRalDataset(
                dataset["test"],
                processor=self.processor,
                target_sample_rate=self.target_sample_rate,
                max_duration=self.max_duration,
            )

        logger.info(
            f"Dataset loaded: train={len(self.train_dataset) if self.train_dataset else 0}, "
            f"val={len(self.val_dataset) if self.val_dataset else 0}, "
            f"test={len(self.test_dataset) if self.test_dataset else 0}"
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
