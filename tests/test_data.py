"""Tests for CoRal data pipeline."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import torch

from danish_asr.data import CoRalDataModule, CoRalDataset, collate_fn
from danish_asr.utils import get_project_hf_cache_dir


def _make_fake_hf_dataset(n: int = 5, sr: int = 16000, duration: float = 2.0):
    """Create a minimal list-of-dicts that mimics a HuggingFace dataset split."""
    num_samples = int(sr * duration)
    items = []
    for i in range(n):
        items.append(
            {
                "audio": {
                    "array": torch.randn(num_samples).numpy(),
                    "sampling_rate": sr,
                },
                "text": f"dette er sætning nummer {i}",
            }
        )
    return items


class TestCoRalDataset:
    """Tests for CoRalDataset.__getitem__ and basic behaviour."""

    def test_getitem_returns_correct_keys(self):
        fake_ds = _make_fake_hf_dataset(n=3)
        dataset = CoRalDataset(fake_ds, target_sample_rate=16000)
        item = dataset[0]

        assert "audio" in item
        assert "text" in item

    def test_getitem_audio_is_float_tensor(self):
        fake_ds = _make_fake_hf_dataset(n=1)
        dataset = CoRalDataset(fake_ds, target_sample_rate=16000)
        item = dataset[0]

        assert isinstance(item["audio"], torch.Tensor)
        assert item["audio"].dtype == torch.float32

    def test_getitem_text_is_string(self):
        fake_ds = _make_fake_hf_dataset(n=1)
        dataset = CoRalDataset(fake_ds, target_sample_rate=16000)
        item = dataset[0]

        assert isinstance(item["text"], str)
        assert len(item["text"]) > 0

    def test_len(self):
        fake_ds = _make_fake_hf_dataset(n=7)
        dataset = CoRalDataset(fake_ds)

        assert len(dataset) == 7

    def test_resampling(self):
        """Audio at 48kHz should be resampled to 16kHz."""
        fake_ds = _make_fake_hf_dataset(n=1, sr=48000, duration=1.0)
        dataset = CoRalDataset(fake_ds, target_sample_rate=16000)
        item = dataset[0]

        # 1s at 16kHz = 16000 samples
        assert item["audio"].shape[-1] == 16000

    def test_truncation(self):
        """Audio longer than max_duration should be truncated."""
        fake_ds = _make_fake_hf_dataset(n=1, sr=16000, duration=5.0)
        dataset = CoRalDataset(fake_ds, target_sample_rate=16000, max_duration=2.0)
        item = dataset[0]

        expected_samples = int(2.0 * 16000)
        assert item["audio"].shape[-1] == expected_samples

    def test_no_truncation_when_shorter(self):
        """Audio shorter than max_duration should not be modified."""
        fake_ds = _make_fake_hf_dataset(n=1, sr=16000, duration=1.0)
        dataset = CoRalDataset(fake_ds, target_sample_rate=16000, max_duration=30.0)
        item = dataset[0]

        expected_samples = int(1.0 * 16000)
        assert item["audio"].shape[-1] == expected_samples


class TestCollateFn:
    """Tests for collate_fn with variable-length audio."""

    def test_collate_pads_to_max_length(self):
        """Batch items should be padded to the longest sequence."""
        items = [
            {"audio": torch.randn(16000), "text": "kort"},
            {"audio": torch.randn(32000), "text": "længere sætning"},
        ]
        batch = collate_fn(items)

        assert batch["audio"].shape == (2, 32000)

    def test_collate_audio_lengths(self):
        """audio_lengths should record original (pre-pad) lengths."""
        items = [
            {"audio": torch.randn(8000), "text": "a"},
            {"audio": torch.randn(24000), "text": "b"},
            {"audio": torch.randn(16000), "text": "c"},
        ]
        batch = collate_fn(items)

        assert batch["audio_lengths"].tolist() == [8000, 24000, 16000]

    def test_collate_texts_preserved(self):
        items = [
            {"audio": torch.randn(100), "text": "hej"},
            {"audio": torch.randn(200), "text": "verden"},
        ]
        batch = collate_fn(items)

        assert batch["text"] == ["hej", "verden"]

    def test_collate_equal_length_no_padding(self):
        """When all items are the same length, no padding should occur."""
        items = [
            {"audio": torch.randn(16000), "text": "a"},
            {"audio": torch.randn(16000), "text": "b"},
        ]
        batch = collate_fn(items)

        assert batch["audio"].shape == (2, 16000)
        assert batch["audio_lengths"].tolist() == [16000, 16000]

    def test_collate_fn_stacks_input_features(self):
        """Whisper input_features (fixed-size mel-specs) should be stacked."""
        items = [
            {"audio": torch.randn(16000), "text": "a", "input_features": torch.randn(80, 3000)},
            {"audio": torch.randn(16000), "text": "b", "input_features": torch.randn(80, 3000)},
        ]
        batch = collate_fn(items)

        assert "input_features" in batch
        assert batch["input_features"].shape == (2, 80, 3000)

    def test_collate_fn_pads_labels_with_ignore_index(self):
        """Labels should be padded with -100."""
        items = [
            {"audio": torch.randn(16000), "text": "a", "labels": torch.tensor([1, 2, 3])},
            {"audio": torch.randn(16000), "text": "b", "labels": torch.tensor([4, 5])},
        ]
        batch = collate_fn(items)

        assert "labels" in batch
        assert batch["labels"].shape == (2, 3)
        assert batch["labels"][1].tolist() == [4, 5, -100]

    def test_collate_fn_empty_batch(self):
        """Empty batch should return empty dict."""
        assert collate_fn([]) == {}

    def test_text_fallback_to_sentence(self):
        """Items with 'sentence' key instead of 'text' should still work."""
        items = [
            {
                "audio": {
                    "array": torch.randn(16000).numpy(),
                    "sampling_rate": 16000,
                },
                "sentence": "dette er en sætning",
            }
        ]
        dataset = CoRalDataset(items, target_sample_rate=16000)
        item = dataset[0]

        assert item["text"] == "dette er en sætning"


class TestWhisperProcessor:
    """Tests for Whisper processor and tokenizer integration."""

    def test_whisper_processor_returns_input_features(self):
        """Processor returning input_features should populate that key."""
        fake_ds = _make_fake_hf_dataset(n=1)
        mock_processor = MagicMock()
        mock_result = MagicMock()
        mock_result.input_features = torch.randn(1, 80, 3000)
        mock_result.input_values = None
        mock_result.attention_mask = None
        mock_processor.return_value = mock_result

        dataset = CoRalDataset(fake_ds, processor=mock_processor)
        item = dataset[0]

        assert "input_features" in item
        assert item["input_features"].shape == (80, 3000)
        assert "input_values" not in item

    def test_tokenizer_produces_labels(self):
        """Tokenizer should produce labels key with token IDs."""
        fake_ds = _make_fake_hf_dataset(n=1)
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3, 4]]))

        dataset = CoRalDataset(fake_ds, tokenizer=mock_tokenizer)
        item = dataset[0]

        assert "labels" in item
        assert item["labels"].tolist() == [1, 2, 3, 4]

    def test_tokenizer_uses_normalized_text_when_configured(self):
        fake_ds = _make_fake_hf_dataset(n=1)
        fake_ds[0]["text"] = "Hej   VERDEN"
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))

        dataset = CoRalDataset(fake_ds, tokenizer=mock_tokenizer, text_normalizer=lambda text: text.lower().strip())
        item = dataset[0]

        mock_tokenizer.assert_called_once_with("hej   verden", return_tensors="pt", padding=False)
        assert item["normalized_text"] == "hej   verden"

    def test_no_input_features_without_processor(self):
        """Without processor, neither input_features nor input_values should be present."""
        fake_ds = _make_fake_hf_dataset(n=1)
        dataset = CoRalDataset(fake_ds)
        item = dataset[0]

        assert "input_features" not in item
        assert "input_values" not in item
        assert "labels" not in item

    def test_combined_processor_and_tokenizer(self):
        """Both processor and tokenizer set should produce input_features and labels."""
        fake_ds = _make_fake_hf_dataset(n=1)

        mock_processor = MagicMock()
        mock_result = MagicMock()
        mock_result.input_features = torch.randn(1, 80, 3000)
        mock_result.input_values = None
        mock_result.attention_mask = None
        mock_processor.return_value = mock_result

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))

        dataset = CoRalDataset(fake_ds, processor=mock_processor, tokenizer=mock_tokenizer)
        item = dataset[0]

        assert "input_features" in item
        assert item["input_features"].shape == (80, 3000)
        assert "labels" in item
        assert item["labels"].tolist() == [1, 2, 3]

    def test_collate_preserves_normalized_text(self):
        items = [
            {"audio": torch.randn(16000), "text": "Hej", "normalized_text": "hej"},
            {"audio": torch.randn(12000), "text": "Verden", "normalized_text": "verden"},
        ]

        batch = collate_fn(items)

        assert batch["normalized_text"] == ["hej", "verden"]


class TestPreprocessedCollateFn:
    """Test that PreprocessedCoRalDataset items work with collate_fn."""

    @staticmethod
    def _make_parquet_dir(tmp_path: Path, n: int = 3) -> Path:
        """Create a temp dir with universal Parquet for testing."""
        import numpy as np
        import soundfile as sf

        from danish_asr.preprocessing import write_universal_parquet

        rng = np.random.default_rng(42)
        rows = []
        for i in range(n):
            waveform = np.clip(rng.standard_normal(8000 + i * 4000) * 0.3, -0.99, 0.99).astype(np.float32)
            buf = io.BytesIO()
            sf.write(buf, waveform, 16000, format="FLAC")
            rows.append(
                {
                    "text": f"sætning {i}",
                    "audio": buf.getvalue(),
                    "audio_samples": len(waveform),
                    "duration_s": len(waveform) / 16000,
                    "subset": "read_aloud",
                    "split": "train",
                    "speaker_id": f"spk_{i}",
                    "gender": "male",
                    "age": "25",
                    "dialect": "copenhagen",
                }
            )
        out_dir = tmp_path / "train"
        out_dir.mkdir(parents=True)
        write_universal_parquet(rows, out_dir / "part-00000.parquet")
        return out_dir

    def test_collate_preprocessed_items(self, tmp_path: Path):
        from danish_asr.data import PreprocessedCoRalDataset

        parquet_dir = self._make_parquet_dir(tmp_path, n=3)
        ds = PreprocessedCoRalDataset(parquet_dir, max_duration=999.0)

        items = [ds[i] for i in range(len(ds))]
        batch = collate_fn(items)

        assert batch["audio"].shape[0] == 3
        assert len(batch["text"]) == 3
        assert batch["audio_lengths"].tolist()[0] == 8000


class TestCoRalDataModule:
    def test_defaults_hf_cache_to_project_drive(self):
        datamodule = CoRalDataModule({"subset": "read_aloud", "hf_cache_dir": None})

        assert datamodule.hf_cache_dir == str(get_project_hf_cache_dir())
