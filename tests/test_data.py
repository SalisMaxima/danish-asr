"""Tests for CoRal data pipeline."""

from __future__ import annotations

import torch

from danish_asr.data import CoRalDataset, collate_fn


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
