"""Tests for CoRal-v3 to Parquet conversion functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.convert_coral_to_parquet import (
    SCHEMA,
    process_audio,
    write_parquet,
    write_stats_tsv,
)


def _make_audio_dict(sr: int = 48000, duration: float = 1.0) -> dict:
    """Create a fake HF audio dict."""
    num_samples = int(sr * duration)
    return {
        "array": np.random.default_rng(42).standard_normal(num_samples).astype(np.float32),
        "sampling_rate": sr,
    }


class TestProcessAudio:
    def test_resamples_to_16khz(self):
        audio = _make_audio_dict(sr=48000, duration=1.0)
        _, audio_size = process_audio(audio)
        assert audio_size == 16000

    def test_returns_int8_array(self):
        audio = _make_audio_dict(sr=16000, duration=0.5)
        audio_int8, _ = process_audio(audio)
        assert isinstance(audio_int8, np.ndarray)
        assert audio_int8.dtype == np.int8

    def test_flac_roundtrip(self):
        """FLAC encoding should produce decodable audio with correct length and close values."""
        import io

        import soundfile as sf

        # Use values clamped to [-1, 1] to avoid FLAC clipping
        rng = np.random.default_rng(42)
        audio = {
            "array": np.clip(rng.standard_normal(8000) * 0.3, -0.99, 0.99).astype(np.float32),
            "sampling_rate": 16000,
        }
        audio_int8, audio_size = process_audio(audio)

        # Decode FLAC back
        flac_bytes = audio_int8.tobytes()
        decoded, decoded_sr = sf.read(io.BytesIO(flac_bytes))
        assert decoded_sr == 16000
        assert len(decoded) == audio_size

        # FLAC uses int16/int24 quantization; allow for rounding error
        max_diff = np.max(np.abs(decoded - audio["array"]))
        assert max_diff < 1e-3, f"Max difference {max_diff} exceeds tolerance"


class TestWriteParquet:
    def test_creates_file(self, tmp_path: Path):
        rows = [
            {
                "text": "hej verden",
                "audio_bytes": [1, 2, 3],
                "audio_size": 16000,
                "corpus": "coral_v3_read_aloud",
                "split": "train",
                "language": "dan_Latn",
            }
        ]
        out_path = tmp_path / "test.parquet"
        write_parquet(rows, out_path)
        assert out_path.exists()

    def test_schema(self, tmp_path: Path):
        rows = [
            {
                "text": "test",
                "audio_bytes": [0],
                "audio_size": 100,
                "corpus": "c",
                "split": "train",
                "language": "dan_Latn",
            }
        ]
        out_path = tmp_path / "schema_test.parquet"
        write_parquet(rows, out_path)

        table = pq.read_table(out_path)
        assert table.schema.equals(SCHEMA)

    def test_row_group_size(self, tmp_path: Path):
        rows = [
            {
                "text": f"sample {i}",
                "audio_bytes": [0] * 10,
                "audio_size": 100,
                "corpus": "c",
                "split": "train",
                "language": "dan_Latn",
            }
            for i in range(250)
        ]
        out_path = tmp_path / "rg_test.parquet"
        write_parquet(rows, out_path)

        pf = pq.ParquetFile(out_path)
        assert pf.metadata.num_row_groups == 3  # 250 rows / 100 per group = 3


class TestWriteStatsTsv:
    def test_format_and_content(self, tmp_path: Path):
        stats = [
            {
                "corpus": "coral_v3_read_aloud",
                "language": "dan_Latn",
                "split": "train",
                "num_samples": 100,
                "total_audio_seconds": 500.0,
            }
        ]
        tsv_path = tmp_path / "stats.tsv"
        write_stats_tsv(stats, tsv_path)

        assert tsv_path.exists()
        lines = tsv_path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "corpus\tlanguage\tsplit\tnum_samples\ttotal_audio_seconds" in lines[0]
        assert "coral_v3_read_aloud\tdan_Latn\ttrain\t100\t500.0" in lines[1]
