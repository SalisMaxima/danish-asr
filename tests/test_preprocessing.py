"""Tests for unified CoRal-v3 preprocessing."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pyarrow", reason="pyarrow not installed; skipping preprocessing tests")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

from danish_asr.preprocessing import (
    FAIRSEQ2_SCHEMA,
    SPLIT_MAP,
    TARGET_SR,
    UNIVERSAL_SCHEMA,
    convert_split,
    process_audio,
    write_fairseq2_parquet,
    write_stats_tsv,
    write_universal_parquet,
)


def _make_audio_dict(sr: int = 48000, duration: float = 1.0) -> dict:
    """Create a fake HF audio dict."""
    num_samples = int(sr * duration)
    return {
        "array": np.random.default_rng(42).standard_normal(num_samples).astype(np.float32),
        "sampling_rate": sr,
    }


# ---------------------------------------------------------------------------
# TestProcessAudio
# ---------------------------------------------------------------------------


class TestProcessAudio:
    def test_resamples_to_16khz(self):
        audio = _make_audio_dict(sr=48000, duration=1.0)
        _, audio_samples = process_audio(audio)
        assert audio_samples == 16000

    def test_returns_flac_bytes(self):
        audio = _make_audio_dict(sr=16000, duration=0.5)
        flac_bytes, _ = process_audio(audio)
        assert isinstance(flac_bytes, bytes)
        assert len(flac_bytes) > 0

    def test_flac_roundtrip(self):
        rng = np.random.default_rng(42)
        audio = {
            "array": np.clip(rng.standard_normal(8000) * 0.3, -0.99, 0.99).astype(np.float32),
            "sampling_rate": 16000,
        }
        flac_bytes, audio_samples = process_audio(audio)

        decoded, sr = sf.read(io.BytesIO(flac_bytes))
        assert sr == 16000
        assert len(decoded) == audio_samples

        # FLAC quantization tolerance
        max_diff = np.max(np.abs(decoded - audio["array"]))
        assert max_diff < 1e-3

    def test_no_resample_when_already_16khz(self):
        audio = _make_audio_dict(sr=16000, duration=0.5)
        _, audio_samples = process_audio(audio)
        assert audio_samples == 8000


# ---------------------------------------------------------------------------
# TestUniversalParquet
# ---------------------------------------------------------------------------


class TestUniversalParquet:
    def _make_rows(self, n: int = 3) -> list[dict]:
        return [
            {
                "text": f"sætning {i}",
                "audio": b"\x00" * 10,
                "audio_samples": 16000,
                "duration_s": 1.0,
                "subset": "read_aloud",
                "split": "train",
                "speaker_id": f"spk_{i}",
                "gender": "male",
                "age": "30",
                "dialect": "copenhagen",
            }
            for i in range(n)
        ]

    def test_creates_file(self, tmp_path: Path):
        rows = self._make_rows(1)
        out = tmp_path / "test.parquet"
        write_universal_parquet(rows, out)
        assert out.exists()

    def test_schema_matches(self, tmp_path: Path):
        rows = self._make_rows(2)
        out = tmp_path / "schema.parquet"
        write_universal_parquet(rows, out)
        table = pq.read_table(out)
        assert table.schema.equals(UNIVERSAL_SCHEMA)

    def test_metadata_columns(self, tmp_path: Path):
        rows = self._make_rows(1)
        rows[0]["speaker_id"] = "spk_42"
        rows[0]["dialect"] = "jutland"
        out = tmp_path / "meta.parquet"
        write_universal_parquet(rows, out)
        table = pq.read_table(out)
        assert table.column("speaker_id")[0].as_py() == "spk_42"
        assert table.column("dialect")[0].as_py() == "jutland"

    def test_text_is_original(self, tmp_path: Path):
        rows = self._make_rows(1)
        rows[0]["text"] = "Hej Verden 123"
        out = tmp_path / "text.parquet"
        write_universal_parquet(rows, out)
        table = pq.read_table(out)
        assert table.column("text")[0].as_py() == "Hej Verden 123"

    def test_audio_is_binary_type(self, tmp_path: Path):
        rows = self._make_rows(1)
        out = tmp_path / "audio.parquet"
        write_universal_parquet(rows, out)
        table = pq.read_table(out)
        assert table.schema.field("audio").type == pa.binary()

    def test_duration_consistency(self, tmp_path: Path):
        rows = self._make_rows(1)
        rows[0]["audio_samples"] = 32000
        rows[0]["duration_s"] = 2.0
        out = tmp_path / "dur.parquet"
        write_universal_parquet(rows, out)
        table = pq.read_table(out)
        samples = table.column("audio_samples")[0].as_py()
        duration = table.column("duration_s")[0].as_py()
        assert abs(duration - samples / TARGET_SR) < 1e-5


# ---------------------------------------------------------------------------
# TestFairseq2Parquet
# ---------------------------------------------------------------------------


class TestFairseq2Parquet:
    def _make_rows(self, n: int = 3) -> list[dict]:
        return [
            {
                "text": f"sample {i}",
                "audio_bytes": bytes([0, 1, 2]),
                "audio_size": 16000,
                "corpus": "coral_v3_read_aloud",
                "split": "train",
                "language": "dan_Latn",
            }
            for i in range(n)
        ]

    def test_schema_matches(self, tmp_path: Path):
        rows = self._make_rows(2)
        out = tmp_path / "f2.parquet"
        write_fairseq2_parquet(rows, out)
        table = pq.read_table(out)
        assert table.schema.equals(FAIRSEQ2_SCHEMA)

    def test_dict_encoded_columns(self, tmp_path: Path):
        rows = self._make_rows(2)
        out = tmp_path / "dict.parquet"
        write_fairseq2_parquet(rows, out)
        table = pq.read_table(out)
        for col in ("corpus", "split", "language"):
            assert pa.types.is_dictionary(table.schema.field(col).type)

    def test_row_group_size(self, tmp_path: Path):
        rows = self._make_rows(250)
        out = tmp_path / "rg.parquet"
        write_fairseq2_parquet(rows, out)
        pf = pq.ParquetFile(out)
        assert pf.metadata.num_row_groups == 3  # 250 / 100 = 3


# ---------------------------------------------------------------------------
# TestConvertSplit
# ---------------------------------------------------------------------------


def _make_fake_hf_dataset(n: int = 5, sr: int = 48000, duration: float = 0.5):
    """Create a mock HF dataset."""
    rng = np.random.default_rng(42)
    num_samples = int(sr * duration)
    items = []
    for i in range(n):
        items.append(
            {
                "audio": {
                    "array": np.clip(rng.standard_normal(num_samples) * 0.3, -0.99, 0.99).astype(np.float32),
                    "sampling_rate": sr,
                },
                "text": f"dette er sætning {i}",
                "speaker_id": f"spk_{i}",
                "gender": "male" if i % 2 == 0 else "female",
                "age": "25",
                "dialect": "copenhagen",
            }
        )

    # Make it behave like a HF Dataset
    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=n)
    mock_ds.__iter__ = MagicMock(return_value=iter(items))
    mock_ds.select = MagicMock(side_effect=lambda r: _make_fake_hf_dataset(len(r), sr, duration))
    return mock_ds


class TestConvertSplit:
    @patch("datasets.load_dataset")
    def test_universal_only(self, mock_load, tmp_path: Path):
        mock_load.return_value = _make_fake_hf_dataset(3)
        stats = convert_split(
            hf_subset="read_aloud",
            corpus_name="coral_v3_read_aloud",
            hf_split="train",
            targets={"universal"},
            universal_dir=tmp_path / "universal",
            rows_per_file=100,
        )
        assert stats["num_samples"] == 3
        part_files = list((tmp_path / "universal" / "read_aloud" / "train").glob("*.parquet"))
        assert len(part_files) == 1

    @patch("datasets.load_dataset")
    def test_both_targets(self, mock_load, tmp_path: Path):
        mock_load.return_value = _make_fake_hf_dataset(3)
        stats = convert_split(
            hf_subset="read_aloud",
            corpus_name="coral_v3_read_aloud",
            hf_split="train",
            targets={"fairseq2", "universal"},
            fairseq2_dir=tmp_path / "fairseq2",
            universal_dir=tmp_path / "universal",
            rows_per_file=100,
        )
        assert stats["num_samples"] == 3
        # Both dirs should have files
        f2_files = list((tmp_path / "fairseq2").rglob("*.parquet"))
        uni_files = list((tmp_path / "universal").rglob("*.parquet"))
        assert len(f2_files) >= 1
        assert len(uni_files) >= 1

    @patch("datasets.load_dataset")
    def test_max_samples(self, mock_load, tmp_path: Path):
        mock_ds = _make_fake_hf_dataset(10)
        mock_load.return_value = mock_ds
        convert_split(
            hf_subset="read_aloud",
            corpus_name="coral_v3_read_aloud",
            hf_split="train",
            targets={"universal"},
            universal_dir=tmp_path / "universal",
            rows_per_file=100,
            max_samples=3,
        )
        mock_ds.select.assert_called_once()

    @patch("datasets.load_dataset")
    def test_corrupt_sample_skipped(self, mock_load, tmp_path: Path):
        mock_ds = MagicMock()
        rng = np.random.default_rng(42)

        def _good_item(i: int) -> dict:
            return {
                "audio": {
                    "array": np.clip(rng.standard_normal(8000) * 0.3, -0.99, 0.99).astype(np.float32),
                    "sampling_rate": 16000,
                },
                "text": f"good {i}",
                "speaker_id": f"spk_{i}",
                "gender": "male",
                "age": "25",
                "dialect": "copenhagen",
            }

        bad_item = {
            "audio": {"array": None, "sampling_rate": 48000},
            "text": "bad",
            "speaker_id": "",
            "gender": "",
            "age": "",
            "dialect": "",
        }
        # 30 good + 1 bad = ~3% skip rate (under 5% threshold)
        items = [_good_item(i) for i in range(30)] + [bad_item]
        mock_ds.__len__ = MagicMock(return_value=len(items))
        mock_ds.__iter__ = MagicMock(return_value=iter(items))
        mock_load.return_value = mock_ds

        stats = convert_split(
            hf_subset="read_aloud",
            corpus_name="coral_v3_read_aloud",
            hf_split="train",
            targets={"universal"},
            universal_dir=tmp_path / "universal",
            rows_per_file=100,
        )
        assert stats["num_samples"] == 30

    @patch("datasets.load_dataset")
    def test_skip_rate_threshold(self, mock_load, tmp_path: Path):
        """Too many skipped samples should raise RuntimeError."""
        mock_ds = MagicMock()
        # All samples are corrupt
        bad_items = [
            {
                "audio": {"array": None, "sampling_rate": 48000},
                "text": f"bad {i}",
                "speaker_id": "",
                "gender": "",
                "age": "",
                "dialect": "",
            }
            for i in range(10)
        ]
        mock_ds.__len__ = MagicMock(return_value=10)
        mock_ds.__iter__ = MagicMock(return_value=iter(bad_items))
        mock_load.return_value = mock_ds

        with pytest.raises(RuntimeError, match="Skip rate"):
            convert_split(
                hf_subset="read_aloud",
                corpus_name="coral_v3_read_aloud",
                hf_split="train",
                targets={"universal"},
                universal_dir=tmp_path / "universal",
                rows_per_file=100,
            )


# ---------------------------------------------------------------------------
# TestWriteStatsTsv (ported from test_parquet_conversion.py)
# ---------------------------------------------------------------------------


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
        assert len(lines) == 2
        assert "corpus\tlanguage\tsplit\tnum_samples\ttotal_audio_seconds\thours" in lines[0]
        assert "coral_v3_read_aloud\tdan_Latn\ttrain\t100\t500.0\t0.138889" in lines[1]


# ---------------------------------------------------------------------------
# TestPreprocessedCoRalDataset
# ---------------------------------------------------------------------------


class TestPreprocessedCoRalDataset:
    """Tests for PreprocessedCoRalDataset loading universal Parquet."""

    @pytest.fixture()
    def parquet_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory with a universal Parquet file."""
        rng = np.random.default_rng(42)
        waveform = np.clip(rng.standard_normal(16000) * 0.3, -0.99, 0.99).astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, waveform, 16000, format="FLAC")
        flac_bytes = buf.getvalue()

        rows = [
            {
                "text": "hej verden",
                "audio": flac_bytes,
                "audio_samples": 16000,
                "duration_s": 1.0,
                "subset": "read_aloud",
                "split": "train",
                "speaker_id": "spk_0",
                "gender": "male",
                "age": "25",
                "dialect": "copenhagen",
            },
            {
                "text": "lang sætning",
                "audio": flac_bytes,
                "audio_samples": 16000,
                "duration_s": 1.0,
                "subset": "read_aloud",
                "split": "train",
                "speaker_id": "spk_1",
                "gender": "female",
                "age": "30",
                "dialect": "jutland",
            },
            {
                "text": "for lang",
                "audio": flac_bytes,
                "audio_samples": 16000,
                "duration_s": 35.0,  # exceeds default max_duration
                "subset": "read_aloud",
                "split": "train",
                "speaker_id": "spk_2",
                "gender": "male",
                "age": "40",
                "dialect": "funen",
            },
        ]

        out_dir = tmp_path / "read_aloud" / "train"
        out_dir.mkdir(parents=True)
        write_universal_parquet(rows, out_dir / "part-00000.parquet")
        return out_dir

    def test_getitem_keys(self, parquet_dir: Path):
        from danish_asr.data import PreprocessedCoRalDataset

        ds = PreprocessedCoRalDataset(parquet_dir)
        item = ds[0]
        assert "audio" in item
        assert "text" in item

    def test_audio_dtype(self, parquet_dir: Path):
        from danish_asr.data import PreprocessedCoRalDataset

        ds = PreprocessedCoRalDataset(parquet_dir)
        item = ds[0]
        assert item["audio"].dtype == torch.float32

    def test_audio_sample_rate(self, parquet_dir: Path):
        from danish_asr.data import PreprocessedCoRalDataset

        ds = PreprocessedCoRalDataset(parquet_dir)
        item = ds[0]
        # 1.0s at 16kHz = 16000 samples
        assert item["audio"].shape[-1] == 16000

    def test_text_normalizer(self, parquet_dir: Path):
        from danish_asr.data import PreprocessedCoRalDataset

        ds = PreprocessedCoRalDataset(parquet_dir, text_normalizer=str.upper)
        item = ds[0]
        assert item["normalized_text"] == "HEJ VERDEN"
        assert item["text"] == "hej verden"

    def test_max_duration_filtering(self, parquet_dir: Path):
        from danish_asr.data import PreprocessedCoRalDataset

        ds = PreprocessedCoRalDataset(parquet_dir, max_duration=30.0)
        assert len(ds) == 2  # third sample has duration_s=35.0

    def test_no_max_duration_filter(self, parquet_dir: Path):
        from danish_asr.data import PreprocessedCoRalDataset

        ds = PreprocessedCoRalDataset(parquet_dir, max_duration=999.0)
        assert len(ds) == 3

    def test_processor_integration(self, parquet_dir: Path):
        from danish_asr.data import PreprocessedCoRalDataset

        mock_processor = MagicMock()
        mock_result = MagicMock()
        mock_result.input_values = MagicMock()
        mock_result.input_values.__bool__ = MagicMock(return_value=True)
        import torch

        mock_result.input_values = torch.randn(1, 16000)
        mock_result.input_features = None
        mock_result.attention_mask = None
        mock_processor.return_value = mock_result

        ds = PreprocessedCoRalDataset(parquet_dir, processor=mock_processor)
        item = ds[0]
        assert "input_values" in item

    def test_file_not_found(self, tmp_path: Path):
        from danish_asr.data import PreprocessedCoRalDataset

        with pytest.raises(FileNotFoundError):
            PreprocessedCoRalDataset(tmp_path / "nonexistent")

    def test_missing_pyarrow_raises_import_error_with_install_hints(self, monkeypatch, tmp_path: Path):
        """Constructing without pyarrow re-raises ImportError with actionable install hints."""
        import builtins
        import sys

        from danish_asr.data import PreprocessedCoRalDataset

        # Remove pyarrow from sys.modules so the deferred import is forced to go
        # through __import__ rather than returning a cached module.
        for mod in [k for k in sys.modules if "pyarrow" in k]:
            monkeypatch.delitem(sys.modules, mod)

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if "pyarrow" in name:
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError) as exc_info:
            PreprocessedCoRalDataset(tmp_path)

        msg = str(exc_info.value)
        assert "uv sync --group omni" in msg
        assert "uv add pyarrow" in msg

    def test_row_group_cache_reduces_reads(self, tmp_path: Path):
        """Cache ensures read_row_group is called once per row-group, not per sample."""
        import io as _io

        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        import soundfile as sf

        from danish_asr.data import PreprocessedCoRalDataset

        # Write a 4-row parquet with row_group_size=2 → 2 row groups (rows 0-1, rows 2-3)
        rng = np.random.default_rng(0)

        def make_flac() -> bytes:
            wave = np.clip(rng.standard_normal(8000) * 0.3, -0.99, 0.99).astype(np.float32)
            buf = _io.BytesIO()
            sf.write(buf, wave, 16000, format="FLAC")
            return buf.getvalue()

        rows = [
            {
                "text": f"s{i}",
                "audio": make_flac(),
                "audio_samples": 8000,
                "duration_s": 0.5,
                "subset": "read_aloud",
                "split": "train",
                "speaker_id": f"spk_{i}",
                "gender": "male",
                "age": "25",
                "dialect": "copenhagen",
            }
            for i in range(4)
        ]
        out_dir = tmp_path / "train"
        out_dir.mkdir(parents=True)
        path = out_dir / "part-00000.parquet"
        arrays = {
            "text": pa.array([r["text"] for r in rows], type=pa.string()),
            "audio": pa.array([r["audio"] for r in rows], type=pa.binary()),
            "audio_samples": pa.array([r["audio_samples"] for r in rows], type=pa.int64()),
            "duration_s": pa.array([r["duration_s"] for r in rows], type=pa.float32()),
            "subset": pa.array([r["subset"] for r in rows], type=pa.string()),
            "split": pa.array([r["split"] for r in rows], type=pa.string()),
            "speaker_id": pa.array([r["speaker_id"] for r in rows], type=pa.string()),
            "gender": pa.array([r["gender"] for r in rows], type=pa.string()),
            "age": pa.array([r["age"] for r in rows], type=pa.string()),
            "dialect": pa.array([r["dialect"] for r in rows], type=pa.string()),
        }
        pq.write_table(pa.table(arrays, schema=UNIVERSAL_SCHEMA), path, row_group_size=2)

        ds = PreprocessedCoRalDataset(out_dir, max_duration=999.0)
        assert len(ds) == 4  # all 4 rows within max_duration

        # Attach a spy to the ParquetFile instance *after* the index is built
        # so init-time duration-column reads are not counted.
        pf = ds._files[0]
        original_read = pf.read_row_group
        call_log: list[int] = []

        def spy(rg_idx, **kwargs):
            call_log.append(rg_idx)
            return original_read(rg_idx, **kwargs)

        pf.read_row_group = spy  # type: ignore[assignment]
        # _rg_cache is already None after __init__; the spy is installed post-init
        # so no artificial reset is needed.

        # ds[0] and ds[1] are both in row group 0 → only one IO call
        _ = ds[0]
        _ = ds[1]
        assert len(call_log) == 1, "Expected 1 read for 2 items from the same row group"

        # ds[2] is in row group 1 → triggers a new IO call
        _ = ds[2]
        assert len(call_log) == 2, "Expected a new read when switching row groups"

        # ds[3] is also in row group 1 → cache hit, no new IO call
        _ = ds[3]
        assert len(call_log) == 2, "Expected cache hit for a second item in the same row group"


# ---------------------------------------------------------------------------
# TestSplitMap
# ---------------------------------------------------------------------------


def test_split_map():
    assert SPLIT_MAP == {"train": "train", "validation": "dev", "test": "test"}
