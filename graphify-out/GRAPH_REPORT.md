# Graph Report - danish_asr  (2026-05-08)

## Corpus Check
- 69 files · ~138,499 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1262 nodes · 2751 edges · 62 communities detected
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 430 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]

## God Nodes (most connected - your core abstractions)
1. `PreprocessedCoRalDataset` - 69 edges
2. `F()` - 68 edges
3. `e()` - 53 edges
4. `t()` - 52 edges
5. `CoRalDataset` - 46 edges
6. `CoRalDataModule` - 45 edges
7. `j()` - 40 edges
8. `g()` - 38 edges
9. `r()` - 37 edges
10. `q()` - 36 edges

## Surprising Connections (you probably didn't know these)
- `PreprocessedCoRalDataset` --uses--> `TestConvertSplit`  [INFERRED]
  src/danish_asr/data.py → tests/test_preprocessing.py
- `PreprocessedCoRalDataset` --uses--> `Tests for unified CoRal-v3 preprocessing.`  [INFERRED]
  src/danish_asr/data.py → tests/test_preprocessing.py
- `PreprocessedCoRalDataset` --uses--> `Create a fake HF audio dict.`  [INFERRED]
  src/danish_asr/data.py → tests/test_preprocessing.py
- `PreprocessedCoRalDataset` --uses--> `Create a mock HF dataset.`  [INFERRED]
  src/danish_asr/data.py → tests/test_preprocessing.py
- `PreprocessedCoRalDataset` --uses--> `Shared infrastructure for HF baseline training scripts (Wav2Vec2, Whisper).`  [INFERRED]
  src/danish_asr/data.py → scripts/hpc/train_common.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (250): _(), a(), Ac(), Ae(), ai(), an(), ao(), ap() (+242 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (122): Write benchmark artifacts in a stable, inspectable format., write_benchmark_outputs(), build_hf_text_lm_corpus(), build_lm_corpus_from_parquet(), build_pyctcdecode_labels(), chunked(), collate_decode_records(), CorpusStats (+114 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (65): collate_fn(), CoRalDataModule, CoRalDataset, _pad_1d_tensors(), PreprocessedCoRalDataset, CoRal Danish ASR dataset and data module., Pad variable-length 1D tensors to uniform length.      Returns (stacked_tensor,, Custom collate function for variable-length audio. (+57 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (63): A(), add(), At(), be(), Bt(), ce(), constructor(), Ct() (+55 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (52): log_gpu_info(), log_line_to_wandb(), log_system_info(), Shared infrastructure for HPC scripts: logging, paths, environment helpers., Log hostname, user, Python version, CUDA_VISIBLE_DEVICES, LSB_JOBID, disk space., Parse a fairseq2 output line and log matching metrics to W&B., Log torch CUDA availability, device names, and VRAM., Configure loguru with console (INFO) and file (DEBUG) sinks.      Returns the pa (+44 more)

### Community 5 - "Community 5"
Cohesion: 0.05
Nodes (47): _backup_score_file(), check_prerequisites(), _get_score_file(), _log_workspace_snapshot(), main(), _MetricParser, _next_backup_path(), Step 4: Evaluation wrapper for omniASR on HPC.  Runs the omniASR eval recipe on (+39 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (32): convert_split(), main(), parse_args(), process_audio(), Unified preprocessing for CoRal-v3: fairseq2 + universal Parquet formats.  Resam, Resample to 16kHz and FLAC-encode once.      Returns:         (flac_bytes, audio, Write rows to a fairseq2-format Parquet file., Write rows to a universal-format Parquet file. (+24 more)

### Community 7 - "Community 7"
Cohesion: 0.08
Nodes (40): _age_group(), bounded_error_rate(), convert_numeral_to_words(), CoRalBenchmarkExample, _dialect(), _empty_filter_stats(), _example_duration_seconds(), example_group_metadata() (+32 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (24): convert_split(), _get_text_normalize(), main(), normalize_text(), process_audio(), Convert CoRal-v3 HuggingFace dataset to omnilingual ASR Parquet format.  Reads C, Convert one HF split to Parquet part files.      Returns stats dict with num_sam, Write language distribution stats TSV with hours column derived from total_audio (+16 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (20): check_prerequisites(), _DuplicateColWarningFilter, ensure_data_symlink(), _init_wandb(), _log_metrics_to_wandb(), main(), _MetricParser, Training wrapper for omniASR on HPC.  Sets up environment, creates data symlink, (+12 more)

### Community 10 - "Community 10"
Cohesion: 0.13
Nodes (32): ArtifactVersionRow, audit(), _build_arg_parser(), build_artifact_report(), build_run_file_report(), _bytes_to_gib(), collect_artifact_rows(), collect_run_file_rows() (+24 more)

### Community 11 - "Community 11"
Cohesion: 0.09
Nodes (18): compute_cer(), compute_metrics(), compute_wer(), ASR evaluation metrics (WER, CER)., Compute Character Error Rate.      Args:         predictions: Model transcriptio, Compute all ASR metrics., Compute Word Error Rate.      Args:         predictions: Model transcriptions, ASRLitModel (+10 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (32): Aa(), De(), Ea(), El(), fo(), fu(), hn(), iu() (+24 more)

### Community 13 - "Community 13"
Cohesion: 0.1
Nodes (19): build_model(), from_config(), ModelConfigFactory, ASR model definitions with LoRA fine-tuning support., OpenAI Whisper for Danish ASR with optional LoRA fine-tuning.      Uses HuggingF, Decorator to register a model class., Build a model from config., Wav2Vec2 for CTC-based ASR with optional LoRA fine-tuning.      Uses HuggingFace (+11 more)

### Community 14 - "Community 14"
Cohesion: 0.08
Nodes (25): check_gpu(), clean_all(), clean_build(), clean_outputs(), clean_pyc(), clean_test(), dtu_vpn(), env_info() (+17 more)

### Community 15 - "Community 15"
Cohesion: 0.1
Nodes (23): _available_llm_hardware(), _has_module(), hpc_smoke(), hpc_sweep(), omniasr(), omniasr_eval(), omniasr_sweep(), Training and hyperparameter tuning tasks. (+15 more)

### Community 16 - "Community 16"
Cohesion: 0.14
Nodes (19): check_auth(), convert_parquet(), download(), download_all(), _hf_env_prefix(), preprocess(), Data management tasks for Danish ASR., Unified preprocessing: resample + FLAC-encode CoRal-v3 once.      Args: (+11 more)

### Community 17 - "Community 17"
Cohesion: 0.11
Nodes (19): ci(), deps_outdated(), deps_tree(), install_hooks(), Code quality and testing tasks., Run ruff linter and formatter., Run tests with coverage report., Run fast unit tests only (excludes slow and data-dependent tests). (+11 more)

### Community 18 - "Community 18"
Cohesion: 0.19
Nodes (10): _DummyCTCModel, _DummyProcessor, _DummySeq2SeqInner, _DummySeq2SeqModel, _DummyTokenizer, Tests for the training module., test_ctc_build_processor_uses_vocab_asset(), test_ctc_validation_prefers_normalized_references() (+2 more)

### Community 19 - "Community 19"
Cohesion: 0.21
Nodes (12): best(), _build_result(), _find_best_run(), _get_default_entity_project(), _get_summary_metric(), _is_better_value(), _normalize_sweep_id(), Utilities for working with W&B sweeps.  Provides a CLI to identify the best run (+4 more)

### Community 20 - "Community 20"
Cohesion: 0.2
Nodes (13): api(), build(), build_cuda(), check_docker_available(), clean(), Docker container and image management tasks., Check if Docker is installed and running., Build docker images (CPU versions). (+5 more)

### Community 21 - "Community 21"
Cohesion: 0.14
Nodes (13): bootstrap(), dev(), python(), Core environment setup and maintenance tasks., Bootstrap a UV virtual environment and install dependencies., Install/sync all dependencies., Install with dev dependencies., Complete development environment setup - one-command setup. (+5 more)

### Community 22 - "Community 22"
Cohesion: 0.29
Nodes (12): _build_html(), _cell_class(), _extract_table(), _find_chrome(), _is_separator_row(), main(), parse_args(), Render a results table from docs/evaluation-results.md to HTML or PNG.  Examples (+4 more)

### Community 23 - "Community 23"
Cohesion: 0.24
Nodes (9): normalize_coral_text(), normalize_ctc_text(), Shared text normalization helpers for ASR baselines., Normalize text following CoRal benchmark conventions.      Applied before model-, Normalize text for CTC training and evaluation.      Keep punctuation as-is for, Tests for text normalization helpers., test_normalize_coral_text_applies_nfkc_and_substitutions(), test_normalize_coral_text_removes_fillers_and_normalizes_spaces() (+1 more)

### Community 24 - "Community 24"
Cohesion: 0.36
Nodes (7): _assert_omni_command_or_windows_noop(), _DummyContext, Tests for training invoke tasks., test_omniasr_builds_expected_command(), test_omniasr_eval_builds_expected_command(), test_omniasr_hpc_uses_legacy_config_for_backward_compatibility(), test_omniasr_windows_fallback_logs_and_skips_run()

### Community 25 - "Community 25"
Cohesion: 0.33
Nodes (5): data_available(), Pytest configuration and fixtures., Check if data is available for testing., Skip test if data is not available (expected in CI)., skip_if_no_data()

### Community 26 - "Community 26"
Cohesion: 0.4
Nodes (3): Documentation building and serving tasks., Serve documentation locally., serve()

### Community 27 - "Community 27"
Cohesion: 0.5
Nodes (4): _has_module(), pull_llm(), Asset management tasks (model checkpoint pre-download)., Pre-download omniASR_LLM_300M_v2 or omniASR_LLM_1B_v2 to FAIRSEQ2_CACHE_DIR.

### Community 28 - "Community 28"
Cohesion: 0.5
Nodes (3): load_module_from_file(), Invoke tasks for Danish ASR.  Tasks are organized into namespaces for better org, Load a module from a file path.

### Community 29 - "Community 29"
Cohesion: 0.67
Nodes (2): Register danish_asr asset cards (datasets) with fairseq2., setup_fairseq2_extension()

### Community 30 - "Community 30"
Cohesion: 0.67
Nodes (1): Regression test: LLM eval configs must have family+arch whenever path is set.  f

### Community 31 - "Community 31"
Cohesion: 0.67
Nodes (1): Upload finetuned model checkpoints to W&B as artifacts.  Run from project root (

### Community 32 - "Community 32"
Cohesion: 0.67
Nodes (1): Pre-download omniASR LLM V2 checkpoint + tokenizer to FAIRSEQ2_CACHE_DIR.  Run o

### Community 33 - "Community 33"
Cohesion: 0.67
Nodes (1): Patch omnilingual-asr wer_calculator.py to handle empty CTC hypotheses.  Fixes k

### Community 34 - "Community 34"
Cohesion: 0.67
Nodes (1): Legacy helper to filter language_distribution_0.tsv to one CoRal-v3 subset.  Dep

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): LM-related helper scripts.

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): True once the first warning occurrence has been seen.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Compact statistics for an LM text build.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Decoded hypothesis with reference metadata.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Normalize transcript text for LM training without changing Danish orthography.

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Parse fairseq2 split names such as ``test`` or ``test_coral_v3_read_aloud``.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Yield rows from fairseq2 parquet shards for the requested corpora/split.

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Build a deterministic LM text corpus from fairseq2 parquet transcripts.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Write normalized LM text, one line per example.

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Write corpus stats as pretty JSON.

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Load a small YAML config file.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Find a cached tokenizer model file using the asset card basename.

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Load the OmniASR tokenizer, preferring an explicit or cached local model file.

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Build pyctcdecode labels in the exact OmniASR logit order.

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Remove special-token text artifacts after beam decoding.

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Load a custom OmniASR CTC checkpoint using fairseq2's registered family.

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Create an OmniASR inference pipeline for CTC decoding.

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Yield fixed-size chunks from a sequence.

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Read UTF-8 text lines without trailing newlines.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Write UTF-8 text lines with trailing newlines.

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Resolve checkpoint, tokenizer, dataset root, and split metadata from an eval con

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Construct a pyctcdecode decoder lazily.

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Compute simple WER summary from aligned prediction/reference lists.

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Resolve a dtype string with a CPU-safe fallback.

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Split decode records into aligned prediction and reference lists.

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Apply greedy CTC decoding to a single logit sequence.

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Decode one CTC logit sequence with greedy or beam search.

## Knowledge Gaps
- **259 isolated node(s):** `Invoke tasks for Danish ASR.  Tasks are organized into namespaces for better org`, `Load a module from a file path.`, `CoRal Danish ASR dataset and data module.`, `CoRal Danish speech dataset wrapper.      Wraps HuggingFace datasets for CoRal r`, `CoRal dataset from preprocessed Parquet (no on-the-fly resampling).      Reads u` (+254 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 29`** (3 nodes): `Register danish_asr asset cards (datasets) with fairseq2.`, `setup_fairseq2_extension()`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (3 nodes): `test_llm_eval_configs.py`, `Regression test: LLM eval configs must have family+arch whenever path is set.  f`, `test_llm_eval_config_has_family_and_arch_when_path_is_set()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (3 nodes): `main()`, `upload_checkpoints.py`, `Upload finetuned model checkpoints to W&B as artifacts.  Run from project root (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (3 nodes): `main()`, `Pre-download omniASR LLM V2 checkpoint + tokenizer to FAIRSEQ2_CACHE_DIR.  Run o`, `pull_llm_assets.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (3 nodes): `main()`, `Patch omnilingual-asr wer_calculator.py to handle empty CTC hypotheses.  Fixes k`, `patch_wer_calculator.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (3 nodes): `main()`, `Legacy helper to filter language_distribution_0.tsv to one CoRal-v3 subset.  Dep`, `make_subset_tsv.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (2 nodes): `LM-related helper scripts.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `True once the first warning occurrence has been seen.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Compact statistics for an LM text build.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Decoded hypothesis with reference metadata.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Normalize transcript text for LM training without changing Danish orthography.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Parse fairseq2 split names such as ``test`` or ``test_coral_v3_read_aloud``.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Yield rows from fairseq2 parquet shards for the requested corpora/split.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Build a deterministic LM text corpus from fairseq2 parquet transcripts.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Write normalized LM text, one line per example.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Write corpus stats as pretty JSON.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Load a small YAML config file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Find a cached tokenizer model file using the asset card basename.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Load the OmniASR tokenizer, preferring an explicit or cached local model file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Build pyctcdecode labels in the exact OmniASR logit order.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Remove special-token text artifacts after beam decoding.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Load a custom OmniASR CTC checkpoint using fairseq2's registered family.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Create an OmniASR inference pipeline for CTC decoding.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Yield fixed-size chunks from a sequence.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Read UTF-8 text lines without trailing newlines.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Write UTF-8 text lines with trailing newlines.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Resolve checkpoint, tokenizer, dataset root, and split metadata from an eval con`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Construct a pyctcdecode decoder lazily.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Compute simple WER summary from aligned prediction/reference lists.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Resolve a dtype string with a CPU-safe fallback.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Split decode records into aligned prediction and reference lists.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Apply greedy CTC decoding to a single logit sequence.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Decode one CTC logit sequence with greedy or beam search.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `set()` connect `Community 1` to `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 9`?**
  _High betweenness centrality (0.379) - this node is a cross-community bridge._
- **Why does `_t()` connect `Community 3` to `Community 0`, `Community 1`?**
  _High betweenness centrality (0.234) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 4` to `Community 1`, `Community 5`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Are the 63 inferred relationships involving `PreprocessedCoRalDataset` (e.g. with `TestCoRalDataset` and `TestCollateFn`) actually correct?**
  _`PreprocessedCoRalDataset` has 63 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `F()` (e.g. with `Kt()` and `ee()`) actually correct?**
  _`F()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `e()` (e.g. with `H()` and `$()`) actually correct?**
  _`e()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `t()` (e.g. with `I()` and `forEach()`) actually correct?**
  _`t()` has 2 INFERRED edges - model-reasoned connections that need verification._
