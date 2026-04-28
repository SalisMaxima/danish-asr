# Graph Report - .  (2026-04-28)

## Corpus Check
- 109 files · ~99,586 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 878 nodes · 1351 edges · 33 communities detected
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 331 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Invoke Task Suite|Invoke Task Suite]]
- [[_COMMUNITY_CTC LM Decode|CTC LM Decode]]
- [[_COMMUNITY_HPC Logging|HPC Logging]]
- [[_COMMUNITY_CoRal Dataset|CoRal Dataset]]
- [[_COMMUNITY_HPC Evaluation|HPC Evaluation]]
- [[_COMMUNITY_Project Docs|Project Docs]]
- [[_COMMUNITY_Fairseq2 Conversion|Fairseq2 Conversion]]
- [[_COMMUNITY_HPC Access|HPC Access]]
- [[_COMMUNITY_Parquet Conversion|Parquet Conversion]]
- [[_COMMUNITY_W&B Sweep Best|W&B Sweep Best]]
- [[_COMMUNITY_HPC Training Runner|HPC Training Runner]]
- [[_COMMUNITY_CoRal DataModule|CoRal DataModule]]
- [[_COMMUNITY_W&B Storage Audit|W&B Storage Audit]]
- [[_COMMUNITY_Lightning ASR Model|Lightning ASR Model]]
- [[_COMMUNITY_Model Factory|Model Factory]]
- [[_COMMUNITY_Text Normalization|Text Normalization]]
- [[_COMMUNITY_Preprocessed Dataset|Preprocessed Dataset]]
- [[_COMMUNITY_WER CER Metrics|WER CER Metrics]]
- [[_COMMUNITY_Train Task Tests|Train Task Tests]]
- [[_COMMUNITY_Pytest Fixtures|Pytest Fixtures]]
- [[_COMMUNITY_LSF Monitoring|LSF Monitoring]]
- [[_COMMUNITY_Invoke Loader|Invoke Loader]]
- [[_COMMUNITY_Fairseq2 Assets|Fairseq2 Assets]]
- [[_COMMUNITY_Checkpoint Upload|Checkpoint Upload]]
- [[_COMMUNITY_WER Patch|WER Patch]]
- [[_COMMUNITY_LM Scripts|LM Scripts]]
- [[_COMMUNITY_Tokenizer Policy|Tokenizer Policy]]
- [[_COMMUNITY_Resume Practices|Resume Practices]]
- [[_COMMUNITY_VS Code Remote|VS Code Remote]]
- [[_COMMUNITY_Parquet Test Fixture|Parquet Test Fixture]]
- [[_COMMUNITY_Skip Failure Test|Skip Failure Test]]
- [[_COMMUNITY_Universal Parquet Fixture|Universal Parquet Fixture]]
- [[_COMMUNITY_Warning Filter State|Warning Filter State]]

## God Nodes (most connected - your core abstractions)
1. `CoRalDataset` - 20 edges
2. `ASRLitModel` - 19 edges
3. `_MetricParser` - 19 edges
4. `PreprocessedCoRalDataset` - 18 edges
5. `resolve_project_path()` - 18 edges
6. `audit()` - 18 edges
7. `main()` - 16 edges
8. `main()` - 16 edges
9. `main()` - 15 edges
10. `_make_fake_hf_dataset()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `build_datasets()` --calls--> `PreprocessedCoRalDataset`  [INFERRED]
  scripts/hpc/train_common.py → src/danish_asr/data.py
- `main()` --calls--> `resolve_project_path()`  [INFERRED]
  scripts/build_wav2vec2_vocab.py → src/danish_asr/utils.py
- `main()` --calls--> `get_project_hf_cache_dir()`  [INFERRED]
  scripts/build_wav2vec2_vocab.py → src/danish_asr/utils.py
- `main()` --calls--> `configure_project_cache_environment()`  [INFERRED]
  scripts/build_wav2vec2_vocab.py → src/danish_asr/utils.py
- `main()` --calls--> `get_device()`  [INFERRED]
  scripts/decode_ctc_with_lm.py → src/danish_asr/utils.py

## Hyperedges (group relationships)
- **omniASR CoRal Training Stack** — readme_danish_asr_project, readme_omniasr_ctc_300m, readme_coral_v3_dataset, finetuning_ctc_recipe, data_prep_unified_preprocessing, training_infra_fairseq2_pipeline [EXTRACTED 1.00]
- **Model Scaling Evaluation Loop** — experiment_e6_shuffle_fix, experiment_e6_1b_scaling, experiment_e6_3b_scaling, evaluation_results_combined_table, experiment_3b_resume_decision_gate, vram_probe_model_scaling_plan [EXTRACTED 1.00]
- **HPC Reproducibility Controls** — dtu_lsf_gpua100_queue, dtu_storage_work3_quota, roadmap_scratch_quota_policy, wandb_artifact_versioning_policy, cleanup_hpc_env_sh, finetuning_checkpoint_resume [INFERRED 0.82]
- **GPU Access Modes** — access_shared_interactive_gpu_nodes, monitoring_shared_gpu_nodes, monitoring_exclusive_gpu_bsub_is, readme_gpu_queues [EXTRACTED 1.00]
- **CUDA Python ML Stack** — python_env_cuda_117, python_env_pytorch_cuda_wheels, python_env_fairseq2_cuda_wheels, python_env_omnilingual_asr [EXTRACTED 1.00]
- **Batch Observability Toolchain** — monitoring_bstat, monitoring_bjobs, monitoring_bpeek, monitoring_nvidia_smi, monitoring_gpu_stats_logging, monitoring_bnvtop [EXTRACTED 1.00]

## Communities

### Community 0 - "Invoke Task Suite"
Cohesion: 0.03
Nodes (97): _has_module(), pull_llm(), Asset management tasks (model checkpoint pre-download)., Pre-download omniASR_LLM_300M_v2 or omniASR_LLM_1B_v2 to FAIRSEQ2_CACHE_DIR., bootstrap(), dev(), python(), Core environment setup and maintenance tasks. (+89 more)

### Community 1 - "CTC LM Decode"
Cohesion: 0.03
Nodes (91): build_lm_corpus_from_parquet(), build_pyctcdecode_labels(), chunked(), collate_decode_records(), CorpusStats, decode_logits_with_argmax(), DecodeResult, _get_cached_tokenizer_path() (+83 more)

### Community 2 - "HPC Logging"
Cohesion: 0.05
Nodes (54): log_gpu_info(), log_line_to_wandb(), log_system_info(), Shared infrastructure for HPC scripts: logging, paths, environment helpers., Log hostname, user, Python version, CUDA_VISIBLE_DEVICES, LSB_JOBID, disk space., Parse a fairseq2 output line and log matching metrics to W&B., Log torch CUDA availability, device names, and VRAM., Configure loguru with console (INFO) and file (DEBUG) sinks.      Returns the pa (+46 more)

### Community 3 - "CoRal Dataset"
Cohesion: 0.05
Nodes (33): collate_fn(), CoRalDataset, _pad_1d_tensors(), CoRal Danish ASR dataset and data module., Pad variable-length 1D tensors to uniform length.      Returns (stacked_tensor,, Custom collate function for variable-length audio., CoRal Danish speech dataset wrapper.      Wraps HuggingFace datasets for CoRal r, _make_fake_hf_dataset() (+25 more)

### Community 4 - "HPC Evaluation"
Cohesion: 0.06
Nodes (47): _backup_score_file(), check_prerequisites(), _get_score_file(), _log_workspace_snapshot(), main(), _MetricParser, _next_backup_path(), Step 4: Evaluation wrapper for omniASR on HPC.  Runs the omniASR eval recipe on (+39 more)

### Community 5 - "Project Docs"
Cohesion: 0.05
Nodes (49): Dead Classification Template Removal, Shared HPC env.sh, CoRal Demographic Metadata, CoRal Collection Methodology, OpenRAIL-D License Restrictions, CoRal Representation Gaps, Roest Wav2Vec2 CoRal Models, 16kHz FLAC Parquet Audio Format (+41 more)

### Community 6 - "Fairseq2 Conversion"
Cohesion: 0.08
Nodes (31): convert_split(), main(), parse_args(), process_audio(), Unified preprocessing for CoRal-v3: fairseq2 + universal Parquet formats.  Resam, Resample to 16kHz and FLAC-encode once.      Returns:         (flac_bytes, audio, Write rows to a fairseq2-format Parquet file., Write rows to a universal-format Parquet file. (+23 more)

### Community 7 - "HPC Access"
Cohesion: 0.06
Nodes (42): DTU Credentials, DTU VPN via OpenConnect, Ed25519 Key Pair, External SSH Key Requirement, Guest UNIX Databar Account, HPC Login Nodes, Invoke HPC Access Shortcuts, HPC Access Gotchas (+34 more)

### Community 8 - "Parquet Conversion"
Cohesion: 0.07
Nodes (25): convert_split(), _get_text_normalize(), main(), normalize_text(), process_audio(), Convert CoRal-v3 HuggingFace dataset to omnilingual ASR Parquet format.  Reads C, Convert one HF split to Parquet part files.      Returns stats dict with num_sam, Write language distribution stats TSV with hours column derived from total_audio (+17 more)

### Community 9 - "W&B Sweep Best"
Cohesion: 0.07
Nodes (34): best(), _build_result(), _find_best_run(), _get_default_entity_project(), _get_summary_metric(), _is_better_value(), main(), _normalize_sweep_id() (+26 more)

### Community 10 - "HPC Training Runner"
Cohesion: 0.1
Nodes (20): check_prerequisites(), _DuplicateColWarningFilter, ensure_data_symlink(), _init_wandb(), _log_metrics_to_wandb(), main(), _MetricParser, Training wrapper for omniASR on HPC.  Sets up environment, creates data symlink, (+12 more)

### Community 11 - "CoRal DataModule"
Cohesion: 0.08
Nodes (25): CoRalDataModule, Lightning DataModule for CoRal Danish ASR dataset., Set the feature processor and optional tokenizer.          Must be called before, Load and prepare CoRal dataset splits., Load from preprocessed Parquet files., Load from HuggingFace (original on-the-fly resampling)., _build_overrides(), main() (+17 more)

### Community 12 - "W&B Storage Audit"
Cohesion: 0.13
Nodes (32): ArtifactVersionRow, audit(), _build_arg_parser(), build_artifact_report(), build_run_file_report(), _bytes_to_gib(), collect_artifact_rows(), collect_run_file_rows() (+24 more)

### Community 13 - "Lightning ASR Model"
Cohesion: 0.12
Nodes (12): ASRLitModel, Lightning module for ASR training (CTC and seq2seq)., _DummyCTCModel, _DummyProcessor, _DummySeq2SeqInner, _DummySeq2SeqModel, _DummyTokenizer, Tests for the training module. (+4 more)

### Community 14 - "Model Factory"
Cohesion: 0.09
Nodes (19): build_model(), from_config(), ModelConfigFactory, ASR model definitions with LoRA fine-tuning support., OpenAI Whisper for Danish ASR with optional LoRA fine-tuning.      Uses HuggingF, Decorator to register a model class., Build a model from config., Wav2Vec2 for CTC-based ASR with optional LoRA fine-tuning.      Uses HuggingFace (+11 more)

### Community 15 - "Text Normalization"
Cohesion: 0.11
Nodes (19): normalize_coral_text(), normalize_ctc_text(), Shared text normalization helpers for ASR baselines., Normalize text following CoRal benchmark conventions.      Applied before model-, Normalize text for CTC training and evaluation.      Keep punctuation as-is for, build_ctc_vocab(), iter_training_texts(), main() (+11 more)

### Community 16 - "Preprocessed Dataset"
Cohesion: 0.15
Nodes (7): PreprocessedCoRalDataset, CoRal dataset from preprocessed Parquet (no on-the-fly resampling).      Reads u, Dataset, Tests for PreprocessedCoRalDataset loading universal Parquet., Constructing without pyarrow re-raises ImportError with actionable install hints, Cache ensures read_row_group is called once per row-group, not per sample., TestPreprocessedCoRalDataset

### Community 17 - "WER CER Metrics"
Cohesion: 0.14
Nodes (16): compute_cer(), compute_metrics(), compute_wer(), ASR evaluation metrics (WER, CER)., Compute Character Error Rate.      Args:         predictions: Model transcriptio, Compute all ASR metrics., Compute Word Error Rate.      Args:         predictions: Model transcriptions, Tests for ASR metrics. (+8 more)

### Community 18 - "Train Task Tests"
Cohesion: 0.42
Nodes (7): _assert_omni_command_or_windows_noop(), _DummyContext, Tests for training invoke tasks., test_omniasr_builds_expected_command(), test_omniasr_eval_builds_expected_command(), test_omniasr_hpc_uses_legacy_config_for_backward_compatibility(), test_omniasr_windows_fallback_logs_and_skips_run()

### Community 19 - "Pytest Fixtures"
Cohesion: 0.33
Nodes (5): data_available(), Pytest configuration and fixtures., Check if data is available for testing., Skip test if data is not available (expected in CI)., skip_if_no_data()

### Community 20 - "LSF Monitoring"
Cohesion: 0.4
Nodes (5): bjobs, bpeek, bstat, LSF Termination Reasons, python -u Unbuffered Output

### Community 21 - "Invoke Loader"
Cohesion: 0.5
Nodes (3): load_module_from_file(), Invoke tasks for Danish ASR.  Tasks are organized into namespaces for better org, Load a module from a file path.

### Community 22 - "Fairseq2 Assets"
Cohesion: 0.67
Nodes (2): Register danish_asr asset cards (datasets) with fairseq2., setup_fairseq2_extension()

### Community 23 - "Checkpoint Upload"
Cohesion: 0.67
Nodes (1): Upload finetuned model checkpoints to W&B as artifacts.  Run from project root (

### Community 24 - "WER Patch"
Cohesion: 0.67
Nodes (1): Patch omnilingual-asr wer_calculator.py to handle empty CTC hypotheses.  Fixes k

### Community 25 - "LM Scripts"
Cohesion: 1.0
Nodes (1): LM-related helper scripts.

### Community 26 - "Tokenizer Policy"
Cohesion: 1.0
Nodes (2): No Text Normalization Policy, omniASR Tokenizer Written V2

### Community 27 - "Resume Practices"
Cohesion: 1.0
Nodes (2): fairseq2 Checkpoint Resume, W&B Resume Best Practices

### Community 28 - "VS Code Remote"
Cohesion: 1.0
Nodes (2): Scientific Linux 7.9, VS Code Remote 1.85

### Community 31 - "Parquet Test Fixture"
Cohesion: 1.0
Nodes (1): Create a temp dir with universal Parquet for testing.

### Community 32 - "Skip Failure Test"
Cohesion: 1.0
Nodes (1): Too many skipped samples should raise RuntimeError.

### Community 33 - "Universal Parquet Fixture"
Cohesion: 1.0
Nodes (1): Create a temporary directory with a universal Parquet file.

### Community 37 - "Warning Filter State"
Cohesion: 1.0
Nodes (1): True once the first warning occurrence has been seen.

## Knowledge Gaps
- **312 isolated node(s):** `Invoke tasks for Danish ASR.  Tasks are organized into namespaces for better org`, `Load a module from a file path.`, `CoRal Danish ASR dataset and data module.`, `CoRal Danish speech dataset wrapper.      Wraps HuggingFace datasets for CoRal r`, `CoRal dataset from preprocessed Parquet (no on-the-fly resampling).      Reads u` (+307 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Fairseq2 Assets`** (3 nodes): `Register danish_asr asset cards (datasets) with fairseq2.`, `setup_fairseq2_extension()`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Checkpoint Upload`** (3 nodes): `upload_checkpoints.py`, `main()`, `Upload finetuned model checkpoints to W&B as artifacts.  Run from project root (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `WER Patch`** (3 nodes): `main()`, `Patch omnilingual-asr wer_calculator.py to handle empty CTC hypotheses.  Fixes k`, `patch_wer_calculator.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `LM Scripts`** (2 nodes): `LM-related helper scripts.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tokenizer Policy`** (2 nodes): `No Text Normalization Policy`, `omniASR Tokenizer Written V2`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Resume Practices`** (2 nodes): `fairseq2 Checkpoint Resume`, `W&B Resume Best Practices`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `VS Code Remote`** (2 nodes): `Scientific Linux 7.9`, `VS Code Remote 1.85`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Parquet Test Fixture`** (1 nodes): `Create a temp dir with universal Parquet for testing.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Skip Failure Test`** (1 nodes): `Too many skipped samples should raise RuntimeError.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Universal Parquet Fixture`** (1 nodes): `Create a temporary directory with a universal Parquet file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Warning Filter State`** (1 nodes): `True once the first warning occurrence has been seen.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `parse_args()` connect `CTC LM Decode` to `HPC Logging`, `HPC Evaluation`, `Parquet Conversion`, `HPC Training Runner`, `W&B Storage Audit`?**
  _High betweenness centrality (0.202) - this node is a cross-community bridge._
- **Why does `configure_project_cache_environment()` connect `CTC LM Decode` to `CoRal DataModule`, `Lightning ASR Model`, `Text Normalization`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `CoRalDataset` (e.g. with `.test_getitem_returns_correct_keys()` and `.test_getitem_audio_is_float_tensor()`) actually correct?**
  _`CoRalDataset` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `ASRLitModel` (e.g. with `test_seq2seq_eval_uses_configured_language()` and `test_epoch_end_metrics_clear_accumulators()`) actually correct?**
  _`ASRLitModel` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `_MetricParser` (e.g. with `test_metric_parser_extracts_multiline_validation_metrics()` and `test_metric_parser_extracts_legacy_validation_metrics()`) actually correct?**
  _`_MetricParser` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `PreprocessedCoRalDataset` (e.g. with `.test_collate_preprocessed_items()` and `.test_getitem_keys()`) actually correct?**
  _`PreprocessedCoRalDataset` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `resolve_project_path()` (e.g. with `.__init__()` and `._setup_preprocessed()`) actually correct?**
  _`resolve_project_path()` has 16 INFERRED edges - model-reasoned connections that need verification._
