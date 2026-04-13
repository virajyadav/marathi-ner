## Context

The repository currently contains no implementation code, so this change defines the initial end-to-end workflow for Marathi NER data preparation, GLiNER fine-tuning, and Hugging Face publication. Existing Marathi corpora such as L3Cube MahaNER and MahaSocialNER provide the likely starting point, but their label schemes, file layouts, split conventions, and redistribution conditions must be normalized before training. The training target should remain compatible with GLiNER's multilingual checkpoints so Marathi support can be improved without building a model from scratch.

## Goals / Non-Goals

**Goals:**
- Create a reproducible pipeline from raw Marathi NER sources to a GLiNER-compatible processed dataset.
- Preserve source provenance and document redistribution assumptions for every upstream dataset used.
- Fine-tune a multilingual GLiNER base model with configuration that can be rerun locally or in a notebook environment.
- Produce evaluation outputs on a held-out Marathi split before publishing artifacts.
- Publish both dataset and model artifacts to Hugging Face with clear cards, metadata, and usage guidance.

**Non-Goals:**
- Creating a large new manually annotated Marathi corpus in the first iteration.
- Supporting arbitrary non-Marathi languages in the initial pipeline.
- Building a custom NER architecture outside the GLiNER ecosystem.
- Automating scheduled retraining or continuous delivery in the first release.

## Decisions

### Decision: Use existing Marathi corpora as the v1 data source
The first release will convert and merge existing Marathi NER datasets rather than require new annotation work.

Rationale:
- Existing corpora provide enough labeled data to validate the workflow quickly.
- The first risk is pipeline correctness and reproducibility, not annotation throughput.

Alternatives considered:
- Manual annotation from scratch. Rejected because it delays the first usable release and adds tooling and reviewer overhead.
- Synthetic annotation with LLMs. Rejected for v1 because quality control would become the dominant uncertainty.

### Decision: Define a normalized intermediate dataset representation before GLiNER export
Raw source formats will be converted into a single internal representation containing text, tokens if available, labeled spans, split, source dataset, and normalized labels.

Rationale:
- This isolates source-specific parsers from training-specific export logic.
- It makes it possible to validate label consistency and provenance before model training.

Alternatives considered:
- Convert each source directly into the final GLiNER format. Rejected because debugging data quality issues becomes harder and source handling becomes tightly coupled to one trainer format.

### Decision: Fine-tune from a multilingual GLiNER checkpoint
Training will start from a multilingual GLiNER model rather than an English-only checkpoint.

Rationale:
- Marathi requires multilingual token and representation coverage.
- This reduces the risk of poor transfer from an English-centric base model.

Alternatives considered:
- English-only GLiNER checkpoints. Rejected because language mismatch is an avoidable baseline weakness.
- Training a new model from scratch. Rejected because data volume is not sufficient for a first release.

### Decision: Keep a fixed evaluation split and require metrics before publishing
The pipeline will reserve a held-out Marathi split and compute evaluation metrics before any model release.

Rationale:
- Fine-tuning quality cannot be judged from training loss alone.
- Fixed evaluation supports future retraining comparisons.

Alternatives considered:
- Publish after ad hoc manual inspection only. Rejected because it is not reproducible.

### Decision: Publish dataset and model as separate Hugging Face artifacts
The processed dataset and fine-tuned model will live in separate Hugging Face repositories linked through their cards.

Rationale:
- Consumers may want the dataset without the trained model or retrain with different settings.
- Separate versioning makes future updates cleaner.

Alternatives considered:
- Publish only the model. Rejected because it hides the training data contract.
- Bundle processed data inside the model repository. Rejected because it makes reuse and discovery worse.

## Risks / Trade-offs

- [Upstream redistribution rights are unclear for one or more datasets] -> Verify licenses before bundling raw or processed content and fall back to documented download-and-convert steps if redistribution is restricted.
- [Merged corpora may use incompatible entity definitions] -> Create an explicit label mapping table and document any lossy normalization decisions.
- [Small data volume may overfit during fine-tuning] -> Use a held-out split, conservative training defaults, and checkpoint evaluation.
- [GLiNER format assumptions may differ across library versions] -> Pin the training dependency version and keep exported schema validation in the pipeline.
- [Social-media data may degrade performance on cleaner text or vice versa] -> Preserve source provenance and support evaluation by source dataset where possible.

## Migration Plan

1. Add the project structure for raw data references, processed outputs, configs, scripts, and documentation.
2. Implement source-specific dataset ingestion and normalization into the intermediate representation.
3. Export the processed dataset into the chosen GLiNER training format and validate the schema.
4. Train and evaluate a multilingual GLiNER checkpoint on the processed Marathi dataset.
5. Generate dataset card and model card content from the final label schema, provenance, and metrics.
6. Push the dataset and model to separate Hugging Face repositories.
7. If a release fails quality checks, retain the processed data artifacts locally and do not publish the model.

## Open Questions

- Which exact upstream Marathi datasets will be included in v1 after license verification?
- Will the project target classic GLiNER training format, GLiNER2 format, or support both?
- Should the first release preserve the original label taxonomy exactly or collapse overlapping labels across corpora?
- What minimum F1 threshold or qualitative criteria should gate Hugging Face model publication?
