## Why

Marathi NER has usable existing corpora, but this project does not yet provide a reproducible way to convert them into GLiNER-ready training data, fine-tune a multilingual GLiNER checkpoint, and publish the resulting artifacts. Establishing that workflow now enables a practical first Marathi GLiNER release with clear provenance, evaluation, and Hugging Face distribution.

## What Changes

- Add a dataset preparation workflow that ingests existing Marathi NER sources, verifies redistribution constraints, normalizes labels, and exports train/validation/test splits in a GLiNER-compatible format.
- Add a fine-tuning workflow that starts from a multilingual GLiNER base checkpoint, trains on the prepared Marathi dataset, and produces reproducible evaluation outputs.
- Add publication support for pushing both the dataset and the fine-tuned model to Hugging Face with dataset cards, model cards, metadata, and usage instructions.
- Add documentation for source provenance, label schema, evaluation metrics, and release criteria so future iterations can extend the dataset or retrain the model consistently.

## Capabilities

### New Capabilities
- `marathi-ner-dataset`: Build a Marathi NER dataset for GLiNER from existing corpora with normalized labels, validated splits, and source provenance.
- `marathi-gliner-finetuning`: Fine-tune a multilingual GLiNER checkpoint on the Marathi dataset and produce reproducible Marathi evaluation results.
- `huggingface-artifact-publishing`: Package and publish the processed dataset and trained model to Hugging Face with complete metadata and documentation.

### Modified Capabilities

## Impact

- Adds data ingestion, normalization, validation, training, evaluation, and publishing workflows to the repository.
- Introduces external dependencies for dataset handling, model training, evaluation, and Hugging Face Hub integration.
- Establishes artifact contracts for processed data, metrics, model checkpoints, and release documentation.
