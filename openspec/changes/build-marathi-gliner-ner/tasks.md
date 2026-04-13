## 1. Dataset foundation

- [x] 1.1 Identify the upstream Marathi NER datasets to support in v1 and document their license or redistribution status
- [x] 1.2 Define the normalized label schema and intermediate dataset representation
- [x] 1.3 Implement source ingestion and validation for each approved Marathi dataset
- [x] 1.4 Implement export of train, validation, and test splits in a GLiNER-compatible format
- [x] 1.5 Add dataset-level validation for spans, labels, provenance, and split integrity

## 2. Training and evaluation

- [ ] 2.1 Select and pin the multilingual GLiNER base checkpoint and training library version
- [ ] 2.2 Implement the fine-tuning workflow against the processed Marathi dataset
- [ ] 2.3 Implement held-out evaluation and checkpoint selection logic
- [ ] 2.4 Persist reproducible run metadata including dataset version and hyperparameters

## 3. Hugging Face release

- [ ] 3.1 Create dataset card and model card templates populated from provenance, label schema, and metrics
- [ ] 3.2 Implement dataset publication to a Hugging Face dataset repository
- [ ] 3.3 Implement model publication to a Hugging Face model repository with evaluation artifacts
- [ ] 3.4 Add release documentation describing prerequisites, credentials, and publication gates
