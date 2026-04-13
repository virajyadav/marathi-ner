## ADDED Requirements

### Requirement: Publication workflow SHALL publish the processed dataset to Hugging Face
The system SHALL support packaging and publishing the processed Marathi NER dataset to a Hugging Face dataset repository with documentation and metadata.

#### Scenario: Publish dataset repository
- **WHEN** a user runs the dataset publication workflow with valid processed artifacts and Hugging Face credentials
- **THEN** the system publishes or updates a Hugging Face dataset repository containing the processed dataset files and dataset card

#### Scenario: Block dataset publication on unresolved redistribution issues
- **WHEN** the processed dataset includes sources with unresolved redistribution constraints
- **THEN** the system prevents publication or requires an explicit documented override before pushing artifacts

### Requirement: Publication workflow SHALL publish the fine-tuned model to Hugging Face
The system SHALL support packaging and publishing the fine-tuned GLiNER Marathi model to a Hugging Face model repository with usage instructions and evaluation metadata.

#### Scenario: Publish model repository
- **WHEN** a user runs the model publication workflow with a trained model, evaluation outputs, and Hugging Face credentials
- **THEN** the system publishes or updates a Hugging Face model repository containing model weights, tokenizer or config artifacts as required, and a model card

#### Scenario: Block model publication without evaluation outputs
- **WHEN** the model publication workflow is invoked without the required evaluation metrics or documentation
- **THEN** the system fails before pushing the model to Hugging Face

### Requirement: Publication workflow SHALL link dataset and model provenance
The system SHALL document the relationship between the published dataset and the published model so users can trace the training source.

#### Scenario: Reference dataset from the model card
- **WHEN** the model card is generated or updated
- **THEN** it identifies the processed Marathi dataset and the evaluation setup used for the fine-tuned model

#### Scenario: Reference model usage from the dataset card
- **WHEN** the dataset card is generated or updated
- **THEN** it describes the intended GLiNER training usage and points to the associated fine-tuned model if one exists
