## ADDED Requirements

### Requirement: Training pipeline SHALL fine-tune a multilingual GLiNER checkpoint on Marathi data
The system SHALL load a configured multilingual GLiNER base model and fine-tune it using the processed Marathi dataset.

#### Scenario: Fine-tune from configured base checkpoint
- **WHEN** a user runs the training workflow with a valid processed dataset and base model configuration
- **THEN** the system starts fine-tuning from the configured multilingual GLiNER checkpoint and saves training artifacts

#### Scenario: Refuse training without prepared data
- **WHEN** the training workflow is invoked before the processed dataset is available or validated
- **THEN** the system fails with an error indicating that dataset preparation must complete first

### Requirement: Training pipeline SHALL evaluate on held-out Marathi data
The system SHALL evaluate the fine-tuned model on a held-out Marathi split and persist reproducible evaluation outputs.

#### Scenario: Generate evaluation metrics after training
- **WHEN** model training completes successfully
- **THEN** the system computes and stores evaluation metrics for the held-out Marathi split

#### Scenario: Compare checkpoints consistently
- **WHEN** multiple checkpoints are produced during training
- **THEN** the system selects or reports checkpoints using a documented evaluation criterion

### Requirement: Training pipeline SHALL capture reproducible training configuration
The system SHALL persist the effective training configuration, including base checkpoint, label set, data version, and training hyperparameters, alongside training outputs.

#### Scenario: Save training configuration with outputs
- **WHEN** a fine-tuning run completes
- **THEN** the resulting artifacts include the effective configuration needed to reproduce that run

#### Scenario: Identify dataset version used for a model
- **WHEN** a published or local model artifact is inspected
- **THEN** the artifact metadata identifies the processed dataset version or build used for training
