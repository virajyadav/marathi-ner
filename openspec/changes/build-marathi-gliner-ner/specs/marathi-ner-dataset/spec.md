## ADDED Requirements

### Requirement: Dataset pipeline SHALL build a normalized Marathi NER corpus
The system SHALL ingest supported Marathi NER source datasets, normalize them into a single intermediate representation, and produce train, validation, and test splits for downstream GLiNER training.

#### Scenario: Convert supported source datasets
- **WHEN** a user runs the dataset preparation workflow with configured Marathi source datasets
- **THEN** the system produces processed dataset artifacts containing examples, labels, split membership, and source provenance in a consistent schema

#### Scenario: Reject unsupported or malformed source data
- **WHEN** a source dataset is missing required fields, contains malformed annotations, or uses an unknown parser
- **THEN** the system fails with a validation error that identifies the affected source and the reason it could not be processed

### Requirement: Dataset pipeline SHALL preserve provenance and redistribution metadata
The system SHALL record the source dataset name, source split if available, and redistribution metadata for every processed dataset build.

#### Scenario: Record source provenance in processed outputs
- **WHEN** the dataset preparation workflow completes successfully
- **THEN** the processed dataset metadata identifies each upstream source used and the normalized label mapping applied to that source

#### Scenario: Surface redistribution constraints
- **WHEN** a source dataset has unknown or restricted redistribution terms
- **THEN** the workflow surfaces that constraint in generated metadata and documentation so publication decisions can account for it

### Requirement: Dataset pipeline SHALL export GLiNER-compatible training data
The system SHALL transform the normalized Marathi corpus into the configured GLiNER-compatible training format and validate that exported examples conform to the expected schema.

#### Scenario: Export training-ready GLiNER data
- **WHEN** dataset preparation finishes for a valid Marathi corpus
- **THEN** the system writes GLiNER-compatible training files for the configured splits

#### Scenario: Fail invalid export schema
- **WHEN** an exported example is missing required fields or contains invalid entity spans
- **THEN** the system fails validation before training artifacts are published
