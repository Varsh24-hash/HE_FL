
# Federated Health Models

A federated learning + homomorphic encryption pipeline for training shared
predictive models across multiple hospitals without any hospital exposing
raw patient data.

Each site trains a model locally on its own data, encrypts the resulting
weights using the CKKS homomorphic encryption scheme, and sends only the
ciphertext to a central aggregator. The aggregator combines encrypted
weights from all sites (mean, and — in the OpenFHE track — outlier-robust
filtering) without ever decrypting individual contributions. Only the final
aggregated model is decrypted, using a secret key that never leaves the
client side.

## What's in this repo

- **Domain-specific training** — local model training + CKKS encryption for
  three health domains: genetic, mental health, and sexual/reproductive
  health (`domains/`)
- **Multi-target / schema-aligned training** — a global feature dictionary
  lets hospitals with different column sets train a shared
  parameter → disease mapping (`multi_target/`)
- **OpenFHE (C++) pipeline** — a lower-level implementation with key
  generation, encryption, server-side robust aggregation (mean/variance,
  outlier filtering, global + cluster masks), and client-side decryption
  (`openfhe_pipeline/`)
- **Azure sync utilities** — upload/download encrypted weight blobs to/from
  Azure Blob Storage (`azure_sync/`)
- **Key management** — export a secret-free "public" keyset bundle
  (`key_management/`)
- **Analysis tools** — feature-importance extraction from trained models and
  BERT-based clustering of inconsistent column names across hospitals'
  schemas (`analysis/`)

## Status / caveats

This is a research/prototype pipeline, not a production system. Notably:
- Credentials must be supplied via environment variables, not hardcoded.
- The OpenFHE and TenSEAL tracks use separate, incompatible key formats.
- No formal security audit has been performed on the aggregation logic.
