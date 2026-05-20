# Feature-Type-Aware Patient Stratification using HDBSCAN

This repository presents an academic extension of the paper:

> **"Accounting for diverse feature-types improves patient stratification on tabular clinical datasets"**

The project investigates whether replacing the original clustering approaches with **HDBSCAN** can improve patient stratification quality on Feature-Type Distributed Clustering (FDC) embeddings generated from heterogeneous clinical datasets.

---

## Overview

Clinical datasets often contain multiple feature types, including:

- Continuous variables
- Ordinal variables
- Nominal variables

Traditional dimensionality reduction approaches frequently ignore these differences and apply a single metric (commonly Euclidean distance) to all features, potentially producing biased low-dimensional embeddings.

The original paper introduces the **Feature-Type Distributed Clustering (FDC)** workflow, which handles each feature type separately before combining them into an intermediate embedding representation.

This project extends the original work by exploring the use of **HDBSCAN** as an alternative clustering method for identifying clinically relevant patient sub-populations.

---

## Objective

The main goal of this project is to evaluate whether **HDBSCAN** can improve:

- Cluster separation
- Cluster compactness
- Clinical interpretability
- Patient stratification quality

when applied to FDC embeddings.

