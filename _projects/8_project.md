---
layout: page
title: Credit Card Fraud Detection using ML & Deep Learning
description: Comparative study of ML and deep learning approaches for credit card fraud detection on imbalanced datasets. Published at Springer ICICC 2022 and AIP 2023.
img: assets/img/8.jpg
importance: 8
category: research
related_publications: true
---

## Overview

Credit card fraud causes billions in annual losses globally. This project performs a **comparative analysis** of machine learning and deep learning approaches for detecting fraudulent transactions in highly imbalanced datasets — a core challenge in financial fraud detection.

---

## Problem

- Fraud transactions are rare (< 0.2% of all transactions) — severe class imbalance
- Real-time detection requires high accuracy and low latency
- False positives (legitimate transactions blocked) carry high customer cost

---

## Methodology

- **Class imbalance handling** — SMOTE, undersampling, cost-sensitive learning
- **Traditional ML** — Logistic Regression, Decision Trees, Random Forest, SVM, k-NN
- **Deep learning** — Artificial Neural Networks, Autoencoders for anomaly detection
- **Evaluation** — precision, recall, F1-score, AUC-ROC (accuracy alone is misleading on imbalanced data)

---

## Key Findings

- Ensemble methods (Random Forest) outperformed single classifiers on imbalanced data
- Deep learning autoencoders effective for anomaly-based unsupervised fraud detection
- SMOTE combined with ensemble methods gave best precision-recall tradeoff

---

## Publications

{% cite gorte2022creditcard %}

{% cite gorte2023creditcard %}

---

## Status

**Published** — Springer ICICC 2022 and AIP Publishing 2023.
