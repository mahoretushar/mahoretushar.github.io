---
layout: page
title: Explainable TabTransformer–RF Model for IoMT Security
description: Hybrid explainable AI model combining TabTransformer and Random Forest for biometric security in IoMT healthcare systems. Published in Academic Press, 2026.
img: assets/img/3.jpg
importance: 3
category: research
related_publications: true
---

## Overview

The **Internet of Medical Things (IoMT)** connects medical devices, wearables, and healthcare infrastructure — but introduces significant biometric security vulnerabilities. This project develops a **hybrid explainable AI model** that combines the representational power of Transformers with the robustness of ensemble methods to secure IoMT systems.

---

## Motivation

IoMT devices collect continuous biometric data (ECG, EEG, fingerprints, etc.). Securing this data against spoofing and unauthorized access requires models that are both **accurate** and **interpretable** — clinicians and security auditors must be able to understand why a decision was made.

---

## Methodology

- **TabTransformer** — applies self-attention to tabular biometric feature data, learning complex inter-feature dependencies
- **Random Forest** — provides robustness, handles non-linearity, and serves as an interpretable ensemble baseline
- **Hybrid architecture** — TabTransformer embeddings fed into a Random Forest classifier
- **Explainability** — SHAP values used to explain individual predictions

---

## Key Contributions

- Hybrid TabTransformer–Random Forest approach for IoMT biometric security
- Explainability layer makes the model audit-ready for clinical environments
- Evaluated on real IoMT biometric datasets with competitive accuracy

---

## Publication

Published in **Recent Advances in Computational Intelligence Applications for Biometrics and Biomedical Devices**, pp. 285–300, Academic Press, 2026.

{% cite pande2026tabtransformer %}

---

## Status

**Published** — Academic Press, 2026.
