---
layout: page
title: research
permalink: /research/
description: Ongoing and completed research projects.
nav: true
nav_order: 2
---

## PhD Research

### Edge-Deployable Crisis Tweet Summarization

> Designing NLP pipelines that generate trustworthy disaster situation summaries
> from social media — running entirely on local hardware, no internet required.

During a disaster, social media generates thousands of posts per minute with critical
real-time information. Field responders often operate without stable internet, making
cloud AI unavailable exactly when it's needed most.

This research builds and evaluates an **edge-deployable NLP pipeline** that:
- Detects crisis information bursts from Twitter/X streams
- Generates abstractive situation summaries using lightweight transformer models
- Runs fully offline on a standard laptop

**Key Findings (V0.6):**

| Model | BERTScore | Edge Feasibility |
|---|---|---|
| Flan-T5 + ChromaDB (RAG) | 0.8403 | ❌ Cloud-dependent |
| DistilBART | ~0.82 | ✅ 353 MB, Flesch 60.7 |

**DistilBART** is the recommended edge model — compact, readable output, no internet needed.
**Flan-T5 + ChromaDB** leads on quality for connected environments.

**Status:** First project of PhD · Notebooks complete · Paper in preparation

**Tech:** Python · HuggingFace Transformers · ChromaDB · Flan-T5 · DistilBART · ROUGE · BERTScore

---

## Research Interests

- Natural Language Processing
- Abstractive Text Summarization
- Retrieval-Augmented Generation (RAG)
- Crisis Informatics & Disaster Response
- Edge AI & On-Device Inference
