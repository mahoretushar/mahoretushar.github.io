---
layout: page
title: Student Attendance Monitoring via Facial Recognition
description: Automated contactless student attendance system using deep learning-based facial recognition. Published at Springer ICICC 2022.
img: assets/img/9.jpg
importance: 9
category: software
related_publications: true
---

## Overview

Manual attendance marking is time-consuming and prone to proxy fraud. This project implements an **automated student attendance monitoring system** using **deep learning-based facial recognition**, enabling real-time, contactless attendance logging from classroom camera feeds.

---

## Motivation

- Manual roll calls consume 5–10 minutes of every lecture
- Proxy attendance is a persistent problem in engineering colleges
- Post-pandemic, contactless solutions gained added relevance
- Results need to integrate with college ERP/MIS systems

---

## System Design

- **Face detection** — MTCNN / Haar Cascade to detect and localize faces in classroom images
- **Face recognition** — deep CNN (FaceNet / VGGFace) to identify students from a registered database
- **Attendance logging** — automatic marking with timestamp, cross-referenced against class schedule
- **Database integration** — structured storage in MySQL/PostgreSQL
- **Dashboard** — faculty view for attendance reports and anomaly alerts

---

## Key Contributions

- End-to-end pipeline from live camera feed to attendance database
- Handles partial occlusion, varying lighting, and multiple faces per frame
- Evaluated on real classroom data from an engineering college environment

---

## Publication

Published in **International Conference on Innovative Computing and Communications (ICICC 2022)**, Vol. 3, pp. 613–620, Springer Nature Singapore.

{% cite wankhade2022attendance %}

---

## Status

**Published** — Springer ICICC 2022.
