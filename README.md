# TimeSenCLIP: A Vision-Language Model for Remote Sensing Using Single-Pixel Time Series

**Anonymous Repository for Double-Blind Review**

---

## 🛰️ Overview

**TimeSenCLIP** is a lightweight vision-language model for remote sensing tasks that revisits the need for spatial context in satellite image classification. Instead of relying on large image patches or text supervision, TimeSenCLIP explores the use of **single-pixel inputs**, enriched with **temporal** and **spectral** information from Sentinel-2 imagery, paired with **cross-view supervision** from ground-level geo-tagged images.

---

## 📝 Abstract

Vision-language models have shown significant promise in remote sensing, particularly for land-use and land-cover (LULC) classification and retrieval through zero-shot learning. However, two key challenges persist: the reliance on large spatial tiles, which increase computational cost, and the dependence on text-based supervision, which is often scarce or unavailable in remote sensing contexts.

In this work, we introduce **TimeSenCLIP**, a lightweight and scalable framework that reconsiders the role of spatial context by evaluating the effectiveness of **single-pixel** inputs for semantic classification. Our method leverages the rich **temporal and spectral information** from Sentinel-2 satellite imagery and incorporates **cross-view learning** using geo-tagged ground-level photographs, reducing reliance on textual captions while preserving semantic alignment between overhead and ground perspectives.

Built upon the **LUCAS** and **Sen4Map** datasets, TimeSenCLIP is evaluated on various classification tasks including LULC, crop type, and ecosystem type. We demonstrate that, when enhanced with temporal and spectral signals, **single-pixel inputs are sufficient for robust thematic mapping**—paving the way for efficient and large-scale remote sensing analysis without sacrificing performance.

---

## 📦 Repository Structure


