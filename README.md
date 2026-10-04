# StressFreeFantasy

[![TabPFN v2](https://img.shields.io/badge/Model-TabPFN%20v2-blue?style=flat-square)](https://github.com/priorlabs/TabPFN)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square)](https://streamlit.io)
[![Data: nflverse](https://img.shields.io/badge/Data-nflverse-2ea44f?style=flat-square)](https://nflverse.com/)
[![DEV Challenge](https://img.shields.io/badge/DEV_Challenge-Hacktoberfest_2026-orange?style=flat-square)](https://dev.to)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

In-context tabular foundation model (TabPFN) for real-time NFL fantasy lineup decisions and Start/Sit dilemmas.

Built for the **DEV Challenge: Build for a Friend (Hacktoberfest 2026)** — competing for **Best Use of TabPFN**.

---

## Overview

Most fantasy platforms rely on static point projections that ignore game script, contextual weather, and actual player ceiling/floor variance. **StressFreeFantasy** pulls actual historical play-by-play data via `nflverse` and uses **TabPFN** to fit an in-context predictive model dynamically to any league's scoring rules.

---

## Key Features

* **In-Context Learning:** Conditioned on 2,000+ real NFL player-weeks per forward pass—adapts instantly to custom scoring changes without gradient-based fine-tuning.
* **Floor & Ceiling Envelopes:** Generates native **P10 (Floor)** and **P90 (Ceiling)** intervals to support both conservative protection and high-upside chasing.
* **ESPN Private League Sync:** Automatic ingestion of roster, matchup lines, defensive ranks, and league scoring tables via `espn-api`.
* **Context-Driven Feature Space:** Ingests Vegas totals, implied team pace, dome status, defensive efficiency against specific positions, and target/carry volume.
* **Start/Sit Pairwise Validation:** Evaluated directly on head-to-head decision accuracy within position groups rather than relying strictly on noisy MAE.

---

## Benchmarks

Temporal out-of-time evaluation across unseen holdout season match weeks:

| Metric            | Baseline (Avg) | Baseline (Form) | TabPFN   |
| :---------------- | :------------- | :-------------- | :------- |
| MAE               | 6.84 pts       | 7.12 pts        | 5.91 pts |
| Correlation (r)   | 0.44           | 0.38            | 0.58     |
| Pairwise Accuracy | 52.8%          | 51.4%           | 63.7%    |

---

## Quickstart

### Prerequisites
* Python 3.10 – 3.14
* Recommended: CUDA-capable GPU (CPU execution fully enabled via `TABPFN_ALLOW_CPU_LARGE_DATASET=1`)

### 1. Setup Environment
```bash
git clone [https://github.com/lucasantonsson/StressFreeFantasy.git](https://github.com/lucasantonsson/StressFreeFantasy.git)
cd StressFreeFantasy

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Application
```bash
streamlit run StressFreeFantasy.py
```

---

## Private ESPN League Authentication

To pull directly from a private ESPN league:

1. Open your league in a browser (Brave, Chrome, Edge).
2. Press `F12` to open Developer Tools.
3. Go to **Application** (or **Storage**) → **Cookies** → `https://fantasy.espn.com`.
4. Copy the values for `espn_s2` and `SWID`.
5. Enter them under the **Live ESPN Sync** tab along with your `League ID`.

*A fallback manual input and CSV uploader are also provided.*

---

## Repository Structure

```text
├── StressFreeFantasy.py   # Main Streamlit engine & TabPFN pipeline
├── requirements.txt       # Dependencies
└── README.md
```

---

## Acknowledgments & AI Transparency

* Built using **TabPFN v2** by Prior Labs.
* Play-by-play historical statistics supplied by the **nflverse** community.
* Code generation, architectural refinement, and debugging were carried out in large parts by Claude and Gemini, under manual direction for system design, validation, and domain integration.
---

## Submission Details

* **Event:** Hacktoberfest 2026: Build for a Friend DEV Challenge
* **Writeup:** [Read the full article on DEV.to](https://dev.to)
* **License:** MIT