# Technical Analyst Agent (LLM-Powered Trading Signal Generator)

## Overview
This repository contains a **Technical Analyst Agent** implemented in Python/Jupyter that:
1) downloads market data (OHLCV),  
2) computes core technical indicators,  
3) generates a regime-based exposure signal with a risk overlay,  
4) runs a backtest (including simple transaction costs), and  
5) optionally generates an **LLM trade note** summarizing signals and evidence.

The project is designed for **reproducible coursework assessment**: an assessor can run either the notebook or the demo script to reproduce the same core outputs.

---

## Repository Structure
```
.
├── Technical_Analyst_Agent_Pingfan_Qian.ipynb   # Full pipeline notebook
├── run_demo.py                                  # Minimal script to reproduce key outputs
├── requirements.txt                             # Python dependencies
├── README.md                                    # Run instructions + file structure (this file)
└── outputs/                                     # Generated outputs (created on run)
    ├── backtest_results.csv
    ├── performance_metrics.json
    ├── googl_technical_analysis.png             # Example chart name (depends on ticker)
    └── trade_note.md                            # Only if LLM enabled + API key set
```

---

## Quick Start

### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the demo script (recommended for assessors)
```bash
python run_demo.py --ticker GOOGL --years 10 --outdir outputs
```

This generates:
- `outputs/backtest_results.csv`
- `outputs/performance_metrics.json`
- `outputs/<ticker>_technical_analysis.png`

### 4) (Optional) Generate an LLM trade note
Set an environment variable before running:
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_MODEL="gpt-4o-mini"
python run_demo.py --ticker GOOGL --period 10y --output_dir outputs --trade_note
```

```bash
# macOS/Linux
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="gpt-4o-mini"
python run_demo.py --ticker GOOGL --period 10y --output_dir outputs --trade_note
```

If enabled, the script writes:
- `outputs/<ticker>_trade_note.md`

---

## Notebook Run (alternative)
Open and run the notebook top-to-bottom:
- `Technical_Analyst_Agent_Pingfan_Qian.ipynb`

Notes:
- Cells containing `!pip install ...` are intended for **Colab**. If running locally, install via `requirements.txt`.
- For LLM trade notes, set `OPENAI_API_KEY` in the environment (or within the notebook) before the LLM cell.

---

## What the Demo Script Produces

### Indicators
- SMA(20), SMA(50), SMA(200)
- RSI(14)
- MACD (12,26,9)
- Bollinger Bands (20, 2σ)

### Signal / Risk Overlay (high-level)
- **Trend regime**: price vs SMA(200)
- **Volatility regime**: annualized vol vs rolling quantile threshold
- Weekly rebalancing (default: Friday close)
- Optional volatility targeting + max leverage cap

### Backtest
- Strategy returns use **lagged exposure** (no look-ahead)
- Transaction costs: configurable basis points (default included)
- Metrics: Total return, CAGR (approx), Sharpe (annualized), Max drawdown, trade days

---

## Marking / Reproducibility Notes
- Outputs are deterministic given the same data download and parameters.
- The backtest avoids look-ahead by applying `position_lag`.
- LLM trade notes are optional; when enabled, prompts are constrained to computed metrics to reduce hallucination risk.

---

## Requirements
- Python 3.9+
- See `requirements.txt`

---

## Author
Pingfan Qian  
MSc Banking & Digital Finance, UCL
