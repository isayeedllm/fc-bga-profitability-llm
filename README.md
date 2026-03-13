# FC-BGA Profitability LLM Analysis

A Flask web app for semiconductor packaging economics analysis with deterministic cost modeling, scenario deltas, verification against LLM output, and downloadable charts/reports.

## What This App Does

- Uploads a single canonical product input CSV and a sensitivity scenario CSV
- Optionally ingests external PDF/XLSX/TXT files for LLM context
- Supports two execution modes:
  - API mode (uses OpenAI API if OPENAI_API_KEY is set)
  - Manual mode (generates prompt for ChatGPT, then accepts pasted JSON)
- Computes deterministic metrics in Python (source of truth):
  - Direct Cost
  - Effective Unit Cost
  - Profit per Unit
  - Scenario deltas for cost and profit
  - Breakeven yield and breakeven price
- Verifies LLM-claimed impacts against recomputed values
- Generates downloadable outputs:
  - LLM report text
  - Full JSON results
  - Yield/cost and price/profit charts
  - Scenario comparison and tornado chart

## Tech Stack

- Python 3
- Flask
- pandas
- numpy
- matplotlib
- openai
- pdfplumber
- python-dotenv

## Project Files

- app.py: Flask routes and UI flow
- llm_pipeline.py: Data cleaning, metric engine, LLM orchestration, verification, plotting
- extract_external.py: External data extraction and normalization
- templates/upload.html: Web UI
- example_prompts/prompt_template.txt: Prompt scaffold
- requirements.txt: Python dependencies

## Quick Start (Windows PowerShell)

1) Create and activate a virtual environment

python -m venv .venv
.\.venv\Scripts\Activate.ps1

2) Install dependencies

pip install -r requirements.txt

3) Optional: configure environment variables

Create a .env file in project root with:

OPENAI_API_KEY=your_key_here
FLASK_SECRET_KEY=your_secret_here

If OPENAI_API_KEY is not set, use Manual mode in the UI.

4) Run the app

python app.py

5) Open browser

http://127.0.0.1:5000

## Input Expectations

### inputs.csv

- Exactly one row (canonical SKU)
- Flexible column naming is supported (for example, Selling Price maps to SellingPrice)
- Common numeric formatting is cleaned automatically (for example, $2,000, 65%, commas)

Typical fields:
- WaferCost or Wafer Cost
- DiesPerWafer or Dies Per Wafer
- UpstreamCost
- AssemblyCost
- MaterialCost
- TestCost
- Overhead
- SellingPrice or Selling Price
- PackagingYield

### sensitivity.csv

- Multiple scenario rows
- Blank trailing rows are ignored
- Flexible naming is supported similarly to inputs.csv

Typical fields:
- Scenario
- Yield
- SellingPrice
- UpstreamCost
- WaferCost
- DiesPerWafer
- AssemblyCost
- MaterialCost
- TestCost
- Overhead

## Output Files

Outputs are written under outputs/session_YYYYMMDD_HHMMSS/.

Typical generated files:
- LLM_Report_*.txt
- llm_results_*.json
- yield_plot_*.png
- scenario_comparison_*.png
- effcost_vs_yield_*.png
- profit_vs_price_*.png
- tornado_*.png

## Notes on Calculation Logic

- Deterministic Python calculations are the authoritative metric layer.
- Scenario deltas and charts are generated from the same metric engine to keep numbers consistent.
- If UpstreamCost is present, it is used directly as the upstream component.
- If UpstreamCost is missing, wafer allocation may be derived from WaferCost / DiesPerWafer.
- Yield values are normalized (for example, 95 becomes 0.95 when needed).

## Troubleshooting

- Git not recognized: install Git for Windows and reopen terminal.
- LLM JSON paste fails: ensure valid JSON or paste content containing a single JSON object.
- UTF-8 decode errors in external text: files are read with replacement fallback; re-exporting as UTF-8 is still recommended.
- Unexpected metrics: check mapped column names and units in CSV files.

## Security

Redact confidential supplier quotes or sensitive business data before sending prompts to public LLMs. For sensitive workloads, use an enterprise/private LLM deployment.
