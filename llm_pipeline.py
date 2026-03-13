import os
import re
import json
import time
import logging
import pandas as pd
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available, plotting will be skipped.")
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from extract_external import (
    extract_from_pdf,
    extract_from_xlsx,
    normalize_facts,
    save_external_json,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROMPT_TEMPLATE_PATH = os.path.join("example_prompts", "prompt_template.txt")
PROMPT_FOR_CHATGPT_PATH = os.path.join("example_prompts", "prompt_for_chatgpt.txt")

def _clean_numeric_string(val):
    """Clean numeric-looking strings (e.g., "$1.8", "1,234") into plain numbers."""
    if pd.isna(val):
        return val
    if isinstance(val, (int, float)):
        return val
    text = str(val)
    # Remove currency symbols, commas, and whitespace
    cleaned = re.sub(r"[^0-9\.\-]", "", text)
    try:
        return float(cleaned)
    except Exception:
        return val


def _clean_dataframe_numbers(df, columns):
    """Convert columns to numeric where possible without raising."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(_clean_numeric_string)
    return df


def _norm_col_key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


CANONICAL_COLUMN_MAP = {
    "scenario": "Scenario",
    "yield": "Yield",
    "packagingyield": "PackagingYield",
    "wafercost": "WaferCost",
    "diesperwafer": "DiesPerWafer",
    "upstreamcost": "UpstreamCost",
    "assemblycost": "AssemblyCost",
    "materialcost": "MaterialCost",
    "testcost": "TestCost",
    "overhead": "Overhead",
    "sellingprice": "SellingPrice",
    "effectivecost": "EffectiveCost",
    "profitperunit": "ProfitPerUnit",
}


def canonicalize_columns(df):
    rename_map = {}
    for col in df.columns:
        key = _norm_col_key(col)
        if key in CANONICAL_COLUMN_MAP:
            rename_map[col] = CANONICAL_COLUMN_MAP[key]
    return df.rename(columns=rename_map)


def _to_float(val, default=None):
    if val is None or pd.isna(val):
        return default
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if not text:
        return default
    text = re.sub(r"[^0-9\.\-]", "", text)
    if not text:
        return default
    try:
        return float(text)
    except Exception:
        return default


def _normalize_yield(yield_val, default=1.0):
    y = _to_float(yield_val, default)
    if y is None:
        y = default
    if y > 1 and y <= 100:
        y = y / 100.0
    return y


def _num_from_row(row, key, default=None):
    if row is None:
        return default
    return _to_float(row.get(key), default)


def compute_case_metrics(base_input_row, scenario_row=None, default_yield=1.0):
    """Compute one scenario metrics using base input + optional scenario overrides."""
    upstream_cost = _num_from_row(base_input_row, "UpstreamCost", None)
    direct_cost_parts = {
        "AssemblyCost": _num_from_row(base_input_row, "AssemblyCost", 0.0),
        "MaterialCost": _num_from_row(base_input_row, "MaterialCost", 0.0),
        "TestCost": _num_from_row(base_input_row, "TestCost", 0.0),
        "Overhead": _num_from_row(base_input_row, "Overhead", 0.0),
    }
    wafer_cost = _num_from_row(base_input_row, "WaferCost", None)
    dies_per_wafer = _num_from_row(base_input_row, "DiesPerWafer", None)
    selling_price = _num_from_row(base_input_row, "SellingPrice", 0.0)

    if scenario_row is not None:
        sc_upstream = _num_from_row(scenario_row, "UpstreamCost", None)
        if sc_upstream is not None:
            upstream_cost = sc_upstream
        for key in ["AssemblyCost", "MaterialCost", "TestCost", "Overhead"]:
            override = _num_from_row(scenario_row, key, None)
            if override is not None:
                direct_cost_parts[key] = override
        sc_wafer = _num_from_row(scenario_row, "WaferCost", None)
        sc_dies = _num_from_row(scenario_row, "DiesPerWafer", None)
        if sc_wafer is not None:
            wafer_cost = sc_wafer
        if sc_dies is not None:
            dies_per_wafer = sc_dies
        sc_price = _num_from_row(scenario_row, "SellingPrice", None)
        if sc_price is not None:
            selling_price = sc_price

    wafer_component = 0.0
    if upstream_cost is None and wafer_cost is not None and dies_per_wafer not in (None, 0):
        wafer_component = wafer_cost / dies_per_wafer

    upstream_component = upstream_cost if upstream_cost is not None else wafer_component
    direct_cost = upstream_component + sum(direct_cost_parts.values())

    yield_val = None
    if scenario_row is not None:
        yield_val = _num_from_row(scenario_row, "Yield", None)
        if yield_val is None:
            yield_val = _num_from_row(scenario_row, "PackagingYield", None)
    if yield_val is None:
        yield_val = _num_from_row(base_input_row, "Yield", None)
    if yield_val is None:
        yield_val = _num_from_row(base_input_row, "PackagingYield", default_yield)

    total_yield = _normalize_yield(yield_val, default_yield)
    effective_unit_cost = compute_effective_unit_cost(direct_cost, total_yield)
    profit_per_unit = selling_price - effective_unit_cost

    return {
        "DirectCost": direct_cost,
        "TotalYield": total_yield,
        "EffectiveUnitCost": effective_unit_cost,
        "ProfitPerUnit": profit_per_unit,
        "SellingPrice": selling_price,
        "WaferCost": wafer_cost,
        "DiesPerWafer": dies_per_wafer,
        "UpstreamCost": upstream_component,
        **direct_cost_parts,
    }


def load_inputs(csv_path):
    """
    Load canonical SKU definition from CSV.
    Returns: pandas.DataFrame (single row)
    """
    df = pd.read_csv(csv_path)
    df = canonicalize_columns(df)
    if len(df) != 1:
        raise ValueError("inputs.csv must contain exactly one row (canonical SKU).")
    df = _clean_dataframe_numbers(
        df,
        [
            "WaferCost",
            "DiesPerWafer",
            "AssemblyCost",
            "MaterialCost",
            "TestCost",
            "Overhead",
            "UpstreamCost",
            "SellingPrice",
            "PackagingYield",
            "Yield",
        ],
    )
    return df


def load_sensitivity(csv_path):
    """
    Load scenario comparison table from CSV.
    Returns: pandas.DataFrame
    """
    df = pd.read_csv(csv_path)
    df = canonicalize_columns(df)
    df = df.dropna(how="all").copy()
    df = _clean_dataframe_numbers(
        df,
        [
            "WaferCost",
            "DiesPerWafer",
            "UpstreamCost",
            "AssemblyCost",
            "MaterialCost",
            "TestCost",
            "Overhead",
            "SellingPrice",
            "Yield",
            "PackagingYield",
            "Volume",
        ],
    )
    return df

def load_external_json(path):
    """
    Load normalized external data from JSON.
    Returns: dict
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(inputs_df, sensitivity_df, external_data):
    """
    Build the LLM prompt string.
    """
    # Load template
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read().strip()

    # Product block
    product_block = "\n".join(
        f"{col}: {inputs_df.iloc[0][col]}" for col in inputs_df.columns
    )

    # Sensitivity block
    if len(sensitivity_df) > 10:
        sens_head = sensitivity_df.head(10)
        stats = sensitivity_df.describe(numeric_only=True)
        sens_block = (
            sens_head.to_markdown(index=False)
            + "\n\n[Table truncated. Summary stats:]\n"
            + stats.to_markdown()
            + "\n(Analyze based on these rows and stats.)"
        )
    else:
        sens_block = sensitivity_df.to_markdown(index=False)

    # External data block
    ext_lines = []
    for srcid, d in external_data.items():
        ext_lines.append(
            f"{srcid}: {d['key']} = {d['value']} {d['units']} (raw: {d['raw']}, page: {d.get('page','?')})"
        )
    ext_block = "\n".join(ext_lines)

    prompt = (
        f"{template}\n\n"
        f"Product Definition:\n{product_block}\n\n"
        f"Sensitivity Table:\n{sens_block}\n\n"
        f"External Data:\n{ext_block}\n"
    )
    return prompt

def call_llm(prompt, use_api=True):
    """
    Call OpenAI LLM or export prompt for manual use.
    Returns: LLM response string or instructions for manual mode.
    """
    if use_api and OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            for model in ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]:
                try:
                    logging.info(f"Calling OpenAI model: {model}")
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=2048,
                    )
                    return response.choices[0].message.content
                except OpenAIError as e:
                    logging.warning(f"Model {model} failed: {e}")
            raise RuntimeError("All OpenAI models failed.")
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            return f"OpenAI API call failed: {e}"
    else:
        # Manual mode: write prompt for user
        os.makedirs(os.path.dirname(PROMPT_FOR_CHATGPT_PATH), exist_ok=True)
        with open(PROMPT_FOR_CHATGPT_PATH, "w", encoding="utf-8") as f:
            f.write(prompt)
        return (
            "No API key detected. Prompt written to example_prompts/prompt_for_chatgpt.txt. "
            "Please paste this prompt into ChatGPT, copy the JSON response, and paste it into the web UI."
        )

def verify_results(llm_json, inputs_df, sensitivity_df, external_data):
    """
    Recompute LLM's claimed impacts and check for reproducibility.
    Returns: list of check dicts.
    """
    checks = []
    try:
        inp = inputs_df.iloc[0]
        base_scenario = sensitivity_df.iloc[0] if not sensitivity_df.empty else None
        default_yield = _normalize_yield(_num_from_row(inp, "PackagingYield", 1.0), 1.0)
        base = compute_case_metrics(inp, base_scenario, default_yield=default_yield)

        scenarios = llm_json.get("ranked_scenarios") or llm_json.get("ranked_levers", [])
        for scenario in scenarios:
            scenario_name = scenario.get("scenario", "")
            impact = scenario.get("impact", {})

            scen_row = sensitivity_df[sensitivity_df["Scenario"] == scenario_name]
            if scen_row.empty:
                checks.append({"scenario": scenario_name, "error": "Scenario not found in sensitivity.csv"})
                continue

            scen = scen_row.iloc[0]
            mod = compute_case_metrics(inp, scen, default_yield=default_yield)

            delta_cost = _to_float(impact.get("delta_effective_cost"), 0.0)
            delta_profit = _to_float(impact.get("delta_profit_per_unit"), 0.0)
            recomputed_delta_cost = mod["EffectiveUnitCost"] - base["EffectiveUnitCost"]
            recomputed_delta_profit = mod["ProfitPerUnit"] - base["ProfitPerUnit"]
            check = {
                "scenario": scenario_name,
                "claimed_delta_effective_cost": delta_cost,
                "recomputed_delta_effective_cost": recomputed_delta_cost,
                "abs_diff_cost": abs(delta_cost - recomputed_delta_cost),
                "pct_diff_cost": (
                    abs(delta_cost - recomputed_delta_cost) / abs(delta_cost)
                    if delta_cost else None
                ),
                "claimed_delta_profit_per_unit": delta_profit,
                "recomputed_delta_profit_per_unit": recomputed_delta_profit,
                "abs_diff_profit": abs(delta_profit - recomputed_delta_profit),
                "pct_diff_profit": (
                    abs(delta_profit - recomputed_delta_profit) / abs(delta_profit)
                    if delta_profit else None
                ),
                "ok": (
                    abs(delta_cost - recomputed_delta_cost) < 0.5
                    and abs(delta_profit - recomputed_delta_profit) < 0.5
                ),
            }
            checks.append(check)
    except Exception as e:
        logging.error(f"Verification failed: {e}")
        checks.append({"error": str(e)})
    return checks

def run_pipeline(inputs_csv, sensitivity_csv, external_files_list, out_dir, use_api=True, manual_llm_json=None):
    """
    Main pipeline: extract, normalize, prompt, call LLM, verify, save outputs.
    Returns: dict with status and output file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log = {"status": "started", "outputs": {}, "errors": []}
    try:
        # 1. Load inputs
        inputs_df = load_inputs(inputs_csv)
        sensitivity_df = load_sensitivity(sensitivity_csv)
        # Normalize dataframes
        inputs_df = normalize_dataframe(inputs_df)
        sensitivity_df = normalize_dataframe(sensitivity_df)
        # Compute base metrics from the first sensitivity row (if present).
        inp = inputs_df.iloc[0]
        base_scenario = sensitivity_df.iloc[0] if not sensitivity_df.empty else None
        default_yield = _normalize_yield(_num_from_row(inp, "PackagingYield", 1.0), 1.0)
        base = compute_case_metrics(inp, base_scenario, default_yield=default_yield)

        # Compute additional metrics
        base_direct_cost = base["DirectCost"]
        base_effective_cost = base["EffectiveUnitCost"]
        base_profit = base["ProfitPerUnit"]
        chart_series = generate_chart_series(base_direct_cost, base["TotalYield"], base["SellingPrice"])
        deltas = compute_scenario_deltas(
            sensitivity_df,
            inp,
            base_effective_cost,
            base_profit,
            default_yield,
        )
        tornado_data = {d['scenario']: d['delta_profit_per_unit'] for d in deltas}
        breakeven_values = {
            "breakeven_yield": breakeven_yield(base_direct_cost, base["SellingPrice"]),
            "breakeven_price": breakeven_price(base_direct_cost, base["TotalYield"])
        }
        scenario_plot_path = os.path.join(out_dir, f"scenario_comparison_{ts}.png")
        if plot_scenario_comparison(inp, sensitivity_df, outpath=scenario_plot_path):
            log["outputs"]["scenario_plot"] = scenario_plot_path

        # 2. Extract and normalize external data
        raw_facts = []
        for f in external_files_list:
            ext = os.path.splitext(f)[1].lower()
            try:
                if ext == ".pdf":
                    raw_facts.extend(extract_from_pdf(f))
                elif ext in [".xlsx", ".xls"]:
                    raw_facts.extend(extract_from_xlsx(f))
                elif ext in [".txt", ".csv"]:
                    # Treat as plain text: each line as a "fact"
                    with open(f, "r", encoding="utf-8", errors="replace") as tf:
                        for i, line in enumerate(tf):
                            raw_facts.append({"page": 1, "line": i+1, "raw": line.strip()})
            except Exception as e:
                logging.warning(f"Extraction failed for {f}: {e}")
        external_data = normalize_facts(raw_facts)
        ext_json_path = os.path.join(out_dir, "external_data.json")
        save_external_json(external_data, ext_json_path)
        # 3. Build prompt
        prompt = build_prompt(inputs_df, sensitivity_df, external_data)
        # 4. LLM call or manual
        if manual_llm_json:
            llm_json = manual_llm_json
            llm_response = "Manual LLM JSON provided."
        else:
            llm_response = call_llm(prompt, use_api=use_api)
            log["llm_response"] = llm_response
            # Try to parse JSON from LLM response
            try:
                llm_json = json.loads(llm_response)
            except Exception:
                llm_json = None
        # 5. If manual mode and no JSON, return prompt and wait for user
        if not llm_json:
            prompt_path = os.path.join("example_prompts", "prompt_for_chatgpt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            log["status"] = "awaiting_manual_llm"
            log["outputs"]["prompt"] = prompt_path
            # include LLM response (if any) for debugging
            if "llm_response" in log:
                log["llm_response_excerpt"] = log["llm_response"][:1000]
            return log
        # 6. Verify
        checks = verify_results(llm_json, inputs_df, sensitivity_df, external_data)
        # 7. Save outputs
        json_path = os.path.join(out_dir, f"llm_results_{ts}.json")
        txt_path = os.path.join(out_dir, f"LLM_Report_{ts}.txt")
        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "llm_json": llm_json,
                    "checks": checks,
                    "timestamp": ts,
                    "external_data": external_data,
                    "base_metrics": base,
                    "chart_series": chart_series,
                    "scenario_deltas": deltas,
                    "tornado_data": tornado_data,
                    "breakeven_values": breakeven_values,
                },
                f,
                indent=2,
            )
        # Save TXT report
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Executive Summary:\n")
            f.write(llm_json.get("executive_summary", "") + "\n\n")
            f.write("Ranked Scenarios:\n")
            scenarios = llm_json.get("ranked_scenarios") or llm_json.get("ranked_levers", [])
            for scenario in scenarios:
                f.write(f"- {scenario.get('scenario','')}: {scenario.get('reason','')}\n")
            breakeven = breakeven_yield(base["DirectCost"], base["SellingPrice"])
            if breakeven is not None:
                f.write(f"\nBreakeven Yield for Profit > 0: {breakeven:.1%}\n")
            breakeven_price_val = breakeven_price(base["DirectCost"], base["TotalYield"])
            if breakeven_price_val is not None:
                f.write(f"Breakeven Price at Current Yield: ${breakeven_price_val:.2f}\n")
            f.write("\nVerification Results:\n")
            for c in checks:
                if "error" in c:
                    f.write(f"ERROR: {c['error']}\n")
                else:
                    f.write(
                        f"{c['scenario']}: ok={c['ok']} "
                        f"(Δcost: {c['claimed_delta_effective_cost']} vs {c['recomputed_delta_effective_cost']:.4f}, "
                        f"Δprofit: {c['claimed_delta_profit_per_unit']} vs {c['recomputed_delta_profit_per_unit']:.4f})\n"
                    )
        # Save yield plot
        plot_path = os.path.join(out_dir, f"yield_plot_{ts}.png")
        plot_result = plot_effective_cost_vs_yield(base["DirectCost"], outpath=plot_path)
        if plot_result:
            log["outputs"]["yield_plot"] = plot_path

        # Save new charts
        effcost_path = os.path.join(out_dir, f"effcost_vs_yield_{ts}.png")
        if plot_and_save_chart(chart_series["effcost_vs_yield"], effcost_path, "effcost_vs_yield"):
            log["outputs"]["effcost_vs_yield"] = effcost_path

        profit_path = os.path.join(out_dir, f"profit_vs_price_{ts}.png")
        if plot_and_save_chart(chart_series["profit_vs_price"], profit_path, "profit_vs_price"):
            log["outputs"]["profit_vs_price"] = profit_path

        tornado_path = os.path.join(out_dir, f"tornado_{ts}.png")
        if plot_and_save_chart(tornado_data, tornado_path, "tornado"):
            log["outputs"]["tornado"] = tornado_path

        log["status"] = "complete"
        log["outputs"]["llm_results_json"] = json_path
        log["outputs"]["llm_report_txt"] = txt_path
        log["llm_json"] = llm_json
        log["checks"] = checks
        log["base_metrics"] = base
        log["chart_series"] = chart_series
        log["scenario_deltas"] = deltas
        log["tornado_data"] = tornado_data
        log["breakeven_values"] = breakeven_values
        return log
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        log["status"] = "error"
        log["errors"].append(str(e))
        return log

def breakeven_yield(direct_cost, target_price):
    """
    direct_cost = Assembly + Material + Test + Overhead + UpstreamCost  (per unit)
    target_price = selling price (USD)
    returns yield_frac needed so that EffectiveUnitCost = direct_cost / yield_frac <= target_price
    """
    if target_price <= 0 or direct_cost <= 0:
        return None
    # We need direct_cost / yield <= target_price  =>  yield >= direct_cost / target_price
    required_yield = direct_cost / target_price
    return min(max(required_yield, 0.0), 1.0)

def breakeven_price(direct_cost, yield_frac):
    """
    direct_cost = per-unit direct cost (Assembly + Material + Test + Overhead + UpstreamCost)
    yield_frac = scenario total yield (0..1)
    returns minimum selling price to break-even (EffectiveUnitCost)
    """
    if yield_frac <= 0:
        return None
    effective_cost = direct_cost / yield_frac
    return effective_cost

def clean_numeric_col(val):
    """
    Clean a single value for numeric columns.
    - If NaN, return val
    - If str: strip $, ,, whitespace; if ends with % convert to decimal; attempt float()
    - Otherwise return val
    """
    if pd.isna(val):
        return val
    if isinstance(val, str):
        val = val.strip()
        val = re.sub(r'[$,]', '', val)
        if val.endswith('%'):
            val = val[:-1]
            try:
                return float(val) / 100.0
            except ValueError:
                return val
        try:
            return float(val)
        except ValueError:
            return val
    return val

def normalize_dataframe(df):
    """
    Apply clean_numeric_col to columns matching keywords.
    Keywords: cost, price, yield, volume, wafer, defect, upstream, assembly, material, test, overhead, dies, selling (case insensitive)
    Return cleaned df.
    """
    keywords = ["cost", "price", "yield", "volume", "wafer", "defect", "upstream", "assembly", "material", "test", "overhead", "dies", "selling"]
    for col in df.columns:
        if any(kw.lower() in col.lower() for kw in keywords):
            df[col] = df[col].apply(clean_numeric_col)
    return df

def compute_direct_cost(row):
    """
    Compute direct cost from row.
    WaferCost_per_die = WaferCost / DiesPerWafer
    DirectCost = WaferCost_per_die + UpstreamCost + AssemblyCost + MaterialCost + TestCost + Overhead
    Raise ValueError if issues.
    """
    try:
        upstream_cost = _num_from_row(row, 'UpstreamCost', None)
        wafer_cost = _num_from_row(row, 'WaferCost', None)
        dies_per_wafer = _num_from_row(row, 'DiesPerWafer', None)
        wafer_cost_per_die = 0.0
        if upstream_cost is None and wafer_cost is not None:
            if dies_per_wafer in (None, 0):
                raise ValueError("DiesPerWafer cannot be zero when WaferCost is provided")
            wafer_cost_per_die = wafer_cost / dies_per_wafer
        upstream_component = upstream_cost if upstream_cost is not None else wafer_cost_per_die
        direct_cost = (
            upstream_component
            + _num_from_row(row, 'AssemblyCost', 0.0)
            + _num_from_row(row, 'MaterialCost', 0.0)
            + _num_from_row(row, 'TestCost', 0.0)
            + _num_from_row(row, 'Overhead', 0.0)
        )
        return float(direct_cost)
    except (TypeError, ZeroDivisionError, ValueError) as e:
        raise ValueError(f"Error computing direct cost: {e}")

def compute_effective_unit_cost(direct_cost, total_yield):
    """
    Return direct_cost / total_yield, validate 0 < total_yield <= 1
    """
    if pd.isna(total_yield):
        raise ValueError("total_yield is NaN")
    total_yield = float(total_yield)
    # Accept percentage-style inputs (e.g., 95 -> 0.95).
    if total_yield > 1 and total_yield <= 100:
        total_yield = total_yield / 100.0
    if not (0 < total_yield <= 1):
        raise ValueError("total_yield must be between 0 and 1")
    return direct_cost / total_yield

def generate_chart_series(direct_cost, base_yield, base_price):
    """
    Return dict with effcost_vs_yield and profit_vs_price as lists of floats.
    """
    # effcost_vs_yield
    base_yield = float(base_yield)
    if base_yield > 1 and base_yield <= 100:
        base_yield = base_yield / 100.0
    min_yield = min(0.2, base_yield * 0.5)
    yields = [min_yield + i * (0.95 - min_yield) / 15 for i in range(16)]
    eff_costs = [direct_cost / y for y in yields]
    
    # profit_vs_price
    prices = [0.5 * base_price + i * (1.5 * base_price - 0.5 * base_price) / 8 for i in range(9)]
    profits = [p - (direct_cost / base_yield) for p in prices]
    
    return {
        "effcost_vs_yield": {"yields": yields, "eff_costs": eff_costs},
        "profit_vs_price": {"prices": prices, "profits": profits}
    }

def compute_scenario_deltas(
    sensitivity_df,
    base_input_row,
    base_effective_cost,
    base_profit,
    default_yield,
):
    """
    Compute deltas for each scenario.
    Return list of dicts with scenario, delta_effective_cost, delta_profit_per_unit
    """
    deltas = []
    for idx, row in sensitivity_df.iterrows():
        try:
            metrics = compute_case_metrics(base_input_row, row, default_yield=default_yield)
            delta_effective_cost = metrics["EffectiveUnitCost"] - base_effective_cost
            delta_profit_per_unit = metrics["ProfitPerUnit"] - base_profit
            scenario_name = row.get('Scenario')
            if pd.isna(scenario_name):
                scenario_name = f"Scenario {idx + 1}"
            deltas.append({
                "scenario": scenario_name,
                "delta_effective_cost": delta_effective_cost,
                "delta_profit_per_unit": delta_profit_per_unit
            })
        except (ValueError, TypeError) as e:
            logging.warning(f"Skipping scenario {row.get('Scenario', 'unknown')}: {e}")
    return deltas

def quick_substrate_impact(direct_cost, yield_frac, delta_substrate):
    """
    Return delta_effective_cost and delta_profit_per_unit when MaterialCost changes by delta_substrate.
    Assume delta_substrate is absolute change in MaterialCost.
    """
    # This is simplified; in full, would need to recompute direct_cost with new material cost
    # For now, assume direct_cost increases by delta_substrate
    new_direct_cost = direct_cost + delta_substrate
    new_effective_cost = new_direct_cost / yield_frac
    delta_effective_cost = new_effective_cost - (direct_cost / yield_frac)
    # Profit delta assuming price fixed: -delta_effective_cost
    delta_profit_per_unit = -delta_effective_cost
    return delta_effective_cost, delta_profit_per_unit

def plot_and_save_chart(series, outpath, chart_type):
    """
    Plot and save chart based on type.
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("matplotlib not available, skipping plot.")
        return None
    plt.figure(figsize=(6,4))
    if chart_type == "effcost_vs_yield":
        yields = series["yields"]
        eff_costs = series["eff_costs"]
        plt.plot(yields, eff_costs, marker='o')
        plt.xlabel("Yield (fraction)")
        plt.ylabel("Effective Unit Cost (USD)")
        plt.title("Effective Unit Cost vs Yield")
        plt.grid(True)
    elif chart_type == "profit_vs_price":
        prices = series["prices"]
        profits = series["profits"]
        plt.plot(prices, profits, marker='o')
        plt.xlabel("Selling Price (USD)")
        plt.ylabel("Profit per Unit (USD)")
        plt.title("Profit per Unit vs Selling Price")
        plt.grid(True)
    elif chart_type == "tornado":
        # series is dict of scenario: delta_profit
        sorted_items = sorted(series.items(), key=lambda x: x[1], reverse=True)
        scenarios = [k for k, v in sorted_items]
        deltas = [v for k, v in sorted_items]
        plt.barh(scenarios, deltas)
        plt.xlabel("Delta Profit per Unit")
        plt.title("Tornado Chart: Scenario Impact on Profit")
        for i, v in enumerate(deltas):
            plt.text(v, i, f"{v:.2f}", va='center')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

def breakeven_yield(direct_cost, selling_price):
    """
    Compute breakeven yield where effective_unit_cost = selling_price.
    Effective unit cost = direct_cost / yield, so yield = direct_cost / selling_price.
    """
    if selling_price <= 0:
        return float('inf')  # impossible
    return direct_cost / selling_price

def quick_self_test():
    """
    Unit test style function.
    """
    # Fake data
    fake_row = {
        'WaferCost': 1000, 'DiesPerWafer': 100, 'UpstreamCost': 0.2, 'AssemblyCost': 1.0, 'MaterialCost': 2.0, 'TestCost': 0.5, 'Overhead': 0.3, 'Yield': 0.95, 'SellingPrice': 5.0
    }
    df = pd.DataFrame([fake_row])
    df = normalize_dataframe(df)
    direct_cost = compute_direct_cost(df.iloc[0])
    base_effective_cost = compute_effective_unit_cost(direct_cost, df.iloc[0]['Yield'])
    base_profit = df.iloc[0]['SellingPrice'] - base_effective_cost
    chart_series = generate_chart_series(direct_cost, df.iloc[0]['Yield'], df.iloc[0]['SellingPrice'])
    # Fake sensitivity
    sens_df = pd.DataFrame([
        {'Scenario': 'Base', 'Yield': 0.95, 'SellingPrice': 5.0, 'WaferCost': 1000, 'DiesPerWafer': 100, 'UpstreamCost': 0.2, 'AssemblyCost': 1.0, 'MaterialCost': 2.0, 'TestCost': 0.5, 'Overhead': 0.3},
        {'Scenario': 'Improved Yield', 'Yield': 0.97, 'SellingPrice': 5.0, 'WaferCost': 1000, 'DiesPerWafer': 100, 'UpstreamCost': 0.2, 'AssemblyCost': 1.0, 'MaterialCost': 2.0, 'TestCost': 0.5, 'Overhead': 0.3}
    ])
    sens_df = normalize_dataframe(sens_df)
    deltas = compute_scenario_deltas(
        sens_df,
        df.iloc[0],
        base_effective_cost,
        base_profit,
        0.95,
    )
    breakeven_y = breakeven_yield(direct_cost, df.iloc[0]['SellingPrice'])
    return {
        "sanity": True,
        "direct_cost": direct_cost,
        "breakeven_yield": breakeven_y,
        "chart_sample": chart_series["effcost_vs_yield"]["eff_costs"][:3]  # first 3
    }

def plot_effective_cost_vs_yield(direct_cost, yields=None, outpath="effective_cost_vs_yield.png"):
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("matplotlib not available, skipping plot.")
        return None
    if yields is None:
        yields = np.linspace(0.2, 0.95, 16)  # you can adjust range
    eff_costs = [direct_cost / y for y in yields]
    plt.figure(figsize=(6,4))
    plt.plot(yields, eff_costs, marker='o')
    plt.xlabel("Yield (fraction)")
    plt.ylabel("Effective Unit Cost (USD)")
    plt.title("Effective Unit Cost vs Yield")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

def plot_scenario_comparison(base_input_row, scenarios_df, outpath="scenario_comparison.png"):
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("matplotlib not available, skipping plot.")
        return None
    if scenarios_df.empty:
        return None

    default_yield = _normalize_yield(_num_from_row(base_input_row, "PackagingYield", 1.0), 1.0)
    scenario_names = []
    costs = []
    profits = []

    for idx, row in scenarios_df.iterrows():
        try:
            metrics = compute_case_metrics(base_input_row, row, default_yield=default_yield)
            scenario_name = row.get("Scenario")
            if pd.isna(scenario_name):
                scenario_name = f"Scenario {idx + 1}"
            scenario_names.append(str(scenario_name))
            costs.append(metrics["EffectiveUnitCost"])
            profits.append(metrics["ProfitPerUnit"])
        except Exception as e:
            logging.warning(f"Skipping scenario row {idx + 1} in scenario chart: {e}")

    if not scenario_names:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(scenario_names, costs, color='skyblue')
    ax1.set_title("Effective Unit Cost by Scenario")
    ax1.set_ylabel("Cost (USD)")
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(scenario_names, profits, color='lightgreen')
    ax2.set_title("Profit per Unit by Scenario")
    ax2.set_ylabel("Profit (USD)")
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

def test_verify():
    """
    Unit test for verify_results.
    """
    # Toy data
    inputs = pd.DataFrame([{
        "SKU": "ABC123",
        "AssemblyCost": 1.0,
        "MaterialCost": 2.0,
        "TestCost": 0.5,
        "Overhead": 0.3,
        "UpstreamCost": 0.2,
        "SellingPrice": 5.0,
        "PackagingYield": 0.98,
    }])
    sens = pd.DataFrame([{
        "Scenario": "Base",
        "Yield": 0.95,
        "PackagingYield": 0.98,
        "Volume": 10000,
    }, {
        "Scenario": "Reduce MaterialCost",
        "Yield": 0.95,
        "PackagingYield": 0.98,
        "MaterialCost": 1.8,
        "Volume": 10000,
    }])
    ext = {
        "src1": {"key": "MaterialCost", "value": 1.8, "units": "USD", "raw": "$1.80", "page": 2, "confidence": 0.95}
    }
    llm_json = {
        "primary_driver": "MaterialCost",
        "ranked_scenarios": [
            {
                "scenario": "Reduce MaterialCost",
                "reason": "Lower substrate price",
                "impact": {"delta_effective_cost": -0.2148227712137487, "delta_profit_per_unit": 0.2148227712137487},
                "sources": ["src1"]
            }
        ],
        "executive_summary": "Reducing material cost improves profit.",
        "next_steps": ["Negotiate with vendor", "Review alternative substrates", "Model impact on yield"]
    }
    checks = verify_results(llm_json, inputs, sens, ext)
    print("Test verify_results:")
    for c in checks:
        print(c)
    assert checks[0]["ok"], "Verification failed for toy scenario"
