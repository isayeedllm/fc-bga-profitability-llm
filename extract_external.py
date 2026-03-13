import os
import re
import pdfplumber
import pandas as pd
import logging

def extract_from_pdf(path):
    """
    Extract candidate facts from PDF using pdfplumber and tabula as fallback.
    Returns: list of dicts {page, line, raw}
    """
    keywords = [
        "wafer", "substrate", "price", "cost", "defect", "defects/cm", "lead time",
        "per wafer", "per unit", "yield", "volume"
    ]
    currency_re = re.compile(r"(\$|USD)?\s?[\d,]+(\.\d+)?")
    percent_re = re.compile(r"\d+(\.\d+)?%")
    facts = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                for j, line in enumerate(text.splitlines()):
                    if any(k in line.lower() for k in keywords):
                        # Find currency/percent patterns
                        for m in currency_re.finditer(line):
                            facts.append({"page": i, "line": j+1, "raw": m.group() + " | " + line.strip()})
                        for m in percent_re.finditer(line):
                            facts.append({"page": i, "line": j+1, "raw": m.group() + " | " + line.strip()})
    except Exception as e:
        logging.error(f"PDF extraction failed: {e}")
    return facts

def extract_from_xlsx(path):
    """
    Extract candidate facts from XLSX using pandas.
    Returns: list of dicts {sheet, row, col, raw}
    """
    keywords = [
        "price", "cost", "wafer", "substrate", "defect", "yield", "volume"
    ]
    facts = []
    try:
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            for col in df.columns:
                if any(k in str(col).lower() for k in keywords):
                    for idx, val in df[col].items():
                        if pd.notnull(val) and isinstance(val, (int, float, str)):
                            facts.append({
                                "sheet": sheet,
                                "row": idx+1,
                                "col": col,
                                "raw": str(val)
                            })
    except Exception as e:
        logging.error(f"XLSX extraction failed: {e}")
    return facts

def normalize_facts(raw_facts):
    """
    Normalize raw facts into external_data dict keyed by srcID.
    Converts currency and percent strings to floats, records units.
    """
    external_data = {}
    for i, fact in enumerate(raw_facts):
        srcid = None
        key = None
        value = None
        units = ""
        confidence = 0.9
        page = fact.get("page") or fact.get("sheet") or "?"
        raw = fact.get("raw", "")
        # Try to extract key
        key_match = re.search(r"(wafer|substrate|price|cost|defect|yield|volume)", raw, re.I)
        if key_match:
            key = key_match.group(1)
        # Value extraction
        val = None
        if "$" in raw or "USD" in raw:
            # Strip common currency formatting
            clean = re.sub(r"[^0-9\.\-]", "", raw)
            try:
                value = float(clean)
                units = "USD"
            except Exception:
                # Fallback to regex extraction
                val_match = re.search(r"(\$|USD)?\s*([\d,]+(\.\d+)?)", raw)
                if val_match:
                    val = val_match.group(2).replace(",", "")
                    try:
                        value = float(val)
                        units = "USD"
                    except Exception:
                        value = val
        elif "%" in raw:
            val_match = re.search(r"([\d\.]+)%", raw)
            if val_match:
                try:
                    value = float(val_match.group(1)) / 100.0
                    units = "fraction"
                except Exception:
                    value = val_match.group(1)
        else:
            # Try to parse as float
            clean = re.sub(r"[^0-9\.\-]", "", raw)
            try:
                value = float(clean)
                units = ""
            except Exception:
                val_match = re.search(r"([\d\.]+)", raw)
                if val_match:
                    try:
                        value = float(val_match.group(1))
                        units = ""
                    except Exception:
                        value = val_match.group(1)
        if not key:
            key = "unknown"
        srcid = f"{str(page).replace(' ','_')}_{key}_{i}"
        external_data[srcid] = {
            "key": key,
            "value": value,
            "units": units,
            "raw": raw,
            "page": page,
            "confidence": confidence,
        }
    return external_data

def save_external_json(external_data, path):
    """
    Save external_data dict to JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        import json
        json.dump(external_data, f, indent=2)
