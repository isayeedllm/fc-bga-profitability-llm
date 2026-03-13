import os
import time
import json
import logging
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash, session

from llm_pipeline import run_pipeline, load_external_json

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"csv", "pdf", "xlsx", "xls", "txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "semipkg_secret")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Security warning for UI
SECURITY_WARNING = (
    "Redact confidential vendor quotes before submitting to public LLMs. "
    "For confidential data use private/enterprise LLM with proper access controls."
)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html", security_warning=SECURITY_WARNING)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        manual_mode = request.form.get("manual_mode") == "on"
        manual_llm_json = None

        # Determine whether this is a follow-up submission with LLM JSON
        existing_session = request.form.get("session_dir")
        existing_out = request.form.get("out_dir")

        if manual_mode and request.form.get("manual_llm_json") and existing_session and existing_out:
            # Second step: use already-uploaded files and provided JSON
            session_dir = existing_session
            out_dir = existing_out
            inputs_path = os.path.join(session_dir, "inputs.csv")
            sens_path = os.path.join(session_dir, "sensitivity.csv")
            ext_paths = [os.path.join(session_dir, p) for p in os.listdir(session_dir) if p not in ["inputs.csv", "sensitivity.csv"]]

            raw_json_text = request.form.get("manual_llm_json")
            try:
                manual_llm_json = json.loads(raw_json_text)
            except Exception as e:
                logging.error(f"Direct JSON parse failed: {e}")
                # Try to extract JSON substring from the pasted text
                start = raw_json_text.find("{")
                end = raw_json_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        manual_llm_json = json.loads(raw_json_text[start:end+1])
                    except Exception as e2:
                        logging.error(f"Substring JSON parse failed: {e2}, extracted: {raw_json_text[start:end+1][:200]}")
                        manual_llm_json = None
                else:
                    logging.error("No JSON braces found in text")
                    manual_llm_json = None
            if manual_llm_json is None:
                flash("Invalid JSON pasted. Please check your ChatGPT output.", "danger")

            result = run_pipeline(
                inputs_path, sens_path, ext_paths, out_dir, use_api=False, manual_llm_json=manual_llm_json
            )
        else:
            # First step: upload files and generate prompt
            inputs_csv = request.files.get("inputs_csv")
            sensitivity_csv = request.files.get("sensitivity_csv")
            external_files = request.files.getlist("external_files")

            ts = time.strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(app.config["UPLOAD_FOLDER"], f"session_{ts}")
            os.makedirs(session_dir, exist_ok=True)
            inputs_path = os.path.join(session_dir, "inputs.csv")
            sens_path = os.path.join(session_dir, "sensitivity.csv")
            inputs_csv.save(inputs_path)
            sensitivity_csv.save(sens_path)

            ext_paths = []
            for f in external_files:
                if f and allowed_file(f.filename):
                    ext_path = os.path.join(session_dir, f.filename)
                    f.save(ext_path)
                    ext_paths.append(ext_path)

            out_dir = os.path.join(app.config["OUTPUT_FOLDER"], f"session_{ts}")
            os.makedirs(out_dir, exist_ok=True)

            result = run_pipeline(
                inputs_path, sens_path, ext_paths, out_dir, use_api=not manual_mode
            )

        # Prepare outputs for UI
        llm_json = result.get("llm_json")
        checks = result.get("checks", [])
        status = result.get("status")
        outputs = result.get("outputs", {})
        base_metrics = result.get("base_metrics", {})
        chart_series = result.get("chart_series", {})
        scenario_deltas = result.get("scenario_deltas", [])
        tornado_data = result.get("tornado_data", {})
        breakeven_values = result.get("breakeven_values", {})
        prompt_path = outputs.get("prompt")
        llm_report_txt = os.path.relpath(outputs.get("llm_report_txt"), app.config["OUTPUT_FOLDER"]).replace("\\", "/") if outputs.get("llm_report_txt") else None
        llm_results_json = os.path.relpath(outputs.get("llm_results_json"), app.config["OUTPUT_FOLDER"]).replace("\\", "/") if outputs.get("llm_results_json") else None
        yield_plot = os.path.relpath(outputs.get("yield_plot"), app.config["OUTPUT_FOLDER"]).replace("\\", "/") if outputs.get("yield_plot") else None
        scenario_plot = os.path.relpath(outputs.get("scenario_plot"), app.config["OUTPUT_FOLDER"]).replace("\\", "/") if outputs.get("scenario_plot") else None
        effcost_vs_yield = os.path.relpath(outputs.get("effcost_vs_yield"), app.config["OUTPUT_FOLDER"]).replace("\\", "/") if outputs.get("effcost_vs_yield") else None
        profit_vs_price = os.path.relpath(outputs.get("profit_vs_price"), app.config["OUTPUT_FOLDER"]).replace("\\", "/") if outputs.get("profit_vs_price") else None
        tornado_plot = os.path.relpath(outputs.get("tornado"), app.config["OUTPUT_FOLDER"]).replace("\\", "/") if outputs.get("tornado") else None

        prompt_text = ""
        if status == "awaiting_manual_llm" and prompt_path and os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read()

        executive_summary = ""
        ranked_scenarios = []
        if llm_json:
            executive_summary = llm_json.get("executive_summary", "")
            ranked_scenarios = llm_json.get("ranked_scenarios") or llm_json.get("ranked_levers", [])

        # Store session info for manual continuation
        if status == "awaiting_manual_llm":
            session["manual_session_dir"] = session_dir
            session["manual_out_dir"] = out_dir

        return render_template(
            "upload.html",
            security_warning=SECURITY_WARNING,
            status=status,
            executive_summary=executive_summary,
            ranked_scenarios=ranked_scenarios,
            checks=checks,
            llm_report_txt=llm_report_txt,
            llm_results_json=llm_results_json,
            yield_plot=yield_plot,
            scenario_plot=scenario_plot,
            effcost_vs_yield=effcost_vs_yield,
            profit_vs_price=profit_vs_price,
            tornado_plot=tornado_plot,
            base_metrics=base_metrics,
            scenario_deltas=scenario_deltas,
            breakeven_values=breakeven_values,
            prompt_text=prompt_text,
            manual_mode=manual_mode,
            session_dir=session.get("manual_session_dir"),
            out_dir=session.get("manual_out_dir"),
        )
    except Exception as e:
        error_message = str(e)
        return render_template(
            "upload.html",
            security_warning=SECURITY_WARNING,
            status="error",
            error_message=error_message,
        )

@app.route("/download/<path:filename>")
def download(filename):
    # Serve from outputs folder
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
