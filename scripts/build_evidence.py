"""Render observed run evidence as a self-contained HTML review page.

This is machine-generated technical evidence, not the student-authored report.
"""
import base64
import html
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT

def build_evidence(root=ROOT):
    root = Path(root)
    report = root / "reports"
    audit = json.loads((report / "quality_audit.json").read_text())
    evaluated = json.loads((report / "test_metrics.json").read_text())
    model = json.loads((root / "models/metadata.json").read_text())
    result, baseline = evaluated["selected"], evaluated["baseline"]
    if audit["status"] != "REAL_DATA_PREPARED" or evaluated["status"] != "TRAINED_REAL_DATA":
        raise ValueError("Evidence page requires actual real-data artifacts")
    sections = []
    def paragraph(text):
        sections.append("<p>" + html.escape(text) + "</p>")
    def table(title, frame):
        sections.append("<h2>" + html.escape(title) + "</h2>" + frame.to_html(index=False, border=0, escape=True, float_format=lambda x: f"{x:,.4f}"))
    sections.append("<h1>Historical flight disruption analysis</h1>")
    paragraph("Machine-generated technical evidence from the uploaded original dataset. Use it to verify the code and support your own writing; it is not a completed group or individual assessment.")
    table("Verified data volume", pd.DataFrame([
        {"Measure": "Original flight records", "Observed value": f"{audit['counts']['raw_rows']:,}"},
        {"Measure": "Original columns", "Observed value": audit["raw_columns"]},
        *[{"Measure": f"{k.title()} partition", "Observed value": f"{v:,}"} for k, v in audit["split_counts"].items()],
        {"Measure": "Exact duplicates removed", "Observed value": audit["counts"]["exact_duplicates_removed"]},
        {"Measure": "Quarantined records", "Observed value": audit["counts"]["quarantined_rows"]}
    ]))
    percentage_metrics = {"prevalence", "precision", "recall", "f1", "alert_rate", "precision_at_budget", "recall_at_budget"}
    def display_metric(key, value):
        if key == "n":
            return f"{int(value):,}"
        if key in percentage_metrics:
            return f"{value:.2%}"
        return f"{value:.4f}"
    table("Final held-out evaluation", pd.DataFrame([
        {"Measure": key.replace("_", " "), "Selected model": display_metric(key, result[key]), "Prevalence baseline": display_metric(key, baseline[key])}
        for key in ["n", "prevalence", "average_precision", "roc_auc", "brier_score", "precision", "recall", "f1",
                    "alert_rate", "precision_at_budget", "recall_at_budget", "lift_at_budget", "scenario_cost_per_1000"]
    ]))
    paragraph(f"Selected family: {model['winner']}. Feature variant: {'engineered' if model['engineered_features'] else 'base'}. Threshold: {model['threshold']:.6f}. Test period: {evaluated['test_date_min']} to {evaluated['test_date_max']}.")
    paragraph("The threshold was selected from validation scores before testing. Precision and recall at the review budget describe a ranked batch of flights; threshold-based metrics describe individual alerts. Cost values are illustrative relative units, not measured money saved.")
    comparison = pd.read_csv(report / "model_comparison.csv")
    table("Model selection evidence", comparison[["model", "cv_ap_before", "cv_ap_after", "cv_ap_std_after", "validation_average_precision", "validation_brier_score"]])
    paragraph("Model-comparison rows use the engineered feature set. The final model then uses the better feature variant from the fixed-parameter chronological ablation; final validation metrics are stored separately.")
    table("Temporal validation folds", pd.read_json(report / "cv_folds.json"))
    table("Before and after missingness", pd.read_csv(report / "missingness_before_after.csv")[["column", "raw_missing", "raw_missing_pct", "clean_missing", "clean_missing_pct"]])
    paragraph("Structural missingness is retained. Numerical predictor imputation occurs within each training fold. Do not interpret all remaining missing cells as cleaning failures.")
    imputation_file = report / "model_input_imputation.csv"
    if imputation_file.exists():
        table("Verified model input imputation", pd.read_csv(imputation_file))
    table("Observed nonzero quality checks", pd.DataFrame([{"Check": k, "Count": v} for k, v in audit["counts"].items() if v and not k.startswith("raw_missing__")]))
    table("Validation period robustness", pd.read_csv(report / "validation_subgroup_metrics.csv")[["group_field", "group", "n", "prevalence", "average_precision", "precision_at_budget", "lift_at_budget"]])
    table("Held-out subgroup checks", pd.read_csv(report / "test_subgroup_metrics.csv")[["group_field", "group", "n", "prevalence", "average_precision", "precision", "recall"]])
    table("Feature ablation", pd.DataFrame([json.loads((report / "feature_ablation.json").read_text())]).drop(columns=["interpretation"]))
    for figure in json.loads((report / "figure_captions.json").read_text()):
        data = base64.b64encode((report / "figures" / figure["filename"]).read_bytes()).decode()
        sections.append('<figure><img alt="' + html.escape(figure["filename"]) + '" src="data:image/png;base64,' + data + '"><figcaption>' + html.escape(figure["caption"]) + '</figcaption></figure>')
    sections.append("<h2>Completion limits</h2>")
    paragraph("Public GitHub repository, Streamlit runtime/deployment, instructor approvals, genuine contribution history and student-authored reports remain separate requirements. This evidence page does not establish completion of the separately supplied RapidMiner rubric.")
    paragraph("Unmapped numeric airport IDs remain explicitly namespaced, with validation-period robustness reported. The data is from 2015 and is not validation for current airline operations.")
    body = "\n".join(sections)
    document = '<!DOCTYPE html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Flight disruption technical evidence</title><style>body{max-width:1150px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui,sans-serif;color:#17252f}h1{font-size:32px}h2{margin-top:40px;font-size:23px}p{max-width:95ch}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0;display:block;overflow:auto}th,td{border:1px solid #d9e0e5;padding:9px 11px;text-align:left}th{background:#203e53;color:white}tr:nth-child(even){background:#f4f7f9}figure{margin:38px 0}img{display:block;max-width:100%;height:auto}figcaption{color:#45545f;font-size:14px;margin-top:10px}@media print{body{margin:0}figure{break-inside:avoid}table{font-size:10px}}</style><body>' + body + '</body></html>'
    path = report / "results_evidence.html"
    path.write_text(document)
    print(path)
    return path

if __name__ == "__main__":
    build_evidence()
