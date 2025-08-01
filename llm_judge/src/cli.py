from __future__ import annotations

"""Command‑line interface: run the Judge over a CSV and print metrics."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .io import load_table, save_table
from .judge import Judge
from .evaluation import precision_recall_f1, metrics_report

REPORTS_DIR = Path("reports")


def _write_report(
    path: Path, metrics: dict[str, float], counts: Counter[str]
) -> None:
    """Write a minimal Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    lines: list[str] = [
        f"# Evaluation Report – {timestamp} UTC",
        "",
        "## Summary counts",
        "| Label | Count |",
        "|-------|-------|",
    ] + [f"| {label} | {cnt} |" for label, cnt in counts.items()] + [
        "",
        "## Macro metrics",
    ] + [f"- **{k.capitalize()}**: {v:.4f}" for k, v in metrics.items()]

    path.write_text("\n".join(lines), encoding="utf-8")


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard format."""
    column_mapping = {
        'Current User Question': 'question',
        'Assistant Answer': 'answer',
        'Fragment Texts': 'fragments',
        'Conversation History': 'conversation_history'
    }
    
    df = df.rename(columns=column_mapping)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM‑as‑a‑Judge runner")
    parser.add_argument(
        "--in", dest="input_path", required=True, help="Input CSV path"
    )
    parser.add_argument(
        "--out", dest="output_path", default=None, help="Output CSV path"
    )
    parser.add_argument(
        "--model", dest="model", default="mistral-small-latest",
        help="Mistral model name"
    )
    args = parser.parse_args()

    # 1. Load and normalize column names
    df = load_table(args.input_path)
    df = _normalize_column_names(df)
    gold = df["Label"].tolist() if "Label" in df.columns else None

    judge = Judge(model=args.model)

    preds: list[str] = []
    cots: list[str] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
        # Convert pandas Series to dict for Judge compatibility
        row_dict = {str(k): str(v) for k, v in row.items()}
        res = judge.evaluate_row(row_dict)
        preds.append(res["label"])
        cots.append(res["chain_of_thought"])

    df["Predicted_Label"] = preds
    df["Predicted_CoT"] = cots

    # Output path determination
    out_path = (
        args.output_path
        or str(Path(args.input_path).with_suffix(".judged.csv"))
    )
    save_table(df, out_path)
    print(f"✅ Judged CSV saved to {out_path}")

    # 4. Metrics + markdown report
    if gold is not None:
        metrics = precision_recall_f1(gold, preds, average="macro")
        print(metrics_report(metrics, title="Macro metrics"))
        counts = Counter(preds)
        report_path = REPORTS_DIR / (Path(args.input_path).stem + "_report.md")
        _write_report(report_path, metrics, counts)
        print(f"📄 Markdown report saved to {report_path}")


if __name__ == "__main__":
    main()
