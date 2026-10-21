"""Generate the Nepal result note and paper table from complete aggregate evidence."""
import argparse
import json
from pathlib import Path


def summarize(report="results/research_checks/nepal_sensitivity.json"):
    data = json.loads(Path(report).read_text())
    if data.get("fits") != 20 or len(data["conditions"]) != 4 or data["clinical_claim_eligible"] is not False:
        raise ValueError("Require all twenty verified, explicitly exploratory Nepal evaluations")
    for condition in data["conditions"]:
        if len(condition["verification"]) != 5 or not all(r["replay"]["passed"] for r in condition["verification"]):
            raise ValueError("Missing successful five-seed persisted replay")
    prep = data["preparation"]
    lines = ["# Nepal ICU sensitivity", "",
        "This completed experiment evaluates frozen WiDS models against a different ICU-mortality endpoint. It is an exploratory sensitivity, without an independent hospital-mortality validation claim.", "",
        f"The checksum-pinned public workbook contains {prep['source_rows']:,} daily rows and {prep['source_admissions']:,} admission identifiers. Selecting one physical row after consistency checks leaves {prep['eligible_admissions']:,} recorded adults with {prep['deaths']:,} ICU deaths. {prep['missing_invalid_or_nonadult_age']:,} identifiers fail the recorded adult-age criterion; no unknown outcome is converted to survival. The selected admission and outcome fields have zero within-identifier conflicts.", "",
        "Two scenarios were fixed before model scoring: demographics uses age, sex and elective surgery; admission core adds first heart rate and systolic pressure. The latter substitute first measurements for WiDS day-one extrema, so the windows are not equivalent. BMI, temperature, lactate, BUN, creatinine, glucose, WBC, SpO2 and platelets remain unavailable. No APACHE score is supplied. Frozen fitted preprocessing fills unsupported inputs; observed-only rules never treat these imputed values as measured evidence.", "",
        "The publisher-linked CCAA dictionary specifies age in years, heart rate in beats/minute and systolic pressure in mmHg. Its Fahrenheit temperature and micromol/L creatinine definitions conflict with large parts of the released numeric scales, and the workbook has no per-record unit flags. WBC scales are also unresolved. These fields were excluded rather than converted using outcome performance or an inferred patient-level unit. The source-level scale audit and analysis amendment are retained.", "",
        "## Results", "",
        "Each condition reuses five frozen no-APACHE checkpoints and 50 matched dropout passes, without training or recalibration. Seed means reuse the same admissions; they are not additional participants.", "",
        "| Scenario | Imputer | Method | AUROC mean (seed SD) | Brier mean (seed SD) | Log loss |",
        "|---|---|---|---:|---:|---:|"]
    for c in data["conditions"]:
        for m in c["metrics"]:
            if m["variant"] == "flat_confidence":
                continue
            lines.append(f"| {c['scenario']} | {c['imputation']} | {m['variant']} | {m['mean']['auc']:.6f} ({m['seed_sd']['auc']:.6f}) | {m['mean']['brier']:.6f} ({m['seed_sd']['brier']:.6f}) | {m['mean']['log_loss']:.6f} |")
    lines += ["", "| Scenario | Imputer | NARS-minus-MC Brier | Family lower | Family upper |",
              "|---|---|---:|---:|---:|"]
    tex = [r"\begin{table}[t]", r"\centering\scriptsize",
        r"\caption{Exploratory Nepal ICU-outcome sensitivity of frozen WiDS hospital-mortality models. Five-seed mean ungated AUROC and Brier, followed by NARS-minus-MC Brier with four-comparison Bonferroni intervals resampling admissions and fitted seeds. Units with unresolved export scales are excluded. Changed outcomes and measurement summaries prevent an external compatibility claim.}",
        r"\label{tab:nepal}", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
        r"Inputs & Imputer & AUROC & Brier & Difference & Family lower & Family upper \\", r"\midrule"]
    for c in data["conditions"]:
        contrast = c["nars_minus_mc"]
        baseline = next(m["mean"] for m in c["metrics"] if m["variant"] == "baseline")
        lines.append(f"| {c['scenario']} | {c['imputation']} | {contrast['brier']:.3e} | {contrast['family_lower_95']:.3e} | {contrast['family_upper_95']:.3e} |")
        label = "Demographic" if c["scenario"] == "demographics" else "Admission core"
        tex.append(f"{label} & {c['imputation']} & {baseline['auc']:.3f} & {baseline['brier']:.3f} & {contrast['brier']:.2e} & {contrast['family_lower_95']:.2e} & {contrast['family_upper_95']:.2e} "+r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    span_zero = all(c["nars_minus_mc"]["family_lower_95"] <= 0 <= c["nars_minus_mc"]["family_upper_95"] for c in data["conditions"])
    if all(c["nars_minus_mc"]["family_upper_95"] < 0 for c in data["conditions"]):
        effects = [c["nars_minus_mc"]["brier"] for c in data["conditions"]]
        finding = f"All four family intervals favor small Brier reductions, with point estimates from {min(effects):.2e} to {max(effects):.2e}. These conditional numerical effects are far below the 0.0001 benefit margin and do not establish clinical benefit or hospital-mortality compatibility."
    elif span_zero:
        finding = "All four family intervals include zero."
    else:
        finding = "Read each contrast in its changed-endpoint exploratory scope; no compatibility acceptance test is applied."
    lines += ["", finding,
        "The 0.0001 symbolic-benefit criterion remains unestablished by the original studies; these additional inspected outcomes do not replace those negative findings. Undefined precision values remain null when no case is predicted positive, rather than being replaced by zero.", "",
        "## Verification and limitations", "",
        "All twenty saved prediction/event bundles pass independent replay and uniform-gate identity. Discrimination, calibration and threshold metrics were separately recomputed from the saved probabilities. Preprocessing cache identities cover the fitted processor, full mapped data and implementation; cached contents are checked on reuse. A duplicate-row optimization failed a real-source equivalence check and was removed; its affected outputs remain in an ignored rejection folder and are excluded. Whole-frame caching preserves the original KNN arithmetic and reduces repeated computation across identical preprocessors.", "",
        "Paired Brier intervals resample admissions and fitted seeds, with Bonferroni adjustment across the four fixed conditions and a fixed inference draw. Per-fit AUROC contrasts have conditional admission intervals. The export has no hospital identifiers or canonical repeat-person linkage, so these intervals do not estimate institution-level dependence, repeat-person dependence or refitting variability beyond the five fitted seeds. Neither source independence nor the first-day hospital-death endpoint equivalence is established. Plan experiment 7 remains unsatisfied.", "",
        "Fixed age/sex strata and decision curves at the existing NGTA research thresholds are also exported by `scripts/nepal_diagnostics.py`. The 400 subgroup metric rows remain private; the public [diagnostic summary](../results/research_checks/nepal_diagnostics.json) includes seed-mean whole-cohort decision curves. These are additional descriptive diagnostics, not clinical threshold validation or a separately registered subgroup analysis.", "", "## Reproduce", "", "```powershell", "python scripts/prepare_nepal.py", "python scripts/evaluate_nepal.py", "python scripts/summarize_nepal.py", "python scripts/nepal_diagnostics.py", "```", "",
        "Place the publisher workbook under `data/private/open_candidates/nepal_registry/`. Original fitted WiDS artifacts are required. Prepared cohorts, preprocessing caches, pass arrays, event logs and rejected outputs stay under ignored private result directories; only aggregate evidence is committed. The pinned protocol is [nepal-analysis.json](nepal-analysis.json), and the aggregate evidence is [nepal_sensitivity.json](../results/research_checks/nepal_sensitivity.json).", "",
        "Sources: [Nepal release](https://doi.org/10.6084/m9.figshare.22263847.v1), [study methods](https://wellcomeopenresearch.org/articles/8-180), [CCAA core dictionary](https://doi.org/10.6084/m9.figshare.15167406.v3)."]
    Path("docs/nepal-results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    Path("paper/figures/nepal_results.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="results/research_checks/nepal_sensitivity.json")
    summarize(**vars(parser.parse_args()))
