"""Blinded, counterbalanced reviewer materials and response analysis.

All supplied tasks are synthetic. Investigator truth files must be kept away
from participants; this module never invents participant responses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


FAULTS = ("extraction", "imputation", "rule_version", "none")


def reviewer_power(reviewers=24, cases=80, simulations=2000, seed=0):
    """Working power sensitivity for crossed reviewer/case correlations.

    Cluster-robust normal intervals approximate the planned response bootstrap.
    This is a design aid, not a replacement for a statistician's protocol review.
    """
    if reviewers < 4 or cases < 8 or simulations < 1:
        raise ValueError("Power design needs at least four reviewers and eight cases")
    rng = np.random.default_rng(seed)
    rows = []
    treatment = (np.arange(reviewers)[:, None] + rng.permutation(cases)[None, :]) % 2
    for reviewer_sd, case_sd in ((.25, .25), (.5, .5), (1., 1.)):
        detected, accepted = 0, 0
        for _ in range(simulations):
            offsets = rng.normal(0, reviewer_sd, (reviewers, 1)) + rng.normal(0, case_sd, (1, cases))
            baseline = expit(.4 + offsets)
            # A fixed 10 percentage-point marginal improvement is the alternative.
            p = np.minimum(baseline + treatment * .1, 1)
            response = (rng.random((reviewers, cases)) < p).astype(float)
            means = [response[treatment == arm].mean() for arm in (0, 1)]
            delta = means[1] - means[0]
            influence = np.where(treatment == 1, (response-means[1])/(treatment == 1).sum(),
                                 -(response-means[0])/(treatment == 0).sum())
            variance = (reviewers/(reviewers-1) * np.square(influence.sum(1)).sum()
                        + cases/(cases-1) * np.square(influence.sum(0)).sum()
                        - np.square(influence).sum())
            detected += delta - norm.ppf(.975) * np.sqrt(max(variance, 0)) > 0
            accepted += delta >= .1 and delta - norm.ppf(.975) * np.sqrt(max(variance, 0)) > 0
        rows.append(dict(reviewer_logit_sd=reviewer_sd, case_logit_sd=case_sd,
                         superiority_power=detected/simulations,
                         superiority_and_observed_10pp_power=accepted/simulations))
    return dict(reviewers=reviewers, evaluation_cases=cases, simulations=simulations, seed=seed,
                baseline="logistic intercept .4 plus crossed random intercepts",
                alternative="10 percentage-point probability improvement, capped at 1",
                method="crossed-cluster normal approximation; confirm design before recruitment",
                sensitivity=rows)


def create_reviewer_package(output_dir, reviewers=24, cases=80, seed=0):
    if reviewers < 4 or reviewers % 2 or cases < 8 or cases % 2:
        raise ValueError("Use even reviewer/case counts for balanced crossover assignments")
    root = Path(output_dir)
    public, private = root / "participant", root / "investigator"
    if root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to overwrite an existing blinded study package")
    public.mkdir(parents=True)
    private.mkdir()
    rng = np.random.default_rng(seed)
    arm_names = dict(zip(("structured", "trace"), rng.permutation(["format_a", "format_b"])))
    order = rng.permutation(cases)
    truth, tasks = [], []
    for index in range(cases + 4):
        case_id = f"practice_{index}" if index >= cases else f"case_{index:03d}"
        fault = FAULTS[index % len(FAULTS)]
        raw = float(rng.uniform(2, 6))
        extracted = raw
        imputed = False
        version = "prototype-1"
        if fault == "extraction":
            raw, extracted = 3., 5.
        elif fault == "imputation":
            raw, extracted, imputed = None, 5., True
        elif fault == "rule_version":
            version = "unapproved-2"
        triggered = extracted >= 4
        base = dict(case_id=case_id, raw_lactate=raw, extracted_lactate=extracted, imputed=imputed,
                    rule="lactate >= 4 mmol/l, observed inputs only", rule_version=version,
                    approved_version="prototype-1", trigger=triggered)
        for arm in ("structured", "trace"):
            task = dict(base, format=arm_names[arm])
            if arm == "trace":
                neural_c, symbolic_c = .4, .8 if triggered else 0
                weight = neural_c/(1-neural_c) + symbolic_c/max(1-symbolic_c, 1e-12)
                task.update(neural_frequency=.5, neural_confidence=neural_c,
                            symbolic_frequency=.8 if triggered else 0, symbolic_confidence=symbolic_c,
                            revised_frequency=(.5*neural_c/(1-neural_c)+(.8*symbolic_c/max(1-symbolic_c,1e-12)))/weight,
                            revised_confidence=weight/(weight+1), rule_source="synthetic prototype for reviewer exercise")
            tasks.append(task)
        truth.append(dict(case_id=case_id, fault=fault, practice=index >= cases,
                          correction={"extraction": "use the raw observed value", "imputation": "suppress the unobserved trigger",
                                      "rule_version": "restore the approved rule version", "none": "no correction needed"}[fault]))
    assignments = []
    for participant in range(reviewers):
        sequence = order if participant % 2 == 0 else order[::-1]
        for position, case in enumerate(sequence):
            arm = "structured" if (case + participant) % 2 == 0 else "trace"
            assignments.append(dict(participant_id=f"reviewer_{participant:02d}", case_id=f"case_{case:03d}",
                                    period=1 if position < cases//2 else 2, order=position+1, format=arm_names[arm]))
    assignments = pd.DataFrame(assignments)
    assignments.to_csv(public / "assignments.csv", index=False)
    responses = assignments.assign(localized_fault="", correction_correct="", completion_seconds="", reassured="")
    responses.to_csv(public / "responses_template.csv", index=False)
    (public / "tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    (private / "truth.json").write_text(json.dumps(dict(arm_names=arm_names, truth=truth), indent=2), encoding="utf-8")
    protocol = dict(seed=seed, materials="synthetic ICU-style records, not patient data",
        practice_cases=4, evaluation_cases=cases, reviewers=reviewers,
        design="each reviewer sees each evaluation case once, half in each format; every case balanced across formats",
        outcomes=["correct localization", "correction accuracy", "completion time", "false reassurance"],
        primary_analysis="crossed reviewer/case bootstrap of balanced format differences",
        mixed_effects_sensitivity="binomial crossed random intercepts for reviewer and case, adjusted for period; variational Bayes credible intervals",
        acceptance="localization gain >= .10 with positive 95% bootstrap lower bound and false-reassurance upper delta <= .05",
        blinding="format identifiers conceal naming, but format richness is visible; never give participants the investigator folder",
        status="prepared only; requires protocol approval, qualified participants and real responses")
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (root / "power.json").write_text(json.dumps(reviewer_power(reviewers, cases, seed=seed), indent=2), encoding="utf-8")
    return protocol


def analyze_reviewer_responses(package_dir, responses_csv, iterations=2000, seed=0):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    root = Path(package_dir)
    key = json.loads((root / "investigator" / "truth.json").read_text())
    assigned = pd.read_csv(root / "participant" / "assignments.csv")
    responses = pd.read_csv(responses_csv)
    identity = ["participant_id", "case_id"]
    if responses.duplicated(identity).any() or len(responses) != len(assigned):
        raise ValueError("Provide exactly one response per assigned participant/case")
    frame = assigned.merge(responses, on=identity, suffixes=("", "_response"), validate="one_to_one")
    if len(frame) != len(assigned):
        raise ValueError("Responses do not match the randomized assignments")
    truth = {row["case_id"]: row["fault"] for row in key["truth"] if not row["practice"]}
    frame["localized"] = (frame.localized_fault == frame.case_id.map(truth)).astype(int)
    frame["trace"] = (frame["format"] == key["arm_names"]["trace"]).astype(int)
    for field in ("correction_correct", "reassured"):
        if not frame[field].isin([0, 1]).all():
            raise ValueError(f"Complete binary {field} responses are required")
    if not frame.localized_fault.isin(FAULTS).all() or not np.isfinite(frame.completion_seconds).all() or (frame.completion_seconds <= 0).any():
        raise ValueError("Responses require valid fault categories and positive finite completion times")
    frame["false_reassurance"] = frame.reassured * frame.case_id.map(truth).ne("none")
    rng = np.random.default_rng(seed)
    reviewers, reviewer_index = np.unique(frame.participant_id, return_inverse=True)
    cases, case_index = np.unique(frame.case_id, return_inverse=True)
    contrasts = {name: [] for name in ("localized", "correction_correct", "completion_seconds", "false_reassurance")}
    treatment = frame.trace.to_numpy()
    for _ in range(iterations):
        r = np.bincount(rng.integers(0, len(reviewers), len(reviewers)), minlength=len(reviewers))
        c = np.bincount(rng.integers(0, len(cases), len(cases)), minlength=len(cases))
        weight = r[reviewer_index] * c[case_index]
        if any(weight[treatment == arm].sum() == 0 for arm in (0, 1)):
            continue
        for name in contrasts:
            values = frame[name].to_numpy()
            contrasts[name].append(float(np.average(values[treatment == 1], weights=weight[treatment == 1])
                                        - np.average(values[treatment == 0], weights=weight[treatment == 0])))
    report = dict(participants=len(reviewers), cases=len(cases), response_count=len(frame),
                  interpretation="human results only; requires completed protocol and recruitment records", outcomes={})
    for name, values in contrasts.items():
        observed = frame.loc[frame.trace.eq(1), name].mean() - frame.loc[frame.trace.eq(0), name].mean()
        report["outcomes"][name] = dict(trace_minus_structured=float(observed),
            lower_95=float(np.percentile(values, 2.5)), upper_95=float(np.percentile(values, 97.5)))
    if frame.localized.nunique() == 2:
        np.random.seed(seed)
        model = BinomialBayesMixedGLM.from_formula("localized ~ trace + C(period)",
            {"reviewer": "0 + C(participant_id)", "case": "0 + C(case_id)"}, frame)
        fit = model.fit_vb()
        position = model.exog_names.index("trace")
        mean, sd = float(fit.fe_mean[position]), float(fit.fe_sd[position])
        report["mixed_effects_sensitivity"] = dict(method="variational Bayes crossed logistic random intercepts",
            trace_log_odds=mean, approximate_credible_lower_95=mean-1.96*sd,
            approximate_credible_upper_95=mean+1.96*sd, optimizer_success=bool(fit.optim_retvals["success"]),
            note="credible intervals are not frequentist confidence intervals or the primary acceptance test")
    else:
        report["mixed_effects_sensitivity"] = {"status": "not estimable: no localization outcome variation"}
    gain, harm = report["outcomes"]["localized"], report["outcomes"]["false_reassurance"]
    report["acceptance_criteria_met"] = gain["trace_minus_structured"] >= .1 and gain["lower_95"] > 0 and harm["upper_95"] <= .05
    (root / "response_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
