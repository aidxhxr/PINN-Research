"""Metric definitions used by the result registry.

These functions read saved numerical results. They do not import training
entrypoints, solve ODEs, fit networks, or sample a posterior.
"""

import json
from pathlib import Path

import numpy as np

REGIMES = ("Normal", "Early Adenoma", "Advanced Adenoma", "Severe APC Loss")


def read_json(path):
    return json.loads(Path(path).read_text())


def relative_errors(recovered, truth, parameters=None):
    """Absolute relative errors; undefined zero-truth scores are rejected."""
    names = list(truth) if parameters is None else list(parameters)
    if not names:
        raise ValueError("The comparison has no parameters")
    errors = {}
    for name in names:
        value, estimate = float(truth[name]), float(recovered[name])
        if value == 0 or not np.isfinite([value, estimate]).all():
            raise ValueError(f"Invalid relative error for {name}: truth={value}, estimate={estimate}")
        errors[name] = abs(estimate - value) / abs(value)
    return errors


def recovery_count(recovered, truth, parameters=None):
    """Count errors strictly below 10%, matching the historical analyses."""
    errors = relative_errors(recovered, truth, parameters)
    return sum(value < 0.1 for value in errors.values()), len(errors)


def common_recovery(control, hybrid):
    """Compare both fits on the same remaining parameters and true values."""
    common = sorted(set(control["recovered"]) & set(hybrid["recovered"]))
    if any(control["true"][key] != hybrid["true"][key] for key in common):
        raise ValueError("Control and hybrid disagree about parameter truth")
    c, denominator = recovery_count(control["recovered"], control["true"], common)
    h, _ = recovery_count(hybrid["recovered"], hybrid["true"], common)
    return {"control": c, "hybrid": h, "denominator": denominator, "parameters": common}


def forward_accuracy(inputs):
    rows, output = [], {}
    for method, labels in (("dense", 100), ("sparse", 40)):
        data = read_json(inputs[method])
        values = np.asarray([data["per_regime"][r]["rel_l2"] for r in REGIMES])
        if values.shape != (4, 7) or not np.isfinite(values).all():
            raise ValueError("Forward accuracy must contain four regimes and seven states")
        mean = float(values.mean())
        np.testing.assert_allclose(mean, data["grand_mean_rel_l2"], rtol=1e-12)
        output[method] = {"mean_relative_l2_percent": 100 * mean, "labels": labels,
                          "state_regime_pairs": int(values.size)}
        rows.append([method, labels, 100 * mean, int(values.size)])
    return {"metrics": output, "columns": ["method", "labels", "relative L2 (%)", "pairs"], "rows": rows}


def inverse_recovery(inputs):
    data = read_json(inputs["recovery"])
    output, rows = {}, []
    for method, records in data.items():
        counts, denominator = [], 0
        for regime in REGIMES:
            record = records[regime]
            count, n = recovery_count(record["recovered"], record["true"])
            if count != record["under10"] or n != 36:
                raise ValueError(f"Saved recovery count disagrees: {method}/{regime}")
            counts.append(count)
            denominator += n
        output[method] = {"counts": counts, "total": sum(counts), "denominator": denominator}
        rows.append([method, *counts, sum(counts), denominator])
    return {"metrics": output, "columns": ["method", *REGIMES, "total", "denominator"], "rows": rows}


def hybrid_comparison(inputs):
    data = read_json(inputs["comparison"])
    output, rows = {}, []
    for variant, records in data.items():
        control = hybrid = denominator = 0
        errors, per_regime = [], {}
        for record in records:
            result = common_recovery(record["control"], record["hybrid"])
            if result["parameters"] != record["common_parameters"]:
                raise ValueError("Saved common-parameter set disagrees with the actual intersection")
            control += result["control"]
            hybrid += result["hybrid"]
            denominator += result["denominator"]
            term = record["term"]
            truth, learned = np.asarray(term["truth"]), np.asarray(term["learned"])
            rmse = float(np.sqrt(np.mean((learned - truth) ** 2)))
            nrmse = rmse / float(np.sqrt(np.mean(truth ** 2)))
            np.testing.assert_allclose([rmse, nrmse], [term["rmse"], term["nrmse"]], rtol=1e-10)
            errors.append(100 * nrmse)
            per_regime[record["regime"]] = {**result, "function_nrmse_percent": 100 * nrmse}
        output[variant] = {"control": control, "hybrid": hybrid, "denominator": denominator,
                           "function_nrmse_percent": errors, "per_regime": per_regime}
        rows.append([variant, control, hybrid, denominator, min(errors), max(errors)])
    return {"metrics": output, "columns": ["term", "control", "hybrid", "denominator",
                                             "min function NRMSE (%)", "max function NRMSE (%)"], "rows": rows}


def fisher_classification(matrix):
    """Reproduce the saved local eigenvector-participation classification."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Fisher matrix must be square")
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-10, atol=1e-8)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues.min() < -1e-8 * max(float(eigenvalues.max()), 1.0):
        raise ValueError("Fisher matrix has a materially negative eigenvalue")
    eigenvalues = np.clip(eigenvalues, 0, None)
    null = eigenvalues < 1.0
    soft = (eigenvalues < eigenvalues[-1] * 1e-4) & ~null
    participation_null = np.sqrt(np.sum(eigenvectors[:, null] ** 2, axis=1))
    participation_soft = np.sqrt(np.sum(eigenvectors[:, soft] ** 2, axis=1))
    verdict = np.where(participation_null > 0.5, "nonidentifiable",
                       np.where((participation_null > 0.2) | (participation_soft > 0.5),
                                "weak", "identifiable"))
    return {"identifiable": int(np.sum(verdict == "identifiable")),
            "weak": int(np.sum(verdict == "weak")),
            "nonidentifiable": int(np.sum(verdict == "nonidentifiable")),
            "near_null_directions": int(null.sum()), "parameters": matrix.shape[0],
            "eigenvalues": eigenvalues.tolist()}


def fisher_information(inputs):
    output, rows = {"full": {}, "reduced": {}}, []
    summaries = read_json(inputs["full_summary"])
    reduced_summaries = read_json(inputs["reduced_summary"])["summaries"]
    with np.load(inputs["full_matrices"], allow_pickle=False) as data:
        full = dict(zip(map(str, data["regimes"]), data["matrices"]))
    for regime, summary, reduced_summary in zip(REGIMES, summaries, reduced_summaries):
        for name, matrix, saved in [("full", full[regime], summary),
                                     ("reduced", np.load(inputs["reduced_" + regime]), reduced_summary)]:
            record = fisher_classification(matrix)
            for key, old in (("identifiable", "n_ident"), ("weak", "n_weak"),
                             ("nonidentifiable", "n_nonident"), ("near_null_directions", "n_null_directions")):
                if record[key] != saved[old]:
                    raise ValueError(f"Fisher classification differs: {name}/{regime}/{key}")
            if name == "reduced":
                record["condition_number"] = record["eigenvalues"][-1] / record["eigenvalues"][0]
                np.testing.assert_allclose(record["condition_number"], saved["cond_number"], rtol=1e-8)
            output[name][regime] = record
            rows.append([name, regime, record["identifiable"], record["weak"],
                         record["nonidentifiable"], record["near_null_directions"]])
    return {"metrics": output, "columns": ["model", "regime", "identifiable", "weak", "nonidentifiable",
                                             "near-null eigen-directions"], "rows": rows}


def anchor_interventions(inputs):
    output, rows = {}, []
    for key in ("wnt", "bcat", "prospective", "counterexample"):
        records = read_json(inputs[key])
        selected = [r for r in records if r.get("dose", 0) == 0]
        output[key] = {}
        for record in selected:
            basal = record["basal_param"]
            err = record["hybrid_param_err"][basal]
            np.testing.assert_allclose(err, record["hybrid_basal_err"], rtol=1e-12)
            label = record["regime"] + "/" + record.get("protocol", "dose=0")
            output[key][label] = {"parameter": basal, "basal_error_percent": 100 * err,
                                  "function_nrmse_percent": 100 * record["term_nrmse"],
                                  "n_conditions": record.get("n_conditions", 11)}
            rows.append([key, record["regime"], record.get("protocol", "dose=0"), basal,
                         100 * err, 100 * record["term_nrmse"]])
    return {"metrics": output, "columns": ["design", "regime", "protocol", "basal parameter",
                                             "basal error (%)", "function NRMSE (%)"], "rows": rows}


def historical_ess(values):
    """Historical single-chain ESS estimator, retained to audit its saved gates.

    Stops summing the biased autocorrelation at the first lag below 0.05.
    This is not a replacement for modern multi-chain sampling diagnostics.
    """
    values = np.asarray(values, dtype=float)
    centered, n = values - values.mean(), len(values)
    variance = np.var(centered)
    if variance <= 0:
        return float(n)
    acf = np.correlate(centered, centered, mode="full")[n - 1:] / (variance * n)
    total = 1.0
    for value in acf[1:]:
        if value < 0.05:
            break
        total += 2 * value
    return float(n / max(total, 1.0))


def hmc_diagnostics(inputs):
    output, rows = {}, []
    for label in ("presentation_normal", "presentation_severe", "hybrid_normal"):
        saved = read_json(inputs[label + "_summary"])
        with np.load(inputs[label + "_samples"], allow_pickle=False) as data:
            values, raw, truth = data["values"], data["raw"], data["true"]
            names = list(map(str, data["names"]))
        intervals = np.percentile(values, [2.5, 97.5], axis=0)
        coverage = int(np.sum((intervals[0] <= truth) & (truth <= intervals[1])))
        ess = [historical_ess(raw[:, index]) for index in range(raw.shape[1])]
        means = values.mean(axis=0)
        for index, name in enumerate(names):
            np.testing.assert_allclose([means[index], ess[index]],
                                       [saved["params"][name]["post_mean"], saved["params"][name]["ess"]],
                                       rtol=1e-9)
        record = {"draws": len(values), "parameters": len(names), "covered_truth": coverage,
                  "median_ess": float(np.median(ess)), "minimum_ess": min(ess),
                  "ess_gate": bool(np.median(ess) >= 200), "coverage_gate": bool(coverage >= .9 * len(names))}
        record["gate_pass"] = record["ess_gate"] and record["coverage_gate"]
        if coverage != saved["meta"]["covers_true"] or record["gate_pass"] != saved["meta"]["gate_pass"]:
            raise ValueError(f"Posterior diagnostics disagree for {label}")
        output[label] = record
        rows.append([label, coverage, len(names), record["median_ess"], record["minimum_ess"], record["gate_pass"]])
    return {"metrics": output, "columns": ["run/regime", "covered truth", "parameters", "median ESS",
                                             "minimum ESS", "passes historical gates"], "rows": rows}


def bayesian_forward(inputs):
    output, rows = {}, []
    for regime in REGIMES:
        saved = read_json(inputs[regime + "_summary"])
        with np.load(inputs[regime + "_predictive"], allow_pickle=False) as data:
            coverage = np.mean((data["lo"] <= data["ref"]) & (data["ref"] <= data["hi"]), axis=0)
            rmse = np.sqrt(np.mean((data["mean"] - data["ref"]) ** 2, axis=0))
            for index, name in enumerate(map(str, data["var_names"])):
                np.testing.assert_allclose([coverage[index], rmse[index]],
                                           [saved["per_state"][name]["coverage95_truth"],
                                            saved["per_state"][name]["rmse"]], rtol=1e-10)
        output[regime] = {"truth_coverage": coverage.tolist(), "rmse": rmse.tolist(),
                          "recorded_energy_ess": saved["ess_U"],
                          "recorded_beta_prediction_ess": saved["ess_pred_beta"]}
        rows.append([regime, float(coverage.mean()), float(coverage[0]), saved["ess_U"], saved["ess_pred_beta"]])
    return {"metrics": output, "columns": ["regime", "mean truth coverage", "beta truth coverage",
                                             "saved energy ESS", "saved beta prediction ESS"], "rows": rows}


def normal_treatment(inputs):
    with np.load(inputs["trajectories"], allow_pickle=False) as data:
        reference, prediction = data["reference"], data["pinn"]
        errors = np.linalg.norm(prediction - reference, axis=1) / np.linalg.norm(reference, axis=1)
        np.testing.assert_allclose(errors, data["relative_l2"], rtol=1e-10)
        np.testing.assert_equal(data["DR"], [0.0, 1.5])
        conditions, states = list(map(str, data["conditions"])), list(map(str, data["states"]))
        output = {condition: dict(zip(states, (100 * error).tolist())) for condition, error in zip(conditions, errors)}
    return {"metrics": output, "columns": ["condition", *states],
            "rows": [[condition, *output[condition].values()] for condition in conditions]}


def training_diagnostics(inputs):
    data, recovered = read_json(inputs["history"]), read_json(inputs["recovered"])
    weighted = np.asarray(data["Ld"]) + np.asarray(data["lam_phys"]) * np.asarray(data["Lp"]) + 20 * np.asarray(data["Lic"])
    np.testing.assert_allclose(weighted, data["loss"], rtol=1e-10)
    errors = {name: (100 * np.abs(np.asarray(data[name]) - recovered["true"][name]) / abs(recovered["true"][name])).tolist()
              for name in ("W", "thetaP")}
    return {"metrics": {"saved_checkpoints": len(data["epoch"]), "epochs": data["epoch"],
                         "final_recorded_adam_loss": data["loss"][-1], "selected_error_percent": errors},
            "columns": ["epoch", "data loss", "weighted physics loss", "weighted IC loss", "W error (%)", "thetaP error (%)"],
            "rows": [[epoch, data["Ld"][i], data["lam_phys"][i] * data["Lp"][i], 20 * data["Lic"][i],
                      errors["W"][i], errors["thetaP"][i]] for i, epoch in enumerate(data["epoch"])]}


def constraint_screen(inputs):
    data = read_json(inputs["transcription"])
    rows = [[row["form"], row["basal_error_percent"], row["function_nrmse_percent"]] for row in data["rows"]]
    errors = [row[1] for row in rows]
    return {"metrics": {"min_reported_basal_error_percent": min(errors), "max_reported_basal_error_percent": max(errors),
                         "provenance_status": "transcribed historical notes; full original run provenance incomplete"},
            "columns": ["constraint", "reported basal error (%)", "reported function NRMSE (%)"], "rows": rows}


CALCULATORS = {function.__name__: function for function in (
    forward_accuracy, inverse_recovery, hybrid_comparison, fisher_information,
    anchor_interventions, hmc_diagnostics, bayesian_forward, normal_treatment,
    training_diagnostics, constraint_screen,
)}
