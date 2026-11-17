"""Strumenti dell'agente: schemi per l'API di Anthropic + esecuzione sicura.

Ogni strumento restituisce un dict JSON-serializzabile; gli errori sono dict {"error": ...} (mai silenziosi).
Le esecuzioni costose passano da una conferma lato UI: il modello NON puo' confermare da solo, perche'
`execute_pending` e' chiamata solo dalla rotta HTTP /api/assistant/confirm.
"""
from __future__ import annotations

import ast
import secrets
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import catalog, config, datasets_tools, evidence, recommender
from .kb.metrics_spec import metric_spec

MOD = {"type": "string", "enum": list(config.MODALITIES), "description": "tabular | ts (time series) | graph (graphs)"}


def _t(name, desc, props, required):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props, "required": required}}


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    _t("list_algorithms", "List the catalog algorithms of a modality with status (executable/wired/described), "
       "strategy and tags. Only 'executable' and 'wired' ones can be launched.", {"modality": MOD}, ["modality"]),
    _t("describe_algorithm", "Algorithm card: paper, source, status, notes and a summary of the measured results.",
       {"modality": MOD, "algorithm": {"type": "string"}}, ["modality", "algorithm"]),
    _t("list_datasets", "List the datasets on disk for a modality (to propose to the user).",
       {"modality": MOD}, ["modality"]),
    _t("analyze_dataset", "Statistics of an available dataset (rows/columns, classes, imbalance, missing values; ts: "
       "length/channels; graphs: nodes/edges/classes). Takes the name, not arbitrary paths.",
       {"modality": MOD, "dataset": {"type": "string"}, "label_col": {"type": "string", "description": "tabular only, optional"}},
       ["modality", "dataset"]),
    _t("recommend_algorithms", "Deterministic recommender: ranks the algorithms with rationale, source of the numbers and status. "
       "Algorithms without measurements are NOT ranked (field unranked).",
       {"modality": MOD, "task": {"type": "string", "description": "ts/graph: generation | view | oversampling"},
        "dataset": {"type": "string"}, "n_rows": {"type": "integer"}, "n_cols": {"type": "integer"},
        "imbalance_ratio": {"type": "number"}, "gpu": {"type": "boolean", "description": "false = GPU not available"},
        "time_budget_s": {"type": "number"}, "need_privacy": {"type": "boolean"}, "need_fairness": {"type": "boolean"},
        "need_constraints": {"type": "boolean"},
        "priority": {"type": "string", "enum": list(recommender.PRIORITIES),
                     "description": "priority profile: balanced | utility | fidelity | diversity | privacy | speed"},
        "executable_only": {"type": "boolean"}, "top_k": {"type": "integer"}}, ["modality"]),
    _t("explain_recommendation", "Explains the recommend_algorithms advice using only numbers computed from the benchmark KB: "
       "score breakdown (weight x axis score, sum = score), strongest and weakest metrics, comparison "
       "with the second and third, similar datasets used as evidence, warnings. If the KB is missing it returns an error: "
       "do not explain in words an advice that has no evidence.",
       {"modality": MOD, "dataset": {"type": "string"},
        "priority": {"type": "string", "enum": list(recommender.PRIORITIES)},
        "algorithm": {"type": "string", "description": "optional: defaults to the top-ranked one"},
        "task": {"type": "string"}, "gpu": {"type": "boolean"}, "time_budget_s": {"type": "number"},
        "need_privacy": {"type": "boolean"}, "need_fairness": {"type": "boolean"}, "need_constraints": {"type": "boolean"},
        "executable_only": {"type": "boolean"}}, ["modality", "dataset"]),
    _t("run_augmentation", "Launches generation ASYNCHRONOUSLY and returns a task_id. With more than one algorithm, "
       "more than one dataset or more than 5 epochs it does NOT start: it requires the user's explicit confirmation in the UI "
       "(it returns status=needs_confirmation; tell the user and wait). Rejects 'described' algorithms. Tabular runs "
       "write to dedicated folders (augmented_ui/, runtime_results_ui/; ts/graph: augmented_ui_<m>/, "
       "runtime_results_ui_<m>/) without resuming previous runs and without touching the benchmark results. Datasets "
       "uploaded by the user (list_datasets: user=true) can be run, but not together with benchmark ones.",
       {"modality": MOD, "algorithms": {"type": "array", "items": {"type": "string"}},
        "datasets": {"type": "array", "items": {"type": "string"}}, "epochs": {"type": "integer", "description": "default 5"},
        "n_samples": {"type": "integer"}, "device": {"type": "string", "enum": ["cpu", "cuda"]}},
       ["modality", "algorithms", "datasets"]),
    _t("get_task_status", "Status of a launched task (RUNNING/SUCCESS/FAILED...) and the last log lines.",
       {"task_id": {"type": "string"}}, ["task_id"]),
    _t("evaluate", "Evaluates (asynchronously) the already generated outputs with the modality's metrics; they must exist.",
       {"modality": MOD, "dataset": {"type": "string"}, "algorithms": {"type": "array", "items": {"type": "string"}}},
       ["modality", "dataset", "algorithms"]),
    _t("compare_results", "Comparison table from the measured CSVs (each row lists its sources). The only way to get numbers to cite.",
       {"modality": MOD, "dataset": {"type": "string"}, "algorithms": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "array", "items": {"type": "string"}}}, ["modality"]),
    _t("explain_metrics", "Verified explanation: for (dataset, algorithm) reads the real values from the CSVs, with the definition and direction of "
       "each metric and its position among the measured algorithms. If you pass claimed_values it compares them with the real data.",
       {"modality": MOD, "dataset": {"type": "string"}, "algorithm": {"type": "string"},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "claimed_values": {"type": "object", "description": "{metric: value} to verify"}},
       ["modality", "dataset", "algorithm"]),
]

GLOSSARY = {  # da evaluate_metrics.py; direzione solo dove certa
    "KL_mean": ("Mean Kullback-Leibler divergence per attribute", "lower"),
    "JS_mean": ("Mean Jensen-Shannon divergence per attribute", "lower"),
    "Q_multi_attr_sim": ("Cosine-based multi-attribute similarity", None),
    "KS_mean": ("Mean Kolmogorov-Smirnov statistic", "lower"),
    "Chi2_mean": ("Mean chi-square test (categorical attributes)", None),
    "WD_mean": ("Mean Wasserstein distance", "lower"),
    "pMSE": ("Propensity MSE: how well a discriminator can tell real from synthetic", "lower"),
    "MAD_mean": ("Normalized mean absolute difference", None),
    "MMD": ("Maximum Mean Discrepancy (multiscale RBF kernel)", "lower"),
    "PCA_WD_mean": ("Wasserstein on the first PCA components", None),
    "Moment_mean_diff": ("Difference of the means", None), "Moment_var_diff": ("Difference of the variances", None),
    "Moment_skew_diff": ("Difference of the skewness", None), "Moment_kurt_diff": ("Difference of the kurtosis", None),
    "Silhouette": ("Separability of the synthetic clusters relative to the real ones", None),
    "Davies-Bouldin": ("Dispersion and separation index", None),
    "Compactness_cls_k": ("Intra-class compactness", None),
    "Precision": ("k-NN Precision: share of synthetic samples inside the real support", "higher"),
    "Recall": ("k-NN Recall: share of the real support covered by the synthetic samples", "higher"),
    "Density": ("Density k-NN", None),
    "Coverage": ("k-NN Coverage: share of real samples with a synthetic neighbour", "higher"),
    "mDCR": ("Mean distance from the nearest real record", None),
    "FE_ratio_mean": ("Mean Feature Entropy ratio", None),
    "DBI_div": ("Davies-Bouldin on the generated data", None),
    "Coverage_div": ("Fraction of real quantiles covered by the synthetic samples", None),
}
DIR_TXT = {"lower": "lower is better", "higher": "higher is better", None: "direction not verified in the project"}


@dataclass
class ToolContext:
    task_manager: Any = None
    pending: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_id: str = ""


# ---------------------------------------------------------------- strumenti di sola lettura
def _list_algorithms(ctx, a):
    c = catalog.get_catalog(a["modality"])
    algos = [{k: x[k] for k in ("key", "name", "status", "strategy", "task", "needs_gpu", "tags")} for x in c["algorithms"]]
    cnt: Dict[str, int] = {}
    for x in algos:
        cnt[x["status"]] = cnt.get(x["status"], 0) + 1
    return {"modality": a["modality"], "counts_by_status": cnt, "algorithms": algos, "warnings": c["warnings"],
            "source": "config.CORE_ALGORITHMS + assistant/kb/benchmark.csv (status) + Algorithms/Algorithms.md (paper)" if a["modality"] == "tabular" else "modalities.list_specs()"}


def _describe(ctx, a):
    x = catalog.find(a["modality"], a["algorithm"])
    out = dict(x)
    if a["modality"] == "tabular":
        t = evidence.tabular_runtime(); m = evidence.tabular_metrics()
        r = t[(t["algo"] == x["key"])] if not t.empty else t
        out["measured_runs"] = {"success": int((r["status"] == "SUCCESS").sum()) if len(r) else 0,
                                "failed": int((r["status"] != "SUCCESS").sum()) if len(r) else 0,
                                "datasets": sorted(set(r["ds"])) if len(r) else [],
                                "source": sorted(set(r["source"]))[:5] if len(r) else []}
        out["measured_quality_datasets"] = sorted(set(m.loc[m["algo"] == x["key"], "ds"])) if not m.empty else []
    else:
        r = evidence.modality_runtime(a["modality"])
        r = r[r["algo"] == x["key"]] if not r.empty and "algo" in r else r
        out["measured_runs"] = {"n": int(len(r)), "source": sorted(set(r["source"]))[:5] if len(r) else []}
    if not out["measured_runs"].get("success", out["measured_runs"].get("n", 0)):
        out["evidence_note"] = "no measured result: I do not know how it behaves here"
    return out


def _list_datasets(ctx, a):
    return datasets_tools.list_datasets(a["modality"])


def _analyze(ctx, a):
    return datasets_tools.analyze_dataset(a["modality"], a["dataset"], a.get("label_col"))


def _recommend(ctx, a):
    kw = {k: v for k, v in a.items() if k != "modality"}
    return recommender.recommend(a["modality"], **kw)


# ---------------------------------------------------------------- esecuzione
def _validate_run(a) -> Dict[str, Any]:
    mod = a["modality"]
    algos = a.get("algorithms") or []
    dss = a.get("datasets") or []
    if not algos or not dss:
        raise ValueError("at least one algorithm and one dataset are required")
    if len(algos) > 20 or len(dss) > 20:
        raise ValueError("too many algorithms or datasets")
    cat = {x["key"]: x for x in catalog.get_catalog(mod)["algorithms"]}
    for k in algos:
        k = str(k).lower().strip()
        if not config.NAME_RE.match(k) or k not in cat:
            raise ValueError(f"algorithm not in the catalog: {k!r}")
        if cat[k]["status"] == "described":
            raise ValueError(f"{k}: only described, not executable")
    names, user_flags = [], []
    for d in dss:
        p = datasets_tools.resolve_dataset(mod, str(d))
        names.append(p.stem)
        user_flags.append(datasets_tools.is_user_dataset(mod, p))
    if any(user_flags) and not all(user_flags):
        raise ValueError("do not mix user-uploaded datasets and benchmark datasets in the same run: "
                         "launch them separately")
    ep = int(a.get("epochs") or 5)
    ns = a.get("n_samples")
    if ep < 1 or ep > 100000 or (ns is not None and (int(ns) < 1 or int(ns) > 10_000_000)):
        raise ValueError("epochs/n_samples out of range")
    dev = a.get("device") or "cpu"
    if dev not in ("cpu", "cuda"):
        raise ValueError("allowed device: cpu | cuda")
    return {"modality": mod, "algorithms": [str(k).lower().strip() for k in algos], "datasets": names,
            "epochs": ep, "n_samples": int(ns) if ns else None, "device": dev, "user_data": bool(user_flags and all(user_flags))}


def needs_confirmation(v) -> Optional[str]:
    why = []
    if len(v["algorithms"]) > config.CONFIRM_MAX_ALGOS:
        why.append(f"{len(v['algorithms'])} algorithms")
    if len(v["datasets"]) > config.CONFIRM_MAX_DATASETS:
        why.append(f"{len(v['datasets'])} datasets")
    if v["epochs"] > config.CONFIRM_MAX_EPOCHS:
        why.append(f"{v['epochs']} epochs")
    return ", ".join(why) or None


def tabular_python() -> str:
    """Interprete di envs/tabular (ENV_PYTHONS['tabular'] di config.py); sys.executable se l'ambiente non c'e'."""
    try:
        import config as root_config
        return root_config.ENV_PYTHONS.get("tabular") or sys.executable
    except Exception:  # noqa: BLE001
        return sys.executable


def ui_dirs() -> Dict[str, str]:
    r = str(config.ROOT)
    return {"augmented": f"{r}/{config.UI_AUGMENTED_DIR}", "results": f"{r}/{config.UI_RESULTS_DIR}",
            "evaluation": f"{r}/{config.UI_EVAL_DIR}"}


def ui_modality_dirs(modality: str) -> Dict[str, str]:
    """Cartelle dedicate alle esecuzioni ts/graph lanciate dalla UI e dall'assistente (run_modality.py --out-dir/--results-dir)."""
    r = str(config.ROOT)
    return {"augmented": f"{r}/{config.UI_OUT_DIRS[modality]}", "results": f"{r}/{config.UI_RES_DIRS[modality]}"}


def _build_cmd(v) -> List[str]:
    root = str(config.ROOT)
    if v["modality"] == "tabular":
        d = ui_dirs()
        # cartelle dedicate e niente ripresa: un rilancio non salta lavoro in silenzio ne' sovrascrive il benchmark
        cmd = [tabular_python(), f"{root}/main.py", "--algorithms", *v["algorithms"], "--datasets", *v["datasets"],
               "--epochs", str(v["epochs"]), "--augmented-dir", d["augmented"], "--results-dir", d["results"],
               "--no-resume"]
        if v.get("user_data"):      # dataset caricati dall'utente: main.py li cerca nella loro cartella
            cmd += ["--datasets-dir", str(config.user_dir("tabular"))]
        if v["n_samples"]:
            cmd += ["--n-samples", str(v["n_samples"])]
        cmd += ["--use-gpu", "--device", "cuda"] if v["device"] == "cuda" else ["--no-gpu"]
        return cmd
    md = ui_modality_dirs(v["modality"])
    cmd = [sys.executable, f"{root}/run_modality.py", "--modality", v["modality"], "--algorithms", *v["algorithms"],
           "--datasets", *v["datasets"], "--epochs", str(v["epochs"]), "--device", v["device"],
           "--out-dir", md["augmented"], "--results-dir", md["results"]]
    if v["n_samples"]:
        cmd += ["--n_samples", str(v["n_samples"])]
    return cmd


def _launch(ctx: ToolContext, task_type: str, cmd: List[str]) -> Dict[str, Any]:
    tm = ctx.task_manager
    if tm is None:
        return {"error": "task manager not available"}
    active = tm.get_active_task()
    if active:
        return {"error": f"a task is already running ({active.task_id}); wait or stop it"}
    task = tm.create_task(task_type, cmd, config.ROOT)
    tm.start_task(task)
    out = {"status": "started", "task_id": task.task_id, "command": " ".join(cmd[1:])}
    if "--augmented-dir" in cmd or "--augmented_dir" in cmd:
        out["output_dirs"] = {"augmented": config.UI_AUGMENTED_DIR, "results": config.UI_RESULTS_DIR,
                              "evaluation": config.UI_EVAL_DIR}
    elif "--out-dir" in cmd:
        m = cmd[cmd.index("--modality") + 1]
        out["output_dirs"] = {"augmented": config.UI_OUT_DIRS[m], "results": config.UI_RES_DIRS[m]}
    return out


def _run(ctx, a):
    v = _validate_run(a)
    why = needs_confirmation(v)
    if why:
        pid = secrets.token_hex(4)
        ctx.pending[pid] = {"kind": "run", "args": v, "created": time.time(), "why": why}
        return {"status": "needs_confirmation", "pending_id": pid, "reason": why,
                "message": "Nothing has been launched. The user must confirm with the button in the UI; "
                           "ask them to do so and wait for their message.", "plan": v}
    return _launch(ctx, "augmentation", _build_cmd(v))


def execute_pending(ctx: ToolContext, pending_id: str, approve: bool) -> Dict[str, Any]:
    """Chiamata SOLO dalla rotta di conferma (azione umana)."""
    p = ctx.pending.pop(pending_id, None)
    if p is None:
        return {"error": "unknown or expired confirmation"}
    if time.time() - p["created"] > 3600:
        return {"error": "confirmation expired (> 1 hour)"}
    if not approve:
        return {"status": "rejected", "message": "the user declined: nothing was launched", "plan": p["args"]}
    return _launch(ctx, p["kind"] == "run" and "augmentation" or "evaluation", p["cmd"] if "cmd" in p else _build_cmd(p["args"]))


def _status(ctx, a):
    tid = str(a["task_id"])
    if not config.NAME_RE.match(tid) or ctx.task_manager is None:
        return {"error": "invalid task_id or task manager not available"}
    t = ctx.task_manager.get_task(tid)
    if not t:
        return {"error": f"unknown task {tid}"}
    d = t.to_dict()
    with t._lock:
        d["log_tail"] = [l.rstrip() for l in t.logs[-15:]]
    return d


def _evaluate(ctx, a):
    mod = a["modality"]
    p = datasets_tools.resolve_dataset(mod, a["dataset"])
    algos = [str(x).lower().strip() for x in a["algorithms"]]
    if not algos or any(not config.NAME_RE.match(x) for x in algos):
        raise ValueError("invalid algorithms")
    missing = []
    for k in algos:
        out = (config.ROOT / config.UI_AUGMENTED_DIR / k / f"{p.stem}_augmented.csv" if mod == "tabular"
               else config.ROOT / config.UI_OUT_DIRS[mod] / f"{p.stem}__{k}{'.npz' if mod == 'ts' else '.pt'}")
        if not out.is_file():
            missing.append(str(out.relative_to(config.ROOT)).replace("\\", "/"))
    if missing:
        return {"error": "outputs to evaluate do not exist (augmented_ui/, augmented_ui_ts/, augmented_ui_graph/): generate first with run_augmentation",
                "missing": missing}
    if mod == "tabular":
        d = ui_dirs()
        cmd = [tabular_python(), f"{config.ROOT}/evaluate_metrics.py", "--dataset", p.stem, "--algorithms", *algos,
               "--augmented_dir", d["augmented"], "--output_dir", d["evaluation"]]
        if datasets_tools.is_user_dataset(mod, p):
            cmd += ["--datasets_dir", str(config.user_dir("tabular"))]
    else:
        if not (config.ROOT / "modalities" / mod / "metrics.py").is_file():
            return {"error": f"modalities/{mod}/metrics.py does not exist yet: metrics not available"}
        if len(algos) > 1:
            return {"error": "for ts/graph evaluate one algorithm at a time"}
        from modalities.base import env_python, get_spec
        ext = ".npz" if mod == "ts" else ".pt"
        gen = f"{config.ROOT}/{config.UI_OUT_DIRS[mod]}/{p.stem}__{algos[0]}{ext}"
        code = ("import json,sys,importlib;m=importlib.import_module('modalities.'+sys.argv[1]+'.metrics');"
                "print(json.dumps(m.evaluate(sys.argv[2],sys.argv[3]),default=float))")
        cmd = [env_python(get_spec(mod, algos[0]).env), "-c", code, mod, str(p), gen]
    return _launch(ctx, "evaluation", cmd)


# ---------------------------------------------------------------- risultati
def _compare(ctx, a):
    mod = a["modality"]
    ds = evidence.canon_ds(a["dataset"]) if a.get("dataset") else None
    algs = {evidence.canon_algo(x) for x in a.get("algorithms") or []} or None
    want = a.get("metrics")
    rows: List[Dict[str, Any]] = []
    if mod == "tabular":
        m = evidence.tabular_metrics(); t = evidence.tabular_runtime()
        if m.empty:
            return {"rows": [], "message": "no evaluation result in evaluation_metrics/"}
        if ds:
            m = m[m["ds"] == ds]
        if algs:
            m = m[m["algo"].isin(algs)]
        for _, r in m.iterrows():
            row = {"dataset": r["ds"], "algorithm": r["algo"], "sources": r["sources"]}
            for c in m.columns:
                if c in ("ds", "algo", "sources"):
                    continue
                if want and c not in want:
                    continue
                v = evidence.fnum(r[c])
                if v is not None:
                    row[c] = v
            tt = t[(t["ds"] == r["ds"]) & (t["algo"] == r["algo"]) & (t["status"] == "SUCCESS")] if not t.empty else t
            if len(tt) and (not want or "total_time_s" in want):
                row["total_time_s_median"] = evidence.fnum(tt["total_time_s"].median())
                row["time_source"] = sorted(set(tt["source"]))[0]
            rows.append(row)
    else:
        d = evidence.modality_runtime(mod)
        if d.empty:
            return {"rows": [], "message": f"no CSV in runtime_results_{mod}/: no measured result"}
        if ds and "ds" in d:
            d = d[d["ds"] == ds]
        if algs and "algo" in d:
            d = d[d["algo"].isin(algs)]
        for _, r in d.iterrows():
            row = {k: (evidence.fnum(v) if not isinstance(v, str) else v) for k, v in r.items()
                   if k not in ("algo", "ds", "error_message") and (not want or k in want or k in ("algorithm", "dataset", "status", "source"))}
            rows.append({k: v for k, v in row.items() if v is not None})
    out = {"modality": mod, "n_rows": len(rows), "rows": rows[:80]}
    if len(rows) > 80:
        out["truncated"] = f"showing 80 rows out of {len(rows)}"
    if not rows:
        out["message"] = "no result for these filters: I do not know"
    return out


def _metrics_doc(mod) -> Optional[str]:
    f = config.ROOT / "modalities" / mod / "metrics.py"
    try:
        return ast.get_docstring(ast.parse(f.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return None


def _explain(ctx, a):
    mod = a["modality"]
    ds = evidence.canon_ds(a["dataset"]); alg = evidence.canon_algo(a["algorithm"])
    want = a.get("metrics")
    claimed = a.get("claimed_values") or {}
    if mod != "tabular":
        d = evidence.modality_runtime(mod)
        r = d[(d.get("ds") == ds) & (d.get("algo") == alg)] if not d.empty and "ds" in d else d
        if d.empty or not len(r):
            return {"error": "no measured result for this pair: I cannot explain", "modality": mod}
        row = r.iloc[-1]
        return {"modality": mod, "dataset": ds, "algorithm": alg, "source": row["source"],
                "values": {k: (evidence.fnum(v) if not isinstance(v, str) else v) for k, v in row.items()
                           if k not in ("algo", "ds", "error_message", "source")},
                "metrics_doc": _metrics_doc(mod) or "modalities/%s/metrics.py not available: unknown definitions" % mod}
    m = evidence.tabular_metrics()
    if m.empty:
        return {"error": "evaluation_metrics/ is empty: no value to explain"}
    same = m[m["ds"] == ds]
    r = same[same["algo"] == alg]
    if not len(r):
        return {"error": f"no evaluation for {alg} on {ds}", "available_for_dataset": sorted(set(same["algo"]))}
    row = r.iloc[0]
    out_m, mism = {}, []
    for c in m.columns:
        if c in ("ds", "algo", "sources") or (want and c not in want):
            continue
        v = evidence.fnum(row[c])
        if v is None:
            continue
        desc, direction = GLOSSARY.get(c, ("metric not documented in evaluate_metrics.py", None))
        e = {"value": v, "definition": desc, "direction": DIR_TXT[direction],
             "source": "evaluation_metrics/diversity_all_datasets.csv" if c in evidence.DIVERSITY_HIGHER_BETTER + [
                 "Silhouette", "Davies-Bouldin", "Compactness_cls_k", "Density", "mDCR", "FE_ratio_mean", "DBI_div", "Coverage_div"]
             else "evaluation_metrics/fidelity_all_datasets.csv"}
        col = same[c].dropna()
        if len(col) > 1 and direction:
            pos = int((col < v).sum() + 1) if direction == "lower" else int((col > v).sum() + 1)
            e["position"] = f"rank {pos} of {len(col)} algorithms measured on {ds}"
        out_m[c] = e
        if c in claimed:
            try:
                ok = abs(float(claimed[c]) - v) <= max(1e-3, 0.005 * abs(v))
            except (TypeError, ValueError):
                ok = False
            e["claimed_value_matches"] = ok
            if not ok:
                mism.append(c)
    unk = [c for c in claimed if c not in out_m]
    res = {"modality": mod, "dataset": ds, "algorithm": alg, "metrics": out_m,
           "note": "kamino-* and variants are averaged" if alg == "kamino" else ""}
    if mism:
        res["MISMATCH"] = f"claimed values do NOT match the data: {mism}"
    if unk:
        res["unverifiable_claims"] = unk
    return res


def _explain_reco(ctx, a):
    mod = a["modality"]
    kw = {k: a[k] for k in ("task", "gpu", "time_budget_s", "need_privacy", "need_fairness", "need_constraints",
                            "executable_only") if k in a}
    r = recommender.recommend(mod, dataset=a["dataset"], priority=a.get("priority") or "balanced", top_k=1000,
                              detail=True, **kw)
    if r.get("evidence") != "kb":
        return {"error": "no evidence from the benchmark KB for this modality: there is no advice to explain "
                         "(see `method`)", "modality": mod, "method": r.get("method")}
    ranking = r["ranking"]
    if not ranking:
        return {"error": "no ranked candidate for these filters", "unranked": r["unranked"][:20],
                "excluded": r["excluded"][:20]}
    if a.get("algorithm"):
        key = evidence.canon_algo(a["algorithm"])
        t = next((x for x in ranking if x["algorithm"] == key), None)
        if t is None:
            return {"error": f"{key} is not among the ranked algorithms", "ranked": [x["algorithm"] for x in ranking]}
    else:
        t = ranking[0]
    ms = []
    for c, sc in (t.get("metric_scores") or {}).items():
        d, ax = metric_spec(mod, c)
        e = {"metric": c, "axis": ax, "direction": d, "score": sc}
        if (t.get("metric_values") or {}).get(c) is not None:
            e["value"] = t["metric_values"][c]
        ms.append(e)
    ms.sort(key=lambda e: (-(e["score"] if e["score"] is not None else -1), e["metric"]))
    others = [x for x in ranking if x is not t and x["rank"] <= 3][:2]
    if t["rank"] > 3:
        others = ranking[:2]
    comp = []
    for o in others:
        dc = {ax: round(t["contributions"].get(ax, 0.0) - o["contributions"].get(ax, 0.0), 6) for ax in r["weights"]}
        comp.append({"algorithm": o["algorithm"], "rank": o["rank"], "score": o["score"],
                     "score_diff": round(t["score"] - o["score"], 6), "contribution_diff_by_axis": dc,
                     "axis_most_favourable": max(dc, key=dc.get) if dc else None,
                     "axis_least_favourable": min(dc, key=dc.get) if dc else None,
                     "other_axis_scores": o["axis_scores"], "other_warnings": o["warnings"]})
    return {"modality": mod, "dataset": a["dataset"], "scope": r["scope"], "profile": r["profile"], "method": r["method"],
            "algorithm": t["algorithm"], "rank": t["rank"], "n_ranked": r["n_ranked_total"], "score": t["score"],
            "weights": r["weights"], "weights_declared": r["weights_declared"], "axes_available": r["axes_available"],
            "axis_scores": t["axis_scores"], "contributions": t["contributions"],
            "contributions_sum": round(sum(t["contributions"].values()), 6),
            "n_metrics": len(ms), "strongest_metrics": ms[:5], "weakest_metrics": ms[::-1][:5],
            "comparison": comp, "similar_datasets": r["similar_datasets"], "evidence": t["evidence"],
            "n_runs": t["n_runs"], "n_seeds": t["n_seeds"], "failure_rate": t["failure_rate"], "warnings": t["warnings"],
            "source": r["kb"]["file"],
            "note": "all numbers are computed from the KB; the causes (why it is better) are not in the data: hypotheses to be labelled as such"}


HANDLERS: Dict[str, Callable] = {
    "list_algorithms": _list_algorithms, "describe_algorithm": _describe, "list_datasets": _list_datasets,
    "analyze_dataset": _analyze, "recommend_algorithms": _recommend, "run_augmentation": _run,
    "get_task_status": _status, "evaluate": _evaluate, "compare_results": _compare, "explain_metrics": _explain,
    "explain_recommendation": _explain_reco,
}


def dispatch(ctx: ToolContext, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    h = HANDLERS.get(name)
    if h is None:
        return {"error": f"unknown tool: {name}"}
    if not isinstance(args, dict):
        return {"error": "invalid arguments"}
    try:
        if "modality" in TOOL_SCHEMAS[[s["name"] for s in TOOL_SCHEMAS].index(name)]["input_schema"]["properties"]:
            if args.get("modality") not in config.MODALITIES:
                return {"error": f"modality must be one of {list(config.MODALITIES)}"}
        return h(ctx, args)
    except (ValueError, KeyError, FileNotFoundError, PermissionError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"internal error in tool {name}: {type(e).__name__}: {e}"}
