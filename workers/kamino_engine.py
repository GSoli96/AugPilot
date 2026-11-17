#!/usr/bin/env python3
"""
kamino_engine.py - Core constraint engine and synthesizer for Kamino (VLDB 2021).

Supports:
1. Parsing Denial Constraints (DCs) in:
   - Kamino original syntax (.ic, e.g., t1&t2&EQ(t1.col,t2.col)&IQ(...))
   - SyDC / FDCD syntax (e.g., not(t1.attr0>t2.attr0 and t1.attr1==t2.attr1 ...), ¬{...})
2. Attribute name matching (case-insensitive and positional fallback attr0 -> col[0]).
3. Exact Kamino weight computation and strategies:
   - 'toggled': Learned weights via LinearNet SGD + toggle_weights (w < 1e-6 -> 1000.0) [Kamino default]
   - 'hard': Uniform hard constraints (all weights = 1000.0)
   - 'raw': Continuous unthresholded SGD learned weights
   - 'dp': Differentially private learned weights with Gaussian mechanism
   - 'uniform': Uniform weights (1.0)
4. Constraint-aware synthesis with Gibbs/Energy acceptance-rejection and MCMC repair.
"""

from __future__ import annotations

import re
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR


# ---------------------------------------------------------------------------
# 1. Operators & Data Structures
# ---------------------------------------------------------------------------
OP_MAP = {
    "==": "==",
    "=": "==",
    "EQ": "==",
    "<>": "!=",
    "!=": "!=",
    "IQ": "!=",
    "<": "<",
    "LT": "<",
    "<=": "<=",
    "LTE": "<=",
    ">": ">",
    "GT": ">",
    ">=": ">=",
    "GTE": ">=",
}


class Predicate:
    def __init__(
        self,
        t1_idx: int,
        col1: str,
        op: str,
        t2_idx: Optional[int] = None,
        col2: Optional[str] = None,
        const: Optional[float] = None,
    ):
        self.t1_idx = t1_idx  # 0 (t1/t0) or 1 (t2)
        self.col1 = col1
        self.op = op
        self.t2_idx = t2_idx
        self.col2 = col2
        self.const = const

    def __repr__(self) -> str:
        if self.t2_idx is not None:
            return f"t{self.t1_idx+1}.{self.col1} {self.op} t{self.t2_idx+1}.{self.col2}"
        return f"t{self.t1_idx+1}.{self.col1} {self.op} {self.const}"


class DenialConstraint:
    def __init__(self, predicates: List[Predicate], raw_text: str = ""):
        self.predicates = predicates
        self.raw_text = raw_text

    def __repr__(self) -> str:
        return f"¬({ ' ∧ '.join(str(p) for p in self.predicates) })"


# ---------------------------------------------------------------------------
# 2. Unified DC Parser (Kamino .ic & SyDC .out / .txt)
# ---------------------------------------------------------------------------
def parse_dc_file(file_path: Path, df_columns: List[str]) -> List[DenialConstraint]:
    """
    Parse Denial Constraints supporting both:
    1. SyDC format: not(t1.col1 op t2.col1 and ...) or ¬{ t0.col1 op t1.col1 ∧ ... }
    2. Kamino original .ic format: t1&t2&EQ(t1.col1,t2.col1)&... or infix forms.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"DC file not found: {file_path}")

    # 1. Positional fallback mapping (attr0 -> df_columns[0])
    col_map = {}
    for idx, c in enumerate(df_columns):
        col_map[f"attr{idx}"] = c
        col_map[f"att{idx}"] = c
        col_map[f"col{idx}"] = c

    # 2. Exact column names take absolute precedence
    for c in df_columns:
        col_map[c.lower()] = c

    dcs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Format 1: SyDC / FDCD format
            if line.startswith("not(") or line.startswith("¬{") or line.startswith("not ("):
                content = line
                if content.startswith("not(") and content.endswith(")"):
                    content = content[4:-1]
                elif content.startswith("not (") and content.endswith(")"):
                    content = content[5:-1]
                elif content.startswith("¬{") and content.endswith("}"):
                    content = content[2:-1]

                parts = re.split(r"\s+(?:and|∧)\s+", content.strip())
                preds = []
                for part in parts:
                    part = part.strip()
                    m = re.match(
                        r"(t[120])\.([a-zA-Z0-9_-]+)\s*(==|<>|!=|<=|>=|<|>|=)\s*(?:(t[120])\.([a-zA-Z0-9_-]+)|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))",
                        part,
                    )
                    if m:
                        t1_str, c1_raw, op_raw, t2_str, c2_raw, const_raw = m.groups()
                        t1_idx = 0 if t1_str in ("t1", "t0") else 1
                        c1 = col_map.get(c1_raw.lower(), c1_raw)
                        op = OP_MAP.get(op_raw, op_raw)
                        if t2_str is not None:
                            t2_idx = 0 if t2_str in ("t1", "t0") else 1
                            c2 = col_map.get(c2_raw.lower(), c2_raw)
                            preds.append(Predicate(t1_idx, c1, op, t2_idx, c2))
                        else:
                            preds.append(Predicate(t1_idx, c1, op, const=float(const_raw)))
                if preds:
                    dcs.append(DenialConstraint(preds, line))

            # Format 2: Kamino original .ic format
            elif "&" in line:
                tokens = [t.strip() for t in line.split("&") if t.strip()]
                preds = []
                for tok in tokens:
                    if tok.lower() in ("t1", "t2", "t0"):
                        continue
                    # Function syntax: OP(t1.A, t2.B)
                    m_fn = re.match(
                        r"(EQ|IQ|LT|LTE|GT|GTE)\s*\(\s*(t[120])\.([a-zA-Z0-9_-]+)\s*,\s*(?:(t[120])\.([a-zA-Z0-9_-]+)|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))\s*\)",
                        tok,
                        re.IGNORECASE,
                    )
                    if m_fn:
                        op_raw, t1_str, c1_raw, t2_str, c2_raw, const_raw = m_fn.groups()
                        t1_idx = 0 if t1_str.lower() in ("t1", "t0") else 1
                        c1 = col_map.get(c1_raw.lower(), c1_raw)
                        op = OP_MAP.get(op_raw.upper(), "==")
                        if t2_str is not None:
                            t2_idx = 0 if t2_str.lower() in ("t1", "t0") else 1
                            c2 = col_map.get(c2_raw.lower(), c2_raw)
                            preds.append(Predicate(t1_idx, c1, op, t2_idx, c2))
                        else:
                            preds.append(Predicate(t1_idx, c1, op, const=float(const_raw)))
                        continue

                    # Infix operator syntax: t1.A op t2.B
                    m_inf = re.match(
                        r"(t[120])\.([a-zA-Z0-9_-]+)\s*(==|<>|!=|<=|>=|<|>|=)\s*(?:(t[120])\.([a-zA-Z0-9_-]+)|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))",
                        tok,
                    )
                    if m_inf:
                        t1_str, c1_raw, op_raw, t2_str, c2_raw, const_raw = m_inf.groups()
                        t1_idx = 0 if t1_str in ("t1", "t0") else 1
                        c1 = col_map.get(c1_raw.lower(), c1_raw)
                        op = OP_MAP.get(op_raw, op_raw)
                        if t2_str is not None:
                            t2_idx = 0 if t2_str in ("t1", "t0") else 1
                            c2 = col_map.get(c2_raw.lower(), c2_raw)
                            preds.append(Predicate(t1_idx, c1, op, t2_idx, c2))
                        else:
                            preds.append(Predicate(t1_idx, c1, op, const=float(const_raw)))

                if preds:
                    dcs.append(DenialConstraint(preds, line))

    return dcs


def build_compiled_dcs(dcs: List[DenialConstraint], col_names: List[str]) -> List[List[Tuple]]:
    """Compile parsed DCs to numerical indices for vectorized evaluation."""
    col_idx = {c: i for i, c in enumerate(col_names)}
    compiled = []
    for dc in dcs:
        preds = []
        valid = True
        for p in dc.predicates:
            if p.col1 not in col_idx:
                valid = False
                break
            c1 = col_idx[p.col1]
            if p.t2_idx is not None:
                if p.col2 not in col_idx:
                    valid = False
                    break
                c2 = col_idx[p.col2]
                preds.append((p.t1_idx, c1, p.op, p.t2_idx, c2, None))
            else:
                preds.append((p.t1_idx, c1, p.op, None, None, p.const))
        if valid and preds:
            compiled.append(preds)
    return compiled


# ---------------------------------------------------------------------------
# 3. Vectorized Violation Checking
# ---------------------------------------------------------------------------
def compute_violation_matrix(X: np.ndarray, compiled_dcs: List[List[Tuple]]) -> np.ndarray:
    """
    Compute violation matrix of size (N, K).
    vio[i, k] is the number of rows j != i that together with i violate DC k.
    """
    N = len(X)
    K = len(compiled_dcs)
    vio_matrix = np.zeros((N, K), dtype=np.float32)
    if N < 2 or K == 0:
        return vio_matrix

    for k, preds in enumerate(compiled_dcs):
        pair_viol = np.ones((N, N), dtype=bool)
        np.fill_diagonal(pair_viol, False)

        for (t1_idx, c1, op, t2_idx, c2, const) in preds:
            if not pair_viol.any():
                break
            if t2_idx is not None:
                v1 = X[:, c1]
                v2 = X[:, c2]
                if t1_idx == 0 and t2_idx == 1:
                    diff = v1[:, None] - v2[None, :]
                elif t1_idx == 1 and t2_idx == 0:
                    diff = v1[None, :] - v2[:, None]
                elif t1_idx == 0 and t2_idx == 0:
                    diff = (v1 - v2)[:, None]
                else:
                    diff = (v1 - v2)[None, :]

                if op == "==":
                    cond = np.isclose(diff, 0.0, atol=1e-8)
                elif op == "!=":
                    cond = ~np.isclose(diff, 0.0, atol=1e-8)
                elif op == "<":
                    cond = diff < -1e-8
                elif op == "<=":
                    cond = diff <= 1e-8
                elif op == ">":
                    cond = diff > 1e-8
                elif op == ">=":
                    cond = diff >= -1e-8
                pair_viol &= cond
            else:
                v1 = X[:, c1]
                if op == "==":
                    cond = np.isclose(v1, const, atol=1e-8)
                elif op == "!=":
                    cond = ~np.isclose(v1, const, atol=1e-8)
                elif op == "<":
                    cond = v1 < const - 1e-8
                elif op == "<=":
                    cond = v1 <= const + 1e-8
                elif op == ">":
                    cond = v1 > const + 1e-8
                elif op == ">=":
                    cond = v1 >= const - 1e-8
                if t1_idx == 0:
                    pair_viol &= cond[:, None]
                else:
                    pair_viol &= cond[None, :]

        vio_matrix[:, k] = pair_viol.sum(axis=1)

    return vio_matrix


def evaluate_candidate_violations(cand: np.ndarray, ref: np.ndarray, compiled_dcs: List[List[Tuple]]) -> np.ndarray:
    """
    Evaluate violations of candidate rows against reference rows.
    cand: (M, D), ref: (N, D) -> returns vios of size (M, K).
    """
    M = len(cand)
    N = len(ref)
    K = len(compiled_dcs)
    vios = np.zeros((M, K), dtype=np.float32)
    if M == 0 or N == 0 or K == 0:
        return vios

    for k, preds in enumerate(compiled_dcs):
        pair_viol = np.ones((M, N), dtype=bool)

        for (t1_idx, c1, op, t2_idx, c2, const) in preds:
            if not pair_viol.any():
                break
            if t2_idx is not None:
                if t1_idx == 0 and t2_idx == 1:
                    diff = cand[:, c1, None] - ref[None, :, c2]
                elif t1_idx == 1 and t2_idx == 0:
                    diff = ref[None, :, c1] - cand[:, c1, None]
                elif t1_idx == 0 and t2_idx == 0:
                    diff = (cand[:, c1] - cand[:, c2])[:, None]
                else:
                    diff = (ref[:, c1] - ref[:, c2])[None, :]

                if op == "==":
                    cond = np.isclose(diff, 0.0, atol=1e-8)
                elif op == "!=":
                    cond = ~np.isclose(diff, 0.0, atol=1e-8)
                elif op == "<":
                    cond = diff < -1e-8
                elif op == "<=":
                    cond = diff <= 1e-8
                elif op == ">":
                    cond = diff > 1e-8
                elif op == ">=":
                    cond = diff >= -1e-8
                pair_viol &= cond
            else:
                v1 = cand[:, c1] if t1_idx == 0 else ref[:, c1]
                if op == "==":
                    cond = np.isclose(v1, const, atol=1e-8)
                elif op == "!=":
                    cond = ~np.isclose(v1, const, atol=1e-8)
                elif op == "<":
                    cond = v1 < const - 1e-8
                elif op == "<=":
                    cond = v1 <= const + 1e-8
                elif op == ">":
                    cond = v1 > const + 1e-8
                elif op == ">=":
                    cond = v1 >= const - 1e-8
                if t1_idx == 0:
                    pair_viol &= cond[:, None]
                else:
                    pair_viol &= cond[None, :]

        vios[:, k] = pair_viol.sum(axis=1)

    return vios


# ---------------------------------------------------------------------------
# 4. Kamino Weight Learning & Strategies
# ---------------------------------------------------------------------------
class LinearWeightNet(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.w = nn.Linear(size, 1, bias=False)
        self.w.weight.data.fill_(0.0)

    def forward(self, n_vios):
        return self.w(n_vios)

    def get_weights(self) -> np.ndarray:
        return self.w.weight.detach().cpu().numpy()[0]


def learn_kamino_weights(
    X: np.ndarray,
    compiled_dcs: List[List[Tuple]],
    strategy: str = "toggled",
    dp_eps: Optional[float] = None,
    dp_delta: float = 1e-3,
    lr: float = 1e-4,
    batch_size: int = 29,
    seed: int = 42,
) -> np.ndarray:
    """
    Learn weights for Denial Constraints based on the specified Kamino strategy:
    - 'hard': Uniform hard constraints (weight = 1000.0)
    - 'uniform': Uniform soft weights (weight = 1.0)
    - 'raw': Raw continuous SGD learned weights
    - 'toggled': Kamino default (toggle_weights: w < 1e-6 -> 1000.0)
    - 'dp': Differentially private learned weights + toggling
    """
    K = len(compiled_dcs)
    if K == 0:
        return np.array([], dtype=np.float32)

    strategy_clean = strategy.lower().replace("-", "_").strip()

    if strategy_clean in ("hard", "uniform_hard"):
        return np.full(K, 1000.0, dtype=np.float32)
    elif strategy_clean in ("uniform", "soft_uniform"):
        return np.full(K, 1.0, dtype=np.float32)

    # Compute violation matrix on clean data
    vio = compute_violation_matrix(X, compiled_dcs)
    n = len(X)

    if strategy_clean == "dp" or dp_eps is not None:
        eps = dp_eps if dp_eps is not None else 1.0
        rng = np.random.default_rng(seed)
        sens = np.sqrt(max(1.0, float(n**2 - n)))
        std = np.sqrt(sens * np.log(1.25 / float(dp_delta))) / float(eps)
        noise = rng.normal(0.0, std, vio.shape)
        vio = np.clip(vio + noise, 0.0, None)

    # Train Kamino LinearNet using SGD + CosineAnnealingLR
    torch.manual_seed(seed)
    model = LinearWeightNet(K)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    mb = min(batch_size, n)
    n_epoch = 1 + n // mb if mb > 0 else 1
    scheduler = CosineAnnealingLR(optimizer, n_epoch)

    vio_tensor = torch.from_numpy(vio.astype(np.float32))

    for _ in range(n_epoch):
        perm = torch.randperm(n)
        for i in range(0, n, mb):
            batch_idx = perm[i : i + mb]
            batch_vio = vio_tensor[batch_idx]

            y = model(batch_vio)
            # Kamino loss: -log(p * e^y) -> since grad w.r.t w is -batch_vio:
            loss = -(torch.log(torch.ones_like(y)) + y).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    raw_weights = model.get_weights().astype(np.float32)

    if strategy_clean == "raw":
        return raw_weights

    # 'toggled' or 'dp': apply Kamino toggle_weights
    # (constraints with 0 / negligible violations on training data are hard constraints)
    toggled_weights = np.where(raw_weights < 1e-6, 1000.0, raw_weights)
    return toggled_weights.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Constraint-Aware Synthesis (Kamino Stage 2 & 3)
# ---------------------------------------------------------------------------
def run_kamino_synthesis(
    df: pd.DataFrame,
    n_synth: int,
    compiled_dcs: List[List[Tuple]],
    weights: np.ndarray,
    epochs: int = 100,
    max_repair_tries: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthesize rows using GMM marginal modeling + Kamino Gibbs/energy constraint enforcement.
    """
    rng = np.random.default_rng(seed)
    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    num_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number) and df[c].nunique() > 5]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    # Model numerical attributes
    gmm = None
    scaler = None
    if num_cols:
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(df[num_cols].fillna(df[num_cols].mean()).values)
        n_comp = min(5, max(1, len(df) // 10))
        gmm = GaussianMixture(n_components=n_comp, covariance_type="full", random_state=seed)
        gmm.fit(X_num_scaled)

    # Model categorical attributes
    cat_dist = {c: df[c].value_counts(normalize=True).to_dict() for c in cat_cols}
    target_dist = df[target_col].value_counts(normalize=True).to_dict()

    # Domain bounds
    constraints = {c: (float(df[c].min()), float(df[c].max())) for c in num_cols}

    ref_array = df.values.astype(float)
    col_idx_map = {c: i for i, c in enumerate(df.columns)}

    synth_rows = []
    # Generate in blocks
    while len(synth_rows) < n_synth:
        needed = n_synth - len(synth_rows)
        batch_n = min(needed * 2, 200)

        # 1. Base generation
        if gmm and scaler:
            num_scaled, _ = gmm.sample(batch_n)
            num_vals = scaler.inverse_transform(num_scaled)
        else:
            num_vals = np.zeros((batch_n, 0))

        # Build candidate DataFrame / array
        cand_df = pd.DataFrame(index=range(batch_n), columns=df.columns)
        for idx, c in enumerate(num_cols):
            c_min, c_max = constraints[c]
            cand_df[c] = np.clip(num_vals[:, idx], c_min, c_max)

        for c in cat_cols:
            keys = list(cat_dist[c].keys())
            probs = list(cat_dist[c].values())
            cand_df[c] = rng.choice(keys, size=batch_n, p=probs)

        t_keys = list(target_dist.keys())
        t_probs = list(target_dist.values())
        cand_df[target_col] = rng.choice(t_keys, size=batch_n, p=t_probs)

        cand_array = cand_df.values.astype(float)

        # 2. Evaluate violations and apply Kamino Gibbs energy acceptance
        # Penalty: E(x) = sum_k w_k * vios_k(x)
        # P_acc(x) = exp(-E(x))
        if len(compiled_dcs) > 0 and len(weights) > 0:
            vios = evaluate_candidate_violations(cand_array, ref_array, compiled_dcs)
            penalties = vios @ weights  # (batch_n,)

            accepted_mask = np.zeros(batch_n, dtype=bool)
            for i in range(batch_n):
                pen = penalties[i]
                if pen <= 1e-6:
                    accepted_mask[i] = True
                elif pen >= 1000.0:
                    # Hard violation -> reject
                    accepted_mask[i] = False
                else:
                    p_acc = np.exp(-pen)
                    accepted_mask[i] = rng.uniform(0.0, 1.0) < p_acc

            valid_cands = cand_df[accepted_mask]
            if len(valid_cands) > 0:
                synth_rows.extend(valid_cands.to_dict("records"))

            # Vectorized MCMC candidate repair for rejected rows
            rejected_indices = np.where(~accepted_mask)[0]
            if len(synth_rows) < n_synth and len(rejected_indices) > 0:
                selected_rej = rejected_indices[:needed]
                rej_df = cand_df.iloc[selected_rej].copy().reset_index(drop=True)
                best_pens = penalties[selected_rej].copy()
                best_rows = [rej_df.iloc[i].copy() for i in range(len(rej_df))]

                for _ in range(max_repair_tries):
                    if (best_pens <= 1e-6).all():
                        break
                    # Generate jittered batch for all candidates simultaneously
                    pert_df = pd.DataFrame([r.to_dict() for r in best_rows], columns=df.columns)
                    for c in num_cols:
                        std_c = df[c].std() or 1.0
                        jitter = rng.normal(0, 0.05 * std_c, size=len(pert_df))
                        c_min, c_max = constraints[c]
                        pert_df[c] = np.clip(pert_df[c].values + jitter, c_min, c_max)

                    pert_arr = pert_df.values.astype(float)
                    pert_vios = evaluate_candidate_violations(pert_arr, ref_array, compiled_dcs)
                    pert_pens = pert_vios @ weights

                    for i in range(len(best_rows)):
                        if pert_pens[i] < best_pens[i]:
                            best_pens[i] = pert_pens[i]
                            best_rows[i] = pert_df.iloc[i].copy()

                for r in best_rows:
                    synth_rows.append(r.to_dict())
                    if len(synth_rows) >= n_synth:
                        break
        else:
            synth_rows.extend(cand_df.to_dict("records"))

    res_df = pd.DataFrame(synth_rows[:n_synth], columns=df.columns)
    return res_df
