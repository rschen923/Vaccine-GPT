"""Evidence-preserving label construction and hard-gated candidate ranking."""
from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd


def _read_optional(path: str | Path | None, columns: list[str]) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def build_candidate_table(
    proteome: str | Path,
    *,
    known_antigens: str | Path | None = None,
    attenuation_evidence: str | Path | None = None,
    fitness: str | Path | None = None,
) -> pd.DataFrame:
    proteins = pd.read_csv(proteome, sep="\t")
    accession_col = "Entry" if "Entry" in proteins else "accession"
    out = proteins.rename(columns={accession_col: "gene_id"}).copy()
    out["known_antigen"] = False
    out["attenuation_grade"] = "unknown"
    out["in_vivo_depleted"] = False
    out["in_vitro_essential"] = False
    out["evidence_ids"] = ""
    antigens = _read_optional(known_antigens, ["gene_id", "evidence_id"])
    if not antigens.empty:
        out.loc[out.gene_id.isin(antigens.gene_id), "known_antigen"] = True
        evidence = antigens.groupby("gene_id").evidence_id.apply(lambda s: ";".join(map(str, s))).to_dict()
        out["evidence_ids"] = out.gene_id.map(evidence).fillna("")
    attenuation = _read_optional(attenuation_evidence, ["gene_id", "grade", "evidence_id", "in_vivo_depleted"])
    if not attenuation.empty:
        out = out.merge(
            attenuation.groupby("gene_id", as_index=False).agg(
                attenuation_grade=("grade", "max"),
                in_vivo_depleted=("in_vivo_depleted", "max"),
            ), on="gene_id", how="left", suffixes=("", "_evidence")
        )
        out["attenuation_grade"] = out["attenuation_grade_evidence"].fillna(out["attenuation_grade"])
        out["in_vivo_depleted"] = out["in_vivo_depleted_evidence"].fillna(False)
        out = out.drop(columns=["attenuation_grade_evidence", "in_vivo_depleted_evidence"])
    if fitness and Path(fitness).exists():
        f = pd.read_csv(fitness)
        if {"gene_id", "in_vitro_essential"}.issubset(f.columns):
            out = out.merge(f[["gene_id", "in_vitro_essential"]], on="gene_id", how="left", suffixes=("", "_fitness"))
            out["in_vitro_essential"] = out["in_vitro_essential_fitness"].fillna(False)
            out = out.drop(columns=["in_vitro_essential_fitness"])
    return out


def choose_thresholds(
    scores: pd.DataFrame,
    *,
    min_known_target_recall: float = 0.60,
    max_antigen_deletion_rate: float = 0.05,
    max_essential_deletion_rate: float = 0.02,
) -> dict[str, float]:
    required = {"attenuation_score", "immunogenicity_score", "essentiality_score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"missing score columns: {sorted(missing)}")
    best = None
    for tau_a in sorted(scores.attenuation_score.unique(), reverse=True):
        for tau_i in sorted(scores.immunogenicity_score.unique()):
            for tau_e in sorted(scores.essentiality_score.unique()):
                chosen = (scores.attenuation_score >= tau_a) & (scores.immunogenicity_score < tau_i) & (scores.essentiality_score < tau_e)
                n = int(chosen.sum())
                if n == 0:
                    continue
                positives = scores.get("known_target", pd.Series(False, index=scores.index))
                antigens = scores.get("known_antigen", pd.Series(False, index=scores.index))
                essentials = scores.get("in_vitro_essential", pd.Series(False, index=scores.index))
                recall = float((chosen & positives).sum() / max(int(positives.sum()), 1))
                antigen_rate = float((chosen & antigens).sum() / n)
                essential_rate = float((chosen & essentials).sum() / n)
                if recall >= min_known_target_recall and antigen_rate <= max_antigen_deletion_rate and essential_rate <= max_essential_deletion_rate:
                    key = (recall, -n, -antigen_rate)
                    if best is None or key > best[0]:
                        best = (key, {"tau_a": float(tau_a), "tau_i": float(tau_i), "tau_e": float(tau_e)})
    if best is None:
        raise ValueError("no thresholds satisfy the locked validation constraints")
    return best[1]


def write_jsonl(frame: pd.DataFrame, path: str | Path, schema_version: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in frame.to_dict(orient="records"):
            record["label_schema_version"] = schema_version
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

