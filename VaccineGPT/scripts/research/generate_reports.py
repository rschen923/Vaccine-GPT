"""Generate auditable report shells from local manifests and validation results.

This generator deliberately reports missing evidence as missing. It never
turns an unmet literature quota into a completed review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build(output: Path, literature_dir: Path, qa_path: Path | None = None) -> None:
    qa = read_json(qa_path or literature_dir / "qa_report.json", {})
    quota = qa.get("quotas", {})
    sections = quota.get("sections", {})
    section_rows = "\n".join(
        f"| {name} | {row.get('included', 0)} | {row.get('reviews', 0)} | "
        f"{row.get('research', 0)} | {'PASS' if row.get('ok') else 'PENDING'} |"
        for name, row in sorted(sections.items())
    ) or "| No retrieved records | 0 | 0 | 0 | PENDING |"
    write(
        output / "01_literature_evidence_status.md",
        f"""# Vaccine-GPT literature evidence status

## Scope

The planned evidence window is **2024-09-19 through 2026-09-19**, with formal
publications preferred and preprints labelled separately. The target is at
least 200 papers for the mathematical/model audit and at least 100 papers for
the data/experiment audit. Each configured section requires at least 20
included papers and at least 5 reviews plus 5 research papers.

This file is an automatically generated status report. It is not a completed
literature review unless all quota rows pass and every claim is linked to an
evidence record.

## Retrieval status

- Papers currently present: **{qa.get('record_count', 0)}**
- Date-invalid records: **{len(qa.get('invalid_dates', []))}**
- Duplicate identifiers: **{len(qa.get('duplicate_keys', []))}**
- Missing provenance: **{len(qa.get('missing_provenance', []))}**
- Overall quota status: **{'PASS' if quota.get('ok') else 'PENDING'}**

| Section | Included | Reviews | Research | Status |
|---|---:|---:|---:|---|
{section_rows}

## Required interpretation

An unmet quota is a research blocker, not permission to pad the corpus.
Network failures and unavailable full text must remain in the retrieval
manifest. Mathematical claims must be labelled as theorem/proof, cited
empirical result, derived proposition, numerical check, hypothesis, or
unresolved conflict.
""",
    )
    write(
        output / "02_architecture_audit.md",
        """# Vaccine-GPT architecture audit and optimization

## Findings addressed in code

1. The authoritative five-layer path now validates batch dimensions, square
   PPI adjacency, node features, and the required feature widths.
2. `node_x` and `PPI_adj` now affect the shared state through a normalized
   graph message, rather than being required-but-unused fields.
3. `S_adj`, `t4_feat`, `N_star`, `ic50`, and `t_obs` now participate in forward
   or loss computation.
4. The primary cross-layer route remains detached and follows
   `tau_TI -> v0`; alpha coupling remains bounded and disabled by default.
5. A contract test checks finite losses, output shape, and graph signal use.

## Risks remaining for evidence review

- Synthetic checks establish software invariants only; they do not establish
  biological validity or task-level performance.
- The current graph implementation assumes one graph per batch. A production
  graph minibatcher must be added when samples have variable node counts.
- The model's mechanistic equations are regularizers and hypotheses until each
  equation is linked to a valid source and its domain assumptions are tested.
- Calibration, uncertainty intervals, external validation, and ablations must
  be completed with real data.

## Validation commands

```powershell
python train.py --smoke
python train.py --epochs 8 --stage-epochs 1
python verification\\verify_math.py
python verification\\verify_l4.py
python verification\\contract.py
```
""",
    )
    write(
        output / "03_data_experiment_and_compute_plan.md",
        """# Public data, experiments, and Tencent Cloud GPU plan

## Data plan

Use versioned, checksum-backed public sources already represented by the
repository manifests, then map each source to the authoritative tensor
contract. Candidate source families include RefSeq/UniProt for sequence,
STRING/other PPI resources for graph edges, VFDB/OGEE/DEG for virulence and
essentiality labels, IEDB for epitope/immunogenicity evidence, and SRA or
single-cell repositories for expression and immune-state validation. License,
release, identifier mapping, negative controls, homology overlap, batch, and
missing-label policy must be recorded per source.

## Computational experiments

1. Grouped strain/species split with homology-aware leakage audit.
2. External-source evaluation held out by source and organism.
3. Ablations for graph signal, each detached coupling, each physical
   consistency term, foundation representation, and PCGrad.
4. Calibration and uncertainty evaluation using held-out confidence labels.
5. Seed and hyperparameter sensitivity with pre-registered acceptance metrics.

## Experimental validation

Before wet-lab execution, obtain institutional biosafety and ethics approval.
Use non-pathogenic or approved strains, randomized treatment assignment,
negative/vehicle controls, positive controls, blinded readout where feasible,
biological replicates, and pre-specified endpoints. Validate predictions with
orthogonal assays rather than treating model scores as mechanistic proof.
Exact sample sizes and assay parameters must be populated from included
recent full-text studies in the evidence table.

## Compute estimate (planning estimate, not a Tencent quotation)

Start with one 40 GB-class GPU for smoke, pilot, ablations, and one-seed
training. Reserve multi-GPU only for the L0 representation pretraining stage
after profiling. Record GPU model, hours, memory peak, storage, data transfer,
checkpoint count, and retry overhead. The executable training commands are:

```powershell
python train.py --data path\\to\\contract_batch.jsonl --epochs 8 --stage-epochs 1 --checkpoint checkpoints\\five_layer.pt
```

The final budget must be recalculated from measured throughput and the chosen
Tencent Cloud instance price at procurement time; this repository does not
invent a current price.
""",
    )
    write(
        output / "04_formula_ledger_template.md",
        """# Formula and theorem audit ledger

Populate one row per equation or theorem after screening the cited full text.

| ID | Code location | Formula/theorem | Type | Assumptions | Derivation/proof source | Domain/units | Numerical check | Contradiction | Status |
|---|---|---|---|---|---|---|---|---|---|
| F-001 | `vaccinegpt/five_layer.py` | `v0 = f(tau_TI)` | derived model coupling | bounded tau, detached upstream state | cite included source or mark hypothesis | dimensionless/declared units | `verification/contract.py` | search included evidence | pending |
| F-002 | `vaccinegpt/losses.py` | non-negative PU risk | cited/derived loss | class-prior assumptions | cite original method | scalar risk | `verification/verify_math.py` | compare competing PU results | pending |
| F-003 | `vaccinegpt/l0_clef.py` | VIB KL | cited objective | Gaussian posterior/prior | cite source | scalar objective | finite-gradient test | report conflicts | pending |

No row may be marked proven solely because the implementation runs.
""",
    )
    formula_rows = [
        {
            "id": "F-001",
            "location": "vaccinegpt/five_layer.py:80",
            "formula": "v0 = clip(0.80 + 0.25 * (tau_TI / 180 - 1), 0.05, 0.99)",
            "type": "derived coupling hypothesis",
            "assumptions": ["tau_TI is finite", "v0 is bounded", "upstream state is detached"],
            "evidence_status": "pending_full_text_formula_audit",
            "biological_claim": False,
        },
        {
            "id": "F-002",
            "location": "vaccinegpt/l0_clef.py:18-39",
            "formula": "KL(q(z|x)||p(z)) with Gaussian posterior",
            "type": "standard variational objective",
            "assumptions": ["Gaussian posterior/prior", "finite log variance"],
            "evidence_status": "implementation_checked; citation_mapping_pending",
            "biological_claim": False,
        },
        {
            "id": "F-003",
            "location": "vaccinegpt/losses.py:50-66",
            "formula": "non-negative PU risk",
            "type": "learning objective",
            "assumptions": ["positive class prior in (0,1)", "unlabeled distribution assumptions"],
            "evidence_status": "implementation_checked; citation_mapping_pending",
            "biological_claim": False,
        },
        {
            "id": "F-004",
            "location": "vaccinegpt/physiology.py:13-17",
            "formula": "finite-difference ODE consistency penalty",
            "type": "numerical regularizer",
            "assumptions": ["trajectory step is nonzero", "RHS matches state convention"],
            "evidence_status": "numerical_invariant_checked; mechanistic_validity_pending",
            "biological_claim": False,
        },
        {
            "id": "F-005",
            "location": "vaccinegpt/five_layer.py:67-88",
            "formula": "detached graph and layer coupling",
            "type": "optimization constraint",
            "assumptions": ["one graph per batch", "detachment is intentional"],
            "evidence_status": "gradient_boundary_checked",
            "biological_claim": False,
        },
    ]
    (output / "formula_ledger.json").write_text(
        json.dumps(formula_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "report_manifest.json":
            manifest.append(
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
            )
    (output / "report_manifest.json").write_text(
        json.dumps({"schema_version": "report-manifest/v1", "artifacts": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--literature", type=Path, default=Path("literature-output"))
    parser.add_argument("--qa", type=Path)
    args = parser.parse_args()
    build(args.output, args.literature, args.qa)
    print(f"reports written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
