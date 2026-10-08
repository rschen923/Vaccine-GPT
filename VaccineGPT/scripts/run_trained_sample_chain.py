from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_and_clean_data import read_jsonl
from scripts.run_real_sample import one_hot_batch
from scripts.train_real_sample_self_supervised import SelfSupervisedProteinEncoder
from shared.lineage import lineage_tag
from src_m2.models import M2Predictor
from src_m2.pipeline import decide_genome


def run(sample_path: Path, m1_checkpoint_path: Path, output_dir: Path, batch_size: int = 8) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = list(read_jsonl(sample_path))
    checkpoint = torch.load(m1_checkpoint_path, map_location="cpu", weights_only=True)
    m1 = SelfSupervisedProteinEncoder()
    m1.load_state_dict(checkpoint["state_dict"], strict=True)
    m1.eval()
    m2 = M2Predictor().eval()
    representations = []
    raw_logits = {head: [] for head in ("H1", "H2", "H3", "H4")}
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            residues, _ = one_hot_batch(batch)
            m1_output = m1(residues)
            representations.append(m1_output["z_gctx"].cpu())
            prediction = m2(m1_output["z_gctx"])
            for head in raw_logits:
                raw_logits[head].extend(prediction[head].cpu().tolist())
    z_gctx = torch.cat(representations)
    if len(z_gctx) != len(records):
        raise RuntimeError("not all sampled records passed through the trained M1 checkpoint")
    decision = decide_genome(
        gene_records=[
            {"gene_id": row["gene_id"], "evidence_level": "E4"}
            for row in records
        ],
        head_logits={},
        pace_b={},
        graph_edges=[],
        operon_neighbors={},
        interaction_neighbors={},
        safety_flags=None,
        calibration_predictions=[],
        calibration_targets=[],
        validated_model=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "trained_m1_real_sample_z_gctx.pt"
    torch.save(
        {
            "gene_ids": [row["gene_id"] for row in records],
            "z_gctx": z_gctx,
            "m1_checkpoint_sha256": hashlib.sha256(m1_checkpoint_path.read_bytes()).hexdigest(),
            "feature_role": "self-supervised one-hot-smoke embedding; not ESM3 or target-trained",
        },
        embedding_path,
    )
    aep_path = output_dir / "trained_m1_unranked_aep.jsonl"
    with aep_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in decision["aep_rows"]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "vaccinegpt-trained-sample-chain-1.0",
        "lineage": lineage_tag(
            "VaccineGPT-v1-selfsup-smoke",
            "FEAT-real-protein-crop-onehot-1.0",
            "LABEL-none-selfsup-1.0",
            f"REAL-CHAIN-{len(records)}",
        ),
        "input_records": len(records),
        "trained_m1_checkpoint": str(m1_checkpoint_path),
        "trained_m1_checkpoint_sha256": hashlib.sha256(m1_checkpoint_path.read_bytes()).hexdigest(),
        "m1_z_gctx_shape": list(z_gctx.shape),
        "m2_logits_evaluated_count": len(raw_logits["H1"]),
        "m2_checkpoint_loaded": False,
        "m2_status": "random_untrained_head_used_only_to_verify_forward_interface",
        "decision_gate": "validated_model=false; all recommendations remain unranked",
        "aep_rows": len(decision["aep_rows"]),
        "biological_prediction_status": "not_evaluated",
        "outputs": {
            "z_gctx": str(embedding_path),
            "aep": str(aep_path),
        },
    }
    (output_dir / "trained_m1_sample_chain_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pass the real 1,000-protein sample through a trained M1 checkpoint and gated M2 forward."
    )
    parser.add_argument("--sample-jsonl", type=Path, required=True)
    parser.add_argument("--m1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.sample_jsonl, args.m1_checkpoint, args.output_dir, args.batch_size),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
