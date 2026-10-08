from __future__ import annotations

import unittest
import sys
import json
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_quality import classify_label_evidence, normalize_protein
from shared.contracts import validate_feature, validate_label
from shared.dataset import merge_annotations
from src_m1.models import (
    M1Encoder,
    freeze_for_species_adaptation as freeze_m1_for_species_adaptation,
    info_nce_loss,
)
from src_m1.clef_compat import (
    CLEFProteinClassifier,
    CLEFSequenceEncoder,
    load_clef_weights,
)
from src_m1.encoders.foundation import (
    CLEFCheckpointAdapter,
    ESM2Adapter,
    ESM3Adapter,
    FoundationModelConfig,
    FoundationFeatureStore,
)
from src_m1.pipeline import FoundationM1Encoder
from src_m1.pretrained import build_pretrained_adapters
from src_m1.graph import build_gene_graph
from src_m1.training import m1_contrastive_objective
from src_m1.genome_v2 import GenomeWindow, make_dna_windows
from src_m2.coverage import calculate_multivalent_coverage
from src_m2.aep import unavailable_aep_row, validate_aep_row
from src_m2.decision import (
    acquisition_scores,
    attention_position_scores,
    antigen_retention,
    conformal_interval,
    fit_pace,
    gradient_contributions,
    normalized_betweenness,
    pareto_tiers,
    prototype_scores,
    safe_score,
)
from src_m2.h5 import implementation_route
from src_m2.models import M2Predictor, SpeciesAdapter, freeze_for_species_adaptation
from src_m2.pipeline import decide_genome
from src_m2.training import fit_supervised_heads
from scripts.train_real_sample_self_supervised import train as train_real_selfsup
from scripts.extract_target_proteome import extract_proteome
from scripts.train_h3_public_ranker import _has_explicit_non_t3ss_pathway
from scripts import predict_clef_effectors
from scripts import extract_esm2_embeddings
from scripts.package_inkstone_dataset import SOURCE_PATHS, package_dataset
from scripts.train_m1_contrastive import validate_esm3_cache_contract


class ArchitectureTests(unittest.TestCase):
    def test_clef_encoder_classifier_and_strict_checkpoint_loading(self):
        encoder = CLEFSequenceEncoder(num_embeds=16, num_hiddens=8, max_length=8)
        batch = {
            "esm_feature": torch.randn(2, 8, 16),
            "valid_lens": torch.tensor([5, 7]),
        }
        residue_values, embeddings = encoder(batch, return_residue_representations=True)
        pooled_values, pooled_embeddings = encoder(batch)
        self.assertEqual(tuple(residue_values.shape), (2, 8, 16))
        self.assertEqual(tuple(embeddings.shape), (2, 8))
        self.assertEqual(tuple(pooled_values.shape), (2, 16))
        self.assertEqual(tuple(pooled_embeddings.shape), (2, 8))
        scores = CLEFProteinClassifier(num_embeds=8)(embeddings)
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.all((scores >= 0) & (scores <= 1)))

        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "clef.pt"
            sequence_state = encoder.state_dict()
            sequence_state["feat_encoder.0.weight"] = torch.ones(4, 5)
            sequence_state["ln_f.weight"] = torch.ones(8)
            torch.save(sequence_state, checkpoint)
            restored = CLEFSequenceEncoder(num_embeds=16, num_hiddens=8, max_length=8)
            self.assertEqual(len(load_clef_weights(restored, checkpoint)), 64)
            self.assertTrue(
                all(
                    torch.equal(source, target)
                    for source, target in zip(encoder.parameters(), restored.parameters())
                )
            )
            adapter = CLEFCheckpointAdapter(
                FoundationModelConfig(
                    model_id="CLEF",
                    local_path=str(checkpoint),
                    device="cpu",
                    extra={"init_kwargs": {"num_embeds": 16, "num_hiddens": 8, "max_length": 8}},
                ),
                "src_m1.clef_compat",
                "CLEFSequenceEncoder",
            )
            self.assertEqual(tuple(adapter.encode(batch).shape), (2, 8))
            with self.assertRaisesRegex(ValueError, "requires a local pretrained checkpoint"):
                CLEFCheckpointAdapter(
                    FoundationModelConfig(model_id="CLEF", device="cpu"),
                    "src_m1.clef_compat",
                    "CLEFSequenceEncoder",
                ).load()
            torch.save({"unexpected.weight": torch.ones(1)}, checkpoint)
            with self.assertRaisesRegex(ValueError, "incompatible CLEF checkpoint"):
                load_clef_weights(restored, checkpoint)

    def test_esm2_token_encoding_preserves_special_tokens_and_residue_lengths(self):
        class FakeAlphabet:
            padding_idx = 0

            @staticmethod
            def get_batch_converter():
                def convert(batch):
                    tokens = torch.zeros((len(batch), 5), dtype=torch.long)
                    for index, (_, sequence) in enumerate(batch):
                        tokens[index, 0] = 1
                        tokens[index, 1 : len(sequence) + 1] = 3
                        tokens[index, len(sequence) + 1] = 2
                    return None, None, tokens

                return convert

        class FakeESM2(torch.nn.Module):
            num_layers = 1

            def forward(self, tokens, repr_layers):
                return {
                    "representations": {
                        self.num_layers: tokens.float().unsqueeze(-1).repeat(1, 1, 4)
                    }
                }

        adapter = ESM2Adapter(FoundationModelConfig(model_id="mock", device="cpu"))
        adapter.model = FakeESM2()
        adapter.alphabet = FakeAlphabet()
        residue_vectors, residue_lengths = adapter.encode_tokens(["ACD", "A"])
        special_vectors, special_lengths = adapter.encode_tokens_with_special_tokens(
            ["ACD", "A"]
        )
        self.assertEqual(tuple(residue_vectors.shape), (2, 3, 4))
        self.assertEqual(tuple(special_vectors.shape), (2, 5, 4))
        self.assertEqual(residue_lengths.tolist(), [3, 1])
        self.assertEqual(special_lengths.tolist(), [5, 3])

    def test_esm3_residue_adapter_enforces_1536_dimension(self):
        from types import ModuleType, SimpleNamespace

        class FakeESM3:
            @staticmethod
            def encode(protein):
                return protein

            @staticmethod
            def logits(protein, config):
                return SimpleNamespace(
                    embeddings=torch.zeros(1, len(protein.sequence) + 2, 1536)
                )

        class FakeProtein:
            def __init__(self, sequence):
                self.sequence = sequence

        class FakeLogitsConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        esm_module = ModuleType("esm")
        esm_module.__path__ = []
        sdk_module = ModuleType("esm.sdk")
        sdk_module.__path__ = []
        api_module = ModuleType("esm.sdk.api")
        api_module.ESMProtein = FakeProtein
        api_module.LogitsConfig = FakeLogitsConfig
        adapter = ESM3Adapter(
            FoundationModelConfig(
                model_id="esm3-sm-open-v1",
                device="cpu",
                extra={"embedding_dimension": 1536},
            )
        )
        adapter.model = FakeESM3()
        with patch.dict(
            sys.modules,
            {
                "esm": esm_module,
                "esm.sdk": sdk_module,
                "esm.sdk.api": api_module,
            },
        ):
            vectors, lengths = adapter.encode_tokens(["ACDE", "FG"])
        self.assertEqual(tuple(vectors.shape), (2, 4, 1536))
        self.assertEqual(lengths.tolist(), [4, 2])

    def test_primary_pretrained_protein_adapter_is_esm3(self):
        config_path = (
            Path(__file__).resolve().parents[1] / "configs" / "pretrained_models.json"
        )
        adapters = build_pretrained_adapters(json.loads(config_path.read_text(encoding="utf-8")))
        self.assertIsInstance(adapters["protein"], ESM3Adapter)
        self.assertEqual(adapters["protein"].config.model_id, "esm3-sm-open-v1")
        self.assertEqual(adapters["protein"].config.extra["embedding_dimension"], 1536)
        foundation_config_path = (
            Path(__file__).resolve().parents[1] / "configs" / "foundation_models.json"
        )
        foundation_adapters = build_pretrained_adapters(
            json.loads(foundation_config_path.read_text(encoding="utf-8"))
        )
        self.assertIsInstance(foundation_adapters["protein"], ESM3Adapter)
        self.assertEqual(
            foundation_adapters["protein"].config.extra["embedding_dimension"], 1536
        )

    def test_foundation_m1_uses_adapter_dimension_and_rejects_mismatch(self):
        class FakeAdapter:
            config = FoundationModelConfig(
                model_id="esm3-sm-open-v1",
                extra={"embedding_dimension": 1536},
            )

            def __init__(self, dimension=1536):
                self.dimension = dimension

            def encode_tokens(self, sequences):
                return (
                    torch.zeros(len(sequences), 4, self.dimension),
                    torch.tensor([4] * len(sequences)),
                )

        encoder = FoundationM1Encoder(FoundationFeatureStore({"protein": FakeAdapter()}))
        self.assertEqual(encoder.residue_dim, 1536)
        self.assertEqual(encoder(["ACDE"])["z_pub"].shape, (1, 128))
        wrong_encoder = FoundationM1Encoder(
            FoundationFeatureStore({"protein": FakeAdapter(dimension=1280)})
        )
        with self.assertRaisesRegex(ValueError, "adapter returned dimension 1280"):
            wrong_encoder(["ACDE"])

    def test_esm3_embedding_cache_contract_rejects_legacy_esm2(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "batch.pt"
            batch = {
                "backend": "esm3",
                "model_id": "esm3-sm-open-v1",
                "embedding_dimension": 1536,
                "residue_embeddings": torch.zeros(2, 4, 1536),
            }
            torch.save(batch, path)
            validate_esm3_cache_contract([path])
            batch.update(
                backend="esm2",
                model_id="esm2_t33_650M_UR50D",
                embedding_dimension=1280,
                residue_embeddings=torch.zeros(2, 4, 1280),
            )
            torch.save(batch, path)
            with self.assertRaisesRegex(ValueError, "expected esm3-sm-open-v1"):
                validate_esm3_cache_contract([path])

    def test_esm3_feature_extraction_records_backbone_and_dimension(self):
        class FakeESM3Adapter:
            def __init__(self, config):
                self.config = config

            def load(self):
                return self

            def encode_tokens(self, sequences):
                lengths = torch.tensor([len(sequence) for sequence in sequences])
                return torch.zeros(len(sequences), max(lengths).item(), 1536), lengths

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "gene_id": "g1",
                        "protein_sequence": "ACDE",
                        "genome_id": "genome-a",
                        "split_group_id": "cluster-a",
                        "split_group_method": "mmseqs2_90",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(extract_esm2_embeddings, "ESM3Adapter", FakeESM3Adapter), patch.object(
                extract_esm2_embeddings, "version", return_value="3.4.post1"
            ):
                manifest = extract_esm2_embeddings.extract(
                    input_path, root / "cache", batch_id="TEST-BATCH"
                )
            self.assertEqual(manifest["backend"], "esm3")
            self.assertEqual(manifest["embedding_dimension"], 1536)
            self.assertEqual(manifest["runtime_package_version"], "3.4.post1")
            batch = torch.load(
                root / "cache" / "esm3_embeddings_000000.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(batch["residue_embeddings"].shape[-1], 1536)

    def test_clef_prediction_writes_rankings_and_keeps_omv_unscored(self):
        class FakeESM2Adapter:
            def __init__(self, config):
                self.config = config

            def load(self):
                return self

            def encode_tokens_with_special_tokens(self, sequences):
                lengths = torch.tensor([len(sequence) + 2 for sequence in sequences])
                tokens = torch.randn(len(sequences), max(lengths).item(), 1280)
                return tokens, lengths

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proteome_path = root / "proteome.jsonl"
            positive_path = root / "positives.jsonl"
            esm_path = root / "esm.pt"
            weights_dir = root / "clef"
            output_dir = root / "output"
            weights_dir.mkdir()
            sequences = {"g1": "ACDE", "g2": "FGHI"}
            with proteome_path.open("w", encoding="utf-8") as stream:
                for gene_id, sequence in sequences.items():
                    stream.write(
                        json.dumps(
                            {
                                "gene_id": gene_id,
                                "protein_sequence": sequence,
                                "gene_symbol": gene_id,
                                "product": "test protein",
                            }
                        )
                        + "\n"
                    )
            with positive_path.open("w", encoding="utf-8") as stream:
                for gene_id, pathway in (("g1", "T3SS"), ("g2", "OMV")):
                    stream.write(
                        json.dumps(
                            {
                                "gene_id": gene_id,
                                "task": "H3",
                                "label": 1,
                                "secretion_or_translocation_pathway": pathway,
                                "sequence_sha256": hashlib.sha256(
                                    sequences[gene_id].encode("ascii")
                                ).hexdigest(),
                            }
                        )
                        + "\n"
                    )
            esm_path.write_bytes(b"mock ESM checkpoint")
            for name in {
                spec["encoder"] for spec in predict_clef_effectors.CLEF_TASKS.values()
            } | {
                spec["classifier"] for spec in predict_clef_effectors.CLEF_TASKS.values()
            }:
                (weights_dir / name).write_bytes(b"mock CLEF checkpoint")

            with patch.object(predict_clef_effectors, "ESM2Adapter", FakeESM2Adapter), patch.object(
                predict_clef_effectors, "load_clef_weights", return_value="0" * 64
            ):
                report = predict_clef_effectors.predict(
                    proteome_path,
                    positive_path,
                    esm_path,
                    weights_dir,
                    output_dir,
                    batch_size=2,
                    top_k=1,
                    device_name="cpu",
                )

            self.assertEqual(report["task_reports"]["T3SS"]["matching_curated_positive_count"], 1)
            self.assertEqual(report["omv_positives_retained_unscored"], 1)
            full_rows = (output_dir / "eib202_h3_clef_effector_predictions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(full_rows), 2)
            omv_rows = (
                output_dir / "eib202_h3_clef_out_of_scope_positives.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(omv_rows), 1)

    def test_t3ss_ranking_excludes_only_explicitly_different_pathways(self):
        self.assertTrue(
            _has_explicit_non_t3ss_pathway(
                {"secretion_or_translocation_pathway": "Type VI secretion system"}
            )
        )
        self.assertFalse(
            _has_explicit_non_t3ss_pathway(
                {"secretion_or_translocation_pathway": "Type III secretion system"}
            )
        )
        self.assertFalse(_has_explicit_non_t3ss_pathway({}))

    def test_target_proteome_extraction_preserves_translation_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            genbank_path = root / "fixture.gb"
            genbank_path.write_text(
                'LOCUS       CP001135             12 bp    DNA     circular BCT\n'
                'VERSION     CP001135.1\n'
                'FEATURES             Location/Qualifiers\n'
                '     CDS             1..12\n'
                '                     /gene="test"\n'
                '                     /locus_tag="ETAE_0001"\n'
                '                     /translation="MKW\n'
                '                     V"\n'
                'ORIGIN\n'
                '        1 atgaaatgggtt\n'
                '//\n',
                encoding="ascii",
            )
            output = root / "proteome.jsonl"
            report = extract_proteome(genbank_path, output)
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["gene_id"], "ETAE_0001")
            self.assertEqual(row["sequence"], "MKWV")
            self.assertEqual(row["genome_accession"], "CP001135.1")
            self.assertEqual(row["label_status"], "unlabelled")
            self.assertEqual(report["records"], 1)
            self.assertEqual(report["exact_duplicate_sequences"], 0)

    def test_m1_protein_contract_and_m2_heads(self):
        torch.manual_seed(42)
        model = M1Encoder(residue_dim=20, max_length=32)
        output = model(
            torch.randn(4, 32, 20),
            torch.ones(4, 32, dtype=torch.bool),
        )
        self.assertEqual(tuple(output["z_pub"].shape), (4, 128))
        self.assertEqual(tuple(output["z_exp"].shape), (4, 128))
        self.assertEqual(tuple(output["z_gctx"].shape), (4, 128))
        self.assertEqual(tuple(output["position_scores"].shape), (4, 32))
        predictions = M2Predictor()(output["z_gctx"])
        self.assertEqual(set(predictions), {"H1", "H2", "H3", "H4"})

    def test_species_adapter_is_identity_initialized_and_freezes_shared_model(self):
        values = torch.randn(3, 128)
        scale_adapter = SpeciesAdapter()
        self.assertTrue(torch.equal(scale_adapter(values), values))
        self.assertEqual(sum(parameter.numel() for parameter in scale_adapter.parameters()), 128)

        lora_adapter = SpeciesAdapter(mode="lora", rank=8)
        self.assertTrue(torch.equal(lora_adapter(values), values))
        self.assertEqual(sum(parameter.numel() for parameter in lora_adapter.parameters()), 2048)

        model = M2Predictor(species_adapter_mode="scale")
        freeze_for_species_adaptation(model)
        self.assertTrue(all(parameter.requires_grad for parameter in model.species_adapter.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.task_heads.parameters()))

    def test_m1_species_adaptation_freezes_large_private_projection_stack(self):
        model = M1Encoder()
        trainable_count = freeze_m1_for_species_adaptation(model)
        self.assertEqual(model.encoder_a.input_dim, 1536)
        self.assertEqual(trainable_count, 1536)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            1536,
        )
        self.assertTrue(model.private_adapter.delta_scale.requires_grad)
        self.assertFalse(model.private_adapter.sequence_projection.weight.requires_grad)
        self.assertFalse(model.private_adapter.private_projection[1].weight.requires_grad)

    def test_info_nce_restricts_private_negatives_by_genome(self):
        anchors = torch.randn(4, 8)
        positives = torch.randn(4, 8)
        result = info_nce_loss(anchors, positives, group_ids=["g1", "g1", "g2", "g2"])
        self.assertTrue(torch.isfinite(result))

    def test_m1_training_uses_paired_tracks_and_graph_edges(self):
        model = M1Encoder(
            residue_dim=20,
            public_modality_dims={"annotation": 8},
        )
        batch = {
            "residue_embeddings": torch.randn(4, 16, 20),
            "sequence_mask": torch.ones(4, 16, dtype=torch.bool),
            "group_ids": ["g1", "g1", "g2", "g2"],
            "public_features": {"annotation": torch.randn(4, 8)},
            "public_feature_masks": {"annotation": torch.ones(4, dtype=torch.bool)},
            "private_features": torch.randn(4, 14),
            "private_feature_mask": torch.ones(4, 14, dtype=torch.bool),
            "edge_index": torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            "edge_type": torch.tensor([0, 0, 0, 0]),
            "edge_weight": torch.ones(4),
        }
        loss = m1_contrastive_objective(model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_pace_fit_and_singular_detection(self):
        fitted = fit_pace([0, 1, 2], [1.0, 0.5, 0.0])
        self.assertEqual(fitted["status"], "fitted")
        self.assertLess(fitted["b"], 0)
        self.assertEqual(fit_pace([1, 1, 1], [1, 2, 3])["status"], "not_fitted")

    def test_conformal_widens_small_calibration_and_unavailable_is_full_interval(self):
        small = conformal_interval(0.5, [0.4, 0.6], [0.5, 0.5])
        self.assertEqual(small["status"], "small_sample_widened")
        self.assertEqual(conformal_interval(None, [], [])["upper"], 1.0)

    def test_graph_retention_and_centrality(self):
        centrality = normalized_betweenness(["a", "b", "c"], [("a", "b"), ("b", "c")])
        self.assertGreater(centrality["b"], centrality["a"])
        score = antigen_retention(
            "a",
            {"a": 0.8, "b": 0.5},
            {"a": ["b"]},
            {},
            centrality,
        )
        self.assertIsNotNone(score)

    def test_pareto_and_veto(self):
        results = pareto_tiers(
            [
                {"gene_id": "a", "att_estimate": 0.8, "ret": 0.8, "safe": 1.0, "consensus_count": 3},
                {"gene_id": "b", "att_estimate": 0.4, "ret": 0.4, "safe": 1.0, "consensus_count": 0},
                {"gene_id": "c", "att_estimate": 0.9, "ret": 0.9, "safe": 1.0, "h4_probability": 0.8},
            ]
        )
        self.assertEqual([row["tier"] for row in results], ["A", "C", "excluded_h4_core_essential"])

    def test_evidence_does_not_promote_source_name_to_gold(self):
        result = classify_label_evidence({"source": "VFDB", "assay": "", "label": 1})
        self.assertEqual(result["evidence_level"], "E4")
        self.assertFalse(result["training_eligible_as_gold"])

    def test_conflicting_evidence_is_preserved_and_quarantined(self):
        rows = merge_annotations(
            [
                {"gene_id": "g", "task": "H1", "label": 1.0, "source": "private", "is_private": True, "assay": "CRISPRi"},
                {"gene_id": "g", "task": "H1", "label": 0.0, "source": "private", "is_private": True, "assay": "CRISPRi"},
            ],
            "H1",
            {"model_version": "m", "feature_version": "f", "label_version": "l", "batch_id": "b"},
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["label_conflict"] for row in rows))
        self.assertTrue(all(row["training_role"] == "quarantine_conflict" for row in rows))

    def test_contract_rejects_dna_on_v1_and_weak_as_gold(self):
        lineage = {"model_version": "m", "feature_version": "f", "label_version": "l", "batch_id": "b"}
        with self.assertRaises(ValueError):
            validate_feature(
                {"gene_id": "g", "view": "dna", "track": "v1_protein", "embedding": [0.0], "lineage": lineage}
            )
        with self.assertRaises(ValueError):
            validate_label(
                {
                    "gene_id": "g",
                    "task": "H1",
                    "label": 1,
                    "evidence_level": "E4",
                    "training_eligible_as_gold": True,
                    "lineage": lineage,
                }
            )

    def test_invalid_sequence_is_reviewed_not_silently_accepted(self):
        result = normalize_protein("ACD*?")
        self.assertEqual(result["quality_status"], "invalid")
        self.assertIn("invalid_characters", result["quality_reasons"])

    def test_safety_requires_all_four_flags(self):
        self.assertIsNone(safe_score({"mge": False})["score"])
        self.assertEqual(
            safe_score(
                {"mge": False, "amr": False, "phase_variation": True, "toxicity": False}
            )["score"],
            0.8,
        )

    def test_v2_window_and_h5_fallback_are_explicit(self):
        windows = make_dna_windows("g1", "ACGT" * 32, window_size=64, overlap=8)
        self.assertEqual(windows[0].element_type, "unclassified_window")
        self.assertEqual(
            implementation_route(0.8, 4000, 0, model_trained=True)["route"],
            "sacB_two_step_exchange",
        )
        self.assertIsNone(implementation_route(None, None, None, False)["route"])

    def test_coverage_needs_all_external_probabilities(self):
        result = calculate_multivalent_coverage(
            ["a"],
            {"s1": 1.0},
            {"a": {"s1": 0.9}},
            {"a": {"s1": 1.0}},
            {"a": {"s1": (0.5, 0.8)}},
        )
        self.assertAlmostEqual(result["coverage"]["lower"], 0.45)
        self.assertAlmostEqual(result["coverage"]["upper"], 0.72)

    def test_aep_contract_and_m3_read_only_interface(self):
        row = unavailable_aep_row("gene-a", "unvalidated model")
        validate_aep_row(row, "v1")
        self.assertEqual(row["implementation_path"], "")

    def test_unvalidated_decision_pipeline_never_ranks_logits(self):
        result = decide_genome(
            [{"gene_id": "g1", "evidence_level": "E4"}],
            head_logits={},
            pace_b={},
            graph_edges=[],
            operon_neighbors={},
            interaction_neighbors={},
            safety_flags=None,
            calibration_predictions=[],
            calibration_targets=[],
        )
        self.assertEqual(result["aep_rows"][0]["tier"], "unranked")
        self.assertEqual(result["virulence_gene_list"], [])
        self.assertEqual(result["biological_use_status"], "blocked_model_not_validated")

    def test_m2_supervised_training_refuses_E4_weak_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "no E1/E2 labels"):
                fit_supervised_heads(
                    torch.randn(12, 128),
                    {"H1": torch.tensor([0.0, 1.0] * 6)},
                    {"H1": ["E4"] * 12},
                    [f"mmseqs-{index}" for index in range(12)],
                    temp,
                    epochs=1,
                    batch_id="TEST-M2-WEAK",
                )

    def test_m2_supervised_training_refuses_positive_only_binary_head(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "positive-only labels"):
                fit_supervised_heads(
                    torch.randn(24, 128),
                    {"H3": torch.ones(24)},
                    {"H3": ["E1"] * 24},
                    [f"mmseqs-{index}" for index in range(24)],
                    temp,
                    epochs=1,
                    batch_id="TEST-M2-POSITIVE-ONLY",
                )

    def test_real_sequence_self_supervised_training_runs_and_reports_no_biological_claim(self):
        rows = [
            {
                "gene_id": f"g{i}",
                "sequence": ("ACDEFGHIKLMNPQRSTVWY" * 3)[:45] + "A" * i,
                "sequence_sha256": f"hash-{i}",
            }
            for i in range(40)
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "sample.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = train_real_selfsup(
                input_path,
                root / "training",
                seed=17,
                epochs=1,
                batch_size=4,
                learning_rate=1e-4,
            )
            self.assertEqual(report["sample_count"], 40)
            self.assertFalse(report["pretrained_weights_used"])
            self.assertIn("no E1/E2 gold labels", report["supervised_m2_training"])
            self.assertTrue((root / "training" / "m1_real_sample_self_supervised.pt").is_file())

    def test_inkstone_packager_excludes_human_data_and_model_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo_root = root / "repository"
            data_root = root / "data"
            repo_root.mkdir()
            for relative in SOURCE_PATHS:
                (data_root / relative).mkdir(parents=True, exist_ok=True)

            genome = data_root / "refseq_bacteria" / "assembly.fna"
            genome.write_text(">gene\nACGT\n", encoding="utf-8")
            iedb = data_root / "raw" / "iedb" / "human_assay.json"
            iedb.parent.mkdir(parents=True)
            iedb.write_text('{"source":"IEDB"}\n', encoding="utf-8")
            raw_bacfitbase = (
                data_root
                / "raw"
                / "public_training_sources"
                / "reference_databases"
                / "BacFITBase"
                / "bacfitbase_v1.zip"
            )
            raw_bacfitbase.parent.mkdir(parents=True)
            raw_bacfitbase.write_bytes(b"unfiltered human-host records")
            literature = (
                data_root
                / "raw"
                / "public_training_sources"
                / "m2_public_candidates"
                / "PMC1"
                / "article.fullTextXML.xml"
            )
            literature.parent.mkdir(parents=True)
            literature.write_text("<article/>", encoding="utf-8")
            curated_candidate = literature.with_name("curated_candidates.jsonl")
            curated_candidate.write_text('{"candidate":"protein"}\n', encoding="utf-8")
            checkpoint = (
                data_root
                / "processed"
                / "vaccinegpt_eib202_proteome_training_261007"
                / "model.pt"
            )
            checkpoint.write_bytes(b"checkpoint")

            bacfitbase = (
                data_root
                / "processed"
                / "public_training_sources_20261007_055028"
                / "bacfitbase_h1_observations.jsonl"
            )
            bacfitbase.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"gene_id": "fish", "host_name": "fish", "host_taxon_id": 8049},
                        {"gene_id": "human-id", "host_taxon_id": 9606},
                        {"gene_id": "human-name", "host_name": "Homo sapiens"},
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            output_dir = root / "inkstone"
            manifest = package_dataset(data_root, output_dir, repo_root)
            packaged_paths = {entry["path"] for entry in manifest["files"]}
            packaged_bacfitbase = output_dir / bacfitbase.relative_to(data_root)
            packaged_rows = [
                json.loads(line)
                for line in packaged_bacfitbase.read_text(encoding="utf-8").splitlines()
            ]

            self.assertIn("refseq_bacteria/assembly.fna", packaged_paths)
            self.assertIn(
                curated_candidate.relative_to(data_root).as_posix(), packaged_paths
            )
            self.assertNotIn("raw/iedb/human_assay.json", packaged_paths)
            self.assertNotIn(raw_bacfitbase.relative_to(data_root).as_posix(), packaged_paths)
            self.assertNotIn(literature.relative_to(data_root).as_posix(), packaged_paths)
            self.assertNotIn(checkpoint.relative_to(data_root).as_posix(), packaged_paths)
            self.assertTrue(
                any(
                    item["path"] == raw_bacfitbase.relative_to(data_root).as_posix()
                    for item in manifest["excluded_files"]
                )
            )
            self.assertEqual([row["gene_id"] for row in packaged_rows], ["fish"])
            self.assertEqual(
                next(
                    entry["human_host_rows_excluded"]
                    for entry in manifest["files"]
                    if entry["path"].endswith("bacfitbase_h1_observations.jsonl")
                ),
                2,
            )
            self.assertEqual(genome.read_text(encoding="utf-8"), ">gene\nACGT\n")
            self.assertTrue((output_dir / "dataset_manifest.json").is_file())
            self.assertTrue((output_dir / "README.md").is_file())

    def test_inkstone_packager_refuses_output_inside_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo_root = root / "repository"
            data_root = root / "data"
            repo_root.mkdir()
            for relative in SOURCE_PATHS:
                (data_root / relative).mkdir(parents=True, exist_ok=True)
            with self.assertRaisesRegex(ValueError, "outside the Git repository"):
                package_dataset(data_root, repo_root / "dataset", repo_root)

    def test_graph_edges_keep_relation_and_evidence(self):
        result = build_gene_graph(
            [
                {"gene_id": "a", "genome_id": "g", "contig_id": "c", "start": 1, "end": 10, "strand": "+"},
                {"gene_id": "b", "genome_id": "g", "contig_id": "c", "start": 51, "end": 60, "strand": "+"},
            ],
            interaction_scores={("a", "b"): (0.8, "STRING")},
        )
        self.assertEqual(result["edge_index"].shape[0], 2)
        self.assertEqual(set(result["edge_type"].tolist()), {0, 3})
        self.assertTrue(all(level == "E3" for level in result["evidence_level"] if level != "E2"))

    def test_decision_readers_and_acquisition_are_deterministic(self):
        scores = prototype_scores(np.asarray([[1.0, 0.0]]), np.asarray([[1.0, 0.0], [0.0, 1.0]]))
        self.assertAlmostEqual(float(scores[0]), 0.5)
        attention = np.asarray([[[[0.5, 0.5], [0.25, 0.75]]]])
        positions = attention_position_scores(attention, np.asarray([[True, True]]))
        self.assertAlmostEqual(float(positions[0, 0]), 0.0)
        self.assertAlmostEqual(gradient_contributions(0.5, 0.5, 0.5)["Att"], float(np.log(0.5)))
        self.assertEqual(
            set(
                acquisition_scores(
                    {"a": (0.2, 0.8), "b": (0.4, 0.6)},
                    {"a": 0, "b": 3},
                    {"a": 0.5, "b": 0.8},
                    {"a": 0.5, "b": 0.8},
                )
            ),
            {"a", "b"},
        )


if __name__ == "__main__":
    unittest.main()
