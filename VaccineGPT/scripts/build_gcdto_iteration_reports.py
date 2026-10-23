from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ITERATION_DECISIONS = {
    1: {
        "baseline_claim": "A frozen protein language-model embedding plus a CLEF-style small encoder is a feasible v1 representation route.",
        "question": "What tasks and scales do protein foundation models actually validate, and where does their transfer stop?",
        "positive_terms": ["embedding", "representation", "protein language", "esm", "sequence"],
        "counter_terms": ["limit", "limitation", "bias", "failure", "generaliz", "domain shift"],
        "delta": "Use frozen ESM-3 as the primary v1 protein feature backend; retain ESM-2 only for published CLEF checkpoint compatibility; add checkpoint provenance gates; do not call embeddings protective-antigen predictions.",
        "falsifier": "On a preregistered homology-cluster test, compare the frozen small-model baseline with ESM-3 on task-specific E1/E2 labels; if the smaller model is non-inferior within a preset margin, evaluate whether it should replace ESM-3 for this hardware.",
    },
    2: {
        "baseline_claim": "Genome-wide DNA is a separate v2 input track; DNA foundation outputs cannot be silently substituted for protein features.",
        "question": "Which DNA/genome foundation-model tasks transfer to bacterial gene-level inference?",
        "positive_terms": ["genome", "dna", "nucleotide", "evo", "regulatory", "sequence"],
        "counter_terms": ["limit", "microbial", "bacterial", "context", "transfer", "generaliz"],
        "delta": "Enforce v1 protein-only input; version v2 DNA windows and mark unclassified windows as unknown; keep Evo2 execution behind a WSL2/CUDA and target-genome validation gate.",
        "falsifier": "On held-out genomes, compare Evo2 window embeddings against a simple sequence-composition/annotation baseline for prespecified regulatory/gene-element labels.",
    },
    3: {
        "baseline_claim": "Public and private modalities must use separate losses/projections and meet only at the M2 decision core.",
        "question": "Does contrastive multimodal alignment provide evidence for cross-view representation without label leakage?",
        "positive_terms": ["contrastive", "multimodal", "paired", "alignment", "protein"],
        "counter_terms": ["negative", "confound", "leakage", "missing", "bias", "limitation"],
        "delta": "Keep public/private tracks separated; require observation masks; restrict private negatives to same-genome pairs; do not use sequence-only crop retrieval as a vaccine metric.",
        "falsifier": "On the same leakage-safe test, compare separated tracks, merged tracks, and single-modality baselines; assess both representation metrics and held-out target-specific E1/E2 task metrics.",
    },
    4: {
        "baseline_claim": "Protective immunity is not equivalent to antigenicity, binding, or database membership.",
        "question": "Which reverse-vaccinology predictions have direct protection evidence, and what labels are available?",
        "positive_terms": ["vaccine", "protective", "antigen", "reverse vaccinology", "immunogenic"],
        "counter_terms": ["not", "challenge", "validation", "experimental", "protect", "limitation"],
        "delta": "Separate E1/E2 target-organism measurements from E3/E4 weak labels; preserve presentation, immunogenicity, and protection as distinct outputs; refuse to train H2 as gold from VFDB/IEDB membership alone.",
        "falsifier": "Prospective target-organism challenge outcomes must improve over homology and annotation-only baselines on a preregistered blind set.",
    },
    5: {
        "baseline_claim": "Fitness/essentiality evidence supports attenuation hypotheses only under matched strain, assay, and condition; it does not establish antigen accessibility.",
        "question": "How do CRISPRi/TnSeq measurements encode conditional fitness and what are their assay boundaries?",
        "positive_terms": ["crispri", "tn-seq", "transposon", "fitness", "essential", "bacteria"],
        "counter_terms": ["condition", "strain", "insertion", "bias", "not", "limitation"],
        "delta": "Use PACE only when timepoints and metadata support fitting; return not_fitted for sparse/singular data; never substitute public cross-species essentiality for an EIB202 gold label.",
        "falsifier": "Blind target-strain mutant/knockdown fitness must correlate with predictions under the prespecified time-course assay; report strain and condition slices.",
    },
    6: {
        "baseline_claim": "Genome annotation coordinates/ORFs are uncertain inputs and partial/pseudogene boundaries must be represented.",
        "question": "How do bacterial annotation systems version gene calls, boundaries, and underlying databases?",
        "positive_terms": ["annotation", "bacterial", "genome", "orf", "database", "prokka", "bakta"],
        "counter_terms": ["partial", "pseudogene", "boundary", "version", "database", "error"],
        "delta": "Keep sequence-role, identifier-recovery, ambiguity, completeness, source version, and exact hash in the data contract; mark unresolved accession mappings for review rather than guessing.",
        "falsifier": "Manually adjudicate a stratified sample of complete/partial/pseudogene and recovered-ID records; estimate error by annotation state before model training.",
    },
    7: {
        "baseline_claim": "A graph layer can model context, but attention weights and predicted interactions are not causal evidence.",
        "question": "What makes biological graph edges and GAT/GATv2 evaluations trustworthy?",
        "positive_terms": ["graph", "attention", "protein", "interaction", "network", "gene"],
        "counter_terms": ["bias", "leakage", "causal", "noise", "benchmark", "limitation"],
        "delta": "Retain edge source/type/evidence; require group-safe graph construction; prohibit converting GAT attention into a causal explanation; keep predicted interactions E3.",
        "falsifier": "Ablate each edge type and compare GCE/no-GCE on held-out genome/cluster groups; graph benefit must exceed a preregistered margin and survive label permutation.",
    },
    8: {
        "baseline_claim": "Conformal intervals can quantify calibration residuals but cannot repair label error or domain shift.",
        "question": "Under what exchangeability and dependence conditions are calibration guarantees valid?",
        "positive_terms": ["conformal", "calibration", "uncertainty", "prediction interval", "coverage"],
        "counter_terms": ["exchangeability", "shift", "dependent", "coverage", "miscalibration", "limitation"],
        "delta": "Return [0,1]/unavailable without a valid calibration set; require at least 100 matched observations for the candidate gate; report calibration strata and domain limitations.",
        "falsifier": "Evaluate empirical coverage and interval width on untouched, group-disjoint target-strain data; report per-condition coverage and predeclared tolerance.",
    },
    9: {
        "baseline_claim": "Attenuation, antigen retention, and safety conflict; Pareto tiers expose this rather than collapsing all candidates into one weighted score.",
        "question": "How do multiobjective methods preserve trade-offs and when are decision weights introduced?",
        "positive_terms": ["pareto", "multiobjective", "multi-objective", "optimization", "trade-off"],
        "counter_terms": ["weight", "preference", "sensitivity", "objective", "uncertainty", "limitation"],
        "delta": "Use Pareto tiers for non-dominated Att/Ret, keep U only for within-tier ordering, and never rank rows with missing H2/Ret/Safe inputs.",
        "falsifier": "Perturb thresholds and scalar exponents ±20%; report stable versus unstable tiers and preserve the full Pareto set for human review.",
    },
    10: {
        "baseline_claim": "Exact duplicate filtering is not homology leakage control; data-source conflicts and unresolved IDs must be visible.",
        "question": "How do redundant sequences, homologs, and metadata-based splits bias biological benchmarks?",
        "positive_terms": ["homology", "redundancy", "duplicate", "leakage", "benchmark", "sequence"],
        "counter_terms": ["similarity", "family", "cluster", "split", "leakage", "identity"],
        "delta": "Preserve raw artifacts; quarantine opposing label rows; distinguish exact SHA matches from MMseqs2 homolog groups; block supervised training when group provenance is missing.",
        "falsifier": "Run exact-sequence, MMseqs2 90%-identity, and genome-held-out audits; compare random-row and grouped metrics to quantify inflation before accepting any gate.",
    },
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rank_for_terms(row: dict[str, Any], terms: list[str]) -> int:
    value = (str(row.get("title") or "") + " " + str(row.get("abstractText") or "")).casefold()
    return sum(value.count(term.casefold()) for term in terms)


def build(root: Path, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    code_ledger = []
    code_path = Path("reference/code_evidence_ledger_261007.jsonl")
    if code_path.exists():
        code_ledger = read_rows(code_path)
    for round_no, decision in ITERATION_DECISIONS.items():
        path = root / "rounds" / f"round_{round_no:02d}_references.jsonl"
        rows = read_rows(path)
        supporting = sorted(
            rows,
            key=lambda row: (
                -rank_for_terms(row, decision["positive_terms"]),
                str(row.get("title") or ""),
            ),
        )[:8]
        challenging = sorted(
            rows,
            key=lambda row: (
                -rank_for_terms(row, decision["counter_terms"]),
                str(row.get("title") or ""),
            ),
        )[:6]
        safe_support = [
            row for row in supporting if rank_for_terms(row, decision["positive_terms"]) > 0
        ]
        safe_challenges = [
            row for row in challenging if rank_for_terms(row, decision["counter_terms"]) > 0
        ]
        round_manifest_rows = [
            row
            for row in read_rows(root / "query_manifest.jsonl")
            if int(row.get("round", -1)) == round_no
        ]
        latest = round_manifest_rows[-1] if round_manifest_rows else {}
        code_rows = [row for row in code_ledger if row.get("round") == round_no]
        manual_review_file = output_dir.parent / f"iteration_{round_no:02d}_manual_review.md"
        manual_review_blocked = output_dir.parent / f"iteration_{round_no:02d}_manual_review_blocked.md"
        report_path = output_dir / f"iteration_{round_no:02d}.md"
        lines = [
            f"# GCDTO 迭代 {round_no:02d}：{latest.get('round_name', 'round_pending')}",
            "",
            f"**问题：** {decision['question']}",
            "",
            f"**基线主张：** {decision['baseline_claim']}",
            "",
            "## 配额与证据边界",
            "",
            f"- 本轮逐篇唯一 PMCID 记录数：{len({row.get('pmcid') for row in rows})}；目标 200。",
            f"- Europe PMC 报告/本轮清单全文 XML 验证数：{latest.get('full_text_xml_verified_count', 0)}。",
            f"- 本轮公开代码库已检查数：{len(code_rows)}；目标至少20。",
            f"- 查询命中数：{latest.get('reported_search_hit_count', '未记录')}；非全文数。",
            "- 候选支持/反题清单由标题摘要关键词检索排序；正文 XML 已下载，但这些条目的主张均**未声称已逐篇人工阅读全文核验**。",
            "- 深度引用只能限于 `research_evidence_261007.md` 中明确标注已人工打开的段落；其它引文状态为摘要/机器索引待人工复核。",
            "",
            "## 正题：候选支持文献（摘要级/全文待逐篇复核）",
            "",
        ]
        if safe_support:
            for row in safe_support:
                doi = row.get("doi")
                link = f"https://doi.org/{doi}" if doi else row.get("fulltext_xml_url", "")
                lines.append(
                    f"- {row.get('authorString','作者未核实')} ({row.get('pubYear','年份未核实')}). "
                    f"[{row.get('title','标题缺失')}]({link}); {row.get('pmcid')}; "
                    f"全文定位候选：{'; '.join(row.get('fulltext_section_titles', [])[:4]) or '全文XML已取，章节名缺失'}。"
                )
        else:
            lines.append("- 本轮没有可安全归纳的主题支持摘要；不得由关键词命中推导支持结论。")
        lines.extend(["", "## 反题：可能限制/相反证据候选（须人工查正文）", ""])
        if safe_challenges:
            for row in safe_challenges:
                doi = row.get("doi")
                link = f"https://doi.org/{doi}" if doi else row.get("fulltext_xml_url", "")
                lines.append(
                    f"- {row.get('authorString','作者未核实')} ({row.get('pubYear','年份未核实')}). "
                    f"[{row.get('title','标题缺失')}]({link}); {row.get('pmcid')}; "
                    "此处仅是限制性词汇检索，不代表已经确认其推翻基线。"
                )
        else:
            lines.append("- 未自动检出反题候选；这不构成不存在反证的结论。")
        lines.extend(
            [
                "",
                "## 合题与架构增量",
                "",
                decision["delta"],
                "",
                "## 可证伪假设与检测器",
                "",
                decision["falsifier"],
                "",
                "## 内联参考索引",
                "",
                f"本轮所有全文参考条目（目标 200）保存在 "
                f"`..\\data\\research_campaign_261007\\rounds\\round_{round_no:02d}_references.jsonl`；",
                "逐篇题名、作者、DOI/PMCID、license、相关实现路径及源码证据见 "
                f"`reference/iterations_261007/iteration_{round_no:02d}_code_review.md` 和 "
                "`reference/code_evidence_ledger_261007.jsonl`。",
                "逐篇全文 SHA-256、全文定位、访问状态与查询来源见总 evidence ledger。未达 200/20 时该轮标记未达相应配额。",
                "",
                "## Δ / skill 自演化记录",
                "",
                f"本轮可复用的通用模式：{decision['delta']}",
                "",
            ]
        )
        lines.append("### 本轮 200 条全文引用")
        lines.append("")
        for index, row in enumerate(rows, 1):
            doi = row.get("doi")
            pmcid = row.get("pmcid")
            url = f"https://doi.org/{doi}" if doi else (
                f"https://europepmc.org/articles/{pmcid}" if pmcid else ""
            )
            author = row.get("authorString") or "作者未核实"
            year = row.get("pubYear") or "年份未核实"
            title = row.get("title") or "标题未核实"
            license_name = row.get("license") or "未核实"
            lines.append(
                f"{index}. {author} ({year}). [{title}]({url}) — "
                f"{pmcid or 'PMCID未核实'}; license `{license_name}`; "
                f"fullTextXML SHA-256 `{row.get('fulltext_xml_sha256')}`."
            )
        lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        summaries.append(
            {
                "round": round_no,
                "references": len({row.get("pmcid") for row in rows}),
                "fulltext_retrieval_quota_met": len({row.get("pmcid") for row in rows}) >= 200
                and latest.get("full_text_xml_verified_count", 0) >= 200,
                "codebase_quota_met": len(code_rows) >= 20,
                "manual_substantive_review_sample_completed": manual_review_file.exists()
                and not manual_review_blocked.exists(),
                "manual_review_blocked": manual_review_blocked.exists(),
                "report": str(report_path),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence-bounded GCDTO iteration dossiers.")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries = build(args.campaign_root, args.output_dir)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
