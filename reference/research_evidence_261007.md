# VaccineGPT public-source evidence status — 2026-10-07

## Coverage declaration

This is a **partial, evidence-bounded source map**, not the ten-round review
requested in the project brief. One independent search pass used Europe PMC and
PubMed for discovery and Crossref for DOI checks. The broad queries returned
3,039 PubMed records and 434 Europe PMC records; these are unfiltered search
hits, **not eligible-paper counts**.

Relevant portions of six scholarly full texts were opened: CLEF, conformal
protein mining, CRISPRi–TnSeq, Bakta, ProtST, and a protein-interaction
benchmark-leakage study. The Evo 2 article endpoint was opened but only its
front matter was inspected. A separate code review pass inspected selected
paths in 22 public repositories. It did not systematically review each
repository's maintenance status, all dependency licenses, model-weight terms,
or dataset terms.

**Status update (2026-10-07):** a subsequent Europe PMC campaign retrieved 200
open fullTextXML items for each of 10 themed rounds and 20 or more verified
public code repositories per round. Across rounds there are 2,000
round-repository references (192 unique repository URLs) and 2,000 full-text
round references (1,806 identifier-deduplicated works). Every round XML
checksum was validated. The **claim review remains sampled**: AI subagents
reviewed selected sections of 90 random round-reference rows across rounds
1–4 and 6–10; after deduplication this is 89 unique sampled works. The round 5
paper-review task was blocked by a biological-risk filter and not retried.
Six previously opened source works are separately summarized below (four
belong to the campaign corpus and were not in the random reviewed sample; two
arXiv works are outside the campaign). Thus 93 campaign works and 95 works
overall have sampled selected-section review, not complete article claim
extraction. The other 1,713 campaign works were not individually
claim-reviewed. This file
therefore remains an evidence-bounded source map and does not claim
GB/T 7714-2025 formatting. See `search_strategy_261007.md`,
`gcdto_campaign_261007_summary.json`, the normalized ledger, per-round
iteration reports, and code-review catalogues.

## Full-text portions inspected

| Work | Identifier and access | Evidence locations actually inspected | Boundaries |
|---|---|---|---|
| Peng et al., “Contrastive-learning of language embedding and biological features for cross modality encoding and effector prediction,” *Nature Communications*, 2025. | DOI [10.1038/s41467-025-56526-1](https://doi.org/10.1038/s41467-025-56526-1); PMID 39900608; PMCID [PMC11791096](https://pmc.ncbi.nlm.nih.gov/articles/PMC11791096/); `open_full_text`. | Full-text XML: Introduction; Results, “CLEF is a contrastive learning-based deep learning framework”; Fig. 1A–C. | Supports contrastive sequence/feature integration and an effector-prediction use case. Does **not** establish protective-antigen prediction. |
| Boger et al., “Functional protein mining with conformal guarantees,” *Nature Communications*, 2025. | DOI [10.1038/s41467-024-55676-y](https://doi.org/10.1038/s41467-024-55676-y); PMID 39747192; PMCID [PMC11695924](https://pmc.ncbi.nlm.nih.gov/articles/PMC11695924/); `open_full_text`. | Full-text XML: Abstract and Introduction, including the discussion of conformal risk, homology/function transfer, and representative calibration data. | Coverage/risk guarantees rely on a valid calibration relationship; they do not establish biological truth or protection. |
| Jana et al., “CRISPRi–TnSeq maps genome-wide interactions between essential and non-essential genes in bacteria,” *Nature Microbiology*, 2024. | DOI [10.1038/s41564-024-01759-x](https://doi.org/10.1038/s41564-024-01759-x); PMID 39030344; PMCID [PMC11371651](https://pmc.ncbi.nlm.nih.gov/articles/PMC11371651/); `open_full_text`. | Full-text XML: Introduction; Results, “Essential–non-essential interaction mapping with CRISPRi–TnSeq”; Fig. 1. | Study organism is *Streptococcus pneumoniae*. Essentiality is assay-, strain-, and condition-dependent; an essential gene is not automatically an accessible antigen. |
| Schwengers et al., “Bakta: rapid and standardized annotation of bacterial genomes via alignment-free sequence identification,” *Microbial Genomics*, 2021. | DOI [10.1099/mgen.0.000685](https://doi.org/10.1099/mgen.0.000685); PMID 34739369; PMCID [PMC8743544](https://pmc.ncbi.nlm.nih.gov/articles/PMC8743544/); `open_full_text`. | Full-text XML: Data Summary; Introduction; Methods, “Annotation workflow”; Fig. 1. | The database is separately versioned; annotation is not a direct validation of surface exposure, virulence, or protection. |
| Xu et al., “ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts,” ICML 2023 preprint. | arXiv [2301.12040](https://arxiv.org/abs/2301.12040); DOI [10.48550/arXiv.2301.12040](https://doi.org/10.48550/arXiv.2301.12040); `open_full_text`. | arXiv HTML §3.2, “Pre-training Tasks: Joint Modeling of Protein Sequences and Biomedical Texts.” | Supports sequence/text contrastive alignment as a method example, not the use of text as experimentally validated vaccine protection labels. |
| “Revealing data leakage in protein interaction benchmarks,” 2024 preprint. | arXiv [2404.10457](https://arxiv.org/abs/2404.10457); DOI [10.48550/arXiv.2404.10457](https://doi.org/10.48550/arXiv.2404.10457); `open_full_text`. | arXiv HTML §3.1, “Splitting by metadata is not enough,” Fig. 4. | The reported leakage rates are specific to that PPI benchmark and must not be transplanted to VaccineGPT data. |

## Additional identified sources not full-text-reviewed

These are candidate reading entries, not verified full-text evidence for claims
in the implementation:

- ESM-2/ESMFold: Lin et al., *Science* 2023, DOI
  [10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574);
  public preprint DOI [10.1101/2022.07.20.500902](https://doi.org/10.1101/2022.07.20.500902).
- ESM-3: Hayes et al., *Science*, DOI
  [10.1126/science.ads0018](https://doi.org/10.1126/science.ads0018); public
  preprint DOI [10.1101/2024.07.01.600583](https://doi.org/10.1101/2024.07.01.600583).
  A discovered URL `arXiv:2403.16324` was unrelated and is explicitly not cited
  as ESM-3.
- Evo 2: Brixi et al., *Nature* 2026, DOI
  [10.1038/s41586-026-10176-5](https://doi.org/10.1038/s41586-026-10176-5);
  PMCID [PMC13128491](https://pmc.ncbi.nlm.nih.gov/articles/PMC13128491/).
- Vaxign-ML: Ong et al., DOI
  [10.1093/bioinformatics/btaa119](https://doi.org/10.1093/bioinformatics/btaa119);
  PMCID [PMC7214037](https://pmc.ncbi.nlm.nih.gov/articles/PMC7214037/).
- Vaxign2: Ong et al., DOI
  [10.1093/nar/gkab279](https://doi.org/10.1093/nar/gkab279);
  PMCID [PMC8218197](https://pmc.ncbi.nlm.nih.gov/articles/PMC8218197/).
- Prokka: DOI
  [10.1093/bioinformatics/btu153](https://doi.org/10.1093/bioinformatics/btu153);
  PMCID [PMC4105932](https://pmc.ncbi.nlm.nih.gov/articles/PMC4105932/).
- NCBI PGAP: DOI
  [10.1093/nar/gkw569](https://doi.org/10.1093/nar/gkw569);
  PMCID [PMC5005942](https://pmc.ncbi.nlm.nih.gov/articles/PMC5005942/).
- Tn-seq review: DOI
  [10.3390/ijms252011298](https://doi.org/10.3390/ijms252011298);
  PMCID [PMC11508858](https://pmc.ncbi.nlm.nih.gov/articles/PMC11508858/).
- GAT: arXiv [1710.10903](https://arxiv.org/abs/1710.10903); GATv2:
  arXiv [2105.14491](https://arxiv.org/abs/2105.14491).
- TAPE: arXiv [1906.08230](https://arxiv.org/abs/1906.08230).
- NSGA-II: DOI [10.1109/4235.996017](https://doi.org/10.1109/4235.996017).

## Public code map

The 22 public repositories below were listed and selected paths were inspected.
License labels refer to repository code only; model weights, databases, and
third-party dependencies may have separate terms.

| Repository | License/status noted in source pass | Paths or relevance |
|---|---|---|
| [PengYueECUST1997/CLEF](https://github.com/PengYueECUST1997/CLEF) | MIT code | `src/CLEF.py`, `src/Feature_transform.py`, `src/Train_method.py`, `Demo/CLEFTrain.py` |
| [facebookresearch/esm](https://github.com/facebookresearch/esm) | MIT code | `esm/pretrained.py`, `esm/model/esm2.py` |
| [Biohub/esm](https://github.com/Biohub/esm) | MIT code; inspect model/weight access separately | `esm/models/esm3.py`, ESM3 README, ESMC interfaces |
| [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2) | Apache-2.0 code | `evo2/models.py`, README; requires Linux/WSL2 CUDA runtime |
| [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) | MIT | `gat_conv.py`, `gatv2_conv.py` |
| [dmlc/dgl](https://github.com/dmlc/dgl) | Apache-2.0 | Graph-learning alternative; no claim that a particular biological graph is valid |
| [ncbi/pgap](https://github.com/ncbi/pgap) | NCBI public code with third-party restrictions | `pgap.cwl`, `wf_common.cwl`, `bacterial_annot/` |
| [ncbi/datasets](https://github.com/ncbi/datasets) | Public-domain notice for NCBI-authored code | `client/`, `datasets.openapi.yaml` |
| [oschwengers/bakta](https://github.com/oschwengers/bakta) | GPL-3.0 | `bakta/`, database versioning |
| [tseemann/prokka](https://github.com/tseemann/prokka) | GPL-3.0 | `bin/prokka`, `db/`, tests |
| [hyattpd/Prodigal](https://github.com/hyattpd/Prodigal) | GPL-3.0 | `main.c`, `gene.c`, training code |
| [eggnogdb/eggnog-mapper](https://github.com/eggnogdb/eggnog-mapper) | AGPL-3.0 | `emapper.py`, `eggnogmapper/` |
| [tseemann/barrnap](https://github.com/tseemann/barrnap) | GPL-3.0 code; separate Rfam terms | `bin/`, `db/`, `LICENSE.Rfam` |
| [soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2) | MIT | `src/clustering/`, `src/linclust/` |
| [ioerger/transit](https://github.com/ioerger/transit) | GPL-3.0 | `src/pytransit/`, fitness-defect routines |
| [scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | BSD-3-Clause | Classification/conformity-score modules |
| [anyoptimization/pymoo](https://github.com/anyoptimization/pymoo) | Apache-2.0 | `pymoo/algorithms/moo/nsga2.py` |
| [OATML-Markslab/ProteinGym](https://github.com/OATML-Markslab/ProteinGym) | MIT code | Benchmark and split definitions; dataset terms need review |
| [songlab-cal/tape](https://github.com/songlab-cal/tape) | BSD-3-Clause | Protein transfer benchmark and task splits |
| [VIOLINet/Vaxign-ML-docker](https://github.com/VIOLINet/Vaxign-ML-docker) | GPL-3.0 | Vaxign-ML implementation/model/benchmark |
| [DeepGraphLearning/ProtST](https://github.com/DeepGraphLearning/ProtST) | Apache-2.0 | Protein/text multimodal training code |
| [biopython/biopython](https://github.com/biopython/biopython) | Biopython License Agreement; some BSD-3-Clause files | `Bio/SeqIO/` |

The separate [VIOLINet/Vaxign2-django](https://github.com/VIOLINet/Vaxign2-django)
repository was retrievable, but no root license file was seen in the inspected
listing; its reuse terms remain unverified.

## Claim-to-source limits

1. CLEF supports a contrastive sequence/biological-feature architecture and
   effector-prediction experiments. It does not show that the same score is a
   protective-antigen or vaccine-efficacy probability.
2. ESM-2, ESM-3, and Evo 2 model different modalities/objectives. A generative
   or sequence representation model is not a protective-antigen classifier.
3. GATv2 attention is not causal explanation. Biological edge provenance,
   graph construction, and leakage-safe splits remain necessary.
4. Tn-seq/CRISPRi fitness is conditional essentiality evidence, not surface
   accessibility or protective immunity.
5. Conformal coverage is conditional on calibration exchangeability and
   cannot compensate for wrong labels, cross-family leakage, or domain shift.
6. Pareto frontiers expose trade-offs; they do not supply a unique biological
   “winner” or an experimental safety decision.
7. Exact duplicates and homolog clusters are different. MMseqs2 ≥90%-identity
   clustering was not available locally at the time of this source pass.

## Campaign execution addendum (2026-10-07)

- Europe PMC was queried through 10 themed, saved requests. For every round,
  200 PMCID records had an OA `fullTextXML` body downloaded and SHA-256 verified:
  **2,000 round-level references**, **1,806 unique works** after DOI→PMID→PMCID→
  normalized-title deduplication. The random byte-integrity audit verified
  2,000/2,000 fulltext bodies and found no title/identifier conflicts.
- Public-code inspections met the per-round quota: **202 round-repository
  references** over ten themes, **192 global unique repositories**. The
  per-round source-path/license status tables are in
  `reference/iterations_261007/iteration_XX_code_review.md`; machine-readable
  evidence rows are in `code_evidence_ledger_261007.jsonl`.
- A deterministic full-text review packet selected 10 articles per round.
  Supplied abstracts and selected sections were AI-subagent reviewed for rounds 1–4 and
  6–10 (90 round-reference rows, 89 unique campaign works). The round 5 paper-review request was blocked by a
  biological-risk filter and was not retried via an alternate route. The
  four previously reviewed campaign works were outside this random sample; 1,713 campaign works remain without individual
  claim-reviewed.
- The ten automated round dossiers contain the per-round 200-reference index,
  candidate positive/negative source lists, synthesis and falsifiable detector.
  Candidate support/limitation classification is automated from titles and
  abstracts, so substantive claims use only the sampled section-review files
  where available.

This meets the requested **retrieval/reference availability** quotas. It does
not mean 2,000 papers were read cover-to-cover or that each substantive claim
was manually validated. The reports preserve this distinction and do not
claim GB/T 7714-2025 conformance.
