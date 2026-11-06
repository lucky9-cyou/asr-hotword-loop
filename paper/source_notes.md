# ECHO literature and claim audit

Checked on **2026-11-06**. This is a working research note, not part of the manuscript. The draft uses direct author–year links; the keys below map them to `references.bib`. Summaries are deliberately narrow. Published papers and preprints are distinguished; no ECHO benchmark result or priority claim is inferred from these sources.

| BibTeX key | Primary source and metadata basis | Claim supported in the draft |
|---|---|---|
| `pundak2018deep` | [arXiv](https://arxiv.org/abs/1808.02480); [author organization page](https://research.google/pubs/deep-context-end-to-end-contextual-speech-recognition/), IEEE SLT 2018 | CLAS learns contextual phrase representations and uses a supplied phrase list. DOI/pages cross-checked in published references; direct IEEE landing page was inaccessible. |
| `zhao2019shallow` | [ISCA proceedings and BibTeX](https://www.isca-archive.org/interspeech_2019/zhao19d_interspeech.html), Interspeech 2019 | Contextual shallow fusion and early subword biasing address vocabulary preference during decoding. |
| `sun2021tree` | [Author preprint](https://arxiv.org/abs/2109.00627), ASRU 2021 noted in record | TCPGen combines a symbolic prefix tree with neural pointer generation. Unverified DOI/pages omitted. |
| `pundak2022audio` | [ISCA proceedings](https://www.isca-archive.org/interspeech_2022/pundak22_interspeech.html), Interspeech 2022 | Audio exemplars paired with text corrections can support correction of later recognition. This is a close predecessor for correction reuse. |
| `chen2023hyporadise` | [NeurIPS official proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6492267465a7ac507be1f9fd1174e78d-Abstract-Datasets_and_Benchmarks.html), NeurIPS 2023 | N-best hypotheses can support generative correction beyond candidate reranking. Official proceedings metadata preferred over arXiv spelling discrepancies. |
| `radhakrishnan2023whispering` | [ACL Anthology and BibTeX](https://aclanthology.org/2023.emnlp-main.618/), EMNLP 2023 | Generative correction can combine acoustic and linguistic representations. |
| `tang2024pinyin` | [ISCA proceedings and BibTeX](https://www.isca-archive.org/interspeech_2024/tang24_interspeech.html), Interspeech 2024 | ChineseHP and pronunciation-aware prompting address Chinese hypothesis correction. |
| `mittal2025recast` | [ACL Anthology and BibTeX](https://aclanthology.org/2025.findings-emnlp.203/), Findings of EMNLP 2025 | Decoder-state retrieval selects keywords to prompt a downstream speech language model. |
| `zheng2026multilevel` | [Publisher record](https://www.sciencedirect.com/science/article/pii/S0167639326001068), Speech Communication 183, article 103458 | A unified prompt combines conversation history with hotwords; phonetic filtering selects relevant candidates. Publisher-indexed abstract and metadata available; direct full-page requests returned 403. No claims based on inaccessible experiments. |
| `guo2026noisycontext` | [Author preprint](https://arxiv.org/abs/2603.24034), March 2026 | Oracle histories and predicted histories produce a contextual mismatch; noisy-context robustness is an explicit research problem. No accepted venue verified. |
| `li2026ontology` | [Author preprint](https://arxiv.org/abs/2606.13464); [full text](https://arxiv.org/html/2606.13464v2), June 2026 v2 | An updatable ontology stores contextual evidence for later hypothesis correction. Text inputs and initial grounded context are treated as reliable in its formulation. No accepted venue verified. |
| `wang2026think` | [Author preprint](https://arxiv.org/abs/2610.26986); [full text](https://arxiv.org/html/2610.26986v1), October 2026 | Context-steered reasoning considers relevance of uncertain hotword candidates using audio before transcription. No accepted venue verified. |

## Positioning that is an ECHO interpretation

The draft distinguishes **where a lexical entry comes from** (a noisy spoken explanation), **what is retained** (a spelling and its source span), and **when it is used** (future acoustic decoding). This is our comparison of the cited methods, not a claim made by their authors. There is substantial overlap with contextual biasing, conversational memory, and correction reuse. Neither “first ASR memory” nor “first use of conversation context” is supported.

The title's “Optimization” denotes changing lexical preferences at inference in the current prototype. It does not establish a learned objective, gradient optimization, or an optimality guarantee. The implementation is an untrained, prefix-conditioned logit adjustment with greedy decoding; it should not be described as TCPGen, a trained contextual module, or a complete beam-search biasing scheme.

## Claims requiring new evidence before submission

- General ASR improvement, entity error-rate reductions, and robustness to unseen names require a benchmark beyond the current synthetic examples and prompt checks.
- Spoken explanations may themselves be misrecognized. Exact quotation of an evidence span checks provenance, not semantic correctness or confidence calibration.
- Existing memory is append-oriented: corrected variants do not automatically retire old entries. Conflict resolution and error propagation need separate evaluation.
- The current interface submits complete utterances through repeated calls. No native streaming latency or chunk-level streaming result should be claimed.
- Source search is a bounded literature review as of the checked date, not proof of novelty or exhaustive coverage. Any new submission should refresh especially the 2026 preprints.
