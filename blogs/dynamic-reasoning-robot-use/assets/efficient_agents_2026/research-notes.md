# Editorial and Evidence Record

Checked: 2026-11-25. Scope: adaptive/efficient reasoning for multi-turn agents, with computer-use as the closest interaction analogy and robot-use as a proposed transfer. No GPU workload, model training or simulator restoration was performed.

## Selection

Five main papers: RPO (ACL 2026 proceedings), MORE (2610.33684v2), TAB (2604.05164v4), RACE (2610.12061v1), STEPWISE (2604.27151v1). The year for RPO is its verified proceedings year; no unverified first-preprint date is asserted. All other selected versions are arXiv preprints; no acceptance status is inferred.

Selection favors a distinct mechanism, a traceable training signal, original full text, and relevant agent evidence. It is an editorial shortlist, not a global quality ranking.

RPO supplies a reviewed, concrete recipe for concentrating gradient computation on a routing decision. MORE supplies mode/budget coupling and an explicitly named PPO-style recipe. TAB tests cross-domain transfer of a multi-turn budget policy. RACE estimates cross-turn support and uses the signal in SFT and RL. STEPWISE provides the most direct GUI setting and an event-triggered monitoring baseline.

PPO preference was applied without relabeling algorithms. MORE Appendix C.3 uses terminal-return REINFORCE++ normalization and token-level clipping without a learned critic. RPO uses per-mode, cross-prompt standardized rewards and a root-only policy-gradient objective. TAB/RACE use GRPO. STEPWISE trains classifiers with supervised cross-entropy. The article's actor–critic PPO section is a proposed adaptation.

## Search and Screen

Primary-source search axes included 2026 adaptive reasoning + PPO, computer-use + efficient reasoning, multi-turn budget allocation, and root-token policy optimization. The search included arXiv, ACL Anthology and official project links; aggregator entries were used only as leads. Queries did not establish exhaustive coverage.

Additional full texts screened:

- The Value of Thinking, 2611.16961v1: PPO-style CDPO and frozen backbone are relevant, but its v1 code URL is a placeholder and episodic evaluation is limited. Excluded from the five rather than filling a PPO quota. Its appendix explicitly discusses surrogate bias.
- Efficient Agentic Reasoning Through Self-Regulated Simulative Planning, 2605.22138v1: explicit planner/world-model decomposition and GRPO, useful but introduces a broader structured-planning question beyond this focused routing/budget article.
- Ares, 2603.07915v1: SFT-to-GRPO per-step effort router, relevant WebArena evaluation; overlaps TAB's effort control role. TAB chosen for explicit unchanged-policy transfer experiments and full cost accounting.
- Adaptive Latent Agentic Reasoning, 2606.02871v1: latent/text switching and AR-GRPO; reserved for a separate hybrid/latent essay.
- Learning When to Think, 2608.20256v2: three-mode GRPO with hard caps; overlaps MORE's mode selection and is primarily single-turn math.
- Earlier Thinkless, AdaCoT, TON, DEER and DyCon remain in the archive/guide. Removed from the main blog only, following the user's five-paper scope.

## Claim Checks

RPO: §3.2–3.3, Algorithms 1–2, Tables 2–5. Success counts are computed per prompt and forced mode; advantage normalization pools across prompts separately for each mode. Root-only loss does not freeze shared parameters. Table 2's memory accounting must not be presented as universal hardware measurements. Full rollouts remain necessary.

MORE: §3.4–3.5, Appendix C.3 and D. Save controller state before rollout and score against that state; update state afterwards. Reward includes lower-band as well as upper-band deviations. Mode utility is partly a specified prior. REINFORCE++ statistics are over valid tokens across the six streams. Do not invent GAE or a critic.

TAB: §3.3, footnote to equation (2), §4.3 and Appendix A. Use actual consumed Solver tokens in the training reward; report Budgeter + Solver cost for transferred workflows and complete specified system cost for math. Same terminal advantage is broadcast to all budget decisions. A separate policy is trained per global budget. Transfer does not establish robotics transfer.

RACE: §3.1–3.3, Table 2, Appendix C. Candidate deletion is compared against the currently compressed context, not always the original trajectory. Future actions and observations remain fixed during teacher-forced scoring. Apply LoGiC to successful on-policy rollouts; mask reasoning tokens but retain original normalization count, and add closure supervision. Likelihood checks are training labels, not an inference-time requirement to see future observations.

STEPWISE: §4, §6, Appendix B. Lightweight monitors see textual rationale/action traces; strong milestone verification also sees two screenshots. Milestones select verification times, not RL rewards. Classification supervision from about 300 trajectories, no end-to-end PPO. Detector F1 and event-vs-periodic controls matter more than overall detector accuracy with sparse events.

No official code was run, and no independent replication of the papers' empirical claims is implied.

## Visual Direction

Preserve the personal site's light research-note identity: paper #f3f2ed, ink #191917, muted #66645e, line #d8d7d0, links #354b58. Use a restrained Helvetica/Arial sans-serif stack with CJK fallbacks and monospace metadata. The exact PI webfont could not be inspected because direct requests returned an access challenge and the UI browser host was unavailable; no exact font match is claimed. Narrow prose measure, wider original figures, light heading weights, no cards or tables.

All 13 figures are original PDF regions rendered at 3.5x. No generated redraws, annotations or added source overlays. See provenance.json for authors, versions, licenses, pages, crops and checksums. The page provides five paper references at its end; figures have descriptive alt text and no image hyperlinks. Source PDFs remain unchanged.
