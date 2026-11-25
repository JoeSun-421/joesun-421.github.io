# Editorial and Evidence Record

Checked: 2026-11-25. Scope: adaptive/efficient reasoning for multi-turn agents, with computer-use as the closest interaction analogy and robot-use as a proposed transfer. No GPU workload, model training or simulator restoration was performed.

## Current Mechanism-Oriented Selection

The latest user clarification supersedes the venue-first constraint. Priorities: recent original releases, clear acceleration mechanism, evidence quality, agent relevance, and an interpretable recipe. Acceptance and PPO are supporting criteria rather than quotas. Five main works now cover five primary interventions: TAB (budget allocation), PUMA (within-generation early exit), RACE (cross-turn support/redundancy), STEPWISE (event-driven strong-model escalation), SpecAgent (draft/verify collaboration replacing serial strong decoding). Original release dates and reviewed-version dates are listed separately in the article.

These classes are editorial intervention categories, not a claim of mutually exclusive methods. The article separates token savings, FLOPs and actual wall time. RPO is supplementary because its headline contribution targets gradient/training cost; SAR is supplementary because its original release is 2025 and it studies a general single-turn reward; CogRouter remains relevant prior work within adaptive effort but is not a sixth main section. All archived papers and historical assets are preserved.

Quality evidence used for the selection:

- TAB: multi-turn formulation; fixed solver versus trainable budgeter; auxiliary-task training followed by unchanged-policy transfer; overhead-inclusive latency; ablations and explicit terminal-credit limitation. Source: v4 §3.3, §4.3, Appendix A. Token consumption in training differs from allocated budget; the blog uses n_t for consumption.
- PUMA: five-backbone/five-benchmark comparison; detector and verification ablations; actual wall-clock breakdown with counterexamples where token-saving baselines are slower; public code; train/test separation discussed in Appendix F. Critical qualification: its PUMA-RL expands question + RD-selected prefix + closing think token and generates only the answer phase. This does not directly demonstrate an autonomously sampled online stopping controller. §3, §5.2, §6.2, Appendix F.
- RACE: matched-backbone SFT/RL ablations and explicit deletion procedure. Main efficiency evidence is reasoning tokens, not robot latency. Fixed reference actions/observations make the likelihood check a proxy, not an intervention rollout.
- STEPWISE: GUI benchmarks, learned monitor ablations and periodic-verification comparison. Monitoring/verification inputs differ; no end-to-end PPO or process reward is asserted.
- SpecAgent: matched Weak/Strong serving and concurrency; quality/resource tables; repair/review/semantic-match ablations; actual wall time plus separately estimated FLOPs/HBM-read. Read-only steps can execute; first mutation is held. No RL training in the main protocol. Physical manipulation changes the environment almost every step and limits available speculative depth. §3.2–3.3, Table 2, §5.

Additional newly screened sources, not main selections:

- Calibration-Free Early Stopping via E-processes, 2610.25212v1 (October): elegant zero-calibration rule and explicit probe cost, but signal-null crossing guarantees do not bound semantic answer-error probability. Predominantly replay/probe-based math evaluation; no wall-clock robot claim. Retained as an alternate early-exit reading.
- Early Stopping in LRMs: A Survey and Empirical Audit, 2611.00166v1: supports evaluating probe overhead, reasoning leakage and retained justification; a survey/audit is not a representative training mechanism.
- ESTAR, 2602.10004v1: classifier + stop-cue SFT + GRPO. §5.3 names MATH-500 for SFT data construction; split scope would need clarification before treating MATH-500 gains as clean held-out evidence. Its principal figures emphasize length; not preferred over PUMA's explicit timing and data-separation account.
- ESPO, 2605.29860: early termination of failed training rollouts. Useful for training resource efficiency, but its rollout-token saving does not directly establish faster deployment.
- AAPT, 2607.28399v1: moves decoding out of event-time critical path with precompiled policy trees; paired controlled evidence, but narrow pre-enumerable scenario, aggregate external-benchmark tie, and late-bound control outside current contract. SpecAgent selected for broader tool-agent cost/quality evidence.
- RECAP, 2609.27156v1: semantic dependency graph and gold-answer-likelihood efficacy reshape GRPO advantages in mathematical reasoning; useful recent training mechanism, but no physical agent transfer or direct timing claim established here.

No independent reproduction, official-code execution, or GPU workload was performed. The shortlist is an evidence-based editorial choice within the checked scope, not a global quality ranking.

## Earlier Venue-Oriented Selection

The user subsequently required the three papers other than RACE and STEPWISE to have verified top-conference acceptance. Current main papers: RPO (ACL 2026), SAR (ICLR 2026), CogRouter (ICML 2026), RACE and STEPWISE. MORE and TAB remain archived but are removed from the main article.

Acceptance was checked at the official proceedings:

- RPO: https://aclanthology.org/2026.acl-long.816/
- SAR: https://proceedings.iclr.cc/paper_files/paper/2026/hash/fc30caeb45721bab13507c50199e6403-Abstract-Conference.html
- CogRouter: https://proceedings.mlr.press/v306/yang26cz.html

SAR first appeared as arXiv:2509.05489 on 2025-09-05; its selection is based on the 2026 conference version, not a claim of first release in 2026. CogRouter uses the official PMLR PDF because its arXiv and proceedings metadata/numbering differ.

SAR source check: §3.1–3.2, Appendix B.1/B.3, Table 9. PPO includes a critic. Conditional scoring reuses log-probabilities; standalone scoring removes the user prompt but retains the system prompt. The SAR-GRPO branch uses Dr. GRPO without std normalization. The plotted token importance is explanatory, not the implemented per-token environment reward. Official code is published, but direct raw requests to the README-named ppo.sh returned 404 in this check; no runnable reproduction claim is made.

CogRouter source check: official §3.3–3.4, Algorithm 1, Appendix E, Table 5. Four-level CoSFT precedes group-relative CoPO. Successful trajectories are expanded with four alternative thoughts at each fixed observation/action; confidence is the mean action-token log probability. Within-step standardization and softmax weights redistribute the trajectory advantage. Failed trajectories are not expanded. There is no learned critic or direct length penalty. No simulator reruns of the alternative thoughts are part of that scoring procedure.

## Superseded Selection

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

The current article displays 14 original PDF figure regions rendered at 3.5x. Historical assets remain preserved and inactive entries are marked in provenance.json. No generated redraws, annotations or added source overlays. See provenance.json for authors, versions, licenses, pages, crops and checksums. The page provides five paper references at its end; figures have descriptive alt text and no image hyperlinks. Source PDFs remain unchanged.
