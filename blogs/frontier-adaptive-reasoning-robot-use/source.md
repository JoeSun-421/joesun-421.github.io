# Adaptive Reasoning for Robot-Use LLMs: From Frontier Models to a Training Recipe

## Abstract

机械臂第一次靠近碗口时，模型可能需要结合两路相机判断遮挡、朝向和抓取位置。拿稳之后，连续几轮对准目标的动作可能只需要根据新观测做小幅修正。给每一轮相同的 reasoning effort，会让简单步骤付出过多延迟，也可能让恢复步骤拿不到足够计算。Claude 和 GPT 公布了推理分配机制，Qwen、GLM 与 DeepSeek 提供了更具体的 post-training 线索。结合 Gemini Robotics 与 PEAR 的具身实验，可以把问题落到一个可检验的决策上：在相同机器人状态下，提高 effort 能否改善完整任务的结果，以及这项收益是否足以覆盖增加的时间。

## Claude and GPT — Allocating Inference Compute

Claude Opus 4.6 的 system card 描述了一种模型内的 adaptive thinking。API 调用方开启 adaptive 模式并传入 effort，模型根据请求内容决定是否使用 extended thinking、以及思考多久。降低 effort 会改变模型调用长思考的倾向。这个接口把产品要承担的延迟和成本交给调用方设定，同时允许不同输入在同一个 effort 下花费不同的 token 数。Opus 5.5 的公开结果仍标明所用的 adaptive thinking 与 effort 设置，复现实验时这两项也应与模型版本一起记录。公开材料解释了接口和可观察行为，足以作为运行时参考。其具体 gating 训练信号仍是未知量。 [Opus 4.6 System Card][claude46] · [Opus 5.5 Model Notes][claude55]

GPT-5 system card 披露了另一个层次的选择。ChatGPT 的 router 根据任务复杂度、工具需求和用户意图在快模型与深度 reasoning 模型之间分流。GPT-5.1 进一步报告模型内的动态分配：Thinking 版本对简单问题更快，对较难问题投入更多计算。后续 API 用 `reasoning.effort` 调节同一模型的计算倾向，档位和默认值随版本变化。这三个层次分别对应系统选择 checkpoint、调用方设定 effort、模型在该设置下决定实际生成长度。对于一个固定 checkpoint 的机器人实验，后两层即可形成可控对照，外置 router 也可以只选择同一 checkpoint 的档位。 [GPT-5 System Card][gpt5] · [GPT-5.1 System Card][gpt51] · [Reasoning API Guide][gpt-api]

## Qwen, GLM and DeepSeek — Training Controllable Modes

Qwen3 报告给出了让一个 checkpoint 同时保持快慢两种能力的顺序。先用 long-CoT cold start 和 Reasoning RL 获得 thinking 能力，再用 continual SFT 融合两种模式。Thinking 数据由 Reasoning RL checkpoint rejection sampling 得到，non-thinking 数据使用独立整理的直接回答。Chat template 用模式标记区分两类样本，direct answer 保留空 thinking block。报告还通过在指定长度结束 thinking 后继续生成答案来做 budget control。迁移到机器人时，训练样本可以用同一 observation 和 action schema 表达两类成功轨迹。需要实测短预算的输出能否闭合为合法动作，因为 completion 被截断时甚至可能没有 JSON。Qwen3.8-27B 的 model card 单独列出 `enable_thinking`、`preserve_thinking` 和 `reasoning_effort`。三者分别控制是否生成思考、历史思考是否保留，以及本轮的推理倾向。 [Qwen3 Technical Report, sections 4.1–4.4][qwen3] · [Qwen3.8-27B Model Card][qwen38]

GLM-4.5 先训练 Reasoning、Agent 和 General Chat experts，再通过 unified training 将它们蒸馏进一个支持 deliberative 和 direct response 的模型。Agentic RL 用可验证的任务结果监督整条轨迹，错误的工具调用格式会令轨迹停止并得到零分。训练对模型生成的位置计算 loss，环境反馈作为 context。这个区分可直接用于仿真数据：控制器返回的实际位姿和成功信号需要进入下一轮输入，policy 更新只针对模型选择的 mode、reasoning 和 action。GLM 的交互轮数实验也提示，增加一次观测可以提供额外证据，机器人应分别计量多看一轮和多生成一段 CoT 的成本。 [GLM-4.5 Technical Report, sections 3.1 and 3.3][glm]

DeepSeek-V3.2 在同一段工具交互中保留 reasoning，收到新的 user message 后清除 reasoning，并保留工具调用和结果。V4.1-Flash 又公开了一套训练档位的配方：在输入中加入标量 effort $b$，对同一任务和同一档位采样多条轨迹，在该组内做 reward mean centering。设 $\ell$ 为一条轨迹的 reasoning token 数，$L_{\mathrm{norm}}$ 为参考长度，$C_{\max}$ 为扣分上限。报告使用以下长度项：

$$
r_{\mathrm{len}}(\ell,b)=-\min\left(C_{\max}, k(b)\frac{\ell}{L_{\mathrm{norm}}}\right).
$$

$k(b)$ 随 effort 指数递减，高档位因而承担更小的长度惩罚。组内比较会让同档轨迹争取更好的结果和成本折中，部署时仍需要从外部指定档位。报告也发现，在不同 coding scaffolds 上增加 effort 一致地拉长轨迹，成功率的变化却会出现平台和回落。迁移到机器人时，effort 的训练与在线分配应分别验证。同样，机器人每一步的 camera frame 都属于同一 episode，历史 reasoning 是否随新 frame 保留需要根据目标 checkpoint 的 template 验证。可以先用完整历史作为基线，把压缩已失效计划作为独立消融。 [DeepSeek-V3.2 Technical Report, section 3.2][deepseek32] · [DeepSeek-V4.1-Flash Technical Report, section 5.1.4 and Appendix B][deepseek41]

## Gemini Robotics and PEAR — Where Thinking Pays Off

Gemini Robotics 1.5 将 embodied reasoning 和 action generation 放在两处。ER 模型读场景、规划子任务并根据反馈更新计划，Thinking VLA 可以在生成动作前用语言把当前子任务拆成短动作。报告的 ER budget sweep 显示，相同的 budget 下，指点任务平均用掉的 thinking tokens 少于视频问答，指点的得分也更早趋于平缓。VLA 的独立 thinking on/off 实验则在多步操作中观察到 task progress 改善。这两组观测对应不同决策粒度，ER benchmark 的预算曲线无法直接给出 pose-level 控制的阈值。Gemini Robotics 2.5 的公开实例继续由 ER 跟踪任务进展、VLA 执行动作，提供了另一种闭环分工的实例。 [Gemini Robotics 1.5 Technical Report, thinking VLA and ER scaling experiments][gemini15] · [Gemini Robotics 2.5 Release][gemini25]

PEAR 则把 camera frames 和机器人 state 交给同一个 Qwen3.8-27B，每轮产出下一步末端位姿增量，再交给 controller 执行。其 system prompt 规定具名 JSON、厘米与弧度、相机坐标方向以及动作何时结束。训练全程用 `xhigh`，仿真成功给 episode-level reward。同任务、同初始状态的 rollouts 组成 GRPO 的 advantage 组，整条轨迹的 reasoning 和 action token 共享该轨迹的 advantage，更新采用不含 PPO clipping 的 REINFORCE policy loss。论文 v2 的固定档位对照里，LIBERO-only checkpoint 从 `xhigh` 切到 `low` 后，思考由每轮约 3.23k 降至 0.86k tokens，成功率从 74.4% 降到 21.9%。这是同一个 BF16 checkpoint 的全程降档结果。它表明直接压低每轮预算的代价很大，至于哪些轮次贡献了这段差距，需要对相同状态做局部干预才能回答。 [PEAR v2, sections 3.1–3.2 and 4.6, Appendices A and H][pear]

## A Turn-Level Recipe for Robot-Use LLMs

### Establish the Closed Loop

先用 LIBERO 跑通同一套 feedback loop。每轮提供任务描述、相机 RGB、当前末端位姿和夹爪状态，让模型输出一个固定 schema 的位姿增量。用 schema 和工作空间检查过滤无效动作，由 controller 执行并返回实际位姿与下一帧。若复用归一化的 OSC action，保留适配层的轴向、旋转、范围和 gripper 约定；若复现 PEAR，改用它的物理单位、逐量增量和停止条件。两种接口都可以做 effort 对照，动作含义或执行时长的变化会形成另一项实验因素。以 simulator success 结束 episode，并记录 timeout、context 耗尽与非法 action 等失败。对本地 Qwen 先在固定输入上核对实际 reasoning token、中止原因和动作格式，确认部署端对不同 effort 的响应。保存原始输出供排查，训练视图单列 reasoning text 和对应 action 参数，并用记录 ID 关联观测、单位与任务结果。

### Collect Matched Branches

用同一 checkpoint、prompt、controller 和观测历史先跑 Always-Fast 与 Always-Think。然后从两类策略访问过的状态抽样，保存可恢复的仿真与 controller state、随机数状态和相同的模型 context。从每个 snapshot 分叉，本轮分别用 Fast 和 Think 采样下一动作，随后用预先固定的 continuation policy 跑到结束，跨多个随机种子估计结果。固定 continuation 可以隔离当前一轮的 effort，重复采样可以区分真实的对照信号与单次动作的随机性。开局定位、持续靠近、接触后验证和失败恢复都值得覆盖。单看位移大小会漏掉夹爪已经闭合、但物体仍留在桌上的情况，因而应同时保存动作前后的图像与可用的进展信号。仿真对象真值用于离线评分，在线 router 的输入限于部署时可获得的观测。

设 $h_t$ 是第 $t$ 轮的观测及历史，$\Delta p(h_t)$ 是这轮切换为 Think 后估计的 episode 成功概率增量，$\Delta T(h_t)$ 是分叉后剩余总耗时的增量。一个可校准的 router 判据是：

$$
\Delta p(h_t) > \lambda \Delta T(h_t).
$$

$\lambda$ 表示愿意用每秒额外时间换取多少成功概率。因为多想一轮也可能省掉后续重试，$\Delta T$ 应计入分叉后的总耗时，允许为负。实际拟合可以先用靠近状态、上一动作执行结果、进展停滞等可观察特征训练小 router，在验证集调阈值。估计效用差的区间覆盖零时，将该状态保留为不确定样本。在线运行还需要为动作输出预留 completion 空间，并在决策超时或观测失效时重新获取状态。高频 actuator loop 由 controller 维持，LLM 的 budget 作用在下一次 pose-level 决策。

### Train and Evaluate the Switch

第一轮实验可以冻结 Qwen，只训练 router。若同一个 checkpoint 在两档都能生成有效动作，而且 learned router 在独立验证任务上超过匹配 Think 比例的随机策略，再把选择蒸馏进模型。SFT 数据包含同一接口下成功的 Fast 直接动作与 Think 轨迹；筛选依据来自 simulator outcome 和动作有效性，轨迹长度用于衡量成本。对接着做的 simulator RL，可以用 matched states 对 mode 选择提供分叉后效用差，用完整 episode 的 success 评估后续行为。若借用 DeepSeek 的 effort-conditioned 长度项，只给 reasoning tokens 计长度成本，并报告各档位的成功率和实际使用比例。只对 switch token 计算一项 loss 时，共享参数仍可能改变动作生成，需要保留固定档位评测来检查退化。

最终把 Adaptive 与 Always-Fast、Always-Think、Random-Matched-Rate 在相同初始状态上成对评估。先于测试定下相对 Always-Think 可接受的成功率差，再报告配对区间、端到端 median 和 p95、reasoning tokens、观察与重试次数，以及 invalid action 和未检测到的失败。把 router、prefill、decode、controller 执行与重试都记入总耗时；分析按任务和初始状态分组，避免将同一 episode 内的多轮当成独立样本。如果低 effort 只在极少状态能保住成功，或者选档的开销抵消了节省，固定高档位仍是这个 checkpoint 的合理基线。这套 recipe 的可行性取决于同一机器人任务中是否存在足够多能被提前识别的低成本决策。

相关研究方法可以参见 [Towards Efficient Closed-Loop Agents via Hybrid Reasoning](../dynamic-reasoning-robot-use/index.html)。

[claude46]: https://www-cdn.anthropic.com/c788cbc0a3da9135112f97cdf6dcd06f2c16cee2.pdf
[claude55]: https://www.anthropic.com/claude/opus
[gpt5]: https://cdn.openai.com/gpt-5-system-card.pdf
[gpt51]: https://cdn.openai.com/pdf/4173ec8d-1229-47db-96de-06d87147e07e/5_1_system_card.pdf
[gpt-api]: https://developers.openai.com/api/docs/guides/reasoning
[qwen3]: https://arxiv.org/html/2505.09388v1#S4.SS3
[qwen38]: https://huggingface.co/Qwen/Qwen3.8-27B
[glm]: https://arxiv.org/html/2508.06471v1#S3
[deepseek32]: https://arxiv.org/html/2512.02556v1#S3.SS2
[deepseek41]: https://arxiv.org/html/2609.19969v1#S5.SS1.SSS4
[gemini15]: https://arxiv.org/abs/2510.03342v1
[gemini25]: https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/
[pear]: https://arxiv.org/html/2611.18562v2
