# ECHO

**Explanations as Context for Hotword Optimization in Speech Recognition**

> 越听，越懂你。把一次解释，变成下一次识别的记忆。

英文论文初稿，先使用 Markdown。作者、单位、目标会议与最终版式尚未指定。

## 阅读顺序

1. [Introduction](introduction.md)：问题动机、解释中也存在 ASR 错误的例子、方法概览与研究问题。
2. [Related Work](related_work.md)：上下文偏置、LLM 纠错、用户纠正与会话记忆，以及 ECHO 的具体区别。
3. [文献核查记录](source_notes.md)：原始来源、支持的论断及需要注意的重叠。
4. [BibTeX](references.bib)：为后续转换 LaTeX 保留的引用元数据。

## 当前写作边界

目前仅写 Introduction 和 Related Work，不虚构 Abstract 中的结果、数据集、对比实验或显著性结论。工程实现与小规模验证记录可参考[项目说明](../README.md)和[验证目录](../artifacts/verification/README.md)，不能当作论文的独立评测结果。

- 核心研究点是从可能有错的口头解释中恢复词汇，并把该词汇用于后续 ASR 解码；不宣称首次使用会话上下文、记忆或用户纠正。
- 每次处理完整音频，流程可重复调用；不是原生流式或伪流式算法。
- 当前模型参数冻结，偏置为启发式；题名中的 Optimization 不代表已经训练了可学习目标或求解了全局最优策略。
- 现有词库主要累加；更正后旧同音词仍需手动删除。首 token 支持不足、解释歧义、错误记忆自我强化和无关热词干扰，均需纳入后续方法与实验。
- 文献检索截至 2026-11-06。预印本与正式出版物在来源记录中区分。

## 后续研究设想

- [Agent 动作与端到端 thinking](agent_and_end_to_end.md)：增删改热词、修订历史，以及把词汇推理纳入端到端模型。
- [ECHO-Bench](../benchmarks/echo_bench/README.md)：多轮评测协议与开发样例，正式数据与 evaluator 待建设。

下一步补 Method、数据采集与对比实验；完成实验证据后再写结果与贡献强度。
