# 验证记录

这些文件记录本地 Demo 的工程验证，不是论文中的独立评测数据集，也不能据此宣称泛化能力或总体 CER/WER 改善。

- `live-verification.json`：两段合成语音的完整独立调用，展示先学习、后偏置。
- `prompt-verification.json`：7 条人工选择的文本回归案例，用于检查 prompt；与 prompt 调整过程相关，不是独立测试集。
- `asr-bias-calibration.json`：合成样例的不同偏置强度及真实 logits/probability 记录。偏置强度用此样例校准，不能把该样例当作独立效果验证。
- `history/llm-verification.json`：早期 prompt 的历史结果，包含失败案例，保留作开发记录。

音频和合成来源保留在 `../../static/samples/`，供 Demo 直接播放。新验证脚本位于 `../../scripts/`，在仓库根目录执行。推理时间受硬件、负载与预热状态影响；token softmax 不是整词置信度。
