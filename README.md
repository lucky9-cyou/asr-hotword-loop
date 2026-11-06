# ECHO

**Explanations as Context for Hotword Optimization in Speech Recognition**

> 越听，越懂你。把一次解释，变成下一次识别的记忆。

把一次解释，变成下一次识别的记忆。每次提交一段完整音频，Qwen3-ASR 使用已有热词完成识别，Qwen LLM 理解转写并更新热词库，供下一次调用使用。

**统一流程，可反复调用；不自动切块、不做伪流式，也不区分“提取模式”和“识别模式”。**

## 论文与目录

论文先以 Markdown 撰写：[论文入口](paper/README.md) · [Introduction](paper/introduction.md) · [Related Work](paper/related_work.md)。

```text
paper/                   论文草稿、引用和 Agent/端到端研究计划
benchmarks/echo_bench/    评测协议与开发样例（非正式测试集）
scripts/                 真实模型验证脚本
artifacts/verification/  演示验证记录，不作为论文实验结果
static/                  Demo 页面及合成语音样例
server.py                会话与完整音频调用 API
asr_engine.py            Qwen3-ASR 解码和 token 偏置
llm_engine.py            解释理解、热词提取和当前 prompt
tests/                   API 隔离测试
```

## 启动

```bash
cd /workspace/workspace/yanglin/asr-hotwords-demo
./run.sh
```

打开 http://localhost:7860 。远程浏览器可通过 `http://<服务器地址>:7860` 上传音频；麦克风要求 HTTPS 或 localhost，可用 SSH 转发：

```bash
ssh -L 7860:127.0.0.1:7860 <用户名>@<服务器地址>
```

默认模型：

- `/workspace/model/Qwen3-ASR-0.6B`，GPU 0。
- `/workspace/model/Qwen3.8-27B`，GPU 1。此为本机模型目录名，配置架构为 `qwen3_5`。

进程需要 GPU 权限。首次加载大模型需要等待，后续复用。服务监听 0.0.0.0，无身份验证，用于可信本机/内网演示。

```bash
ASR_DEVICE=cuda:0 LLM_DEVICE=cuda:1 PORT=7860 ./run.sh
# 可选择已有的小模型：
LLM_MODEL=/workspace/model/Qwen3-4B-Instruct-2507 ./run.sh
```

当前脚本复用本机环境，换机器需调整环境路径：ASR 使用 `envs/lm-tts`，`.deps` 隔离安装 qwen-asr0.0.6 与 transformers4.57.6；LLM 常驻子进程使用 `envs/sglang-omni` 的 transformers5.12.1，避免版本冲突。不调用云端模型。新增依赖清单为 `requirements-local.txt`。

## 怎么体验

1. 点击“示例 1”提交一段完整的名字介绍，或者录音后停止提交。页面展示转写、LLM 输出和新增热词。
2. 点击“示例 2”或提交另一段音频，页面对比无偏置与有偏置转写；本次调用同样会继续让 LLM 更新热词。
3. 可继续第三、第四次调用。每次调用开始时读取热词快照，本次学到的新词从下一次调用生效。

录音最长 30 秒，到时自动停止并提交；上传超过 30 秒直接提示，不自动切割。没有后台音频队列。文本入口可单独测试 LLM（明确跳过 ASR，不会生成虚构概率）。热词支持手动删除或重建会话；会话存于内存，服务重启后清空，最多 32 条。

内置 WAV 是本地 Qwen3-TTS 生成的合成语音，来源见 `static/samples/provenance.json`。示例 1 说“我姓李，名字的第二个字是春风和煦的最后一个字”，示例 2 再次提到这个名字。页面现场调用模型，不使用预置转写或概率。

## LLM 提示词

当前完整提示词在 `llm_engine.py` 的 `SYSTEM_PROMPT`，每次提取会重新读取文本，无需重载模型即可调整。

输入中的姓名、解释词以及解释末尾的单字都可能被 ASR 写错。LLM 先尝试恢复解释词的合理写法，再确定所指汉字，只修正有依据的位置。例如“鱼明洞，鱼是年年有鱼的鱼”可以根据“年年有余”修正为“余明洞”，不能凭姓名常见程度把“明洞”改成“明栋”。用户明确否定或强调谐音表达时应尊重该说明。

`evidence` 必须原样摘自转写；`word` 允许纠错后与原文不同。不会用固定规则或写死答案代替模型输出。LLM 仍可能误解歧义表达，不能保证每次纠错；新提取的更正名暂不会自动删除此前已学到的错误同音名，可在页面手动移除。

## 偏置与概率

`HotwordLogitsProcessor` 在每次生成 token 前修改 logits：热词前缀匹配时加完整偏置；热词起始只加 0.2 倍，并检查其原始分数是否接近最高候选。实现包含 token/字符前缀匹配及分数门限，属于演示启发式。它没有完整的多候选路径保留与奖励回退，因此热词首字本来概率很低时仍可能无法触发。

图表展示同一音频、同一偏置解码前缀下，加偏置前后的完整词表 softmax 概率。只显示部分候选，不重新归一化；不是整词置信度，也不是两条独立转写路径的整体概率。无偏置/有偏置转写则分别由 greedy decode 得到，双路推理有额外耗时。

真实示例曾验证 bias=8 使两处“李旭”变为“李煦”；首个“煦”token 概率从 0.0785% 到 70.0729%，详见 [偏置校准记录](artifacts/verification/asr-bias-calibration.json)。该效果不代表任意音频都会改善。

## API 与验证

- `GET /api/status`：模型状态；`workflow=manual_calls`，`streaming=false`。
- `POST /api/warmup`：加载两个模型。
- `POST /api/sessions`：创建会话。
- `POST /api/sessions/{id}/transcribe`：multipart `file`（0.1–30 秒 WAV）和 `bias`（0–8），执行完整识别、学习与更新流程。
- `/api/sessions/{id}/chunk` 保留为兼容别名，名字不代表自动切块。
- `POST /api/sessions/{id}/learn`：JSON `{"text":"转写"}`，直接测试 LLM。
- `GET /api/events/{id}`：SSE 状态事件，不是音频流式解码。
- `DELETE /api/sessions/{id}/hotwords/{word}`：删除热词。

```bash
# API 隔离测试（测试替身，不加载模型）
../envs/lm-tts/bin/python -m pytest tests/test_server.py -q
# 真模型检查，需要已启动服务、GPU 和预热
../envs/lm-tts/bin/python scripts/verify_prompt.py
../envs/lm-tts/bin/python scripts/verify_live.py
```

[prompt-verification.json](artifacts/verification/prompt-verification.json) 记录新提示词实际回归结果；[live-verification.json](artifacts/verification/live-verification.json) 记录完整音频连续调用结果。[早期 LLM 记录](artifacts/verification/history/llm-verification.json) 保留早期提示词测试历史，不代表当前提示词表现。

当前版本已通过 18 项 API 隔离测试和 7 个真实 Qwen3.8-27B 提示词回归，包括解释词也被错写、明确否定、无解释的姓名提取和普通句不提取。上述案例通过不代表所有歧义输入都能正确恢复。
