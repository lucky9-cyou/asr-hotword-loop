# ASR Hotword Loop · Qwen 动态热词实验室

本地运行的「Qwen3-ASR → Qwen LLM → 会话热词库 → 后续 ASR token 偏置」演示。

## 启动

```bash
cd /workspace/workspace/yanglin/asr-hotwords-demo
./run.sh
```

打开 http://localhost:7860 。远程机器可打开 http://<服务器地址>:7860 上传音频；浏览器麦克风通常要求 HTTPS 或 localhost，推荐 SSH 转发：

```bash
ssh -L 7860:127.0.0.1:7860 <用户名>@<服务器地址>
```

然后在电脑浏览器打开 http://localhost:7860 。GPU 运行需要进程能访问 `/dev/nvidia*`；受限沙箱需要外部 GPU 权限。

默认使用 `/workspace/model/Qwen3-ASR-0.6B`（GPU 0）和 `/workspace/model/Qwen3.8-27B`（GPU 1）。后者是目录内模型名称，模型配置为 `qwen3_5` 架构。首次加载需等待，共约 60 GB GPU 权重/运行空间，后续请求复用模型。服务监听 0.0.0.0，面向可信本机/内网演示，无身份验证。

```bash
ASR_DEVICE=cuda:0 LLM_DEVICE=cuda:1 PORT=7860 ./run.sh
# 可显式改用更小模型：
LLM_MODEL=/workspace/model/Qwen3-4B-Instruct-2507 ./run.sh
```

ASR 复用现有 `envs/lm-tts`，项目 `.deps` 隔离安装 qwen-asr0.0.6 和 transformers4.57.6。LLM 通过持久子进程复用 `envs/sglang-omni` 的 transformers5.12.1，避免版本冲突，不调用云端模型。

## 演示步骤

1. 页面创建独立会话，选择 4 / 6 / 8 秒音频块。
2. 点击“跑一遍示例”，或持续录音，先说「我姓李，名字的第二个字是春风和煦的最后一个字」。解释指向的是 **李煦**。也可以在文本入口测试「我叫李旭，是春风和煦的煦」。
3. 观察 LLM 输出、证据与热词库；只有真实模型提取到的词才加入。
4. 后续说「请把这份文件发给李煦」。页面同时展示无偏置与有偏置转写。
5. 选择某块、某个解码步骤，比较候选 token 的偏置前后概率。可调偏置强度、删除热词或重建会话。

也可以输入转写文本直接测试 LLM，随后上传/录入音频。文本输入明确跳过 ASR，不会产生虚构概率。识别本身正确、热词不相关、声学支持不足时，两路结果可以相同。

## 伪流式行为

浏览器持续采集 PCM，按所选时长组成 WAV，顺序上传。一块处理顺序为：读取当前热词快照 → 同音频无偏置解码/有偏置解码 → LLM 读取最近若干块转写 → 更新热词 → 返回。下一块自动使用新库，热词不会追溯影响当前块。录音可在模型推理时继续，队列过长时停止采集，避免无限堆积。网页以 SSE 显示各阶段。

这是独立音频块伪流式，不是 Qwen 原生增量流式；没有跨块 acoustic KV cache、重叠窗或 VAD 断句，句中切块可能损失字词。LLM 从最近转写上下文提取，以支持解释跨块。双路解码用于观察效果，有额外耗时。会话热词存于服务内存，重启即清空，最多 32 条。

## 概率和偏置

`HotwordLogitsProcessor` 在自回归解码中加 logits：热词前缀匹配时加入完整偏置；热词起始只加入 0.2 倍偏置并要求模型原始分数接近最高候选。前缀支持 token 和解码字符匹配，并有分数门限，减轻不相关热词强插入。它是演示启发式，不是完整上下文偏置 trie/beam-search 系统。

图表的 `prob_before` / `prob_after` 是**同一音频、同一偏置解码前缀下**，加偏置前后完整词表 softmax 的真实 token 概率。仅展示 top 候选及被加分 token，不重新归一化；这些条形总和不要求为 100%。它们不是整词置信度，也不是两条独立解码路径最终结果的概率。原始/偏置两路转写则来自各自独立的 greedy decode。

## API

- `GET /api/status`：模型与就绪状态。
- `POST /api/warmup`：加载两个模型。
- `POST /api/sessions`：创建会话。
- `POST /api/sessions/{id}/chunk`：multipart `file`（0.1–30 秒 WAV）和 `bias`（0–8）。
- `POST /api/sessions/{id}/learn`：JSON `{"text":"转写"}`，直接测试 LLM。
- `GET /api/events/{id}`：SSE 阶段事件。
- `DELETE /api/sessions/{id}/hotwords/{word}`：删除热词。

主要代码：`server.py` 会话闭环；`asr_engine.py` 解码与概率；`llm_engine.py` LLM 和版本隔离；`static/` 页面。

## 已验证与当前限制

- 14 项会话/API 集成测试通过：下一块生效、删除、会话隔离、输入校验、LLM 失败时保留 ASR 等。
- 已用真实 Qwen3-ASR 验证合成 followup 音频：bias=8 将两处“李旭”改成“李煦”；首次“煦”token 的同前缀概率从 0.0785% 到 70.0729%。详见 `static/samples/evaluation.json`。
- 已用真实 Qwen3.8-27B 验证明确解释用字与“词语的最后一个字”表达；普通天气句不增加热词。详见 `llm-verification.json`。
- **已知失败案例**：当 ASR 输出“我叫李旭，姓李。旭是春风和煦的旭”，LLM 仍可能选择“李旭”。这类解释本身也损坏的输入尚未可靠解决，不能保证每次纠错。
- 当前热词库支持累加、手动删除和重建会话；新提取的更正名不会自动撤销此前已学到的错误同音名。若第一句话跨块造成错误名先入库，可手动移除。
- 完整语音闭环可运行 `python verify_live.py`（使用 lm-tts Python），结果记录到 `live-verification.json`。模型实际测试和 API 隔离测试是不同层级，后者使用 stub 保证可复现。
