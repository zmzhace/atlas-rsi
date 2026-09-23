# Atlas-RSI：受约束、可复现的自我改进控制面

[![CI](https://github.com/zmzhace/atlas-rsi/actions/workflows/ci.yml/badge.svg)](https://github.com/zmzhace/atlas-rsi/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Atlas-RSI 将当前较成熟的 RSI / Agent 自改方法组合成一条能实际运行的闭环：候选生成、隔离求解、隐藏评测、安全锚点、多随机种子统计、保守晋升、失败记忆、离线 replay、断点续跑与回滚。

它不是“让模型任意改写自己”。系统给予语义策略足够的探索自由，同时锁死目标、权限、隐藏集、安全门、审计与发布权。

## 已实现能力

| 研究方法 | 工程实现 |
|---|---|
| A-Evolve-Training | 不可变 Constitution、同基线 fork、受限 mutation surface、policy-only promotion |
| DGM / Hyperagents | 候选谱系、分支档案、父子版本、失败分支保留 |
| KSI / PAST-Bench | promoted lesson、dead-end registry、SQLite 事件日志与路径证据 |
| Dream-RSI | 从已执行分支离线估计 mutation 方向，避免重复昂贵实验 |
| RQGM | evaluator 仅可在 epoch 边界、通过 anchor 且经人工批准后升级 |
| Self-Harness / SIA | Harness、知识和搜索策略默认可改；权重单独注册、验签、门控发布 |
| Agent Control | 容器权限边界、预算、隐藏测试、双评审、回滚和人工发布门 |

## 方法论：改什么、限制什么、自由放在哪里

默认可改：

- `harness`：提示、上下文、验证、重试、工具编排；
- `knowledge`：成功经验、失败模式与检索策略；
- `search_policy`：探索轴、候选分配与 replay 建议。

默认锁定：

- 目标、Constitution、系统权限和审计日志；
- holdout、anchor 和 private reviewer；
- 模型权重与 evaluator；
- 网络访问和宿主机能力（Docker runner 默认断网、drop capabilities、只读根文件系统）。

高自由度放在“如何解决问题”，零自由度放在“谁定义成功、谁能发布、能访问什么”。权重和 evaluator 属于高风险面，即使显式启用，也仍要求 anchor、指标阈值、完整性校验与 `operator_approved=True`。

## 快速运行

```bash
git clone https://github.com/zmzhace/atlas-rsi.git
cd atlas-rsi
python3 -m pip install -e .
atlas-rsi run examples/experiment.json
atlas-rsi inspect runs/example.sqlite3
```

输出包括：

- `runs/example.sqlite3`：追加式事件与逐轮快照；
- `runs/example-report.md`：轮次轨迹、晋升、拒绝原因、replay 建议与最终机器状态。

恢复中断实验时，把配置中的 `resume` 改为 `true`。引擎按 `completed_rounds` 恢复；没有发生晋升的轮次也不会重复执行。

## 接入真实 Agent

`solver_command` 是外部 Agent 的稳定适配层。每次调用会收到：

- `ATLAS_TASK_ID`：公开任务标识；
- `ATLAS_SEED`：本次随机种子；
- `ATLAS_HARNESS_PATH`：候选参数 JSON；
- 工作目录：只包含公开 fixture，求解结束后才注入 `.atlas-private`。

生产实验建议使用 Docker：

```json
{
  "runner": {
    "type": "docker",
    "image": "your-agent-image@sha256:PINNED_DIGEST",
    "memory": "2g",
    "cpus": "2",
    "pids_limit": 256
  },
  "solver_command": ["python", "/agent/solve.py"]
}
```

`local` runner 仅用于受信代码和快速开发。模型生成代码应使用容器或更强的外部 sandbox。

每个任务可配置一个主评价器和一个独立复核器：

```json
{
  "test_command": ["python", ".atlas-private/test_solution.py"],
  "review_command": ["python", ".atlas-private/review_solution.py"]
}
```

两者都通过，任务才计为成功。

## 使用 LLM 生成候选

内置 OpenAI-compatible chat-completions 适配器；它只能提交声明式 payload，不能改 Constitution，也看不到 holdout 或 anchor 描述。

```json
{
  "proposer": {
    "type": "openai-compatible",
    "model": "YOUR_MODEL",
    "base_url": "https://YOUR_ENDPOINT/v1",
    "api_key_env": "OPENAI_API_KEY",
    "allowed_surfaces": ["harness", "knowledge", "search_policy"]
  }
}
```

候选 payload 会合并到当前 incumbent，因此每一轮都是可审计的增量 mutation。

## 晋升逻辑

```text
同一 incumbent fork 候选
  → solver 只看公开任务
  → solver 结束后注入隐藏 evaluator/reviewer
  → dev + holdout + anchor × seeds
  → 置信区间保守增益 + safety + cost gate
  → 单一胜者晋升，其他分支进入 dead-end registry
  → 快照、报告、replay policy；保留回滚路径
```

达到 `minimum_trials_for_ci` 后，系统使用“候选下置信界 − incumbent 上置信界”判断是否满足最小提升；样本不足时退回直接分数差，并在配置层明确这一事实。

## 权重与评价器发布

`CheckpointRegistry` 只管理外部训练产物，不自行训练或加载权重。它记录 SHA-256、父 checkpoint、指标与 anchor 状态；发布要求：

1. `allow_weight_updates=true`；
2. anchor 全部通过；
3. 指标达到阈值；
4. 人工明确批准；
5. 使用前再次 `verify()` 检查文件未被替换。

评价器更新同样默认关闭，并且只能在 epoch 边界通过固定 anchor 与人工批准后生效。

## 验证

```bash
python3 -m compileall -q rsi_lab demo.py tests
python3 -m unittest discover -s tests -v
atlas-rsi-demo
```

测试覆盖核心晋升/拒绝/回滚、统计区间、无晋升轮次恢复、SQLite、隐藏测试时序、双评审、超时、checkpoint 篡改检测、replay、候选 provider 与完整 CLI 实验。

## 诚实边界

Atlas-RSI 是可运行的研究控制面，不是已经实现通用递归智能跃迁的系统。它不提供 GPU 训练器、云级恶意代码隔离或对未知任务的正确性证明；这些应由外部训练平台、sandbox 与领域 evaluator 提供。见 [SECURITY.md](SECURITY.md)。

## License

MIT
