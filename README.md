# Atlas-RSI：受约束的集成式自我改进控制面

[![CI](https://github.com/zmzhace/atlas-rsi/actions/workflows/ci.yml/badge.svg)](https://github.com/zmzhace/atlas-rsi/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Atlas-RSI 把当前 RSI 研究中最可靠的部件组合成一个小型、可运行、可审计的控制面。它不追求“无限自改”，而是让系统只在明确授权的表面上产生候选改进，并通过隐藏集、安全锚点和回滚门槛决定是否晋升。

> 当前状态：研究原型。默认只演化声明式 Harness 参数，不执行任意模型生成代码，也不会自行修改目标、权限或安全边界。

## 汇总了哪些方法

| 研究路线 | Atlas-RSI 中的对应部件 |
|---|---|
| A-Evolve-Training | 不可变 Constitution、固定基底、memory-free candidate、policy-only promotion |
| DGM / Hyperagents | 候选谱系档案；允许 Harness 与搜索策略形成不同分支 |
| KSI / PAST-Bench | 可审计知识库、失败登记、持久化路径证据 |
| Dream-RSI | 用历史评测记录重放候选策略，不重复执行昂贵任务 |
| RQGM | 评价器只能在 epoch 边界更新，并必须通过固定 anchor |
| Self-Harness / SIA | Harness 是默认可变面；权重更新有独立接口且默认关闭 |
| Agent Control | 明确权限、预算、审计、晋升门和一键回滚 |

## 核心边界

默认允许改变：

- `knowledge`：可复用经验与失败教训；
- `harness`：提示、工具选择、重试、验证和上下文策略；
- `search_policy`：下一轮探索轴、预算和候选分配。

默认禁止改变：

- 最终目标和 Constitution；
- 文件、网络和凭据权限；
- 安全锚点与隐藏评测；
- 审计日志；
- 模型权重和评价器。

权重更新和评价器演化必须由操作员显式打开，并分别通过模型发布门和 anchor gate。

## 一轮循环

```text
失败分析 / 知识检索
        ↓
选择改进目标与探索轴
        ↓
从同一已晋升基线 fork N 个声明式候选
        ↓
隔离评测：dev + holdout + safety + cost + anchors
        ↓
晋升门：隐藏集提升、无安全回退、成本受控
        ↓
更新谱系、知识库、dead-end registry 和 rolling policy
        ↓
保留上一版本，可随时回滚
```

## 运行演示

无需第三方依赖：

```bash
git clone https://github.com/zmzhace/atlas-rsi.git
cd atlas-rsi
python3 demo.py
python3 -m unittest discover -s tests -v
```

也可以安装为本地包：

```bash
python3 -m pip install -e .
atlas-rsi-demo
```

演示域使用声明式 Harness 参数，不执行候选生成的任意代码。真实接入时，应把 `Domain` 接口连接到容器化 runner，并保持隐藏集和安全锚点对 proposer 不可见。

## 设计原则

1. **非对称自由**：可比较性与安全相关的轴零自由；产生能力增益的语义轴高自由。
2. **候选不直接覆盖基线**：每轮候选从当前已晋升版本 fork，失败候选只进入档案。
3. **评测与生成隔离**：proposer 看不到 holdout 和 anchors。
4. **以证据晋升**：不能只凭 Agent 的自我判断或 dev 分数。
5. **评价器慢于 Agent 演化**：只允许在 epoch 边界替换评价器。
6. **权重更新慢于 Harness 演化**：先证明非参数改进，再进入高成本训练门。
7. **完整可回滚**：每次晋升保留父版本、分数、理由和 Constitution 指纹。

## 下一阶段

- 接入真实 coding-agent 容器 runner；
- 增加统计置信区间和多随机种子评测；
- 加入 Dream replay policy optimizer；
- 增加模型适配器训练与 checkpoint registry；
- 引入双评价器和人工审批的高风险发布门；
- 发布完整 benchmark fixtures 与可复现实验报告。

## License

MIT
