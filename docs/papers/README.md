# Agent Skill 自演化前沿研究与重构参考论文集 (Reference Papers)

本目录归档了将 **SkillOpt** 重构升级为 **SkillPivot** 及后续演化研究所需的核心参考文献、完整技术报告与理论依据。

---

## 一、 核心参考论文清单

| 论文标识 | 论文标题 | 归档文件名 | 核心参考价值 |
| :--- | :--- | :--- | :--- |
| **SkillPivot**<br>*(arXiv:2609.29154)* | A Wrong Turn Does Not Ruin the Journey: Deviation-Guided Skill Self-Evolution for LLM Agents | [SkillPivot_arXiv_2609.29154.md](./SkillPivot_arXiv_2609.29154.md) | **本次重构的直接目标规范**。<br>定义了偏差断点检测、教师接管、成对后缀差分与先验防回归门控。 |
| **WikiSkill**<br>*(arXiv:2608.27454)* | WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution | [WikiSkill_arXiv_2608.27454.md](./WikiSkill_arXiv_2608.27454.md) | **系统工程与知识持久化参考**。<br>Google 提出的三层架构（Raw / Wiki / Skills），保证失败经验永不丢失。 |

---

## 二、 核心论文要点快速速查

### 1. SkillPivot 算法落地要点 (arXiv:2609.29154)

```text
SkillPivot 执行流水线
├── 阶段 1: Student Rollout ──► 收集学生交互多步轨迹 (1 至 T 步)
├── 阶段 2: Deviation Detection ──► 计算势能 V_t = w1·Exec_t + w2·Prog_t + w3·Div_t
│   ├── Exec_t: 工具执行状态码 (0/1)
│   ├── Prog_t: 基于 sentence-transformers/all-MiniLM-L6-v2 的 Observation 目标增量 (仅记历史最高超额值)
│   ├── Div_t: Action 嵌入向量的多样性距离 (惩罚死循环)
│   └── 波谷判定: 寻找最低点 t* 并校验后续步骤的持续低迷
├── 阶段 3: Teacher Rollout ──► 强模型接收 1 至 t*-1 前缀，在相同环境状态下接管走完后半程 (最多 10 步)
├── 阶段 4: Paired Suffix Contrast ──► 比对 [学生失败后缀] 与 [教师成功后缀]，提炼最小原子 Delta
└── 阶段 5: Regression-Gated Verifier ──► 先验语义审查 (No-Regression ≥ 0.5 且 Generalizable-Rule ≥ 0.5)
```

* **推荐超参数配置**：
  * 学生推理步数上限：20 步
  * 教师接管步数上限：10 步（接管失败可重试最多 3 次）
  * 审校评分阈值：$\theta = 0.5$
  * 嵌入模型：`sentence-transformers/all-MiniLM-L6-v2`

---

### 2. WikiSkill 架构设计要点 (arXiv:2608.27454)

* **三层工作区解耦**：
  1. `raw/`：不可变执行轨迹存储；
  2. `wiki/`：持久模式与演化审计日志（`patterns/`, `logs.md`, `skill-impact.md`）；**即使技能触发回滚，Wiki 永远不回滚**；
  3. `skills/`：供推理智能体加载的可执行技能代码（`SKILL.md` + `PURPOSE.md`）。
* **隔离设计原则**：
  * Inference Agent 执行任务期间严禁直接读取 Wiki 库，防止走捷径而破坏可执行技能本身的提炼质量。

---

## 三、 2026 年 9 月至 10 月前沿关联文献速览

在深入开发与拓展时，可进一步参考以下最新同向演进成果：

1. **Skill-V** *(arXiv:2610.11781, 2026.10)*:  
   * **核心要点**: 提出“可证伪契约（Falsifiable Contracts）”与结果锚定演化，终结技能库“只增不减、过度膨胀、规则冲突”的加法式自演化通病。
2. **Rep2Skill** *(arXiv:2609.39149, 2026.09)*:  
   * **核心要点**: 摆脱纯文本反思的浅层性与幻觉，利用模型隐层内部表征（Representation Trajectories）几何距离定位失败点并引导单模型自迭代。
3. **GSO (Generalizable Skill Optimization)** *(arXiv:2609.39148, 2026.09)*:  
   * **核心要点**: 揭露自演化技能在留出测试集上的过拟合陷阱；主张优化“元指南（Meta-Guide）”并在执行时即时动态生成针对当前任务的技能。
4. **CoSkill** *(arXiv:2609.04865, 2026.09)*:  
   * **核心要点**: 将技能演化建模为多智能体联合强化学习（Multi-Agent RL），推理 Agent 与元技能 Agent 端到端共适应。

---

## 四、 本地开发与代码引用指南

新 Session 或本地开发时，可直接在 Python 代码与文档中通过相对路径查看与引用：
* SkillPivot 全文: `docs/papers/SkillPivot_arXiv_2609.29154.md`
* WikiSkill 全文: `docs/papers/WikiSkill_arXiv_2608.27454.md`
