# Tau 的 RSI 改进方案：从可验证的 Skill 演化开始

> 状态：设计稿，尚未实现 RSI。基线为 `huggingface/tau` 的 `775ed0519edab1ebedd340a1819940a2d98fc889`（`tau-ai` 0.4.12）。本文中的 RSI 指 recursive self-improvement，即利用前一批任务的外部反馈改进后续任务的 agent 行为。

## 1. 研究依据和目标边界

- 参考论文：[Evo-Harness: Context-to-Harness Skill Compilation for Self-Evolving Agents](https://arxiv.org/abs/2608.15071)；[作者公开实现](https://github.com/A-EVO-Lab/a-evolve/tree/release/evo-harness)。Tau 本身的分层意图见 [roadmap #1](https://github.com/huggingface/tau/issues/1)。
- 论文中的 solver 模型及代码保持冻结，在线更新的是外部 Skill harness。流程为 `Select → Inject → Execute → Reflect → Compile → Update`。只在失败或负面反馈后抽取候选经验，再依据证据合并、修订或跳过；依赖外部验证器提供成功与否。**不能把模型自评当作任务成功证据。**
- 第一阶段实现“Skill 级 RSI”：跨任务积累可审查、可回滚的操作经验。第二阶段才考虑“代码级 RSI”：由 agent 提议修改自身实现，隔离运行测试和基准，经人工批准后合入。论文没有验证自动修改 agent 源码，因此两者的效果和风险要分开报告。
- 验收目标是同一模型、同一工具集、同一任务集和预算下，演化版在留出任务上的外部验证通过率高于冻结基线，且没有超过预设的回归、成本和延迟上限。单次案例或模型自述不算验收。

## 2. 源码审阅结论

已按包扫描全部 120 个 Python 源文件的语法树、类和函数，并沿模型流、Agent 循环、会话、扩展、资源、CLI/RPC/TUI 主调用链审阅实现与测试；下表给出影响 RSI 的实际入口。仓库约 5.9 万行 Python 源码，主要体量在 `tau_coding` 的 TUI 与本地模型集成，因此首阶段避免改动这些区域。

| 层 | 当前实现及对 RSI 的含义 |
| --- | --- |
| `tau_ai` | `provider.py` 定义统一模型接口；`fake.py` 可做确定性测试；各厂商适配器与 `stream.py` 将响应规整为事件。第一阶段不改 provider，不让演化算法依赖某家 API。 |
| `tau_agent` | `loop.py` 的 `run_agent_loop` 处理模型、工具、终止与错误，`harness.py` 管消息队列、取消和运行状态；`events.py` 提供 `agent_start/end`、`turn_start/end` 和工具事件。这里保持通用，不塞入 Skills 或评测策略。 |
| 会话存储 | `session/entries.py` 已有 `CustomEntry`，`session/storage.py` 提供 JSONL 持久化与原子追加，`session/memory.py` 重建活动分支。可用带命名空间的自定义条目记录任务与选用的 Skill 版本；大型轨迹/评测结果放独立存储，条目只留引用和哈希。 |
| Skill 资源 | `tau_coding/skills.py` 从用户和受信任项目的 `*/SKILL.md` 加载，`system_prompt.py` 把可见 Skill 的名称、描述、路径列入提示词，正文由模型读取或 `/skill:name` 展开；`resources.py` 定义优先级。现有机制不会自动按任务选择和注入演化 Skill 正文。 |
| 扩展 | `extensions/api.py` 可注册 `input`、`turn_end`、`tool_call`、`tool_result`、`agent_settled` 等事件，写 `append_entry`，增加提示词章节、命令和工具；`runtime.py` 隔离处理异常并管理代际。第一版适合做 opt-in 扩展。 |
| CodingSession | `session.py` 在 `prompt()` 中先跑 `input` hooks，再展开 prompt，随后执行 harness；`agent_settled` 在运行和持久化收尾后发出。`reload()` 要求空闲并重建资源与扩展，适合作为版本切换边界。`session_preparation.py` 负责候选会话的准备和接管。 |
| 入口与界面 | `cli.py`、`rpc.py`、`tui/app.py` 消费同一个 CodingSession。第一版不增加 TUI 控件，可先提供扩展命令与 `tau -e PATH` 显式启用，保持 print/RPC/TUI 行为一致。 |

### 当前缺口

1. 缺少任务级 ID、输入指纹、Skill 版本快照和反馈的统一记录；仅凭会话文本无法可靠归因。
2. 缺少受预算约束的 Skill 选择器；全部 Skill 索引虽然简短，正文注入仍需主动读取或手动调用。
3. 缺少以外部测试/评测为真值的反馈输入、失败反思、批量编译、候选验证、版本化发布和回滚。
4. `input` hook 可以转换用户文本，但会与 prompt 模板、显式 `/skill:name` 展开、扩展输入、运行中 steer/follow-up 交互；若直接注入，必须精确限定首轮、保留原始用户请求、记录所选版本并防止重复注入。更稳妥的产品化落点是 `CodingSession.prompt()` 中“输入展开后、harness 启动前”的单一任务准备接口，等原型证明需要后再添加；不把选择策略放进 `tau_agent`。
5. 当前 `agent_settled` 表示运行结束，不表示任务通过。正确性必须来自隔离的验证器，且需区分失败、取消、工具异常、超时与未知结果。

## 3. 拟议架构

```text
任务输入 ──► 任务准备/Skill 选择 ──► 冻结的 Tau solver ──► 外部验证器
                  │                         │                  │
                  └─ 版本快照/预算             └─ 轨迹/成本        └─ 结构化反馈
                                                                    │
           下一批任务 ◄── 原子发布/回滚 ◄── 候选评测 ◄── 反思与批量编译
```

**组件位置。** 在 `src/tau_coding/extensions/` 下增加独立可选的 `evo_harness/` 扩展包；负责事件捕获、选择和审计。反思、编译与 benchmark runner 放 `src/tau_coding/evo/` 或独立实验包，避免把训练循环和验证器塞进交互会话。首版由 CLI/脚本显式跑一批任务，再由扩展消费已发布版本。`tau_agent` 与 `tau_ai` 保持不变。必要时只在 `tau_coding/session.py` 加一个最小的任务准备接缝，先以扩展原型验证。

**任务记录契约，建议 JSON Schema/Pydantic v1。**

| 对象 | 必需字段 |
| --- | --- |
| `TaskRun` | `run_id`、`task_id`、输入哈希、任务类别、仓库/基准版本、模型与 provider、温度等配置、工具集哈希、基线 commit、Skill 集版本、选用条目及顺序、起止时间、token/成本、状态。 |
| `VerifierFeedback` | `run_id`、验证器名称/版本、命令或检查项、退出码、通过/失败/未知、失败摘要、原始证据文件引用与哈希。只接受受信任验证器写入；不得让 solver 自填。 |
| `CandidateLesson` | `lesson`、`trigger`、`evidence`（任务、失败检查项、轨迹片段引用）、`scope_hint`、反例/适用限制、来源运行 ID。缺证据不进入编译。 |
| `SkillRevision` | 稳定 ID、版本、作用域 `general/topic`、触发条件、正文、证据引用、父版本、作者/生成器、候选评测结果、状态 `draft/approved/rejected`、正文哈希。 |

**选择。** 先用可解释的关键词、任务类型与触发条件检索，按固定预算（如最多 3 条、正文合计最多 2,000 tokens；阈值由配置给出）排序。保留一般经验与任务类型经验两个层次，记录候选、分数、淘汰原因和最终注入顺序。检索不得查看验证答案或未来任务。显式用户 Skill 的优先级高于自动选择。

**注入。** 先在实验 runner 中给每个任务创建新 CodingSession，把所选 Skill 作为带来源的提示词章节注入，确保单任务内版本冻结。交互原型可在 `input` hook 中对空闲、普通用户首轮输入进行一次性结构化包装，但不得处理 `/skill:`、命令、扩展源输入、运行中的 steer/follow-up；包装必须可见且保留原文。若这些限制难以满足，应直接引入 `CodingSession` 的任务准备接缝。无论哪条路径，日志记录实际生效的提示词和 Skill 哈希，不记录凭据。

**反思和编译。** 仅对验证失败或明确负反馈的记录提取经验；每批结束由独立 evolver 比较候选与现有 Skills，执行 `ADD/MERGE/REVISE/SKIP`，去重并检查证据、范围、长度和冲突。模型输出只是候选提案。正式发布前，自动执行留出集回归与人工审阅；通过后写入暂存目录，使用原子替换更新当前 Skill manifest。运行中的任务继续使用旧快照，空闲时 `/reload` 或下一会话才生效。

**存储与回滚。** `~/.tau/evo/` 保存版本 manifest、Skill 修订、任务记录和验证证据；会话 `CustomEntry(namespace="tau.evo", ...)` 只存引用、哈希及生效版本。仅保存脱敏摘要，不把环境变量、密钥、私有文件正文复制进 Skill。支持 `evo status`、`evo diff`、`evo rollback <version>`；回滚更新 manifest，不删除历史任务/会话。项目级资源仍遵守 Tau 的 project trust；自动生成的 Skill 不应写入受信任项目目录或覆盖用户的同名 Skill。

## 4. 分阶段落地与验收

### P0：基线与可复现实验

- 固定 Tau commit、Python/依赖锁、模型配置、工具集、任务顺序和预算；用 git worktree/容器隔离每个 coding 任务。先选几十个有确定性检查的本地任务，划分开发、留出和重复运行集。
- 记录 no-skill、手写固定 Skill、仅检索历史案例、自动演化 Skill 四组；成功率由外部测试/检查器给出，报告 token、延迟、成本和失败类别。
- 退出条件：相同输入能复现记录，验证器能明确给出通过/失败/未知，基线不依赖人工目测。

### P1：记录与 Skill 选择（不更新 Skill）

- 增加可选扩展、任务记录 schema、版本 manifest 与选择器；利用 `agent_settled` 和 `append_entry` 捕获运行引用，验证结果由独立 runner 回填。
- 测试：FakeProvider + fake tools 检查选择顺序、预算、空集、显式 Skill 优先、重复注入保护、取消/中断、恢复时快照一致；Windows 和 POSIX 路径。
- 退出条件：能追踪一次运行用了哪几个确切版本的 Skill，以及验证器为什么判成某一结果。

### P2：失败反思与批量编译

- 实现 `CandidateLesson` 提取、`ADD/MERGE/REVISE/SKIP`、冲突/重复/敏感信息检测；候选保存在 draft，不自动加入活跃 Skill。
- 测试：失败含证据才产出候选；成功、自评成功、证据缺失和验证器未知不推动更新；相同批次幂等；损坏的候选被隔离。
- 退出条件：每条修订可从 Skill 正文追溯到失败证据，人工可审查 diff 与反例。

### P3：门控发布与回滚

- 在留出任务上比较候选与当前版本；按预设阈值检验整体通过率、关键任务回归、成本和上下文占用。通过后经人工确认发布；原子切换版本，支持一键回滚。
- 测试：运行中不切换、发布失败不破坏旧版本、reload/恢复/分支会话行为、并发写入、不同项目的信任边界。
- 退出条件：重复实验可观察到净收益且无超阈值回归；否则保持上一版本并记录失败原因。

### P4：代码级 RSI（独立研究线）

- 仅在 P1–P3 已稳定后，允许 agent 在隔离 worktree 中提出 Tau 自身补丁。先要求具体问题和预期收益，再运行格式、静态检查、单测、任务基准、安全/凭据检查；生成可审查 diff，人工审核后才合入。
- 与 Skill 演化分开存版本、报告和指标。agent 不能自动修改基准、验证器、批准规则或当前正在运行的 solver；禁止自我打分后自动发布。

## 5. 需要特别守住的边界

- **反馈污染：** 同一任务的 verifier 信息只能在任务完成后进入 evolver，不能进入该任务 solver 上下文；留出集不能参与候选生成。
- **提示词注入：** 工具输出、仓库文件和失败日志都是不可信输入。提取 `CandidateLesson` 时仅保留必要引用和摘要，发布前审查 Skill 指令。
- **策略漂移：** 若 Skill 改变了工具使用或验证流程，记录有效工具、提示词和配置哈希；不能把 provider/模型升级混入 Skill 实验。
- **长上下文与过拟合：** 限制总 Skill 数和 token；按任务类型选取；用留出项目、重复运行和负例评估迁移收益。
- **可信边界：** Tau 扩展是本机 Python 代码，不是沙箱；Shell/项目资源遵循现有信任语义。代码级修改的执行必须用独立 worktree/容器和资源配额。

## 6. 首个实现 PR 建议

第一个 PR 仅交付 P1 的最小纵切：`evo_harness` 可选扩展、任务/Skill 版本 schema、固定预算的确定性选择、只读 manifest、一次运行的审计记录以及 FakeProvider 回归测试。暂不调用 evolver、不写自动生成 Skill、不改 Agent loop，也不增加 TUI。补 `dev-notes/` 设计记录；用户可见命令出现时再更新 `website/content/`。用 `uv run pytest`、`uv run ruff check .`、`uv run ruff format --check .`、`uv run mypy` 验证。

## 7. 本次 Windows 基线记录

- `uv sync --dev --locked` 成功，使用 uv 管理的 Python 3.12.16。
- 在未修改任何 Python 源码前运行全量 `uv run --locked pytest -q`：`2184 passed, 62 failed, 5 skipped`。首个失败已单独复现：`tests/test_cli.py::test_run_print_mode_prints_final_assistant_text` 比较提示词中的工作目录时，期望 `D:/Rust/...`，实际为 `d:/rust/...`；其他失败尚未逐项归因，不能算作 RSI 引入的回归或直接断言同根因。
- 重点子集（Skills、扩展、Agent loop、Harness）为 `192 passed, 1 failed`；失败项 `test_failed_setup_cleans_every_registration_after_entry_symlink_retarget` 在创建符号链接时遇到 Windows `WinError 1314`（当前进程没有相应权限）。
- `uv run --locked ruff check .` 通过；`uv run --locked ruff format --check .` 通过（205 个文件）；`uv run --locked mypy` 在 `project_trust.py` 和 `session/storage.py` 出现 14 个 `fcntl/flock` 及 `unused-ignore` 类型错误，属于当前 Windows 环境下的未通过项。
- 后续 P1 PR 需要在相同 Windows 基线和 CI 平台各跑一次；本设计稿只新增 Markdown，未实现功能或改变测试结果。

