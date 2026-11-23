# Ouro 昇腾 prefix cache 提交前验证

PR 1 的评审问题是，在一张昇腾卡上用原生 BF16 运行 Ouro 时，普通同步单流 prefill 能否安全复用整页前缀，热请求省了多少工作，冷请求付出了什么成本。主实验使用 Ouro-1.4B、`ascend_eager`、`last_exited`、greedy、四次循环和 16-token 页。这个测试分支以实验提交 `fca4514` 为基线，生产代码在 `vllm_rlt/`。正式服务用例和运行时配置回归集中在本分支第一个测试提交，供精简 PR 复用。

## 在 910B3 与 910B4 上运行同一套检查

在已有 fork 检出中执行下面的命令。独立工作树可与已有的 FP16 工作树并存。

```bash
git fetch https://github.com/NEMOyota/vllm-rlt.git refs/heads/test/ascend-prefix-pr1-validation
git worktree add --detach ../rlt-prefix-dual-tests FETCH_HEAD
cd ../rlt-prefix-dual-tests
```

两台机器检出同一测试提交，激活各自的 CANN / torch-npu 环境，指定本地的同一份 Ouro-1.4B 权重。完整运行还会调用 `git`、`pre-commit`、项目的 CPU 开发依赖（pytest、Ruff 等）以及 torch-npu 的设备 profiler。下面的卡号是示例，请替换为分配给实验的空闲卡。

```bash
# 910B3 八卡服务器，选一张分配到的卡
export MODEL=/models/Ouro-1.4B
ASCEND_RT_VISIBLE_DEVICES=0 python -m benchmarks.ascend_prefix_pr1_suite all --chip 910B3

# 910B4 服务器，在该机的 shell 中执行
export MODEL=/models/Ouro-1.4B
ASCEND_RT_VISIBLE_DEVICES=2 python -m benchmarks.ascend_prefix_pr1_suite all --chip 910B4
```

两个型号使用相同的阶段、精度、负载和判定规则。PR 1 每个 worker 使用逻辑卡 `npu:0`。B3 八卡节点的整机 HBM 和主存可以单列为机器规格，每份性能结果对应所选的一张卡。另选空闲卡复测时使用新的结果目录，分别保留每张卡的数据。

入口打印 `RESULTS` 的绝对路径。使用 `--chip` 及单卡选择器时默认创建 `results/ascend-prefix-pr1-<型号>-card<选择器>-<UTC时间>/`。测试前的独立进程记录运行时设备名、可见卡数、逻辑卡、该卡 HBM 字节数及操作系统可见物理内存。容器可重映射卡号，报告保留传给 `ASCEND_RT_VISIBLE_DEVICES` 的原值。型号冲突或指定多卡会使硬件阶段记为 INCOMPLETE。若运行时仅返回家族名，仍执行检查并保留数据，型号标为 unverified，不能据此声明指定型号已全量通过。可结合 `environment/npu-smi.txt` 和平台分配记录核实。省略 `--chip` 可沿用原有入口。

全量测量包含两种精度的完整矩阵、模型加载和设备 trace 导出，适合独占实验卡夜跑。可以先查看命令，或在已设定 MODEL 的机器上先采集设备记录与原生 BF16 服务基线。

```bash
python -m benchmarks.ascend_prefix_pr1_suite all --chip 910B3 --list
ASCEND_RT_VISIBLE_DEVICES=0 python -m benchmarks.ascend_prefix_pr1_suite --chip 910B3 \
  --only environment --only bf16-burst --only bf16-paced
```

如果 B3 节点给你分配了多张卡，可以先选一张跑全量，再在其他空闲卡上按顺序复测服务基线。比如两张复测卡是 3 和 7 时，每张卡会生成各自的报告。

```bash
for card in 3 7; do
  ASCEND_RT_VISIBLE_DEVICES="$card" python -m benchmarks.ascend_prefix_pr1_suite --chip 910B3 \
    --only environment --only bf16-burst --only bf16-paced
done
```

### 核对历史流量和最终候选提交

`--reference-traffic` 可指向本套件的完整结果目录，也兼容旧目录的 `{burst,paced}/traffic.json`。新目录取 `bf16-burst/traffic.json` 和 `bf16-paced/traffic.json`。要在 B3 复现一份已完成的 B4 负载，把这两份文件按原目录结构复制到 B3，设置 `B4_RESULTS` 为其父目录。每种流量在启动 HTTP 服务前核对对应归档中的 primer、文本、token 摘要、计划到达时刻、输出长度和 KV 池大小。burst 与 paced 各自使用对应的清单。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m benchmarks.ascend_prefix_pr1_suite all \
  --chip 910B3 --reference-traffic "$B4_RESULTS"
```

默认验证本测试工作树的已提交 HEAD。整理完 PR 1 后，在测试工作树里指定最终候选所在的本地仓库和已提交 ref，填入实际路径及 SHA。

```bash
python -m benchmarks.ascend_prefix_pr1_suite all \
  --repo "$PR1_REPO" --candidate "$PR1_SHA"
```

将 `PR1_REPO` 设置为候选仓库的路径，`PR1_SHA` 设置为该仓库中要验证的提交。入口对候选提交建立临时本地克隆，先运行候选自己的 pre-commit 和 NPU 回归，再放入指定测试提交的 benchmark 和额外测试。新 worker 从这个临时检出加载运行时。`review.json` 分别记录 `candidate` 与 `harness`。外部脚本版本不能代替候选提交号。本地未提交修改不会进入候选快照。

也可以只在 CPU 机器运行一次 `python -m benchmarks.ascend_prefix_pr1_suite cpu`，在各自机器选卡后分别运行 `npu --chip 910B3` 和 `npu --chip 910B4`。将这三份相同候选 SHA 和测试 SHA 的报告附到 PR。部分运行只声明所选阶段。重跑用 `--only 阶段名` 生成新报告，保留原报告和失败记录。

### 生成两款卡的服务对照表

将两端结果拷到能同时读取它们的机器上。`B3_RESULTS` 和 `B4_RESULTS` 分别指向运行输出目录。至少需要各自已完成的 `environment`、`bf16-burst` 和 `bf16-paced` 三个阶段。生成表格可在 CPU 上执行。

```bash
python -m benchmarks.ascend_prefix_pr1_compare "$B3_RESULTS" "$B4_RESULTS" > dual-chip.md
```

工具核对两端实际型号、候选及测试提交、unpacked 设置、模型文件指纹和两份服务流量清单。表中分别给出每张卡的软件版本、单次运行状态、各负载的输出分歧和卡内 off/main 的配对 TTFT 变化。未通过服务校验的负载保留问题计数，收益栏留空。两台可同时先跑，事后用此命令核对清单。需要运行前校验时，可复制流量归档。

### 上游测试依据

RLT 的昇腾提交移除了产品测试对特定型号名的断言，并在 packed prefill 改动中加入后端与调度回归。本套件沿用产品的能力检查，在实验报告中核对具体型号。[RLT 型号处理](https://github.com/InferMatrixLab/vllm-rlt/commit/3f0bb13b082f03febe71064e91754440d3896741) · [RLT packed prefill](https://github.com/InferMatrixLab/vllm-rlt/commit/756f7bf5d1b4ff2f9ac9e760d572de1c25bbcfe6)

vLLM 的核心用例验证块复用和逐出。vLLM-Ascend 的双卡用例对共享长前缀、不同后缀的提示分别开关缓存，比对 greedy 输出。本项目在支持的单卡配置上采用同输入对照，并增加逐循环页检查、实际命中计数和独立 HTTP 混合负载。[vLLM 块级测试](https://github.com/vllm-project/vllm/blob/b26e0840134ea4b4c87c34bd5a0e0aa61397c638/tests/v1/core/test_prefix_caching.py#L3036) · [vLLM-Ascend 端到端测试](https://github.com/vllm-project/vllm-ascend/blob/ae9a697311b02421ca6cb09157f0414a5521a71a/tests/e2e/pull_request/two_card/test_prefix_caching.py)

## 全量运行覆盖

| 阶段 | 输入与检查 | 核验材料 |
|---|---|---|
| `pre-commit`、`config` | 候选分支的 `pre-commit run --all-files` 包含 Ruff、格式和 CPU pytest。额外的 CPU 测试通过 NPU 模型桩验证受支持配置能到达 packed 后端选择，并检查异步、多流、静态缓存、staged、增量分配、错误后端/KV、退出模式、推测、抢占、graphs 和分离部署的拒绝路径。 | hook 日志与配置测试 JUnit |
| `npu-regression`、`npu-contract` | 收集候选仓库全部 `npu` 标记用例，两个新增入口单列。以 BF16 真权重比较冷跑与缓存复用，检查实际命中 0/16 token、取消后共享页复用、请求 ID 重用、结束原因、退出深度、活跃页释放与不支持的采样参数。对显式关闭 packed 的路径也执行检查。 | 三类 pytest 阶段分别保存 XML 和日志。硬件测试意外跳过记为 INCOMPLETE |
| `stock-http` | 通过 `python -m vllm_rlt.entrypoints.serve --enable-prefix-caching` 启动正式服务。先执行 donor，随后发两条共享长前缀、不同后缀的请求，检查普通完成、SSE、usage、断连后的服务和正常退出。正式接口返回文本和 usage。逐请求命中来自下方独立的 worker 审计。 | `upstream-http/ascend_eager-sync-prefix/` |
| `environment` | 记录实际芯片、可见卡、CANN、Torch、torch-npu、Python、HF32 设置及权重/配置/分词器文件指纹。可用时保存 npu-smi 输出。 | `environment/` |
| `bf16-combo`、`fp32-combo` | 页级 oracle 逐层、逐循环检查发布缺口、共享地址、私有后缀隔离、共享 KV 内容和绝对位置因果掩码。9 个引擎场景各跑 2 轮。 | 对应 `.jsonl` 的 ORACLE、CONTROL、CASE、REVIEW |
| `bf16-matrix`、`fp32-matrix` | 同一组 30 例，3 轮。主线路径分别执行 off、miss、reuse。覆盖页边界、整段重复、冷热并发、驻留、紧池逐出。逐请求检查 16 个生成 token、固定退出深度、结束原因、实际命中、同模式重复运行。FP32 参照仅替换线性层，固定 256 行，KV/SDPA 仍为 BF16。 | `cases.csv`、`review.json.failed_comparisons`、`raw.jsonl` |
| `bf16-burst`、`bf16-paced` | 原生 BF16，真实流式 HTTP 和独立模型 worker。32 条目标请求，冷热及长短混合，客户端并发上限 4，seed 168，每模式 2 轮。burst 突发，paced 平均 3 请求/秒。每轮新服务、相同 primer，off/main 第二轮逆序。核查 worker 的线性层权重与 KV dtype、chunk 设置、backend、packing、同步单流、逐请求命中与输出。 | `traffic.json`、各轮 `worker.json` / `server.log`、`service.csv`、`raw.jsonl` |
| `bf16-benefit` | 30 例，各预热 1 轮、测量 6 轮，无 profiler。比较 off 与 miss 的启用开销、miss 与 reuse 的命中收益，以及 off 与 reuse 的净收益。逐轮轮换测试臂，相同输入、输出长度和循环策略。 | `CASE_DETAILS.paired_ttft` 含逐对差值及正向轮数。失败场景保留原始时间并排除收益汇总 |
| `drift` | 独立重放 4 个已知敏感场景和本次 BF16 矩阵失败项。在第一个不同输出位置确认此前生成历史一致，记录请求、两边 token ID、top-2 分数、分差、LM-head 输入差异和同一对候选的 CPU FP64 点积。 | `drift/<场景>.json`。FP64 对照只解释输出头，其他层的因果仍需独立证据 |
| `profile` | 单请求原生 BF16，默认 p512_s8。同一目标分别做开启缓存的 miss 与 reuse，先预热两种形状，再用 torch-npu 分别抓 ops-only 和 full，共 4 个短窗口。只采集目标 prefill 步，核对 prefill 行数、命中及最终 token。full 打开 shape、stack、memory。解析 trace、设备 kernel 表及 full 内存记录。 | `profile/{ops,full}-{miss,reuse}/evidence.json` 和原始 profiler 文件 |

实验分支当前允许显式 `--no-packed-prefill`，所以默认 `--unpacked support` 要求它真实命中并通过结果比对。如果精简候选明确改成拒绝这个配置，使用 `--unpacked reject` 验证它确实报告 packed 配置错误。开关描述待验证的产品行为，不改变产品代码。

## 如何写 PR 的测试结果

先打开结果目录中的 `PR_RESULTS.md` 和 `review.json`。总表包含候选/工具 SHA、阶段状态，以及原生 BF16 服务的逐轮 P50、P95、含客户端排队的 P95、实际命中 token 和输出 tok/s。四类请求分别查看 `service.csv` 的 `hot_short`、`hot_long`、`cold_short`、`cold_long`。HTTP TTFT 从发送请求开始。`arrival_ttft_ms` 从计划到达时刻开始。服务输出固定 8/16 token，预先种入共享前缀，默认 KV 池可容纳整条流量。报告中标明这是合成混合负载，尾部百分位同时给出每类样本数。

- **自动结果。** 0 表示所选阶段通过，1 表示有测试或数值差异，2 表示有未完整执行的检查。已完成项目保留，单项 FAIL 后继续采集其他阶段。意外中断时，尚未完成的项目留作 NOT_RUN。只有 PR 1 的全部自动检查通过才置 `full_pr1_pass=true`。
- **正确性。** 页地址、共享 KV 内容、写入覆盖、掩码、资源回收错误要复现并修复。每个 BF16 反例列出比较双方和首处分歧。固定行 FP32 的结果单列。短服务回放恰好输出一致，仍应在 PR 中保留矩阵里的反例。若分歧只在服务出现，使用相应轮次 `traffic.json`、`worker.json` 与 `SERVICE_CHECKS` 定位实际请求和输出，必要时按该批形状另行重放。代码不会自动把已知分歧改成通过。
- **性能。** 用相同 dtype、设备、模型、负载和执行配置比较，列绝对值、逐轮变化及错误数。分开报告冷请求开销和热请求收益。输出或路径检查有差异的场景标为诊断数据。用 profiler 的输入形状、设备执行和 prefill 行数解释无 profiler 的计时结果。
- **提交。** 给出复现命令、受支持的配置、已知限制、未跑检查，链接可供审阅的结果及 profile。按照上游 PR 模板，由贡献者本人填写目标、关键变量、改动如何影响它、目标用户与需求，以及为什么需要提交 PR。[贡献说明](https://github.com/InferMatrixLab/vllm-rlt/blob/ce200f0/CONTRIBUTING.md) · [精度政策](https://github.com/InferMatrixLab/vllm-rlt/blob/ce200f0/docs/precision-policy.md) · [开发指南](https://github.com/InferMatrixLab/vllm-rlt/blob/ce200f0/docs/developer-guide.md) · [PR 模板](https://github.com/InferMatrixLab/vllm-rlt/blob/ce200f0/.github/PULL_REQUEST_TEMPLATE.md)

## PR 2 的独立实验

```bash
python -m benchmarks.ascend_prefix_pr1_suite pr2
```

该入口复用 BF16 30 例矩阵以及 burst/paced，并保留 `main → grouped` 的单独比较。它还各做一轮单独的调度观测。查看 `observe-{burst,paced}/round-1-main/opportunities.json` 和同目录 `worker.json` 的 `prefill_steps`。记录包括同一步的成员、请求摘要、循环深度、起点、剩余长度、Q 行数和可见 KV 长度。分母为目标请求参与 prefill 的“请求 × 步”次数，剔除 primer。已被主线整齐冷批覆盖的成员另计。阈值 0/2/4/8 给出同起点、同批、存在一个长度相容伙伴的成员数。阈值是包含关系。这些数值衡量机会，不等于能同时成组的数量或实测速率。观测轮的计时只供诊断。近似长度计算仍属于后续设计。
