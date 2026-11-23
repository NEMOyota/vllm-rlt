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

### 在 910B3 复测 910B4 已出现 token 分歧的四个场景

在已激活 CANN 和 torch-npu 的 910B3 容器内，选一张空闲卡和本地 Ouro-1.4B 权重。结果写到终端打印的 `RESULTS` 目录。也可传入 `--out` 指定新的目录。

```bash
export MODEL=/models/Ouro-1.4B
ASCEND_RT_VISIBLE_DEVICES=0 bash docs/recipes/ascend_910b3_b4_repro.sh
```

| 场景 | 每条目标输入长度 | 同批目标数 | 开缓存并预热后各目标预期命中 token 数 |
|---|---:|---:|---|
| `p1024_s8` | 1032 | 1 | 1024 |
| `total512_p384` | 512 | 1 | 384 |
| `c4_h3` | 144 | 4 | 128、128、128、0 |
| `c4_h4` | 144 | 4 | 128、128、128、128 |

原生 BF16 和固定 256 行 FP32 线性层分别在独立进程中各跑一轮。两种精度使用相同的 seed 168、目标 token ID、同批顺序、BF16 KV 和 SDPA。每个缓存状态都新建引擎，greedy 生成 16 个 token，固定四次循环。每种精度内部比较关闭缓存（`off`）、开启缓存且预热不匹配（`miss`）及开启缓存且预热匹配（`reuse`）。BF16 出现分歧后仍继续执行 FP32 对照。

在 `PR_RESULTS.md` 看各场景两项比较中有分歧的请求数。在 `b4-cases-bf16/<场景>.json` 和 `b4-cases-fp32/<场景>.json` 看原始输入、各目标实际命中量、首步 prefill 行数、完整输出和首个不相同 token 的位置及 ID。FP32 文件还记录每个缓存状态实际执行的线性层数。运行环境、权重指纹和实测设备型号随结果保存。

已有上述两组结果时，选择同一张 B3 和同一份权重，单独补跑 FP16。下面这条命令采集环境和四个场景，每个场景按 `off / miss / reuse` 各跑一次。脚本会打印新的结果目录。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 bash docs/recipes/ascend_910b3_b4_repro.sh --fp16-only
```

FP16 worker 在临时工作树中应用随测试提交固定的 dtype 放行补丁，把同一 checkpoint 加载为 FP16，按实际行数执行线性层。报告保存 `fp16-runtime.patch` 和 SHA-256。在 `b4-cases-fp16/<场景>.json` 中，`arms.off.observed_dtypes`、`arms.miss.observed_dtypes` 和 `arms.reuse.observed_dtypes` 记录权重、K/V 缓存、SDPA 的 Q/K/V 与输出的实测 dtype，以及 SDPA 调用次数。每条目标请求的 16 个输出 token 都与本次 FP16 的 `off` 比较。首个分歧和两边的 token ID 在 `differences` 中。目标步出现 NaN 或 Inf 时，报告将该场景记为 FAIL 并写明请求和步号。

### 先跑冷请求、完整重复与分支后缀

下面两行分别在对应服务器执行，沿用上面设置的 `MODEL`，把 0 换成已分配的单卡编号。这组检查也已纳入 `all` 和 `npu`。要同时跑固定行 FP32 线性层参照，在同一命令再加 `--only fp32-replay`。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m benchmarks.ascend_prefix_pr1_suite --chip 910B3 --only bf16-replay
ASCEND_RT_VISIBLE_DEVICES=0 python -m benchmarks.ascend_prefix_pr1_suite --chip 910B4 --only bf16-replay
```

每个场景使用两个初始缓存为空的引擎。一个关闭缓存，一个开启缓存。两边都依次执行同一条冷请求、完整重复、共享前缀后更换末尾的分支。三条请求各自生成 16 个 token，逐条等到结束再提交下一条。输入来自本地 Ouro 分词器编码的固定英文段落，分支在 token 序列上构造。16-token 页只复用完整页，完整重复时仍要重算末尾以取得输出头需要的隐藏状态。

| 场景 | 输入 token | 分支的第一个不同位置（从 0 计） | 每步 token 预算 | 开缓存时冷 / 重复 / 分支的预期命中 |
|---|---:|---:|---:|---|
| short_full | 71 | 48 | 71 | 0 / 64 / 48 |
| short_split | 71 | 48 | 16 | 0 / 64 / 48 |
| long_full | 1100 | 1068 | 1100 | 0 / 1088 / 1056 |

`short_full` 和 `short_split` 共用输入。分步场景的热分支必须按 `[48,64)`、`[64,71)` 执行，attention 的 Q/KV 长度依次为 16/64、7/71。该分支在缓存开、关两边的每个 prefill 步都审计 packed 页表、绝对位置因果掩码和四个循环各层的连续写入区间。长场景的分叉落在页内，第 1056 到 1067 位这 12 个相同 token 也需要重算。

每个场景都以同一预算和精度比较三条请求各自的开关输出，另比对每个引擎内冷请求与完整重复的输出。`PR_RESULTS.md` 展示命中量、调度区间、活跃页和逐 token 对照结果。`{bf16,fp32}-replay/review.json` 保留完整输出、退出深度、首处分歧的位置和双方 token ID。另列的六项 `budget_comparisons` 比较相同请求在整段与分步下的输出，便于核对关闭缓存时是否也受计算形状影响。这六项是预算诊断，缓存一致性按同一预算验收。请求结束时活跃页须归零，共享页可由前缀索引保留。TTFT 和吞吐取 `bf16-benefit`、`bf16-burst` 和 `bf16-paced` 的独立测量。

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

RLT 的 Dory PR 用真权重运行 cold、hit、branch，并给分支保留自己的 cache-off 输出作对照。这里把这套验收方法用于 Ouro 的同步单卡路径，并记录每一步的调度区间。[Dory PR #190](https://github.com/InferMatrixLab/vllm-rlt/pull/190)

vLLM 的核心用例验证块复用和逐出。vLLM-Ascend 的双卡用例对共享长前缀、不同后缀的提示分别开关缓存，比对 greedy 输出。本项目在支持的单卡配置上采用同输入对照，并增加逐循环页检查、实际命中计数和独立 HTTP 混合负载。[vLLM 块级测试](https://github.com/vllm-project/vllm/blob/b26e0840134ea4b4c87c34bd5a0e0aa61397c638/tests/v1/core/test_prefix_caching.py#L3036) · [vLLM-Ascend 端到端测试](https://github.com/vllm-project/vllm-ascend/blob/ae9a697311b02421ca6cb09157f0414a5521a71a/tests/e2e/pull_request/two_card/test_prefix_caching.py)

## 全量运行覆盖

| 阶段 | 输入与检查 | 核验材料 |
|---|---|---|
| `pre-commit`、`config` | 候选分支的 `pre-commit run --all-files` 包含 Ruff、格式和 CPU pytest。额外的 CPU 测试通过 NPU 模型桩验证受支持配置能到达 packed 后端选择，并检查异步、多流、静态缓存、staged、增量分配、错误后端/KV、退出模式、推测、抢占、graphs 和分离部署的拒绝路径。 | hook 日志与配置测试 JUnit |
| `npu-regression`、`npu-contract` | 收集候选仓库全部 `npu` 标记用例，两个新增入口单列。以 BF16 真权重比较冷跑与缓存复用，检查实际命中 0/16 token、取消后共享页复用、请求 ID 重用、结束原因、退出深度、活跃页释放与不支持的采样参数。对显式关闭 packed 的路径也执行检查。 | 三类 pytest 阶段分别保存 XML 和日志。硬件测试意外跳过记为 INCOMPLETE |
| `stock-http` | 通过 `python -m vllm_rlt.entrypoints.serve --enable-prefix-caching` 启动正式服务。先执行 donor，随后发两条共享长前缀、不同后缀的请求，检查普通完成、SSE、usage、断连后的服务和正常退出。正式接口返回文本和 usage。逐请求命中来自下方独立的 worker 审计。 | `upstream-http/ascend_eager-sync-prefix/` |
| `environment` | 记录实际芯片、可见卡、CANN、Torch、torch-npu、Python、HF32 设置及权重/配置/分词器文件指纹。可用时保存 npu-smi 输出。 | `environment/` |
| `bf16-replay`、`fp32-replay` | 冷请求、完整重复、分支后缀三路回放。三种长度与预算组合各跑缓存开、关，检查逐请求命中、逐步 prefill、16 个输出 token、退出深度、结束原因和页释放。 | `review.json`、逐请求的 `raw.jsonl` |
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

## PR 1 与 PR 2 的提交数据和实验脚本

两份 PR 分别记录最终候选 SHA。上游 PR 模板要求贡献者填写验证计划和结果，并在仓库根目录执行 `pre-commit run --all-files`。下面是针对这次两款芯片支持范围的验证安排。每个候选跑一次 CPU 检查。声明覆盖 910B3、910B4 时，两款卡分别跑单卡真权重测试。报告正文给出绝对值、样本数和异常，附可下载的原始证据。[上游 PR 模板](https://github.com/InferMatrixLab/vllm-rlt/blob/ce200f0d2085fb041c66691245e29aef2a9ed714/.github/PULL_REQUEST_TEMPLATE.md)

两个工作树的总入口都叫 `benchmarks/ascend_prefix_pr1_suite.py`。PR 1 使用 `test/ascend-prefix-pr1-validation` 的 `all`。PR 2 使用 `exp/ascend-prefix-pr2-grouping` 的 `pr2`。PR 1 工作树里同名的 `pr2` 子命令只运行早期的 benchmark 合批原型。下面列出实验脚本的文件名，Python benchmark 均在 `benchmarks/`，pytest 文件均在 `tests/`。[PR 2 固定版本运行说明](https://github.com/NEMOyota/vllm-rlt/blob/9a2d5100c5344cd6d1cb0f210ea19ebcb6ccbb67/docs/recipes/ascend_prefix_pr2.md)

| 提交和问题 | 要跑的阶段 | 对应实验脚本 | 提交时展示的数据 |
|---|---|---|---|
| 两份 PR 的环境与回归 | 各候选的 `pre-commit`、`config`，各型号的 `environment` | `ascend_prefix_pr1_suite.py`、`ascend_prefix_pr1_hardware.py`、`ascend_prefix_pr1_diagnostics.py` | 候选与测试 SHA、单卡型号、CANN/Torch/torch-npu、HF32、模型指纹、pytest 通过/失败/跳过数量。 |
| PR 1 的设备功能和正式服务入口 | `npu-regression`、`npu-contract`、`stock-http` | `test_ascend_prefix_cache_contract.py`、`test_ascend_serving.py`，及候选自带 NPU 用例 | JUnit 和服务冒烟结果，包含共享页、取消与复用、SSE、断连恢复。 |
| PR 1 的缓存机制与数值一致性 | `{bf16,fp32}-replay`、`{bf16,fp32}-combo`、`{bf16,fp32}-matrix`、`drift` | `ascend_prefix_pr1_replay.py`、`ascend_main_prefix_combo.py`、`ascend_main_prefix_matrix.py`、`ascend_prefix_pr1_diagnostics.py` | 冷/重复/分支回放、实际命中和逐步区间、每请求 16 个输出、分页与掩码检查、30 例通过数。单列首处分歧、双方 ID、可取得的 top-2 依据和失败场景。 |
| PR 1 的缓存收益 | `bf16-benefit`、`bf16-burst`、`bf16-paced`、`profile` | `ascend_main_prefix_matrix.py benefit`、`ascend_main_prefix_service.py`、`ascend_prefix_pr1_diagnostics.py profile` | 引擎内 off/miss/reuse 的冷开销和命中收益。附服务 off/main 每轮的完成数、HTTP 错误、输出分歧、TTFT P50/P95、输出 tok/s 以及代表性 prefill trace。 |
| PR 2 的入口回归与 attention 正确性 | 在 PR 2 候选上复跑 `npu-regression`、`npu-contract`、`stock-http`，再跑 `pr2-oracle` | 同上入口测试、`test_ascend_prefix_prefill_groups.py`、`test_ascend_prefix_pr2_benchmark.py`（CPU） | 原有默认路径回归、设备 oracle 对独立稠密 attention 的比较、页表、有效行、因果掩码与填充隔离。 |
| PR 2 的合批贡献 | `pr2-geometry`、`pr2-reference`、`pr2-profile` | `ascend_prefix_pr2.py` | 六组受控形状的 main/exact/similar 逐 token 对照，真实合批数、节省的 SDPA 次数、额外填充、各轮 TTFT 和设备 trace。单请求、整齐冷批和离群请求也要报告。 |
| PR 2 的服务效果 | `pr2-burst`、`pr2-paced`、`observe-burst`、`observe-paced` | `ascend_main_prefix_service.py`、`ascend_main_prefix_service_worker.py`、`ascend_prefix_opportunities.py` | 同一混合 HTTP 流量下三种模式各三轮，逐轮 P50/P95、吞吐、实际近似组数及 `similar_service`。另附 `opportunities.json` 的同批机会与分母。零触发也列入结果，受控形状的加速单独报告。 |

### 运行两份候选

以下命令假定已设置 `MODEL` 和 `ASCEND_RT_VISIBLE_DEVICES`。在 B3 的 shell 中设置 `export CHIP=910B3`，B4 使用 `export CHIP=910B4`。PR 1 的全量命令从其测试工作树运行。

```bash
python -m benchmarks.ascend_prefix_pr1_suite all --chip "$CHIP"
```

在已有 fork 中取得 PR 2 工作树。若已经有该工作树，可直接从其根目录执行后面的 Python 命令。

```bash
git fetch https://github.com/NEMOyota/vllm-rlt.git refs/heads/exp/ascend-prefix-pr2-grouping
git worktree add --detach ../rlt-pr2-submit-tests FETCH_HEAD
cd ../rlt-pr2-submit-tests

# 每个 PR 2 候选一次。CPU 范围包含 pre-commit 和配置检查。
python -m benchmarks.ascend_prefix_pr1_suite cpu

# 每款卡运行这两条命令。第一条验证默认路径，第二条测合批。
python -m benchmarks.ascend_prefix_pr1_suite --chip "$CHIP" \
  --only npu-regression --only npu-contract --only stock-http
python -m benchmarks.ascend_prefix_pr1_suite pr2 --chip "$CHIP"
```

用同卡 PR 1 的 burst/paced 清单复测 PR 2 时，在最后一条命令追加 `--reference-traffic "$PR1_RESULTS"`。若提交候选在另一工作树，每个套件命令都可以用 `--repo` 和 `--candidate` 指向该 PR 的最终提交。两份 PR 都保留 `PR_RESULTS.md`、`review.json`、`hardware.json`、环境与模型指纹、相关 JUnit、CSV/JSONL 及代表性 trace 的可访问链接。大型 trace 单独打包。

### 精度与结论

原生 BF16 是上游主推理精度。这里的 `fp32` 和 `pr2-reference` 使用 FP32 线性层固定 256 行，KV 与 SDPA 为 BF16。历史黄区 FP16 服务使用本地补丁，本分支的四场景 FP16 回放使用随结果归档的实验补丁。报告中提供补丁版本、worker 的实际权重/KV dtype、线性层设置和相同精度的对照。当前两个候选分支的 NPU 产品入口要求 BF16 权重。已运行的 BF16 分歧记 FAIL 并附可复现输入。未跑或设备条件不足记 NOT_RUN/INCOMPLETE。原生 BF16 和精度参照分别报告。[上游精度政策](https://github.com/InferMatrixLab/vllm-rlt/blob/ce200f0d2085fb041c66691245e29aef2a9ed714/docs/precision-policy.md)

PR 1 归因于同卡缓存关闭到开启的差值。PR 2 三种模式都开启缓存，main 到 exact 测等长合批，exact 到 similar 测近似合批的增量。两份 PR 的服务实验都是实 HTTP 服务回放可复现的合成混合负载。任何服务性能结论须同时给出该负载的正确性和真实触发情况。
