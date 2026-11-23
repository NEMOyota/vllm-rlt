# Ouro 在昇腾主线上的 prefix cache 夜跑

准备提交 PR 1 时，可使用[提交前全量验证](ascend_prefix_pr1_validation.md)运行正式服务回归、原生 BF16 矩阵及 HTTP 对照。

在实验工作树和服务器已有的 CANN、torch-npu 环境中执行。选择一张空闲卡。

```bash
export MODEL=/models/Ouro-1.4B
export ASCEND_RT_VISIBLE_DEVICES=2
bash docs/recipes/ascend_main_prefix_experiments.sh fp32-all
```

夜跑在全部实验臂使用同一精度。Ouro 的每一个 `nn.Linear` 使用 FP32 输入、权重和偏置，行数固定为 256，尾块补零，有效行输出转回 BF16。覆盖解码层、退出门和 LM head。KV 和 SDPA 使用 BF16。`off`、`miss`、`reuse` 分别指缓存关闭、开启未命中和开启复用。每个阶段只执行选定的线性层精度。

| 阶段 | 默认负载 | 结果 |
| --- | --- | --- |
| combo | 页级 oracle、9 个组合场景、2 轮 | `combo.jsonl` |
| full | 历史 29 例加 `c4_h3`、3 轮、主线与合批各三个缓存臂 | `full/review.json`、`cases.csv` |
| benefit | 同一组 30 例，交替主线与合批，预热 1 轮、测量 6 轮 | `benefit/review.json`、`cases.csv` |
| burst | 每种服务模式 32 条冷热、长短请求，并发 4，2 轮 | `burst/review.json`、`service.csv` |
| paced | 相同请求与并发，平均每秒到达 3 条，2 轮 | `paced/review.json`、`service.csv` |

启动时打印 `RESULTS` 根目录。每阶段完成后 `stages.tsv` 追加阶段名和退出码。0 表示该阶段核对通过，1 表示发现差异，2 表示运行不完整。单项检查失败仍会保留原始记录并继续后续阶段。性能配对仅汇总通过本阶段检查的用例。可用 `BENEFIT_REPEATS` 调整收益测试轮数，`CHUNK_ROWS` 修改所有阶段共同使用的固定行数。每个阶段另存软件版本、提交和模型指纹。

## 1　主线组合

```bash
bash docs/recipes/ascend_main_prefix_experiments.sh combo --linear-fp32-chunk 256 --repeats 2
```

检查共享页、缺口阻止发布、packed 页表、后缀绝对位置掩码、SDPA 调用及每层每个循环深度的写入范围。每条 `COMBO_CASE.prefill_rows` 给出实际目标 prefill 行数。`COMBO_ARM.linear_work` 记录 FP32 臂实际执行的模块数和未调用模块。`COMBO_REVIEW` 分列 token 分歧、路径检查失败和跨轮漂移。

## 2　全量正确性

```bash
bash docs/recipes/ascend_main_prefix_experiments.sh fp32
```

沿用 seed 168 和旧实验的 token 输入。覆盖长短前缀、非整页、整段重复、冷热并发、驻留和紧池逐出，同时检查每个请求的 16 个生成 token、固定退出深度、结束原因以及真实合批调用。看 `FP32_REVIEW.failed_comparisons` 区分同精度下的缓存开关分歧和合批分歧。需要重现历史原生 BF16 时，可单独使用 `bf16` 入口。

## 3　主线真实收益

`benefit` 默认覆盖 13 个场景，每个场景先预热 1 轮，再测 6 轮。每个缓存状态下主线与合批相邻运行，交换先后顺序，同时轮换三个缓存状态。每条计时都使用新缓存并在提交目标请求前完成预热 donor 或逐出操作。计时从提交请求到首个实际输出 token，以及完成全部输出。模型加载、引擎初始化、donor 和预热不计入延迟。

`cases.csv` 给出 TTFT、端到端延迟、TPOT 和吞吐。`review.json` 的 `paired_ttft_saved_ms` 汇总通过场景的配对中位数，`CASE_DETAILS.paired_ttft` 保存逐轮差值及正向轮数。正值表示节省。三个比较分别回答不同的问题。

| 字段 | 含义 |
| --- | --- |
| `main_off_to_reuse` | 主线打开缓存并命中，相比关闭缓存的端到端首 token 收益 |
| `main_miss_to_reuse` | 缓存同为开启时，命中相对未命中的收益 |
| `reuse_grouping` | 相同命中、相同请求和主线 SDPA 下，冷请求与后缀合批额外节省的时间 |

`off_grouping`、`miss_grouping` 用于检查控制路径。合批候选仅在实验中装载，读取每个请求自己的 packed 页表并复用主线 SDPA。比较输出或路径的检查失败时保留原始计时，收益栏留空。PASS 表示数据可比，收益正负由实测决定。

```bash
bash docs/recipes/ascend_main_prefix_experiments.sh benefit --linear-fp32-chunk 256 --case c4_h2 --case c4_h0 --repeats 6
bash docs/recipes/ascend_main_prefix_experiments.sh benefit --linear-fp32-chunk 256 --all-cases
```

## 4　服务模式混合流量

`service` 自动在本机空闲端口启动真实 HTTP 服务和独立推理工作进程，依次重放关闭缓存、主线缓存、缓存加合批三种模式。每轮使用新服务，第二轮反向运行。默认每种模式每轮 32 个目标请求，并发 4、2 轮。长短共享前缀与长短私有前缀各占约四分之一，分别生成 8 或 16 个 token。每次都先运行相同的预热请求并种入共享前缀。默认 KV 池能容纳整条流量，以便检验预置前缀的命中范围。

默认一次提交整批负载并限制客户端在途数。`--rps` 可改成固定种子的泊松到达。建议分别保留突发和有到达间隔的结果。下面的命令每种模式每轮重放 96 条请求，可用于观察长尾。

```bash
bash docs/recipes/ascend_main_prefix_experiments.sh service --linear-fp32-chunk 256 --requests 96 --concurrency 8 --rps 3 --rounds 2
```

`traffic.json` 给出请求文本、到达时刻及由实际 tokenizer 得到的可复用范围。`service.csv` 按冷热和长度列出完成量、错误数、实际命中 token、TTFT p50/p95/p99 和端到端 p95。TTFT 从发出 HTTP 请求算到首个 token 事件，原始结果也记录首个非空文本和从计划到达时刻开始的延迟。吞吐按目标请求实际耗时计算，包含负载中的空闲间隔。

服务在测量后导出工作进程的逐请求 token ID 和实际命中数，核对流的结束标志、usage、命中范围、输出及 KV 释放。流量时序可能改变组批和 BF16 输出。出现差异时，`SERVICE_CHECKS` 分开列出 HTTP/命中检查和 token 分歧，各模式的 `server.log`、`worker.json` 及原始计时保留。聚合的合批调用次数可在 `SERVICE_RUN.state.group_calls` 查看。

`SERVICE_REVIEW.grouped_calls_per_round` 给出每轮真实合批次数。若某轮缺少可合并的请求，仍可看端到端负载表现。只有每轮都观察到合批且核对通过时，汇总才会列出合批的配对收益。

HTTP 实验在实际持有模型的独立工作进程安装线性层配置。每次从 worker 核对分块大小和执行计数。CSV 和汇总带有 `precision` 和 `linear_fp32_chunk`，便于按相同精度比较。

## 5　单请求 hot_long 和 256 行分块边界

在同一份工作树、空闲卡和夜跑 checkpoint 上执行。把 `--service-raw` 指向已有的服务原始日志。脚本从 MANIFEST 恢复真实请求和 `primer_long`，用本机 tokenizer 核对两份 token ID 的 SHA-256，并用 SERVICE_RUN 核对原请求的首 token。

```bash
export MODEL=/models/Ouro-1.4B
export ASCEND_RT_VISIBLE_DEVICES=2
bash docs/recipes/ascend_main_prefix_experiments.sh hot-long \
  --service-raw /workspace/rlt-prefix-night-Vjc1lL/results/main-prefix-fp32-r256-gMZbiLg7/paced/raw.jsonl \
  --request-id target-5 --repeats 6 --audit-repeats 3
```

| 目标长度 | 复用 token | 实际后缀行数 | 单次逐 token 投影的完整输入块数 → 命中后的块数 |
| ---: | ---: | ---: | --- |
| 242 | 208 | 34 | 1 → 1 |
| 256 | 208 | 48 | 1 → 1 |
| 257 | 208 | 49 | 2 → 1 |

242 是原请求的 token 序列。两个对照在它末尾追加原请求最后 15 个 token 中的前 14 或全部 15 个，257 和 256 仅差最后一个 token。只在同一长度内比输出；242 的首 token 也与存档比较。此实验使用主线同步引擎、并发 1、固定 4 次循环、生成 1 个 token，所有线性层用 FP32 定长 256 行，KV 与 SDPA 用 BF16。

每条目标请求都用新引擎和相同大小的 KV 池。先执行与目标分开的预热请求。off 关闭缓存，reuse 用归档的长前缀预热，miss 仅替换预热请求的第一个 token。三臂的目标输入相同。待预热完成并同步后，计时从提交目标请求开始，到引擎返回首 token 为止。每种组合先预热一次，再做 6 轮配对，轮换长度并交替三臂的顺序。

最后单独重放 3 轮采集算子数据。按投影类型统计实际执行的 FP32 `F.linear` 块数。只为当前层 K/V 读取用的两个 `index_select` 以及 BF16 SDPA 记录 NPU stream event，取每次重放内相应区间耗时之和，再报跨重放中位数。读数据量按返回张量的字节数记录。时间可能含流内等待；页表和掩码构造、Q 收集、输出写回等工作另占时间。诊断重放独立于首 token 的计时样本。

启动时打印结果目录。看 `review.json` 中三个长度的 `paired_ttft.miss_to_reuse`，正值表示命中节省时间。`audit_miss_to_reuse` 给出诊断轮逐对的 KV 读取和 SDPA 设备流时间差。各臂的 `q_proj_tiles`、`linear_tiles`、`kv_read_p50_ms`、`sdpa_p50_ms` 同时列在 `cases.csv`。Ouro-1.4B 有 24 层、4 次循环，257 行完整输入预期共 192 个 Q 投影块，三个长度的命中臂各 96 个。逐轮顺序、命中边界、模块执行、计时、K/V 和 SDPA 调用数均在 `raw.jsonl`。检查失败的长度保留逐轮数据并标出原因；意外运行错误会报告 INCOMPLETE 和出错位置。

### 固定 256 行与固定 64 行的配对对照

在上述 `hot-long` 命令末尾加 `--compare-chunk 64` 即可。两套配置共用同一份模型及 FP32 权重，每条请求从预热 donor 到目标生成始终使用该臂的固定块大小。每个长度和缓存状态的两种块大小紧邻运行，6 轮中各有 3 次先跑。预热、6 轮 TTFT 计时、3 轮算子采集和独立的 16-token 输出检查分别记录。也可用 `--compare-chunk 128` 与 256 行基线比较。

| 输入总长 | 命中后缀 | 固定 256 行时完整输入 → 后缀的块数 | 固定 64 行时完整输入 → 后缀的块数 |
| ---: | ---: | ---: | ---: |
| 242 | 34 | 1 → 1 | 4 → 1 |
| 256 | 48 | 1 → 1 | 4 → 1 |
| 257 | 49 | 2 → 1 | 5 → 1 |

表中块数对应一次逐 token 投影。每个长度的 `review.json` 同时给出以下结果。

- `by_chunk.64.paired_ttft.miss_to_reuse` 给出固定 64 行时开缓存命中的收益。
- `paired_256_to_candidate.reuse` 给出同一个已命中请求从 256 换成候选块大小后节省的 TTFT。正值表示候选更快。`miss` 和 `off` 两项用同一口径给出完整输入路径的变化。`by_chunk` 中也有各臂的绝对 TTFT。
- `parity16` 把两个块大小的 off、miss、reuse 共六条输出逐 token 与该长度的 256 行 off 比较。出现分歧时报告请求、位置和 token ID，并保留逐轮原始记录。跨块大小的性能配对只在检查通过的长度上汇总。
- `linear_padded_rows` 是全部线性层实际 tile 行数减去各层有效输入行数的总和。`q_pad_p50_ms` 是一次重放中全部 Q 投影的尾块补零操作在设备流上的时间之和，再取跨重放中位数。它只覆盖 Q 投影。KV 读取和 SDPA 仍各自计时，所有算子标记都位于单独的采集轮。

下一步判断时先看命中请求相对 256 行的绝对收益及冷请求的开销，再用候选块大小跑 30 例正确性和混合流量。三种长度的配对均使用高精度线性层配置。
