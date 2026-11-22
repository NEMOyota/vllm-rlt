# Ouro 在昇腾主线上的 prefix cache 夜跑

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
