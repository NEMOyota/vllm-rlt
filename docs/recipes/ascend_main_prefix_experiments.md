# Ouro 在昇腾主线上的 BF16、收益和 HTTP 压测

在主线组合实验的工作树运行。使用服务器现有的 CANN、torch-npu 和本地 Ouro-1.4B 权重。卡号按本机情况调整。

```bash
export MODEL=/models/Ouro-1.4B
export ASCEND_RT_VISIBLE_DEVICES=2

bash docs/recipes/ascend_main_prefix_experiments.sh bf16
bash docs/recipes/ascend_main_prefix_experiments.sh benefit
bash docs/recipes/ascend_main_prefix_experiments.sh service
```

三次运行各自新建结果目录，启动时打印绝对路径。可用 `--out-dir` 命名一个新目录。退出码 0 表示该次检查通过，1 表示实验完成且发现差异，2 表示运行不完整。每个目录有 `raw.jsonl` 和 `review.json`，记录提交、权重指纹、设备、软件版本和运行参数。固定使用原生 BF16、同步单流、packed SDPA 和四轮 greedy 解码。

## 2　原生 BF16 全量测试

`bf16` 默认跑历史 29 例，加上三条命中的 `c4_h3`，共 30 例，每例 3 轮。每轮都跑主线和合批候选两个版本，每个版本分别关闭缓存、开启但未命中、开启并尝试复用。沿用 seed 168 和原始 token 输入，覆盖长短前缀、非整页、整段重复、冷热并发、缓存驻留和紧池逐出。

检查每个请求的 16 个输出 token、命中数、prefill 起点、固定退出深度、结束原因和实际合批调用数。记录原生开关缓存分歧、同一缓存状态下合批引入的变化及跨轮漂移。已知的三个历史分歧会按本次实际结果计入。单个用例的分歧不会中止后续用例。阅读 `BF16_REVIEW` 和 `CASE_DETAILS`。

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
bash docs/recipes/ascend_main_prefix_experiments.sh benefit --case c4_h2 --case c4_h0 --repeats 6
bash docs/recipes/ascend_main_prefix_experiments.sh benefit --all-cases
```

## 4　服务模式混合流量

`service` 自动在本机空闲端口启动真实 HTTP 服务和独立推理工作进程，依次重放关闭缓存、主线缓存、缓存加合批三种模式。每轮使用新服务，第二轮反向运行。默认每种模式每轮 32 个目标请求，并发 4、2 轮。长短共享前缀与长短私有前缀各占约四分之一，分别生成 8 或 16 个 token。每次都先运行相同的预热请求并种入共享前缀。默认 KV 池能容纳整条流量，以便检验预置前缀的命中范围。

默认一次提交整批负载并限制客户端在途数。`--rps` 可改成固定种子的泊松到达。建议分别保留突发和有到达间隔的结果。下面的命令每种模式每轮重放 96 条请求，可用于观察长尾。

```bash
bash docs/recipes/ascend_main_prefix_experiments.sh service --requests 96 --concurrency 8 --rps 3 --rounds 2
```

`traffic.json` 给出请求文本、到达时刻及由实际 tokenizer 得到的可复用范围。`service.csv` 按冷热和长度列出完成量、错误数、实际命中 token、TTFT p50/p95/p99 和端到端 p95。TTFT 从发出 HTTP 请求算到首个 token 事件，原始结果也记录首个非空文本和从计划到达时刻开始的延迟。吞吐按目标请求实际耗时计算，包含负载中的空闲间隔。

服务在测量后导出工作进程的逐请求 token ID 和实际命中数，核对流的结束标志、usage、命中范围、输出及 KV 释放。流量时序可能改变组批和 BF16 输出。出现差异时，`SERVICE_CHECKS` 分开列出 HTTP/命中检查和 token 分歧，各模式的 `server.log`、`worker.json` 及原始计时保留。聚合的合批调用次数可在 `SERVICE_RUN.state.group_calls` 查看。

`SERVICE_REVIEW.grouped_calls_per_round` 给出每轮真实合批次数。若某轮缺少可合并的请求，仍可看端到端负载表现。只有每轮都观察到合批且核对通过时，汇总才会列出合批的配对收益。
