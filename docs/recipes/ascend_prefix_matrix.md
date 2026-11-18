# 910B2 Prefix Cache 性能测试

本实验覆盖单卡 Ouro-1.4B、BF16、同步 `ascend_eager`、固定四轮 greedy 的引擎路径。使用已经跑通基线的容器、分配的 `ASCEND_RT_VISIBLE_DEVICES` 和本地权重目录 `$MODEL`。停掉占用同一张卡的 HTTP 服务，在仓库根目录运行。

## 建立基线

先运行已有的真机生命周期回归，再跑快速集。回归会检查两个请求同时引用前缀页时，取消一个请求后另一个仍能完成，且共享 KV 内容逐元素一致。

~~~bash
python -m pytest -q --run-npu -m npu tests/test_ascend_prefix_cache.py
python -m benchmarks.ascend_prefix_matrix --model "$MODEL" --suite quick \
  --repeats 3 --out results/quick.json
~~~

完整基线默认每组输出 16 个 token。每个场景每种缓存配置各预热一轮，再测五轮。完整集共 29 个场景，合计 522 次目标请求批次（含预热）。查看单个场景可用 `--list`；`--case` 可重复指定。输出文件请使用新名字，便于保留基线。

~~~bash
python -m benchmarks.ascend_prefix_matrix --model "$MODEL" --suite full \
  --warmups 1 --repeats 5 --out results/baseline.json
~~~

| 维度 | 场景 | 观测重点 |
| --- | --- | --- |
| 固定后缀 8 token | 共享 128、256、384、512、1024 token | 共享长度对 TTFT 的影响 |
| 页边界 | 0、15、16、17、127、128、129 token，页大小 16 | 实际命中按整页截断 |
| 完全相同的请求 | 总长 128、512，后缀为 0 | 保留末 token 用于首个输出，预期命中 112、496 |
| 固定总输入 512 | 共享 0%、25%、50%、75%、100% | 区分总输入长度和共享比例；100% 按整页保留末 token |
| 固定前缀 128 | 剩余 1、8、32、128 token | 命中后不同 query 长度的 attention 开销 |
| 单客户端与并发 | C1、C2、C4 同时到达，C4 包含 0、1、2、4 个预期命中请求 | 同批热、冷请求的延迟和整批吞吐 |
| 保留与淘汰 | `resident_p64`、`evicted_p64` | 同一目标、同容量小池；比较留驻与无关流量填满池后被淘汰 |

`quick` 选取 `p128_s8`、`p512_s8`、`repeat_p128`、`c2_h1`、`evicted_p64`。显式指定 `--case` 或 `--shape` 时只运行选中的负载。下面的自定义形状同时提交四个请求，其中三个与预热请求共享 2048 个 token，每个请求有 32 个后缀 token。用 `--output-tokens 1` 单独研究首 token，或设为 32、64 研究生成阶段。

~~~bash
python -m benchmarks.ascend_prefix_matrix --model "$MODEL" \
  --shape 2048 32 4 3 --output-tokens 1 --repeats 10 --out results/custom.json
~~~

## 如何读报告

每轮都会在同一模型上重建空缓存，先跑一个等长的预热请求，再按场景提交同一批确定性目标 token ID。三组轮换执行顺序。构造引擎、预热、填满缓存池的请求均在目标计时窗口之外。

| 名称 | 计时前的缓存状态 | 回答的问题 |
| --- | --- | --- |
| `off` | 关闭缓存 | 现有推理路径的基线 |
| `miss` | 开启缓存，预热请求第一页不同 | 开启缓存本身的开销 |
| `reuse` | 开启缓存，按场景留下共享前缀或制造淘汰 | 实际命中和未命中的混合效果 |

同名 CSV 每个场景一行。先看 `status`。脚本以逐请求的 prefill 起点和命中数核对预期，并检查批次 KV 计数和各组输出 token。有分歧时会给出请求位置和首个不同的输出位置，位置从 1 开始。正确性失败时收益列留空。已完成场景的原始测量会及时写入 JSON，运行异常会记下当时的场景、轮次和缓存配置。

末行 `MATRIX_REVIEW` 分别列出命中后比 miss 慢、miss 比 off 慢的最多三个场景，以及需要先解决的检查失败。`extra_ms` 为中位 TTFT 多花的毫秒数，`slower_pairs/pairs` 表示几轮配对出现了相同方向的差异。这个清单用于选取复测与 trace 场景。

- `reuse_vs_miss_ttft_p50_saved_ms` 是 miss 的中位 TTFT 减去 reuse 的中位 TTFT，正值表示命中组更快。`reuse_vs_off` 衡量启用复用带来的净收益，`miss_vs_off` 衡量开启缓存后的冷请求表现。
- TTFT 从每个请求提交到引擎返回首个 token，E2E 从提交到完成。p50/p95 按该组所有计时请求合并计算。JSON 中还保留逐轮配对差值。少量样本的 p95 适合初筛，在候选慢点上增加 `--repeats` 后再判断。
- 吞吐按所有目标输出 token 总量除以各批次总耗时计算，单位为 token/s。它衡量一次批量到达的处理速度。TPOT 按每请求首 token 到末 token 的时间除以后续 token 数，只有一个输出时为空。
- `reuse_hit_tokens_per_request` 对混合场景是全批均值。JSON 的 `cases.<场景>.summary.arms.reuse` 还分别给出共享前缀请求和无关请求的延迟。原始 `trials` 保留每个请求的实际命中、首个 prefill 位置、输入摘要和生成 token ID。
- 小池场景会记录填满缓存池前可查到的前缀长度以及目标实际命中量。`peak_used_blocks` 描述目标批次占用、无法回收的物理 KV 块数；每轮的总容量见 `num_blocks`。

## 和以后版本比较

在修改相关逻辑后，用相同的卡、环境、权重和参数重跑。基线 JSON 会校验权重文件 SHA-256、设备和运行设置、场景、各请求的输入摘要与输出 token。程序逐一比较 `off`、`miss`、`reuse` 的同名组。正的延迟节省或吞吐增幅表示新版本更快。早期脚本的单行 `PREFIX` 结果可作为趋势参考；历史对照使用本矩阵保存的 JSON。

~~~bash
python -m benchmarks.ascend_prefix_matrix --model "$MODEL" --suite full \
  --warmups 1 --repeats 5 --baseline results/baseline.json --out results/candidate.json
~~~

也可直接读取已完成的两份报告，在原容器中离线比较：

~~~bash
python -m benchmarks.ascend_prefix_matrix --read-report results/candidate.json \
  --baseline results/baseline.json
~~~

## 从异常结果定位代码

CSV 中用场景名和 `status=FAIL` 定位行；原始终端中的 `场景名 FAIL` 后面附有首条分歧。用下面的命令直接离线读取原矩阵 JSON，按场景区分预热与计时轮，打印出错轮次、类别和首条证据。只读报告无需权重或 NPU。

~~~bash
python -m benchmarks.ascend_prefix_diagnose --report results/prefix-baseline.json
~~~

需要复现时，在相同环境里加上权重和一个新的输出文件。默认处理原报告标记的场景并各跑两轮，也可以重复指定 `--case`。以下示例保留原来四个目标的顺序、seed、16 个输出及四个循环深度，比较关闭缓存、开启未命中、命中复用和无缓存分块四条路径。

~~~bash
python -m benchmarks.ascend_prefix_diagnose --report results/prefix-baseline.json \
  --case c4_h4 --model "$MODEL" --out results/diag-c4-h4.json
~~~

`ARCHIVE` 的 `target` 从 0 编号，`first_diff_1based` 从 1 编号。`DIAG` 给出重放各组输出的对比、分歧位置和两名候选 token 的分数。完整 JSON 还保存每次调度的请求、阶段、范围、循环数、分配页表、共享页内容检查和候选分差。先区分三种现象：

- 命中数或起点与预期不符，检查缓存查找和调度起点；起点正确但物理页不符、私有尾页重叠或共享页摘要改变，检查分配与写入。
- 若页检查通过，且复用与无缓存分块给出同一输出、与完整 prefill 分歧，优先比较算子形状与首次分歧时的 logits。分数接近本身不能证明 BF16 是根因。
- 一次重放通过仅说明本轮未复现。根据原来的预热、计时轮和失败频率，重新检查原矩阵中的严格逐 token 对照，再恢复性能验收。

诊断运行会读回页内容和 logits，不能用其耗时做性能结论。共享页检查覆盖 donor 释放前与目标结束后两个端点；它能发现保留下来的改写，无法单独证明 donor 计算出的 KV 数值或每次设备读取地址正确。

当单请求的 `off` 和 `split` 输出相同、`reuse` 分歧时，可直接使用上一步保存的诊断 JSON。下面的实验增加 `aligned` 对照。它让提供前缀的请求（donor）先算 384 个 token，再算剩余 128 个。`reuse` 和 `aligned` 的目标请求都命中 384 个 token，都只计算 128 个后缀 token。

~~~bash
python -m benchmarks.ascend_prefix_boundary --diag results/prefix-diag-01.json \
  --model "$MODEL" --case total512_p384 --out results/boundary-384.json
~~~

先看末行 `BOUNDARY` 的 `comparable`。为真表示旧配置的输出、输入摘要、命中数和起点与至少一轮历史重放对应，且本次四组都执行了预期的前缀和后缀区间。`kv` 比较请求释放前同一逻辑前缀的 BF16 字节，按循环深度、模型层和 K/V 分别计算摘要。`aligned_donor__split_target.equal` 为真且 `output_first_diff.split__aligned` 为空，表示分块生成的 donor 前缀与目标自己分块计算的前缀逐位相同，后续生成的 token 也相同。如果前缀逐位相同但输出仍有分歧，下一步检查目标读取的逻辑位置、后缀计算和首次 decode。同样的对照也支持 `--case p1024_s8`。

| 观测 | 下一步验证 |
| --- | --- |
| 页边界、完全重复或压力场景命中长度不符 | 看首次 prefill 偏移、KV 页发布覆盖和淘汰前后的查找结果 |
| 只在 reuse 出现输出分歧 | 找首个不同 token；在可复现的单请求上用原 `benchmarks.ascend_prefix_cache` 的无缓存分块对照区分续算数值路径与共享页读写 |
| 命中正确，短后缀 TTFT 却变慢 | 比较 miss/reuse 的 KV 查找、完整 prefill、续算 attention 及 NPU 算子时间线 |
| miss 比 off 慢，或只在 C4 混合负载退化 | 分别比较 off/miss 的查找与分配时间，或单个热/冷请求的等待、实际计算形状与批次占用 |

针对单个预设场景单独采集诊断 trace。默认输出一个 token，聚焦首 token。为复现正式测量的输出上限，可显式传 `--output-tokens 16`。每个缓存配置独立预热，只追踪目标批次。Profiler 会扰动耗时，正式的速度结论取上面的未采集 trace 的报告。

~~~bash
python -m benchmarks.ascend_prefix_trace --model "$MODEL" --case p128_s8 \
  --arms miss reuse --output-tokens 16 --out results/trace-p128
~~~

在输出的 `trace_view.json` 中搜索 `RLT/KVCacheManager.lookup_prefix`、`RLT/AscendEagerAttention._batched_prefill` 和 `RLT/AscendEagerAttention._forward`，再对齐同一时间段的设备算子。若需要量化开启缓存后的未命中开销，改用 `--arms off miss`。主机范围包含提交和等待，直接相加会重复计算。

## 来源与范围

- RLT 自身的前缀复用测试包含完整页复用、继续生成、引用计数和淘汰，本矩阵沿用逐 token 的基线对照。[RLT 前缀回归](../../tests/test_prefix_growth.py) · [原始 910B2 脚本](../../benchmarks/ascend_prefix_cache.py)
- vLLM Ascend 的双卡回归用相同长前缀及不同问题比较开关缓存后的输出，系统测试比较不同前缀共享比例的 TTFT。这里借鉴请求构造和对照方式，910B2 单卡 Ouro 的结果独立测量。[共享前缀正确性测试](https://github.com/vllm-project/vllm-ascend/blob/80a705e62778bee2d49ffa72becf557961f7c181/tests/e2e/pull_request/two_card/test_prefix_caching.py#L60-L93) · [共享比例性能配置](https://github.com/vllm-project/vllm-ascend/blob/80a705e62778bee2d49ffa72becf557961f7c181/tests/SIT/cases/models/configs/Qwen/Prefix-Cache-Qwen3-32B-Int8.yaml#L31-L70)
- 本矩阵使用确定性合成 token ID、单卡引擎和批量到达请求。HTTP 层、自然语料、持续在线到达及多卡部署各有不同的测量口径。
