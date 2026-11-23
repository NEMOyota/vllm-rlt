# 昇腾 prefix cache 命中后的同批 prefill attention

两轮正确性、三轮性能及多卡执行见[并行验收](ascend_prefix_parallel.md)。

本分支在 Ouro 同步、单流、BF16、`ascend_eager` 的 packed prefill 路径中实现近似长度合批。prefix cache 命中消去部分 Q 行，attention 仍需读取当前请求的可见 KV。例如，两条请求各命中 48 个 token，分别剩 23、20 行。单独计算需要 2 次 SDPA，每个头共有 `23×71 + 20×68 = 2993` 个矩形分数位置。合批后使用 `[2, heads, 23, 71]` 的分数几何，每层每个循环深度只调用 1 次 SDPA，代价是多出 273 个位置（9.12%）。这里的矩形面积衡量补齐开销，设备实际执行受因果掩码和算子实现影响。

## 开启方式

在原有的昇腾 prefix-cache 服务命令中添加 `--ascend-prefix-prefill-slack 4`。0 只合并 Q 和 KV 长度都相同的请求。省略该参数运行现有调度。下面给出单卡示例，卡号与本地权重路径按分配替换。

```bash
export MODEL=/models/Ouro-1.4B
export ASCEND_RT_VISIBLE_DEVICES=0
python -m vllm_rlt.entrypoints.serve --model "$MODEL" --device npu --dtype bfloat16 \
  --attention-backend ascend_eager --enable-prefix-caching --single-stream \
  --ascend-prefix-prefill-slack 4 --max-num-seqs 4 \
  --prefill-chunk-size 512 --max-num-batched-tokens 2048
```

服务端只在已调度的一个 prefill 步里分组，分别处理各循环深度。按 Q 长度、KV 上下文长度排序后，把成员加入相容的组。组内 Q 和 KV 各自的最大长度差都须在阈值内，每头矩形面积的额外开销最多为未补齐面积的 1/8。每组临时输入与掩码的估算上限为 64 MiB，SDPA 内部工作区由后端决定。4 行、12.5% 和 64 MiB 都是本次实验的保守取值。常规等长冷批、单行 decode、未分组成员和不适用的 KV 存储布局沿用已有路径。

每个成员依自己的页表取 KV，有效查询按原始绝对位置做因果 attention。全冷子批使用 SDPA 原生因果标志，同起点后缀共享二维掩码，起点不同则构建逐请求掩码。填充的 Q/KV 索引只引用本请求已初始化的位置，结果按有效行散回原始顺序。KV 写入、模型线性层和请求准入沿用原有路径。后端直接消费已经调度的请求，无需等待下一次到达。形状索引按设备和流复用，每份新描述符重新绑定当前页表。vLLM-Ascend 的 TND 路径同样传递逐请求实际 Q/KV 长度和页表，本实现沿用 RLT 已有的 Torch SDPA 和按循环深度选择的 KV 页。[vLLM-Ascend attention](https://github.com/vllm-project/vllm-ascend/blob/ae9a697311b02421ca6cb09157f0414a5521a71a/vllm_ascend/attention/attention_v1.py#L987)

## 910B3 和 910B4 运行

两台机器检出同一提交，在各自的 CANN / torch-npu 环境中设置相同版本的 `MODEL` 和分配到的单卡。B3 八卡节点每份报告使用一张卡。两个命令分别在对应机器上执行。

```bash
# 910B3
ASCEND_RT_VISIBLE_DEVICES=0 python -m benchmarks.ascend_prefix_pr1_suite pr2 --chip 910B3 --model "$MODEL"
# 910B4
ASCEND_RT_VISIBLE_DEVICES=2 python -m benchmarks.ascend_prefix_pr1_suite pr2 --chip 910B4 --model "$MODEL"
```

快速验证 48-token 共享前缀和 23/20 行后缀，在已选卡的终端运行下列命令。单请求对照使用 `--case single`。单机 CPU 可执行 `OMP_NUM_THREADS=1 python -m pytest -q tests/test_ascend_prefix_prefill_groups.py tests/test_ascend_prefix_pr2_benchmark.py`。

```bash
python -m benchmarks.ascend_prefix_pr2 --model "$MODEL" --case near --repeats 3
```

### 验收与读数

所有主比较都开启 prefix cache。`main` 走原有后端，`exact` 为 slack 0，`similar` 为 slack 4。`main → exact` 衡量等长合批，`exact → similar` 衡量放宽形状的增量。使用最终候选时可在套件命令末尾加 `--repo "$PR2_REPO" --candidate "$PR2_SHA"`，其中仓库须含 PR2 产品代码。用 `--reference-traffic "$PR1_RESULTS"` 校验已有 BF16 流量清单。

| 阶段 | 验收与证据 |
|---|---|
| `environment`、`pr2-oracle` | 记录运行时型号、软件版本与权重指纹。设备单测用独立 FP32 稠密因果计算对照 BF16 attention，覆盖不同前缀起点、跨页、非相邻成员、GQA、共享页、填充槽 NaN、页分配变化及单行混合。记录真实 SDPA 调用数，验证 KV 内容保持一致。 |
| `pr2-geometry` | 六组为近似后缀、等长后缀、冷热混合、长度离群、整齐冷批和单请求。每种模式预热后轮换测三轮。断言实际命中、真实 prefill 行数、组数、减少的调用和填充位置。每个请求生成 16 token，逐 token 对比三种模式与同模式重放，检查四次循环、正常结束和 KV 释放。`raw.jsonl` 保存提示、逐请求 TTFT、首个分歧和逐轮差值。 |
| `pr2-reference` | 同一矩阵使用 FP32 线性层固定 256 行作数值诊断，KV 与 SDPA 为 BF16。原生 BF16 的分歧保留在各自报告。 |
| `pr2-burst`、`pr2-paced` | 三种模式用正式 CLI 选择后端，真实 HTTP 和独立模型 worker 回放相同合成负载。32 条冷热长短请求，并发 4，seed 168，burst 与平均 3 请求/秒各做三轮，轮换模式顺序。检查逐请求 token、命中和 worker 的实际开关。读 `service.csv` 看各类请求的完成数、P50/P95、含客户端排队的 P95 和输出吞吐。读 `review.json` 看实际触发和配对 TTFT，`raw.jsonl` 保留所有轮次。 |
| `observe-burst`、`observe-paced` | 单独观测运行。打开 `round-1-main/opportunities.json`，在同一步的真实成员上按 0/2/4/8 重放分组规则。每层每个深度的预测区分没有同批、已有整齐冷批、可等长合批和需要补齐的组。实际执行以计时轮 worker 的计数为准。 |
| `pr2-profile` | 对近似后缀的三种模式分别预热，采集目标 prefill 单步的 ops-only/full 设备 trace，校验设备 kernel，full 另外校验形状、栈与内存记录。性能结论取自关闭 profiler 的轮次。 |

正确性要求每个对照请求的命中、输出、循环和结束原因一致。数值或控制检查失败时保留证据，排除对应性能结论。服务量化看同卡、同工作量下各轮 TTFT、P95、吞吐，同时列出新增分数位置与省下的 SDPA 调用。只有实际执行了近似合批，才用这些样本归因近似策略的增益。零触发本身就是负载结果。受控场景节省调用可以证明机制生效，是否适合默认开启由 B3、B4 上目标流量的收益和回退决定。

`similar_service` 单列近似相对等长的服务结果。`no_similar_groups` 表示本负载没有触发近似合批。`limited_samples` 表示触发不连续或轮次少于三，保留全轮差值。`favorable_direction` 要求每轮均触发、多数轮 TTFT P50 变快，且配对 P95 节省量、输出吞吐增量的中位数均非负。`mixed_or_flat` 表示该组结果未达到此判据。这是小样本方向性判据，PR 中同时展示原始轮次和样本数。
