# Ascend prefix cache 并行验收

在已激活 CANN、torch-npu 和项目测试依赖的 Linux 容器中，分别检出两个候选。测试入口使用已提交的 Git 快照。两个分支的结果目录分别带有完整候选 SHA。

```bash
git clone -b test/ascend-prefix-pr1-validation --single-branch https://github.com/NEMOyota/vllm-rlt.git rlt-pr1
git clone -b exp/ascend-prefix-pr2-grouping --single-branch https://github.com/NEMOyota/vllm-rlt.git rlt-pr2
cd rlt-pr2
export MODEL=/models/Ouro-1.4B
python -m benchmarks.ascend_prefix_parallel --pr1-repo ../rlt-pr1 --pr2-repo . \
  --chip 910B3 --cards 0,1,2,3,4,5,6,7
```

在 910B4 上把后两个参数改成 `--chip 910B4 --cards 0`，卡号使用平台实际分配值。只提供一个仓库参数可以只跑对应的 PR。`--list` 输出任务分配。`--device-only` 适用于已另行保存同一候选 CPU 回归证据的情况。

三组分别为原生 BF16、原生 FP16 和固定 256 行的 FP32 Linear 参照（BF16 KV/SDPA）。FP16 权重从同一 checkpoint 加载。引擎、后端及 HTTP CLI 的 dtype 放行补丁应用在单独的临时工作树，补丁及摘要随报告保存。正式服务进程中的 FP32 线性层由测试 worker 安装并核对实际执行次数。

- **引擎阶段。** 卡号按参数顺序使用。前四张均分 PR1 的 30 例，接下来两张均分 PR2 的六组形状，最后两张分别负责两个候选的入口与设备回归。设备数较少时，任务映射到已有卡并串行排队。每个场景的三个精度和各对照模式在同卡执行。
- **HTTP 阶段。** 全部引擎任务结束后才启动。burst 和 paced 各占一张卡，两个 PR 依次回放同一流量。各并发 worker 及其服务子进程使用各自的卡选择器、CPU 亲和集合、端口和目录。PR2 的机会统计来自实际计时回放，保留 host 观测开销，跨模式使用同一观测设置。
- **轮数。** 正确性专用实验执行两个独立轮次。矩阵、合批和 HTTP 各完成三轮配对测量，前两轮给出正式 token 正确性结论，第三轮同时核对性能样本的可比性。benchmark 的整场预热设为零，构造命中的 donor 请求属于用例前置条件。每款卡核对一次型号，每个候选各跑一次自己的 CPU 和原生 NPU 基础回归。
- **读报告。** 从终端打印的目录打开 `PR_RESULTS.md`。根目录的 `plan.json`、`review.json` 记录分配、硬件和完成状态。每项工作在 `<PR>/<SHA>/<卡型>/<精度>/<任务>/` 保存命令、日志和数据。矩阵分片按场景、轮次和模式保留原始 token、性能和首处分歧。HTTP 的 `data/service.csv` 按轮列出 P50、P95 和吞吐，PR2 的 `data/round-<n>-<模式>/opportunities.json` 记录机会和分母。正确性失败保留原始时延，并继续运行其余工作。缺少或重复测量记 INCOMPLETE。
