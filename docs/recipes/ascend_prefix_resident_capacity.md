# 实验 01 · 固定 KV 池下的 W4 驻留与性能

实验分支为 `experiment/ascend-prefix-resident-capacity`，框架基线为 `a4b1e4b07617`。一次运行使用同一台机器、同一张物理卡。所有臂保持 5280 块、16-token 页、每步 1024 token、256-token chunk、4 条普通预热和 32 条目标请求。每臂每轮启动独立进程。默认历史 seed 为 797543261。

## 准备

在两台服务器原有的 CANN 9.0.0 容器内分别运行下列 Bash 命令。复用已安装的 torch_npu 和项目依赖。以下用新的克隆目录，原工作区可继续使用。

```bash
git clone --single-branch --branch experiment/ascend-prefix-resident-capacity \
  https://github.com/NEMOyota/vllm-rlt.git rlt-resident-01
cd rlt-resident-01
git rev-parse HEAD
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# B3 使用历史 W4 检查点目录
export CHIP=910B3 MODEL=/models
# 在 B4 上改用下一行
# export CHIP=910B4 MODEL=/models/Ouro-1.4B

export CARD=3 ASCEND_RT_VISIBLE_DEVICES=3
test -f "$MODEL/config.json"
export RUN_ROOT="/workspace/rlt-pr1/results/prefix-resident-capacity/$CHIP/card-$CARD/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_ROOT"
export ENTRY=benchmarks.ascend_prefix_resident_capacity
run01() {
  local name="$1"
  shift
  python -m "$ENTRY" --device 0 --checkpoint "$MODEL" --output "$RUN_ROOT/$name" "$@"
}
```

`--device` 使用进程内编号。映射单张物理卡时一律用 0。八卡环境可把独立实验分到不同卡，但同一配对的各臂、各轮应由同一次命令在同一卡上运行。不要在同一卡上同时启动两个实验。输出目录按实验使用新名称。

## 执行

```bash
# CPU 可运行，打印全部九组独立配置和理论容量
python -m "$ENTRY" --plan --rounds 1 --groups A B C D E F G H I

# 第一阶段，A/B/C/I，六轮，每轮四臂
run01 core --case hot

# 第二阶段，九组，六轮，独立产生完整配对报告
run01 full --case hot --groups A B C D E F G H I

# 全冷，对比 C、I
run01 cold --case cold

# 21 热、11 冷；默认执行交错、冷优先、热优先，每种顺序 B/C/I 各六轮
run01 mixed --case mixed --hot-requests 21 --order all

# 强制回收，冷请求先到，占满五个全冷槽位，然后尝试原 donor 前缀
run01 evict --case evict --rounds 1

# 完整页表和引用审计，先核对 C、I，再检查回收
run01 audit-hot --case hot --groups C I --mode audit --rounds 1
run01 audit-evict --case evict --mode audit --rounds 1

# 独立 msprof 运行；结果保存在该臂的 profiles/ 目录
msprof --help
run01 profile --case hot --groups I --mode profile --rounds 1
```

单独测试一种混合顺序可用 `--order interleaved`、`--order cold-first` 或 `--order hot-first`。其他种子使用 `--seed`。超时默认每个子进程 7200 秒，包含模型加载、预热、目标请求和 profile 导出，可按服务器速度调整 `--timeout-s`。某臂 OOM、超时、异常或出现输出分歧仍会继续其余轮次并返回非零状态。profile 采集整个独立进程，包括模型加载及预热；仅 performance 模式进入正式吞吐统计。

## 读结果

- `environment.json` 保存 Git SHA、检查点配置和各权重分片的完整 SHA-256、设备型号、物理编号、环境版本。`config.json` 保存独立参数和所有轮次的执行顺序。
- 每个臂在 `模式/负载-顺序/seed-*/round-*/组/` 保存 `job.json`、`device.json`、`result.json`、`run.log`。`requests.jsonl` 保留提交、准入、首次 host dispatch、首 token 和完成时间，单位为距本组目标开始的秒，以及每个请求的实际命中、完整输出和退出深度。
- `steps.jsonl` 给出每步的阶段、请求数、Q 行数、KV 长度、token 预算利用率、FIA 和各类 SDPA 计数。只有 prefill 和 recurrent 会执行 Ouro attention。对观察到的批量 recurrent gather 路径，输出按当步形状估算的**单层 K+V** 字节数；独立 profile 可检查实际分配。
- `residency.jsonl` 按提交、成功准入和释放记录活跃、独占、共享、仅缓存和空闲块。正式轮记下不可变页表引用，计时结束后再按物理页号去重。audit 另外生成 `kv_audit.jsonl`，核对四深度的页号、逻辑前缀、索引引用、请求引用、逐块守恒，并记录发布与淘汰明细。
- 每臂采用相同 4 条普通计算预热，预热生成 2 token。同步排空后释放预热索引；ON 组执行完整 4096-token donor、生成 1 token。OFF 组的正式 donor 成本为 0。输出 `output_tok_s` 衡量目标组，`inclusive_output_tok_s` 另计本组一次 donor；两者均以目标请求生成的 token 作分子。
- 等待分为提交到准入的 `queue_*_ms` 和准入到首次执行提交的 `execution_wait_*_ms`。使用 `resident_mean`、`residence_seconds` 和 `steps.jsonl` 确认新增驻留是否变成更大的执行批。对比 `capacity_gap`、`budget_deferrals`、实测命中和 NPU 高水位诊断未达到理论峰值的原因。budget deferrals 为尝试次数，同一请求可以多次等待。
- `summary.csv` 列出每轮绝对指标，`group_summary.csv` 汇总各档位正式轮的绝对中位、最小和最大值，`paired_rounds.csv` 列出两臂的原始值和变化百分比，`paired_summary.csv` 汇总有效轮的中位和范围。变化按“分子/分母 − 1”计算，延迟的负数代表降低。`report.md` 单列理论和观测驻留。只有同轮、同输入、同设备、完整且未被标记分歧的 performance 配对参与正式汇总。
- `exceptions.jsonl` 保留异常与 BF16 首个不同 token（从 1 起算），同组跨轮、相同驻留下的缓存开关、相同开关下的驻留变化、同时改变两因素均单列。关联臂的实测数据保留为 DIAGNOSTIC。如进程被硬超时终止，仅能保留终止前已落盘的文件和 TIMEOUT 状态。

建议先回传 core、audit-hot 和 audit-evict 的完整目录；其余负载同样保留全量原始数据。归档当前两机各自的运行目录：

```bash
tar -C "$(dirname "$RUN_ROOT")" -czf "$RUN_ROOT.tar.gz" "$(basename "$RUN_ROOT")"
sha256sum "$RUN_ROOT.tar.gz"
```
