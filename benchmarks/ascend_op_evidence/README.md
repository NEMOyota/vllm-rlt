# Ouro Ascend 算子正确性诊断包

本包仅重放 PR1 的两个原生 BF16 反例。实际推理使用 `606a60ce9bac`，原工作负载和引擎 trial 来自 `136a775e02c8`。压缩包自带 Git bundle，服务器需要 Git、Python 和原 CANN / PyTorch 环境。所有推理与采集均在服务器本地执行。

## 准备

在原报告的容器中，激活 CANN 9.0.0、torch 2.9.0、torch_npu 2.9.0.post2。从压缩包根目录或实验分支根目录执行。两台服务器分别设置 CHIP 和本地权重路径。原报告中 B3 的模型路径为 `/models`，B4 为 `/models/Ouro-1.4B`。脚本核对配置和权重的 SHA-256。

```bash
export CHIP=910B3                          # 在另一台改为 910B4
export MODEL=/models                         # 指向含 config.json 的原始模型目录
export ORIGINAL_RESULTS=/workspace/rlt-pr1/results-$CHIP
export OP_WORK=/workspace/rlt-pr1/op-evidence-$CHIP-original
RUN="$PWD/benchmarks/ascend_op_evidence/run.sh"
bash "$RUN" prepare
```

同一个 OP_WORK 只需 prepare 一次。此前创建的采集目录会保留，独立重复实验请使用新的 OP_WORK。脚本按用例绑定物理卡 3 或 0，进程内设备为 npu:0；请确保容器允许访问相应物理设备。先在原环境复现。比较新 CANN 时设置新的 OP_WORK 和准确的 OP_CANN_VERSION，再独立执行 prepare 和后续步骤。

## 无探针复现

```bash
bash "$RUN" baseline "$CHIP" c4_h1
bash "$RUN" baseline "$CHIP" total512_p256
```

`c4_h1` 使用 seed 421559278、卡 3，期望命中 [128,0,0,0]，target-3 的首个输出 532 → 30。`total512_p256` 使用 seed 114281796、卡 0，期望命中 [256]，target-0 前八个输出相同，第九个 302 → 447。两轮按原 trial 顺序每臂建立独立引擎并预热 donor，完整生成 16 个 token。原请求 ID 列表可用 OP_INPUT_IDS 传入，JSON 字段为 donor、miss、targets、fillers，targets 保持原提交顺序。原结果目录下只有哈希时，脚本调用原工作负载生成器并显式传 seed，核对 inputs.json、result.json 或 matrix/raw.jsonl 中可用的指纹。每次都保存完整 IDs 与校验结果。

每个 case 的 summary.json 保存两轮 off↔miss、off↔reuse 的首次分歧（1-based），以及各请求的实际命中数与输出 IDs。无探针运行不拦截模型；同历史 hidden/logits 由下一步独立采集，并再次核对所有请求的所有 IDs 与命中数。标记未复现时先检查 baseline 下的 environment.json、fingerprints.json 和 round-*.json。

## 定位、采集与重放

先在一台卡上深抓，另一台对相同用例、相同逻辑调用重复。对两个用例分别设置 C。

```bash
C=c4_h1                                 # 另一例用 total512_p256
bash "$RUN" scan "$CHIP" "$C"           # 三臂，自动生成 summary.json 与 selection.json
bash "$RUN" profile "$CHIP" "$C"        # 独立 msprof 应用采集；亦可在最后仅传 off、miss 或 reuse
bash "$RUN" dump "$CHIP" "$C"           # 定点保存 off、reuse 两份整批原始调用
bash "$RUN" replay "$CHIP" "$C"         # 使用原卡，独立于 Ouro
bash "$RUN" op "$CHIP" "$C" off         # 独立 NPU 重放成功后执行
bash "$RUN" op "$CHIP" "$C" reuse
bash "$RUN" pack "$CHIP" "$C"
```

`scan` 比较目标请求的首个共同后缀位置、末 prompt 位置及直到首次分歧的 decode 位置，记录真实 packed 行、绝对位置、循环深度、层、模块与调用次数。在有缓存前缀的单请求用例中还逐深度/层对齐整段共享 K/V 与冷算 K/V 的位级指纹。summary 中的“最早”仅涵盖已观察的逻辑位置。若最早差异出现在 Linear 的输入、attention、KV 或其他模块，保留这条上游线索。自动选择最早观测到输出差异的 F.linear 作为独立重放边界。没有这样的调用时，保留 scan 结果，跳过 dump、replay 和 op。

`profile` 按本机 msprof --help 选择参数并记录实际 argv 和日志，保留 PROF_*、op_summary 和时间线。外部 msprof 的应用级 trace 可以覆盖进程启动及收尾，目标窗口的 Linear 调用带 MSTX 标记（运行时支持时）。若磁盘有限，使用下列独立采集方式，在目标 prefill 前开启 profiler，在指定 CODA 后停止；采集窗口之外继续完成原生成。两种方式分别运行，避免嵌套 profiler。

```bash
bash "$RUN" window "$CHIP" "$C" reuse
```

在 MindStudio Insight 中按 selection.json 的模块、轮次、深度及目标行定位，在 observations.json 取两臂各自的 `rlt/.../tileN` 标记。kernels.csv 给出真实名称、stream/task、block dim 和原文件行号。仅当导出包含相同设备 stream 的时间区间且恰有一个 MatMul 匹配时才自动关联；其余显示 INSPECT_TIMELINE，请用 MSTX 的设备轨道和框架到内核的关联线核对，必要时补充记录。独立 op 采集各形状只调一次 F.linear。保留 op 目录供 Insight 对比 kernel、block dim、tiling 和本机实际可导出的字段；日志里的 debug_line 提示列在 exports.json。

## 第二例的精度对照

```bash
for phase in baseline scan; do
  bash "$RUN" "$phase" "$CHIP" total512_p256 all fp32-linear || break
done
bash "$RUN" pack "$CHIP" total512_p256
```

此路径在原脚本的固定 256 行 FP32 Linear 下重放。scan 各臂的 result.json 记录实际执行的 Linear 模块及调用次数，observations.json 记录内层 F.linear、SDPA 的实际 dtype，trial 记录 K/V dtype。head.pt 保存对应步骤的完整 logits 和进入 LM head 的 hidden，summary 保存候选分数与真实采样 ID。原 BF16 标记是 off 打平、reuse 的 logit[447] − logit[302] 为 +0.0625。

## 交付给算子开发者

每个 case 生成一个 `$OP_WORK/$CHIP-$C-evidence.tar.gz`。包内包含脚本、环境与补丁、输入、关键 CSV、两臂各自的完整 input/weight/bias/output 和精度对照数据。大体积原始 PROF_* 保存在 OP_WORK，路径列在 package-manifest.json。解包后无需安装 Ouro 或 vllm-rlt 即可运行：

```bash
# 在证据包解压目录中，指定本机能访问的物理卡
ASCEND_RT_VISIBLE_DEVICES=3 python scripts/replay_op.py \
  bf16/dump/off/operator.pt bf16/dump/reuse/operator.pt --device npu:0 --out replay-local.json
```

边界是 F.linear(input, weight, bias)，其逻辑计算使用 weight.T。重放保留整批 shape、stride、storage offset、可获取的 NPU format 和目标行，单独用 CPU FP64 计算该行全部通道，再舍入至真实输出 dtype。报告区分输入/权重/偏置是否一致、独立输出是否逐位复现，以及相对参考值的误差。能复现形状相关的差异后，是否超出算子精度规格仍需结合该 dtype/内核的容差判定。

另一台卡完成 scan 后，可固定第一台卡选中的同一逻辑调用，再执行 dump 和 replay。会重新核对本机的行号和张量指纹。

```bash
OP_SELECT_FROM=/path/from/first-host/selection.json bash "$RUN" compare "$CHIP" "$C"
```
