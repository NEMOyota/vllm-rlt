# 昇腾 PR1 双机验收

两台服务器各自检出此分支，并激活已有的 CANN 9.0.0、PyTorch 2.9.0 和 torch_npu 2.9.0.post2 环境。将 `MODEL` 设为本机 Ouro-1.4B 权重和 tokenizer 所在目录。

```bash
git clone --single-branch -b test/ascend-pr1-classified https://github.com/NEMOyota/vllm-rlt.git
cd vllm-rlt
export MODEL=/models/Ouro-1.4B
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

在 B3 设置 `export CHIP=910B3`，在 B4 设置 `export CHIP=910B4`。以下命令只测引擎性能。

```bash
python benchmarks/ascend_pr1_acceptance.py run --chip "$CHIP" --categories 3 \
  --cases W1,W2,W3,W4,W5 --out "results-$CHIP-perf-isolated"
```

每种精度依次运行 W1～W5。每个负载按固定种子分卡，每个种子使用独立进程。全部卡上当前负载的进程退出后，才启动下一个负载。在同一种子的进程内，每次 off、miss、reuse 测量结束后都会关闭引擎、回收循环引用、释放 NPU 分配器的空闲缓存并检查旧 K/V 张量已释放。计时覆盖实际请求，共享前缀照常在预热后供目标请求复用。

各用例目录下的 `replay-memory.jsonl` 按轮次记录进程号、建池前后和回收后的显存字节数；成功回收的记录为 `stage=after_cleanup, kv_released=true`。回收后的已分配显存仍包含该进程的模型权重。若某次建池 OOM，可用最后一条记录和 `result.json` 的异常栈定位。

要测七类全量，使用 `python benchmarks/ascend_pr1_acceptance.py run --chip "$CHIP"`。全量结果在 `results-$CHIP/`。重跑时用 `--out` 指定新目录。查看 `summary.csv` 的 `bf16_first_issue` 列可定位首个分歧，`followups.json` 列出同卡补测 FP16 和固定 256 行 FP32 Linear 的用例。

两台均使用固定的十个种子，按顺序轮转到选定的卡。将两个全量结果目录放在同一台机器上后合并报告。

```bash
python benchmarks/ascend_pr1_acceptance.py report results-910B3 results-910B4 > PR_RESULTS.md
```

脚本在临时 Git 检出中使用正式候选 `606a60ce9bacdd5df1d25168d38665b3a0ef106d`、回归基线 `4821d47cec92aa1d7d2ec4be5f589d19b2794ae3` 和历史测试工具 `136a775e02c8c63ac5335970189a1d50b6c3d4f2`。分支附带的完整 Git bundle 包含这三个提交，脚本会从中补齐本地缺少的对象。[固定种子](./pr1-fixed-seeds.json) · [完整测试包](./pr1-fixed10-tests.bundle)
