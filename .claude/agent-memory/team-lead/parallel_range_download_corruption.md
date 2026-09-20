---
name: parallel-range-download-corruption
description: 并行 Range 分块下载的并发追加竞态导致"尺寸正确内容错乱"损坏——探针抽样校验不够，必须全量流式比对或 gzip -t 终检
metadata:
  type: feedback
---

# 并行 Range 下载器的并发追加竞态损坏（2026-09-01 ProteinMPNN pdb_2021aug02 实测）

**规则**：
1. append-resume 式分块下载器（`curl -r start+got-end >> file`）**绝对禁止两个实例同时写同一分块目录**——并发追加会产生"长度恰等于期望值但内容错乱"的坏块（两个写者从同一 got 出发交错追加，或 kill 时机凑巧拼出 8MB）。
2. 分块下载完成后**必须做流级完整性终检**（`gzip -t` / `tar -tzf` / md5），仅靠"每块尺寸==期望"判据会把坏块当成功。
3. 修复坏块时用"多探针远端比对"（每块 5×16KB）只能覆盖 ~1% 字节，会漏掉块内中部损坏；**确定性收敛只有全量流式比对**：并行拉全部块逐字节比对 + 不匹配就地 dd/seek 修补（16.8GB @ ~5MB/s ≈ 1h，192 线程）。
4. zlib 流式解压可精确定位 gzip 第一个损坏块（解压死在哪块坏在哪块），但每轮迭代要全量重解压且修复串行慢，只适合损坏点极少时；损坏点多时直接上全量比对。

**Why**：国内到 files.ipd.uw.edu 单连接仅 ~20-30KB/s，需 384 并发 Range 才有 ~5MB/s；下载中途调并发数重启实例时，若旧实例没杀干净（xargs 会持续重生 dl_chunk bash，bash 内 40 次重试循环会重生 curl），两个实例同写一块即触发竞态。kill 顺序必须：主脚本 → xargs → dl_chunk bash → curl（按 PPID 逐层，且注意 xargs 命令行含 "dl_chunk" 字样会被宽泛 grep 误杀）。

**How to apply**：
- 写此类下载器时加**每块锁文件**（`flock chunk_i.lock`）或唯一临时名+原子 rename，从根上防双写者。
- 杀多实例进程树：`ps -eo pid,ppid,cmd` 找父链，先杀主脚本再杀 xargs 再杀 bash worker 最后杀 curl，循环到 0 且稳定。
- 完成判据链：全部块尺寸正确 → merge 后总尺寸正确 → **gzip -t 通过** → 才算下载成功。
- 下载大文件优先用 tarball 单流（本例 16.8GB）而非 70 万小文件直拉；提取到本地盘（overlay）避免网络 FS 小文件 IO。

关联：[[hf-mirror-download-technique]]（12 路并行的老经验，本次扩展到 384 路+损坏修复协议）
