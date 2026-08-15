# arm_grasp 兄弟目录快照 — 2026-08-14

来源：`/home/jetson/Desktop/guosai/arm_grasp/`（独立工作区之外的源快照）
捕获时间：2026-08-14 22:28（Jetson 本地时间，最后修改时间戳）

## 这是什么

工作区内的 `国赛/arm_grasp/` 是当前在用版本，2026-08-15 在此基础上做了
红条 ratio 放宽、深度采样窗口扩大、`fixed_depth` 兜底等现场调参。

本分支 `backup/sibling-arm-grasp-2026-08-14` 是**调参之前**的兄弟目录
完整源快照，用于：
- 现场出问题需要"回到 08-14 状态"时直接 checkout
- 对比 08-14 → 08-15 的 diff（已在 commit `04f5704` 落地）
- 防止兄弟目录本身被误删/损坏（云端多一份独立备份）

## 与 main 的关系

| 路径 | main (HEAD `04f5704`) | 本分支 |
|---|---|---|
| `国赛/arm_grasp/` | 最新（08-15 调参后） | 同样最新（分支基线继承） |
| `arm_grasp_sibling_snapshot/` | 不存在 | 兄弟目录 08-14 旧版快照 |

## 排除项

rsync 拷贝时已排除：
- `build/`、`install/`、`log/`（colcon 产物）
- `__pycache__/`、`*.pyc`
- `*.pdf`、`*.jpg`、`*.png`（大体积二进制）
- `.DS_Store`、`.claude/`、`新建 文本文档.txt`

如需恢复整个包：
```bash
rsync -a --exclude='.git' <本分支根>/arm_grasp_sibling_snapshot/ \
  /home/jetson/Desktop/guosai/arm_grasp/
```
