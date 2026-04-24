# Git/CI 运维经验

## 2026-03-18: 自定义模型库源码版本控制与 CI 打包

### 问题现象
- 4 个 adaptation 含克隆的第三方仓库源码（LTX-2、MiVOLO、TRELLIS、IBM Materials）
- 嵌套的 `.git/` 目录导致 git 将其识别为 submodule，报 `modified content` 警告
- `.gitignore` 整个排除这些目录后，CI checkout 后目录为空，`package_adaptations.py` 打包的 zip 不含源码
- 用户下载 zip 后无法运行（缺少 editable install 依赖的源码）

### 根本原因
1. **git submodule 识别**：含 `.git/` 子目录的目录会被 git 视为 submodule（mode 160000），即使没有 `.gitmodules` 文件
2. **CI 不初始化 submodule**：`actions/checkout@v4` 不含 `submodules: recursive`，checkout 后 submodule 目录为空
3. **源码修改需要提交**：adapter 对源码做了 NPU 适配修改，这些修改需要纳入版本控制

### 解决方案（三层防护）

**第一层：`.gitignore` 仅排除嵌套 `.git/`**
```
# 之前（错误）：排除整个目录
adaptations/lightricks_ltx_2_3/LTX-2

# 之后（正确）：仅排除嵌套 .git
**/LTX-2/.git
```
源码文件可正常 `git add`，不会被排除。

**第二层：删除嵌套 `.git/` 目录**
```bash
rm -rf adaptations/lightricks_ltx_2_3/LTX-2/.git
```
删除后 git 不再将目录识别为 submodule，消除 `modified content` 警告。

**第三层：CI 兜底 clone**
`deploy-dashboard.yml` 新增步骤：
- 检查源码目录是否存在且非空
- 若缺失 → `git clone --depth 1` + `rm -rf .git`
- 若已存在 → 跳过（说明已在 git 中跟踪）

**第四层：`package_adaptations.py` 跳过 `.git` 目录**
```python
dirs[:] = [d for d in dirs if d != ".git"]
```
防止 CI 克隆残留的 `.git` 目录被打入 zip。

### 操作流程
```
1. Adapter 克隆第三方仓库到 adaptations/{name}/{repo}/
2. 对源码做 NPU 适配修改
3. 删除嵌套 .git/：rm -rf adaptations/{name}/{repo}/.git
4. git add adaptations/{name}/{repo}/  （.gitignore 不再排除）
5. git commit
6. CI checkout 获取源码 → package_adaptations.py 打包含源码到 zip
```

### 已知的自定义模型库
| adaptation | 源码目录 | 仓库 URL | 大小 |
|-----------|---------|---------|------|
| lightricks_ltx_2_3 | LTX-2/ | github.com/Lightricks/LTX-2 | ~11MB |
| iitolstykh_mivolo_v2 | mivolo_src/ | github.com/WildChlamydia/MiVOLO | ~4MB |
| microsoft_trellis_image_large | trellis_src/ | github.com/microsoft/TRELLIS | ~42MB |
| ibm_research_materials_pos_egnn | ibm_materials/ | github.com/IBM/materials | ~8MB (.py/.txt/.md，数据文件被 .gitignore 排除) |

### 注意事项
- **`ibm_materials`** 的 `models/` 子目录含 ~977MB 数据文件（.csv/.npy），已被全局 `*.csv` 和 `*.npy` 规则排除，提交安全
- **`trellis_src`** 的 `.git` 目录已在本地删除（原来就是下载的拷贝，非 git clone）
- 新增自定义模型库时，需在 `.gitignore` 中添加 `**/{dir_name}/.git` 规则，并在 CI `CUSTOM_REPOS` 列表中添加条目
