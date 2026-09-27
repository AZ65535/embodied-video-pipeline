# embodied-video-pipeline

把具身智能采集的原始视频清洗成帧图片，整理成 **LeRobot v3.0 格式**的数据集。

```
videos/*.mp4 ──clean_video.py──▶ output/images/*.jpg ──format_dataset.py──▶ dataset/ ──export_lerobot.py──▶ lerobot_dataset/
               抽帧 + 过滤模糊帧        带原始帧号的文件名         帧清单（中间产物）        LeRobot v3.0 数据集
```

| 步骤 | 脚本 | 职责 | 依赖 |
|---|---|---|---|
| 1 | `clean_video.py` | 抽帧、过滤模糊帧 | OpenCV |
| 2 | `format_dataset.py` | 生成帧清单 | 标准库 |
| 3 | `export_lerobot.py` | 组装成 LeRobot v3.0 数据集 | lerobot |

第二步的产物 `dataset/` 是**中间产物**，供第三步消费。它可读、可查、便于调试，但不是
LeRobot 数据集。

## 环境

```bash
pip install -r requirements.txt
```

前两步需要 `opencv-python`、`numpy`、`tqdm`、`pytest`；第三步另外需要 `pillow` 和
`lerobot`（含 torch，约 1~2 GB），**只跑前两步可以不装**。

装 lerobot 时 pip 会用 `opencv-python-headless` 覆盖已有的 `opencv-python`（两者共用
`cv2/` 目录），见第三步的注意事项。

若 pip 配置的镜像不可用，显式指定官方源：

```bash
pip install -r requirements.txt -i https://pypi.org/simple/
```

---

## 第一步：抽帧 + 过滤模糊帧

```bash
python clean_video.py --input ./videos --output ./output --fps 5
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` / `-i` | 必填 | 视频目录，**递归**扫描 `.mp4`（大小写不敏感） |
| `--output` / `-o` | `output` | 图片落在 `OUTPUT/images/` |
| `--fps` | `5` | 目标采样帧率 |
| `--blur_threshold` | `100.0` | Laplacian 方差低于此值的帧判为模糊，丢弃 |

多进程并行处理所有视频，结束后打印 Benchmark 汇总表。

**产物**：`output/images/{前缀}_frame_{原始帧号:06d}.jpg` 与 `output/images/_meta.json`。

### 输出名前缀

规则：**相对 `--input` 的路径去掉后缀，路径分隔符换成下划线**。

| 输入 | 前缀 |
|---|---|
| `videos/1.mp4` | `1` |
| `videos/a/1.mp4` | `a_1` |
| `videos/a/b/1.mp4` | `a_b_1` |

因为视频是递归扫描的，子目录名会进入前缀。这样做是为了避免 `videos/a/1.mp4` 与
`videos/b/1.mp4` 写成同一批文件名 —— 多进程并发写入时撞名会静默丢帧。

若加上目录前缀后仍然撞车（如同时存在 `a_1.mp4` 和 `a/1.mp4`），脚本直接报错退出，
不会覆盖。

### 文件名中的原始帧号

被丢弃的模糊帧在时间轴上留下**空洞**，因此文件名记录的是原始帧号而非连续序号：
`1_frame_000018.jpg` 的 `18` 是它在原视频里的帧号；若目录中没有 `..._000012.jpg`，
说明那一帧被判为模糊丢掉了。

这个信息只在此处出现一次。`format_dataset.py` 会把它转存进 `meta/frames.jsonl` 的
`source_frame` 字段，之后文件按 episode 内位置重命名。丢失它意味着真实时间轴无法还原。

### `_meta.json`

记录本次的处理参数和每个视频的原始帧率，供第二步读取：

```json
{
  "target_fps": 5,
  "blur_threshold": 100.0,
  "videos": {
    "1": {"orig_fps": 30.00005, "frame_step": 6, "sampled": 1000, "saved": 818,
          "skipped_blurry": 182, "elapsed": 56.288}
  }
}
```

三个设计点：由**父进程**在收齐所有结果后统一写入（多进程下子进程各写各的会互相覆盖）；
先写 `.tmp` 再原子替换，避免崩溃留下半截 JSON；重复运行是**合并**而非覆盖，可以分批向
同一输出目录追加。

---

## 第二步：生成帧清单

```bash
python format_dataset.py --input output/images --output dataset
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` / `-i` | `output/images` | 上一步的图片目录 |
| `--output` / `-o` | `dataset` | 输出目录 |
| `--fps` | `5` | 应与第一步一致 |
| `--camera` | `front` | 生成 `observation.images.<camera>` |
| `--link-mode` | `auto` | `auto`/`hardlink`/`symlink`/`copy` |
| `--chunks-size` | `1000` | 每个 chunk 容纳多少 episode |
| `--force` | 关 | 输出目录已存在时允许覆盖 |
| `--dry-run` | 关 | 只扫描出报告，不写任何文件 |

**仅使用 Python 标准库**，不依赖 `lerobot` / `pyarrow` / 任何图像库 —— JPEG 宽高由脚本
自己解析 SOF 标记读出。

### 产物结构

一个源视频对应一个 episode。

```
dataset/
├── data/.gitkeep                     # 占位；上游这里放 parquet
├── videos/chunk-000/observation.images.front/
│   ├── episode_000000/frame_000000.jpg ...
│   ├── episode_000001/ ...
│   └── episode_000002/ ...
└── meta/
    ├── info.json          # 对齐上游字段，非标准信息全在 extras 里
    ├── episodes.jsonl     # 一行一个 episode
    ├── frames.jsonl       # 一行一帧（本项目扩展，上游没有此文件）
    ├── tasks.jsonl        # 任务标注，无标注时为空文件
    └── README.md          # 自动生成，记录与上游 LeRobot 的逐项差异
```

`info.json` 里的 `codebase_version` 是 `v2.0`，但**这不是最终交付物** —— 真正的
LeRobot 数据集由第三步生成，格式为 v3.0。这里保留 LeRobot 风格的目录只是为了让中间
产物易读、可对照。

### 原始帧率的三级来源

计算 `timestamp` 需要视频的原始帧率，但第二步未必拿得到。优先级：

| 优先级 | 来源 | 精度 |
|---|---|---|
| 1 | `_meta.json` 中 `clean_video.py` 记录的值 | 精确 |
| 2 | 从帧号间隔反推（`gcd(相邻帧号差) × --fps`） | ±2.5 fps |
| 3 | 名义值 `frame_index / fps` | **时间轴偏小**，仅最后兜底 |

每一帧实际使用了哪个来源，都写进了 `info.json` 的
`extras.source_fps_origin_by_video`，生成的 `meta/README.md` 中也有对应说明。

脚本同时做**交叉校验**：`_meta.json` 记录的帧率对应的步长应当等于从帧号反推的步长，
不符则告警（通常意味着素材与记录对不上）；各视频原始帧率不统一时也会主动检出。

### 文件落盘方式

默认 `auto`：硬链接 → 软链接 → 复制，依次尝试。

Windows 上**硬链接不需要管理员权限**（软链接需要开发者模式），且不占额外空间 ——
`dataset/` 里的图片与 `output/images/` 里的是同一个文件。代价是修改任一边都会影响另一边，
需要独立副本请用 `--link-mode copy`。重复运行会识别已存在的链接并跳过，不会重复占空间。

---

## 第三步：组装成 LeRobot v3.0 数据集

```bash
python export_lerobot.py --input dataset --output lerobot_dataset
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` / `-i` | `dataset` | 第二步的产物目录 |
| `--output` / `-o` | `lerobot_dataset` | 数据集输出目录 |
| `--repo-id` | `local/embodied-videos` | 数据集标识（lerobot 必填，本质是 HF Hub 路径格式） |
| `--camera` | `front` | 相机名，生成 `observation.images.<camera>` |
| `--fps` | 取 `info.json` 的值 | 处理帧率 |
| `--task` | 空串 | 每帧任务标注；无标注则留空 |
| `--force` | 关 | 输出目录已存在时**整目录删掉重建** |

`meta/` 下的 parquet 与 `info.json` 全部由 lerobot 自己生成，脚本只负责喂数据。这样格式
正确性由官方代码保证；lerobot 升级格式时跟着升级 API 即可，无需重新逆向。

### 加载方式

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/embodied-videos", root="lerobot_dataset")
frame = ds[0]
frame["observation.images.front"]    # (3, H, W) float32，取值 [0, 1]
frame["timestamp"]                   # 真实时间轴上的时间戳，见下
frame["source_frame"]                # 原始视频里的帧号
```

### 数据集结构

```
lerobot_dataset/
├── data/chunk-000/file-000.parquet         # 帧数据，图片字节直接嵌在里面
└── meta/
    ├── info.json                           # features 定义，codebase_version = v3.0
    ├── stats.json                          # 各特征的统计量
    ├── tasks.parquet                       # 任务标注；--task 为空时是一行空任务
    └── episodes/chunk-000/file-000.parquet # 每集的起止与分片位置
```

### 时间轴：`timestamp` 需要手工注入

`clean_video.py` 丢弃了模糊帧，时间轴上有空洞，因此真实时间戳应当是
`source_frame / 原始帧率`，而不是按保留帧数顺序数出来的 `frame_index / fps`。

而 lerobot 的 `add_frame()` 把 timestamp 硬编码为 `frame_index / fps`。它本意支持调用方
自己传入 —— 函数体中有这样一行：

```python
timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
```

但这行永远取不到：同一函数里字段校验跑在它前面，会把 `timestamp` 当作多余字段直接报错。
这是 lerobot 0.4.4 的 bug（`datasets/lerobot_dataset.py` 第 1182 与 1189 行的顺序问题）。

因此 `export_lerobot.py` 绕开校验，在 `add_frame()` 之后改写缓冲区中刚追加的那一项
（`inject_real_timestamp()`）。parquet 仍由 lerobot 自己的代码写出，`stats.json` 也按正确
值计算，无需事后修改文件。

偏差的实际量级：三个视频分别为 **36.4 / 20.0 / 14.8 秒**，各占其时长的约 18%。若按
lerobot 默认值走，视频 1 会显示为 163.6 秒，而实际为 199.8 秒。

注入失败时脚本直接报错退出，不会静默退回标称值 —— 否则会产出时间轴错误但表面正常的
数据集。

| 列 | 含义 |
|---|---|
| `timestamp` | `source_frame / 原始帧率`，**真实时间轴**（含模糊帧造成的空洞） |
| `source_frame` | 原始视频里的帧号，**不连续**；跳过的号即被判为模糊的帧 |

`source_frame` 独立保留，是因为它不只是时间轴的中间量：将来按帧号 join 动作数据，或
重算时间轴（如原始帧率被修正）都需要它。

**注意**：`timestamp` 修好了，但 lerobot 的 `delta_timestamps` 在本数据集上仍会取到错误
的帧（它按帧索引偏移而非时间戳查找）。需要时序窗口请自行按 `source_frame` 计算，详见
「已知限制」第 2 条。

### `--force` 是整目录重建

lerobot 的 `create()` 要求输出目录**不存在**，因此 `--force` 的语义是「删掉整个目录重新
生成」，而非「往上覆盖」。重建结果必然是干净的，不存在上一轮的残留文件。（第二步的
`--force` 行为不同，会保留旧文件，见「已知限制」。）

### 图片以 PNG 无损存储，体积会增大

lerobot 的存图路径硬编码为 `.png`，JPEG 会被解码为像素后无损编码成 PNG。好处是**零画质
损失**（实测逐像素一致），代价是 1080p 照片 PNG 压缩率很低，**落盘体积约为原始 JPEG 的
3~4 倍** —— 2318 帧从 1.2 GB 增至约 4.4 GB。

### 注意事项

**安装 lerobot 会覆盖已有的 cv2。** lerobot 依赖 `opencv-python-headless`，它与
`opencv-python` 共用同一个 `cv2/` 目录，安装后 cv2 会被替换为 headless 版（无 GUI 功能）。
实测前两步不受影响（只用到 `VideoCapture`/`cvtColor`/`Laplacian`/`imwrite`），但若依赖
GUI，建议装进独立 venv，或事后恢复：

```bash
pip uninstall -y opencv-python-headless
pip install --force-reinstall --no-deps opencv-python
```

**一个无害的 DeprecationWarning。** 写入时会看到
`Conversion of an array with ndim > 0 to a scalar is deprecated`。原因是自定义标量特征经
`np.stack` 后变为 `(N,1)`，`datasets` 库靠 `int(array)` 兜住。无法回避（lerobot 要求
`shape` 精确等于 `(1,)`），目前不影响结果，但将来 numpy / datasets 收紧后可能变为错误。

---

## 辅助工具：模糊阈值标定

```bash
python calibrate_blur.py --input ./videos --export-dir ./review
```

不在三步管线里，回答一个问题：`--blur_threshold` 该设多少。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` / `-i` | 必填 | 视频目录，递归 |
| `--fps` | `5` | 应与 clean_video 一致 |
| `--sample` | `0` | 每视频最多分析多少帧；0 = 全量 |
| `--threshold` | `100` | 可重复传入以对比 |
| `--export-dir` | 无 | 导出复核帧 |
| `--export-band` | `3` | 临界帧上界：导出 `[阈值, 阈值×band]` |
| `--export-max` | `150` | 每视频每类最多导出多少帧；0 = 不截取 |
| `--workers` | CPU 核数 | 并行进程数 |

分数由 `clean_video.blur_score()` 计算，采样帧号也与 clean_video 一致 —— 两者不一致的话，统计出的分布就对不上实际过滤行为。

### 复核帧怎么用

```
review/
├── below/      # 分数 < 阈值，被判为模糊的
├── critical/   # 分数在 [阈值, 阈值×band]，刚过线的
└── scores.jsonl
```

文件名以分数开头（`var0000183.42_1_frame000123.jpg`），所以**按名称排序就是按质量排序**。`below/` 末尾与 `critical/` 开头相接，那里就是阈值所在的位置：看有没有该留的被丢了（误杀）、该丢的被留了（漏网）。

### 不自动推荐阈值

Laplacian 方差分布通常不是双峰的，任何自动选阈值的做法都是假装有依据。工具只给事实，线画在哪由人决定 —— 这也正是要导出复核帧的原因。

### 实测：仓库里这三个视频的问题

| | 视频 1 | 视频 2 | 视频 3 |
|---|---|---|---|
| 方差中位数 | 168.7 | 407.4 | 633.9 |
| 阈值 100 的丢弃率 | 18.2% | 11.2% | 9.5% |
| 阈值 300 的丢弃率 | 85.1% | 36.2% | 23.2% |

两点结论：

1. **各视频中位数相差 3.8 倍**，同一个绝对阈值对它们的严格程度完全不同 —— 「统一阈值」实际上意味着各视频的过滤标准不一致。工具在相差 2 倍以上时会主动告警。
2. **阈值 300 对视频 1 是灾难性的**（丢 85.1%），对视频 3 却只丢 23.2%。所以「调高一点更保险」这个直觉在这批素材上不成立。

复核帧里还能看到：部分被判为模糊的帧肉眼看是清晰的 —— 低对比或大面积平滑的画面，Laplacian 方差天然就低，与是否模糊无关。

---

## 测试

```bash
pytest            # 154 个用例，约 2 秒
pytest -m slow    # 5 个用例，含真实调用 lerobot，约 10 秒
```

第三步的测试分两层：不碰 lerobot 的纯逻辑（读清单、组装特征、校验帧文件）在默认套件中，
保持秒级；真正调用 lerobot 写入再读回的用例带 `slow` 标记，用于锁住 `finalize()` 这一
调用 —— 漏掉它 parquet 不会写 footer，文件看似存在但无法加载。

测试**不依赖** `videos/` 和 `output/`（两者都在 `.gitignore` 中），换台机器克隆后可直接
运行。测试图片是三个不到 1KB 的真实 JPEG，见 `tests/fixtures/README.md`。

两个 `main()` 的多进程端到端刻意未测 —— 运行慢、依赖进程调度、容易假失败，而其中逻辑已
被 `process_video` 和 `write_images_meta` 分别覆盖。

---

## 已知限制

### 1. 不能直接训练策略

LeRobot 是**模仿学习**数据集，核心是 `(observation, action)` 配对，而本项目产出的只有
observation —— 没有 `action`、`state`、`reward`。

数据集**可以被 `lerobot` 的 loader 加载**（第三步做到的），可用于感知预训练、世界模型、
视频预测，但**训练不了策略**。这不是软件层面的问题：动作数据无法从视频推出，需要另外的
来源（机器人日志、遥操作记录等）。

接入点已经留好：`build_features()` 是唯一描述数据集有哪些特征的地方，帧的组装也集中在
一处。将来有动作数据时按 `source_frame` 对齐 join 即可 —— 这也正是 `source_frame` 被
作为正式列写入数据集的原因。

### 2. 不能用 `delta_timestamps` 取时序窗口

lerobot 把 `delta_timestamps` 的秒数按 `round(d * fps)` 换算成**帧索引偏移**，而不是按
时间戳查找：

```python
delta_indices[key] = [round(d * fps) for d in delta_ts]     # utils.py:890
```

于是「取当前帧前后 0.2 秒」实际变成「取文件中前后一帧」。本数据集的帧间隔因模糊帧被丢弃
而不均匀（0.2 / 0.4 / 0.6 秒不等），因此**取到的帧不在请求的时间点上，且不会报错**。

实测（`fps=5`，请求前后各 0.2 秒，换算为帧偏移 `[-1, 0, +1]`）：

| 查询帧 | 该帧时间戳 | 「0.2 秒前」应为 | 实际取到的时间戳 |
|---|---|---|---|
| 第 10 帧（均匀区） | 2.0 | 1.8 | 1.8 / 2.0 / 2.2 ✅ |
| 第 51 帧（有空洞） | 13.6 | 13.4 | **10.0** / 13.6 / 13.8 ❌ |

第 51 帧的「0.2 秒前」实际是 3.6 秒前。

需要时序窗口时，请自行按 `source_frame` 计算帧号，不要用 `delta_timestamps`。

这不是可以简单修掉的缺陷：lerobot 整套时序查询都假设帧位于 `1/fps` 的均匀网格上，而
「丢弃模糊帧」这个设计恰好破坏了这个前提。要根治，需要改成保留模糊帧并加标记位，让下游
自行决定是否丢弃 —— 那是对第一步的重新设计。

### 3. `--blur_threshold 100` 的标定尚无定论

Laplacian 方差是**绝对量**，随分辨率、对比度和光照变化。`100` 是 OpenCV 教程中的常见值，
通常针对 480p/720p 素材。

标定工具已经就绪（见「辅助工具」一节，含实测数据），**但阈值最终改成多少还没定** ——
复核帧需要人工翻过才能决定。在那之前 `clean_video.py` 的默认值仍是 `100`。

### 4. 重跑 `--force` 不会清理多余的旧文件

若更换参数导致 episode 数变少，之前生成的 `episode_XXXXXX` 目录会残留在原处。

### 5. 硬链接的双向性

见「文件落盘方式」。`dataset/` 与 `output/images/` 共享同一份数据。
