# embodied-video-pipeline

把具身智能采集的原始视频清洗成帧图片，再整理成**仿 LeRobot v2.0 格式**的数据集骨架。

两步流水线，两个脚本各管一段：

```
videos/*.mp4  ──clean_video.py──▶  output/images/*.jpg  ──format_dataset.py──▶  dataset/
                   抽帧 + 过滤模糊帧            带原始帧号的文件名                仿 LeRobot 目录结构
```

## 环境

```bash
pip install -r requirements.txt
```

`opencv-python`、`numpy`、`tqdm`、`pytest`。

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
| `--blur_threshold` | `100.0` | Laplacian 方差低于此值的帧判为模糊、丢弃 |

多进程并行处理所有视频，结束后打印 Benchmark 汇总表。

**产物**：`output/images/{前缀}_frame_{原始帧号:06d}.jpg`，外加一个 `output/images/_meta.json`。

### 输出名前缀怎么定

规则：**相对 `--input` 的路径去掉后缀，路径分隔符换成下划线**。

| 输入 | 前缀 |
|---|---|
| `videos/1.mp4` | `1` |
| `videos/a/1.mp4` | `a_1` |
| `videos/a/b/1.mp4` | `a_b_1` |

视频是**递归**扫描的，所以子目录会带上目录名（第一行可见，平铺时命名和以前完全一样）。

这是为了防止 `videos/a/1.mp4` 和 `videos/b/1.mp4` 写成同一批文件名 —— 它们是在多进程里并发写的，撞名会静默丢帧。

如果加上目录前缀仍然撞车（例如同时存在 `a_1.mp4` 和 `a/1.mp4`，两者都得到 `a_1`），脚本会**直接报错退出**，不会覆盖。

### 为什么文件名里带的是原始帧号，不是连续序号

因为被丢弃的模糊帧在时间轴上留下了**空洞**。`1_frame_000018.jpg` 的 `18` 是它在原视频里的帧号；如果目录里没有 `..._000012.jpg`，说明那一帧被判成模糊丢掉了。

这个信息只在这里出现一次 —— `format_dataset.py` 会把它转存进 `meta/frames.jsonl` 的 `source_frame` 字段，之后文件就按 episode 内位置重命名了。**丢了它，真实时间轴就永久丢失了。**

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

几个设计点：由**父进程**在收齐所有结果后统一写一次（多进程下子进程各写各的会互相覆盖）；先写 `.tmp` 再原子替换，避免崩溃留下半截 JSON；重复运行是**合并**而非覆盖，所以可以分批往同一个输出目录里追加。

---

## 第二步：整理成数据集

```bash
python format_dataset.py --input output/images --output dataset
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` / `-i` | `output/images` | 上一步的图片目录 |
| `--output` / `-o` | `dataset` | 数据集输出目录 |
| `--fps` | `5` | 应与第一步一致 |
| `--camera` | `front` | 生成 `observation.images.<camera>` |
| `--link-mode` | `auto` | `auto`/`hardlink`/`symlink`/`copy` |
| `--chunks-size` | `1000` | 每个 chunk 容纳多少 episode |
| `--force` | 关 | 输出目录已有数据集时允许覆盖 |
| `--dry-run` | 关 | 只扫描出报告，不写任何文件 |

**仅使用 Python 标准库**，不依赖 `lerobot` / `pyarrow` / 任何图像库 —— 连 JPEG 宽高都是自己解析 SOF 标记读出来的。

### 产物结构

一个源视频 = 一个 episode。

```
dataset/
├── data/.gitkeep                     # 占位：上游这里放 parquet，见「已知限制」
├── videos/chunk-000/observation.images.front/
│   ├── episode_000000/frame_000000.jpg ...
│   ├── episode_000001/ ...
│   └── episode_000002/ ...
└── meta/
    ├── info.json          # 对齐上游字段，非标准信息全在 extras 里
    ├── episodes.jsonl     # 一行一个 episode
    ├── frames.jsonl       # 一行一帧（本项目的扩展，上游没有这个文件）
    ├── tasks.jsonl        # 任务标注，无标注时为空文件
    └── README.md          # 自动生成，记录与上游 LeRobot 的逐项差异
```

### 原始帧率的三级来源

`timestamp` 需要知道视频的原始帧率，但第二步未必拿得到。优先级：

| 优先级 | 来源 | 精度 |
|---|---|---|
| 1 | `_meta.json` 里 `clean_video.py` 记下的值 | 精确 |
| 2 | 从帧号间隔反推（`gcd(相邻帧号差) × --fps`） | ±2.5 fps |
| 3 | 名义值 `frame_index / fps` | **时间轴会偏小**，仅最后兜底 |

每一帧用的是哪个来源，都写进了 `info.json` 的 `extras.source_fps_origin_by_video`，生成的 `meta/README.md` 里也有人话说明。

同时会**交叉校验**：`_meta.json` 记的帧率对应的步长，应当等于从帧号反推的步长 —— 不符就告警，因为这通常意味着素材和记录对不上。还会自动检出「各视频原始帧率不统一」的情况。

### 文件落盘方式

默认 `auto`：硬链接 → 软链接 → 复制，依次尝试。

Windows 上**硬链接不需要管理员权限**（软链接需要开发者模式），且不占额外空间 —— `dataset/` 里的图片和 `output/images/` 里的是同一个文件。代价是改一边会影响另一边；需要独立副本就 `--link-mode copy`。重复运行会识别出已存在的链接并跳过，不会重复占空间。

---

## 测试

```bash
pytest            # 92 个用例，约 1 秒
pytest -m slow    # 3 个用例，需要仓库里的真实素材，约 3 秒
```

测试**不依赖** `videos/` 和 `output/`（两者都在 `.gitignore` 里），换台机器克隆下来直接能跑。测试用的图片是三个不到 1KB 的真实 JPEG，见 `tests/fixtures/README.md`。

刻意没测两个 `main()` 的多进程端到端 —— 跑得慢、依赖进程调度、容易假失败，而其中的逻辑已被 `process_video` 和 `write_images_meta` 分别覆盖。

---

## 已知限制

### 1. 这个数据集不能直接训练策略

LeRobot 是**模仿学习**数据集，核心是 `(observation, action)` 配对。本项目产出的只有 observation —— 没有 `action`、`state`、`reward`，`data/` 下也没有 parquet。

它可以用于感知预训练、世界模型、视频预测，但**训练不了策略**，也无法被 `lerobot` 的 loader 直接加载。要走到那一步，下一步是补 `data/*.parquet`，那时绕不开 `pyarrow`。

### 2. `--blur_threshold 100` 没有针对实际素材标定过

Laplacian 方差是**绝对量**，随分辨率、对比度和光照变化。`100` 是 OpenCV 教程里的常见值，通常针对 480p/720p 素材；1080p 以上的清晰帧常在 500–3000 区间，这个阈值可能几乎只拦得住严重糊帧。

判断方法：统计抽样帧的方差分位数，看 `100` 落在分布的什么位置；再把被判为模糊的帧存出来肉眼复核。这个标定尚未做过。

### 3. 重跑 `--force` 不会清理多余的旧文件

如果换了参数导致 episode 数变少，之前生成的 `episode_XXXXXX` 目录会留在原地。

### 4. 硬链接的双向性

见上文「文件落盘方式」。`dataset/` 与 `output/images/` 共享同一份数据。
