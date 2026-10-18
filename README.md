# embodied-video-pipeline

把具身智能采集的原始视频清洗成帧图片，整理成 **LeRobot v3.0 格式**的数据集。

```
videos/*.mp4 ──①clean_video.py──▶ output/images/*.jpg ──②format_dataset.py──▶ dataset/ ──③export_lerobot.py──▶ lerobot_dataset/
               抽帧 + 记录模糊分数      带原始帧号的文件名         帧清单（中间产物）         组装           LeRobot v3.0 数据集
                                                                      ▲                                       （帧编成 mp4）
                                       动作记录（如 trajectory.h5）───┘
                                               import_actions.py（可选，见下）
```

**动作那一支是可选的**：有动作记录，产物里就带上 `action` / `observation.state`，数据集
才谈得上训练策略；没有，就只产 observation（能训世界模型那一类，见「已知限制 1」）。

面向**数据量大、录制时间长**的素材设计：帧不嵌进 parquet（那有两个会爆的硬限制，见第三步），
长视频自动切成时长有上限的 episode。

| 步骤 | 脚本 | 职责 | 依赖 |
|---|---|---|---|
| 1 | `clean_video.py` | 抽帧、记录每帧的模糊分数 | OpenCV |
| 2 | `format_dataset.py` | 生成帧清单，按阈值标记模糊帧 | 标准库 |
| — | `import_actions.py` | **可选**：把外部动作记录转成侧车，供第三步消费 | numpy（droid 格式另需 h5py） |
| 3 | `export_lerobot.py` | 组装成 LeRobot v3.0 数据集 | lerobot |
| — | `run_pipeline.py` | 一条命令串行执行三步 | 标准库 |
| — | `ui.py` | **可选**：本地网页界面（运行 + 模糊复核） | gradio（`requirements-ui.txt`） |

第二步的产物 `dataset/` 是**中间产物**，供第三步消费。它可读、可查、便于调试，但不是
LeRobot 数据集。`import_actions.py` 排在第二步之后、第三步之前——它只写 `dataset/meta/actions/`，
不碰别的产物。

脚本之间共用的产物文件名、清单字段和纯工具（帧文件名解析、JPEG 头解析、模糊判定、JSON
读写）集中在 `pipeline_common.py`。它**只依赖标准库**，所以第二步、标定工具和动作导入都
不会因为 import 公共层而被动拉起 cv2。

**模糊帧一律保留，只做标记。** 被丢弃的帧会让时间轴出现空洞，而 lerobot 的整套时序逻辑
都假设帧等间隔排列 —— 空洞会让它的 `delta_timestamps` 静默取到错误的时间点。因此过滤是
**下游的事**：数据集里带一列 `blur_score`，用不用、阈值设多少，由消费方决定。
阈值参数在第二步（`--blur-threshold`），第一步不带这个参数。

## 环境

```bash
pip install -r requirements.txt
```

**需要 Python 3.12 或更高**（lerobot 0.6 起的要求；本项目在 Python 3.12 的 conda 环境
`line2` 上开发验证）。第一步需要 `opencv-python`、`numpy`、`tqdm`；第二步只用标准库；
第三步另外需要 `pillow` 和 `lerobot`（含 torch，约 1~2 GB），**只跑前两步可以不装**。
跑测试需要 `pytest`；本地界面另需 `requirements-ui.txt` 里的 Gradio。

第三步装的是 `lerobot[dataset]`——0.6 把 `datasets`/`pyarrow`/`torchcodec` 挪进了这个
可选组，裸装 `lerobot` 写不出数据集。另外 `av` 需要显式安装：lerobot 0.6.1 把它从依赖里
拿掉了，但数据集代码仍然无条件 import 它（上游的打包疏漏，requirements.txt 里已注明）。

**装完还要卸掉一个**：`pip uninstall -y torchcodec`。它在 Windows 上找不到匹配的 FFmpeg
共享库，留着每次 import 都会崩一段 traceback 再退回 pyav；卸掉只剩一行提示，行为完全一样
（我们本来就用 pyav）——它只影响训练时读数据的速度，与写数据集无关。

装 lerobot 时 pip 会用 `opencv-python-headless` 覆盖已有的 `opencv-python`（两者共用
`cv2/` 目录），见第三步的注意事项。

若 pip 配置的镜像不可用，显式指定官方源：

```bash
pip install -r requirements.txt -i https://pypi.org/simple/
```

## 快速开始

装好依赖后，一条命令跑完三步：

```bash
python run_pipeline.py --input videos
```

```
videos/ ──①──▶ output/images/ ──②──▶ dataset/ ──③──▶ lerobot_dataset/
```

任何一步失败都会立刻停下，已生成的产物保留；修好后可以接着跑。常用开关：

| 参数 | 说明 |
|---|---|
| `--skip-existing` | 跳过产物已完成的步骤（不会处理新增素材） |
| `--force` | 允许覆盖已有产物（第三步会整个目录删掉重建） |
| `--dry-run` | 只打印将执行的命令，不执行 |
| `--fps` | 处理帧率，第一步与第二步共用（默认 5） |
| `--output` / `--dataset` / `--lerobot` | 三个产物的目录（默认 output / dataset / lerobot_dataset） |
| `--blur-threshold` / `--max-episode-seconds` / `--link-mode` | 透传给第二步 |
| `--camera` / `--task` / `--vcodec` / `--repo-id` | 透传给第三步（`--camera` 同时用于第二步） |

需要更细的控制（单独跑某一步、标定阈值、接动作数据）时，用下面的步骤脚本和辅助工具。

## 可视化界面（可选）

```bash
python -m pip install -r requirements-ui.txt
python ui.py
```

浏览器打开终端打印的 `http://127.0.0.1:7860`（端口被占用时会顺延）。两个页面：

- **运行管线**：填输入/输出目录，勾选要跑的步骤，运行/停止，实时看日志。第 2 步完成后会自动刷新数据集摘要（episode 数、帧数、阈值、尺寸、每源视频分布、磁盘占用），也可以在跑第 3 步前手动刷新做预检。
- **模糊复核**：读取 `<第 1 步输出根目录>/images/_scores.jsonl`，按分数浏览缩略图，调整阈值实时看各视频的标记比例；确认后可一键用该阈值重跑第 2 步（只改 `is_blurry`，不删帧）。

界面只监听 `127.0.0.1`，不上传数据；Gradio 是可选依赖，不装也不影响三步 CLI 和 `run_pipeline.py`。若 pip 镜像超时，加 `-i https://pypi.org/simple/`。部分 Windows 环境的系统代理会拦截 Gradio 的 localhost 健康检查，`ui.py` 已自动把 `127.0.0.1,localhost` 加入 `NO_PROXY`。

---

## 第一步：抽帧 + 记录模糊分数

```bash
python clean_video.py --input ./videos --output ./output --fps 5
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` / `-i` | 必填 | 视频目录，**递归**扫描 `.mp4`（大小写不敏感） |
| `--output` / `-o` | `output` | 图片落在 `OUTPUT/images/` |
| `--fps` | `5` | 目标采样帧率 |

**这里没有阈值参数。** 第一步是纯机械操作：解码、按目标帧率采样、算每帧的 Laplacian
方差、全部存盘。不做任何取舍判断，因此这一步跑完的产物是不可逆素材的完整保留
（抽帧是最慢的一步）。

多进程并行处理所有视频，结束后打印 Benchmark 汇总表，**报每个视频的分数中位数** ——
中位数不需要阈值就能看出素材有多模糊。

**产物**：`output/images/{前缀}_frame_{原始帧号:06d}.jpg`、`_meta.json`、`_scores.jsonl`。

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

文件名记录的是原始帧号而非连续序号：`1_frame_000018.jpg` 的 `18` 是它在原视频里的帧号。
按 30fps 采样到 5fps 时步长为 6，所以正常产物是 `..._000000`、`..._000006`、`..._000012`
这样连续的（**不跳号**）。

这个信息只在此处出现一次。`format_dataset.py` 会把它转存进 `meta/frames.jsonl` 的
`source_frame` 字段，之后文件按 episode 内位置重命名。丢失它意味着真实时间轴无法还原。
一旦有帧缺号，时间轴就有空洞，`delta_timestamps` 会开始取错帧。

### `_scores.jsonl`

每行一帧的模糊分数，是模糊信息从第一步流向数据集的**唯一通道**（第二步刻意只依赖
标准库，重新算分需要 cv2，而且 JPEG 有损会让分数与 `calibrate_blur.py` 对不上）：

```json
{"video": "1", "source_frame": 0, "blur_score": 185.4321}
{"video": "1", "source_frame": 6, "blur_score": 92.77}
```

分数是原始值，不是「是否模糊」的布尔量 —— 阈值还没标定，判断留给第二步和下游。

### `_meta.json`

记录处理参数和每个视频的原始帧率，供第二步读取：

```json
{
  "format": 2,
  "tool": "clean_video.py",
  "target_fps": 5,
  "videos": {
    "1": {"orig_fps": 30.00005, "frame_step": 6, "sampled": 1000,
          "score_median": 168.7, "elapsed": 56.288}
  }
}
```

三个设计点：由**父进程**在收齐所有结果后统一写入（多进程下子进程各写各的会互相覆盖）；
先写 `.tmp` 再原子替换，避免崩溃留下半截 JSON；重复运行是**合并**而非覆盖，可以分批向
同一输出目录追加。`_scores.jsonl` 用同一套写法，合并粒度是「按视频整段替换」。

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
| `--blur-threshold` | `100` | 低于此分数在清单里标为模糊。**只标记，不删帧** |
| `--max-episode-seconds` | `60` | 单个 episode 的时长上限，超长的视频切成多段；`0` = 不切 |
| `--link-mode` | `auto` | `auto`/`hardlink`/`symlink`/`copy` |
| `--chunks-size` | `1000` | 每个 chunk 容纳多少 episode |
| `--force` | 关 | 输出目录已存在时允许覆盖 |
| `--dry-run` | 关 | 只扫描出报告，不写任何文件 |

**仅使用 Python 标准库**，不依赖 `lerobot` / `pyarrow` / 任何图像库 —— JPEG 宽高由脚本
自己解析 SOF 标记读出，模糊分数则直接读第一步算好的 `_scores.jsonl`，不重新计算。

### 长视频会切成多个 episode

**一个源视频不一定等于一个 episode。** 超过 `--max-episode-seconds` 的视频按固定长度
切成连续小段，每段一个 episode；`source_frame` 不会因切分而重置，所以跨段的绝对位置
不丢。

切分**不是**为了绕开某个技术限制（视频存储下没有单 episode 的大小上限），而是为了
**训练时的样本独立性**：episode 越长，一个 batch 里来自同一段连续视频的样本越多，
时间上越相关，对 SGD 不利。生态里常见 episode 是几秒到几十秒（[LeRobot 的 AL0HA 例子](https://huggingface.co/lerobot/act_aloha_sim_insertion_human/discussions/1)
是 10 秒 / 500 帧），视频预训练管线通常在 5~240 秒之间截断。

`timestamp` 相对**本段起点**计时（不是视频起点），与 lerobot 写的 `frame_index/fps`
一致 —— 否则两者会出现系统性的偏差。

### 阈值只做标记

`--blur-threshold` 决定 `frames.jsonl` 里 `is_blurry` 这一列怎么填，**不改变帧数**。
判据是**严格小于**（恰好等于阈值算清晰）。三处（第一步、第二步、标定工具）共用
`pipeline_common.is_blurry()`，口径只有一份。

换阈值只需重跑这一步（秒级），不用重新解码视频 —— 这正是不把阈值放在第一步的原因。

改完之后汇总行会报「按此阈值有多少帧会被标为模糊」。若上游没有留下 `_scores.jsonl`，
`blur_score` 与 `is_blurry` 记为 `null` 并给出提示，不会让流程失败。

### 产物结构

一个源视频对应一个或多个 episode（见上）。

```
dataset/
├── data/.gitkeep                     # 占位；上游这里放 parquet
├── videos/chunk-000/observation.images.front/
│   ├── episode_000000/frame_000000.jpg ...
│   ├── episode_000001/ ...
│   └── episode_000002/ ...
└── meta/
    ├── info.json          # 对齐上游字段，非标准信息全在 extras 里
    ├── episodes.jsonl     # 一行一个 episode，含 segment_index / segment_count
    ├── frames.jsonl       # 一行一帧，含 blur_score / is_blurry（本项目扩展）
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
| `--vcodec` | `h264` | 视频编码器，见下 |
| `--force` | 关 | 输出目录已存在时**整目录删掉重建** |

`meta/` 下的 parquet 与 `info.json` 全部由 lerobot 自己生成，脚本只负责喂数据。这样格式
正确性由官方代码保证；lerobot 升级格式时跟着升级 API 即可，无需重新逆向。

### 帧编成视频，不嵌进 parquet

相机特征声明为 `dtype: "video"`，帧交给 lerobot 编码成 mp4（按大小轮转，**一个文件
可容纳多个 episode**），parquet 里只放数值列。这是 lerobot 的默认模式，也是生态惯例 ——
[DROID](https://droid-dataset.github.io/droid/the-droid-dataset) 原始数据是 8.7 TB 的 mp4，
[LeRobot v3.0](https://dctx-team.github.io/lerobot-zh/zh/lerobot-dataset-v3/) 的设计目标之一
就是「chunked parquet + compressed videos」。

**为什么不用「图片字节嵌进 parquet」**（`dtype: "image"`）—— 有两个硬限制：单个 arrow
批次超过约 2 GB 会因 int32 偏移溢出而失败；写一个 episode 时，全部帧要同时驻留内存
（1080p × 1000 帧 ≈ 6.2 GB）。视频模式两者都没有，代价是**有损编码**，解码也比读图片慢。

**实测体积**（2674 帧 1080p 示例素材）：视频模式 **220 MB**；同样内容用图片模式会写到
5 GB 以上，并因第 1 条限制失败。约 24 倍。

### `--vcodec` 怎么选

| 编码器 | 编码速度（1080p 实测） | 解码代价 | 备注 |
|---|---|---|---|
| `h264`（默认） | ~8.5 fps | 约为读图片的 **2.2 倍** | 各种尺寸都能用（宽高须为偶数，见下方说明） |
| `libsvtav1` | ~17.9 fps | 约为读图片的 **5 倍** | **16×32 这种小图会让进程直接崩** |
| `auto` | 最快（走硬件编码器） | 视硬件而定 | 硬件编码器有最小尺寸要求，小图会失败 |

编码速度是 1080p、`g=2`、`crf=30` 下的实测值；默认选 `h264`，因为尺寸鲁棒、解码最快、
软件编码到处可用。素材都是大尺寸且机器有硬件编码器时，`--vcodec auto` 能明显加快写入。

> 编码器使用 `yuv420p`，要求宽高都是偶数。普通摄像头 / h264 素材不会踩到；ProRes、
> RGB、4:4:4 等非 4:2:0 素材若带奇数尺寸，先在输入侧补齐：
> `ffmpeg -i 原视频.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" 新视频.mp4`

### 流式编码会丢帧，脚本会检查

`streaming_encoding=True` 时帧直接喂给编码线程（省掉先落 PNG 再编码的一趟），但
**编码队列满时 lerobot 会丢帧，且只打一条 warning**。丢帧意味着视频比 parquet 少几帧、
帧网格重新变得不均匀 —— 正是这个管线要消灭的失败模式。

两重防护：队列深度调到 300 给突发留余量；每个 episode 写完后核对丢帧计数，不为 0 就
**报错退出**（而不是静默产出坏数据集）。

### 加载方式

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/embodied-videos", root="lerobot_dataset")
frame = ds[0]
frame["observation.images.front"]    # (3, H, W) float32，取值 [0, 1]，按需从视频解码
frame["timestamp"]                   # 相对本 episode 起点的时间戳，见下
frame["source_frame"]                # 原始视频里的帧号（跨 episode 唯一）
frame["blur_score"]                  # Laplacian 方差；下游按它自行过滤模糊帧
frame["action"]                      # (A,) 动作；只有接了动作数据才有（见「接动作数据」）
frame["observation.state"]           # (A,) 状态；同上
```

### 数据集结构

```
lerobot_dataset/
├── data/chunk-000/file-000.parquet         # 数值列（timestamp/source_frame/blur_score…）
├── videos/observation.images.front/
│   └── chunk-000/file-000.mp4              # 帧编码成的视频，按大小轮转；一个文件可含多个 episode
└── meta/
    ├── info.json                           # features 定义，codebase_version = v3.0
    ├── stats.json                          # 各特征的统计量
    ├── tasks.parquet                       # 任务表；--task 为空时只有 task_index=0
    └── episodes/chunk-000/file-000.parquet # 每集在 parquet 与 mp4 里的起止位置
```

视频文件按 `video_files_size_in_mb`（lerobot 默认 200 MB）轮转，一个文件可容纳多个
episode；parquet 同理按 `data_files_size_in_mb`（默认 100 MB）轮转。

### 时间轴：由构造保证均匀

lerobot 的 `add_frame()` 把 timestamp 硬编码为 `frame_index / fps`，且它整套时序查询
（`delta_timestamps`）都按帧索引偏移换算。换句话说：**它假定帧等间隔排列**。

本管线让这个假设成立：第一步不丢任何采样帧，帧因此严格落在 `1/fps` 的网格上，
`frame_index / fps` 就是真实时间（**相对本 episode 起点**）。`export_lerobot.py`
不需要做任何补偿。

| 列 | 含义 |
|---|---|
| `timestamp` | `frame_index / fps`，即相对本 episode 起点的秒数 |
| `source_frame` | 原始视频里的帧号；**在同一个源视频内跨 episode 唯一**（切分不会让它重置），但不同视频的帧号会重叠 —— 定位一帧要 `source_video` + `source_frame` |
| `blur_score` | 该帧的 Laplacian 方差，**下游按它自行过滤** |

要还原绝对时间：第二步产物 `dataset/meta/episodes.jsonl` 里每集都有
`first_source_frame`，除以该集的 `source_fps` 再加上 `timestamp` 即可（`source_fps`
未知时只能得到近似值）。

`source_frame` 独立保留，是因为它不只是时间轴的中间量：将来按帧号 join 动作数据，或
重算时间轴（如原始帧率被修正）都需要它。

#### 一个隐含前提：原始帧率是处理帧率的整数倍

步长取 `round(原始帧率 / 目标帧率)`，只有整除时 `步长 / 原始帧率` 才恰好等于 `1/目标帧率`。
例如 30fps / 25fps 素材采样到 5fps 时满足（30/5=6、25/5=5），时间轴精确。换成 24fps
素材采样到 5fps 时，
步长是 5，但 `5/24` 秒 ≠ `1/5` 秒，每帧偏约 8.3 ms 且随帧数累积。

`export_lerobot.py` 会在写入前检查这一点并告警（同时也检查帧号是否连续），**但不中断** ——
数据集仍然有效，只是时间戳有偏差。要精确就换一个能整除的 `--fps`。

#### 帧号有缺口时会告警

若帧清单的 `source_frame` 不均匀（手工裁剪的素材，或不同批次的产物混在一起），
`delta_timestamps` 会取到错误的时间点。这种情况在写入前也会被检查出来并告警。

要修就重跑前两步 —— 第一步会保留全部采样帧，正常产物不会有缺口。

#### `delta_timestamps` 的用法限制

lerobot 把 `delta_timestamps` 的秒数按 `round(d * fps)` 换算成帧索引偏移，而不是按时间戳
查找。只有帧等间隔时，「取前后 0.2 秒」才等价于「取前后一帧」；本管线保留全部采样帧，
所以这个前提成立。

`timestamp` 不能列进 `delta_timestamps`：视频型数据集要用 `item["timestamp"].item()` 定位
视频帧，窗口化后它变成多元素张量，会抛 `RuntimeError`。窗口取位请用 `source_frame` 验证。

**下游必须自己处理模糊帧。** 数据集完整保留了模糊帧（这是时间轴正确的前提），要用就得
按 `blur_score` 过滤：

```python
ds = LeRobotDataset("local/embodied-videos", root="lerobot_dataset")
keep = ds.hf_dataset.filter(lambda row: row["blur_score"] >= 100)
```

### `--force` 是整目录重建

lerobot 的 `create()` 要求输出目录**不存在**，因此 `--force` 的语义是「删掉整个目录重新
生成」，而非「往上覆盖」。重建结果必然是干净的，不存在上一轮的残留文件。（第二步的
`--force` 行为不同，会保留旧文件，见「已知限制」。）

**编码参数。** 脚本只传 `vcodec`，其余用 lerobot 默认值：`g=2`（每 2 帧一个关键帧）、
`crf=30`。短 GOP 是为随机访问优化的，压缩率会低一些。要调就改 `write_dataset()` 里的
`RGBEncoderConfig(...)`。

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

## 接动作数据（可选）

```bash
python import_actions.py --format droid --input videos --dataset dataset
```

排位在**第二步之后、第三步之前**（不在三步管线里，但第三步依赖它的产物）。它回答一个问题：
这份数据集能不能训练策略。只有 observation 的话只能训世界模型、视频预测那一类（见「已知限制 1」）。

### 为什么单开一个脚本

动作记录是**另一个来源**（DROID 是 `trajectory.h5`，别处可能是 CSV / npz / JSON），这些格式是
数据源的「方言」——管线不该认识它们。所以先用一个适配器把方言翻成统一格式：

```
<素材目录> ──import_actions.py──▶ dataset/meta/actions/<源视频>.npz
   （各方言）                            action (N, A) / state (N, A)
```

好处是**转完就能把原始素材归档掉**：第三步只面对侧车，不会在导出时又回头去读几百 GB 的原始
记录。换数据源也只是加一个适配函数（`SOURCES` 字典 + 一个 `find_*_sources`）。

### 侧车格式

每个源视频一个 `.npz`，文件名就是帧清单里的 `source_video`（即第一步算出的文件名前缀）。
里面两列：

| 数组 | 含义 |
|---|---|
| `action` | 该帧**下发的指令** |
| `state` | 该帧**实际的状态** |

**第 i 行对应源视频第 i 帧**，不是抽帧后的第 i 帧——所以第三步的 join 直接是
`actions[source_frame]`，不做任何步长算术。这正是当初把 `source_frame` 保留成正式列的原因。

### DROID 格式取哪几列

8 维 = 7 关节 + 1 夹爪，与官方 `lerobot/droid_1.0.1` 的 `action` / `observation.state` 定义一致：

| 数据集列 | h5 来源 |
|---|---|
| `action` | `action/joint_position` (7) ⊕ `action/gripper_position` (1) |
| `observation.state` | `action/robot_state/joint_positions` (7) ⊕ `action/robot_state/gripper_position` (1) |

**两者不能混用**：`action/` 是下发的指令，`action/robot_state/` 是实际状态，实测两者平均差
约 1.5°（跟踪误差）。用错了不会报错，只会让模型学到「命令和结果是一回事」。

### 动作必须齐备

一旦侧车里有了某个源视频，第三步就要求**它的每一帧都能取到动作**——缺了在落盘前报错，
不填默认值。理由和 `blur_score` 一样：静默填 0 等于告诉模型「此刻机器人什么也没做」，而且
不会报错。

只给一部分源视频动作是不行的（lerobot 要求声明的特征每帧都有）。要么补齐，要么把那些素材
从 `videos/` 里排除。

---

## 辅助工具：模糊阈值标定

`--blur-threshold` 的值有两种确定方式，读取的都是第一步的 `_scores.jsonl`：

- **可视化界面（推荐）**：在「模糊复核」页调整阈值，实时看各视频的标记比例和缩略图；
  确认后直接重跑第二步。
- **命令行工具**：`calibrate_blur.py`，适合导出复核帧离线查看，或在没有 Gradio 的
  环境里使用。

命令行工具的用法：

```bash
python calibrate_blur.py --input output/images --export-dir ./review
```

**它读的是第一步的产物，不解码视频。** 每帧的分数已经存在 `_scores.jsonl` 里，这里只做
统计；通常零点几秒就能跑完。标定与第二步读的是同一份分数，口径一致。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` / `-i` | `output/images` | clean_video.py 的产物目录（含 `_scores.jsonl`） |
| `--threshold` | `100` | 可重复传入以对比 |
| `--export-dir` | 无 | 导出复核帧 |
| `--export-band` | `3` | 临界帧上界：导出 `[阈值, 阈值×band]` |
| `--export-max` | `150` | 每视频每类最多导出多少帧；0 = 不截取 |

复核帧是从 `output/images/` **拷贝**的，不是重新解码出来的。

### 复核帧怎么用

```
review/
├── below/      # 分数 < 阈值，被判为模糊的
├── critical/   # 分数在 [阈值, 阈值×band]，刚过线的
└── scores.jsonl
```

文件名以分数开头（`var0000183.42_1_frame000123.jpg`），所以**按名称排序就是按质量排序**。`below/` 末尾与 `critical/` 开头相接，那里就是阈值所在的位置：看有没有该留的被丢了（误杀）、该丢的被留了（漏网）。

### 不自动推荐阈值

Laplacian 方差分布通常不是双峰的，所以工具不自动推荐阈值，只给出分布和复核帧，
由使用人自己判断。低对比或大面积平滑的画面方差天然偏低，肉眼看未必模糊 —— 复核帧
正是用来区分这种情况的。

---

## 测试

```bash
pytest            # 226 个用例，约 5 秒
pytest -m slow    # 8 个用例，含真实调用 lerobot，约 10 秒
```

第三步的测试分两层：不碰 lerobot 的纯逻辑（读清单、组装特征、校验帧文件）在默认套件中，
保持秒级；真正调用 lerobot 写入再读回的用例带 `slow` 标记，用于锁住 `finalize()` 这一
调用 —— 漏掉它 parquet 不会写 footer，文件看似存在但无法加载。

**默认测试**不依赖 `videos/` 和 `output/`（两者都在 `.gitignore` 中），换台机器克隆后
可直接运行；`-m slow` 的用例依赖真实素材，缺失时会自动跳过。测试图片是三个不到 1KB
的真实 JPEG，见 `tests/fixtures/README.md`。

`clean_video.py` / `format_dataset.py` 的 `main()` 多进程端到端刻意未测 —— 运行慢、
依赖进程调度、容易假失败，而其中逻辑已被 `process_video`、`write_images_meta` 和
`write_blur_scores` 分别覆盖。

---

## 已知限制

### 1. 动作数据得自己提供

LeRobot 是**模仿学习**数据集，核心是 `(observation, action)` 配对。本项目**采集**的只有
observation —— 动作推不出来，必须有另外的来源（机器人日志、遥操作记录等）。

所以分两种情况：

- **有动作记录**（走 `import_actions.py`，见「接动作数据」）→ 产物里带 `action` 和
  `observation.state`，**能训练策略**（ACT / diffusion policy / SmolVLA 之类）。
- **没有** → 只有 observation，能训感知预训练、世界模型、视频预测那一类，训不了策略。

`import_actions.py` 目前只带一个 DROID 适配器。换数据源要自己加一个 `find_*_sources`
（从素材目录找到动作文件）和一个 `load_*`（读出来并并成 `(N, 8)`）——转换本身是格式适配，
不涉及管线逻辑。

### 2. 重跑 `--force` 不会清理多余的旧文件

第二步的 `--force` 会保留旧文件：若改了 `--max-episode-seconds` 导致 episode 数变少，
上次运行生成的 `episode_XXXXXX` 目录会残留在原处；帧数变多时也可能留下位置错位的旧文件
（帧文件名按 episode 内位置编号）。**改了切分长度就先把 `dataset/` 整个删掉再跑。**

第三步的 `--force` 不同，它是整目录 `rmtree` 重建，不会留残渣。

### 3. 只支持单相机

一个 mp4 对应一个 episode。同一集的多路视角（如 DROID 的外部 + 腕部相机）会被拆成多个
独立 episode，语义上是错的。当前管线尚未实现多相机支持。
