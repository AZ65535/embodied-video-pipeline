#!/usr/bin/env python
"""把外部的动作记录转成管线能读的「动作侧车」。

不在三步管线里，是给**第三步**准备输入的（排在第二步之后、第三步之前）：

    <原始素材目录> ──import_actions.py──▶ dataset/meta/actions/<源视频>.npz

每个源视频一个文件，文件名就是帧清单里的 `source_video`（即 clean_video.py 算出的
文件名前缀）。npz 里两列，都是 (N, A)：

| 列 | 含义 |
|---|---|
| `action` | 源视频第 i 帧**下发的指令** |
| `state`  | 源视频第 i 帧**实际的状态** |

**第 i 行对应源视频第 i 帧**（不是抽帧后的第 i 帧）—— 所以第三步的 join 直接是
`actions[source_frame]`，不需要任何步长算术。这也是当初把 `source_frame` 保留成正式列
的原因。

为什么单开一个脚本而不是让第三步自己读 h5：外部格式是数据源的「方言」，管线不该认识它。
转成侧车之后，原始素材可以归档掉，第三步只面对一个统一的格式。换数据源只加一个适配函数。

用法：
    python import_actions.py --format droid --input videos --dataset dataset
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 前缀规则与链路清单的文件名都来自公共层。这里不再 import clean_video：
# 那个模块会连带拉起 cv2/tqdm，而本脚本只需要 pathlib 和 numpy。
from pipeline_common import (
    ACTIONS_DIRNAME,
    ACTION_ARRAY,
    EPISODES_JSONL,
    INFO_JSON,
    META_DIRNAME,
    STATE_ARRAY,
    episode_key_for_dir,
    read_jsonl,
    setup_logging,
    video_output_prefix,
)

# DROID 的字段映射。8 维 = 7 关节 + 1 夹爪，对齐官方 lerobot/droid_1.0.1 的
# action / observation.state 定义。
#
# 注意 action 与 state 来自**不同的组**：`action/` 是下发的指令，`action/robot_state/`
# 是实际状态。实测两者平均差约 1.5°（跟踪误差），不是同一个东西，不能混用。
DROID_ACTION_FIELDS = ("action/joint_position", "action/gripper_position")
DROID_STATE_FIELDS = (
    "action/robot_state/joint_positions",
    "action/robot_state/gripper_position",
)


def stack_fields(h5, fields: Tuple[str, ...]) -> np.ndarray:
    """把 h5 里几个字段并成 (N, A)。标量字段（形状 (N,)）补成一列。"""
    columns = []
    for name in fields:
        if name not in h5:
            raise ValueError(f"动作文件里没有字段 {name}；现有的是：{sorted(h5)}")
        data = np.asarray(h5[name][:])
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        columns.append(data)
    return np.concatenate(columns, axis=1)


def load_droid(action_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读出 (action, state)，各自 (N, 8)，float32。"""
    import h5py  # 只有这个数据源需要，放函数里，别拖慢别的格式

    with h5py.File(action_file, "r") as h5:
        action = stack_fields(h5, DROID_ACTION_FIELDS)
        state = stack_fields(h5, DROID_STATE_FIELDS)
    if action.shape != state.shape:
        raise ValueError(
            f"{action_file} 的 action {action.shape} 与 state {state.shape} 行数不一致"
        )
    return action.astype(np.float32), state.astype(np.float32)


# ---- 各数据源的适配器：源视频前缀 → 动作文件 ----
#
# DROID 的目录约定：trajectory.h5 与各相机的 mp4 同目录（都在 episode 目录下）。
# 按**实际文件**配对，而不是拿扁平化的名字去猜，避免 a/b_c 与 a_b/c 这类撞名。
def find_droid_sources(
    input_dir: Path, group_by: str = "file"
) -> Dict[str, Path]:
    """扫出「动作侧车键 → trajectory.h5」。

    file 模式每个视频一个键（与 clean_video 的源视频前缀一致）；dir 模式每个
    含 mp4 的目录一个键（与 format_dataset 的 episode 键一致）。
    """
    sources: Dict[str, Path] = {}
    if group_by == "file":
        for video in sorted(input_dir.rglob("*")):
            if not video.is_file() or video.suffix.lower() != ".mp4":
                continue
            action_file = video.parent / "trajectory.h5"
            if action_file.is_file():
                sources[video_output_prefix(video, input_dir)] = action_file
        return sources

    for action_file in sorted(input_dir.rglob("trajectory.h5")):
        if not action_file.is_file():
            continue
        parent = action_file.parent
        if not any(
            path.is_file() and path.suffix.lower() == ".mp4"
            for path in parent.iterdir()
        ):
            continue
        try:
            relative = parent.relative_to(input_dir)
        except ValueError:
            relative = Path(parent.name)
        key = episode_key_for_dir(relative)
        if key in sources:
            raise ValueError(
                f"目录 {relative} 与 {sources[key].parent} 映射到同一个 episode 键 "
                f"{key!r}；请重命名目录后重试"
            )
        sources[key] = action_file
    return sources


SOURCES = {
    "droid": find_droid_sources,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把外部动作记录转成 dataset/meta/actions/<源视频>.npz，供第三步使用。"
    )
    parser.add_argument(
        "--format",
        default="droid",
        choices=sorted(SOURCES),
        help="动作记录的格式（决定怎么从素材目录里找到它；默认 droid）",
    )
    parser.add_argument(
        "--group-by",
        choices=("file", "dir"),
        default="file",
        help="与 clean_video.py 一致：file=每个视频一集，dir=同目录多相机属于一集（默认 file）",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="原始素材目录，与 clean_video.py 的 --input 同一个",
    )
    parser.add_argument(
        "--dataset", "-d", default="dataset",
        help="第二步的产物目录（读它的帧清单，把 npz 写进它的 meta/actions/；默认 dataset）",
    )
    parser.add_argument(
        "--force", action="store_true", help="已存在的 npz 也重新转换"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("import_actions")

    input_dir = Path(args.input)
    dataset_dir = Path(args.dataset)
    if not input_dir.is_dir():
        logger.error("素材目录不存在：%s", input_dir)
        return 1

    try:
        episodes = read_jsonl(
            dataset_dir / META_DIRNAME / EPISODES_JSONL, missing_hint="请先跑 format_dataset.py"
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    info_path = dataset_dir / META_DIRNAME / INFO_JSON
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("读不了 %s：%s", info_path, exc)
            return 1
        extras = info.get("extras") if isinstance(info, dict) else None
        recorded = extras.get("group_by") if isinstance(extras, dict) else None
        if recorded and recorded != args.group_by:
            logger.error(
                "数据集是按 --group-by %s 生成的，本次却传了 %s；"
                "请改用 --group-by %s",
                recorded,
                args.group_by,
                recorded,
            )
            return 1

    # 以帧清单为准：它才是第三步真正会读的 episode 集合。
    # 新清单用 episode 作权威键；旧清单没有 episode 字段时回退 source_video。
    wanted: List[str] = []
    for row in episodes:
        name = row.get("episode")
        if not isinstance(name, str) or not name:
            name = row.get("source_video")
        if isinstance(name, str) and name and name not in wanted:
            wanted.append(name)
    wanted.sort()

    logger.info(
        "帧清单里有 %d 个 episode，正在素材目录里找 %s 格式的动作记录…",
        len(wanted),
        args.format,
    )
    try:
        sources = SOURCES[args.format](input_dir, args.group_by)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    out_dir = dataset_dir / META_DIRNAME / ACTIONS_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    converted: List[str] = []
    skipped: List[str] = []
    missing: List[str] = []
    failures: List[Tuple[str, str]] = []
    rows = None

    for name in wanted:
        action_file = sources.get(name)
        if action_file is None:
            missing.append(name)
            continue
        dst = out_dir / f"{name}.npz"
        if dst.exists() and not args.force:
            skipped.append(name)
            continue
        try:
            action, state = load_droid(action_file)
        except Exception as exc:  # h5 损坏、字段缺失等等
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        np.savez_compressed(dst, **{ACTION_ARRAY: action, STATE_ARRAY: state})
        converted.append(name)
        if rows is None:
            rows = action.shape[1]

    # ---- 报告 ----
    logger.info("=== 完成 ===")
    logger.info(
        "转换 %d 个，跳过（已存在）%d 个，缺动作记录 %d 个，出错 %d 个",
        len(converted), len(skipped), len(missing), len(failures),
    )
    if rows is not None:
        logger.info("动作维度：%d（action 与 state 同维）", rows)
    for name, why in failures:
        logger.error("  出错 %s：%s", name, why)
    for name in missing:
        logger.error("  缺动作记录：%s", name)
    logger.info("输出目录：%s", out_dir.resolve())

    # 缺动作不在这里报错：第三步的预检会统一拦（那儿才知道哪些帧真的要动作）。
    # 这里只把事实说清楚，让人能提前发现素材与动作对不上。
    if missing or failures:
        logger.warning(
            "有源视频没有动作记录 —— 第三步会因此拒绝生成带动作的数据集（见 README）"
        )
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
