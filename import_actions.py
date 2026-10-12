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

# 复用第一步的前缀规则。两处各写一套的话，迟早会漂移成「看起来一样但不一致」——
# 而这里一旦不一致，表现是动作静默配错帧，不报错。
from clean_video import video_output_prefix

ACTIONS_DIRNAME = "actions"
EPISODES_JSONL = "episodes.jsonl"

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


def read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        raise ValueError(f"缺少文件：{path}（请先跑 format_dataset.py）")
    rows: List[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    return rows


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
def find_droid_sources(input_dir: Path) -> Dict[str, Path]:
    """扫出「源视频前缀 → trajectory.h5」。前缀规则必须与 clean_video.py 一致。"""
    sources: Dict[str, Path] = {}
    for video in sorted(input_dir.rglob("*")):
        if not video.is_file() or video.suffix.lower() != ".mp4":
            continue
        action_file = video.parent / "trajectory.h5"
        if action_file.is_file():
            sources[video_output_prefix(video, input_dir)] = action_file
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


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("import_actions")

    input_dir = Path(args.input)
    dataset_dir = Path(args.dataset)
    if not input_dir.is_dir():
        logger.error("素材目录不存在：%s", input_dir)
        return 1

    try:
        episodes = read_jsonl(dataset_dir / "meta" / EPISODES_JSONL)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    # 以帧清单为准：它才是第三步真正会读的源视频集合
    wanted: List[str] = []
    for row in episodes:
        name = str(row.get("source_video", ""))
        if name and name not in wanted:
            wanted.append(name)
    wanted.sort()

    logger.info("帧清单里有 %d 个源视频，正在素材目录里找 %s 格式的动作记录…", len(wanted), args.format)
    sources = SOURCES[args.format](input_dir)

    out_dir = dataset_dir / "meta" / ACTIONS_DIRNAME
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
        np.savez_compressed(dst, action=action, state=state)
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
