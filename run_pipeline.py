#!/usr/bin/env python
"""一条命令跑完三步管线。

    python run_pipeline.py --input videos

等价于依次执行：

    python clean_video.py --input videos --output output --fps 5
    python format_dataset.py --input output/images --output dataset --fps 5
    python export_lerobot.py --input dataset --output lerobot_dataset

任何一步失败都会立刻停下，已生成的产物保留；修好后可以接着跑。
默认**不覆盖**已有产物：第二步 / 第三步的产物已存在时，需要 --force 或 --skip-existing。

相对路径按**当前目录**解析。子脚本始终从项目目录调用，因此不受当前目录影响。
"""

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from pipeline_common import (
    BLUR_SCORES_FILENAME,
    DEFAULT_REPO_ID,
    IMAGES_META_FILENAME,
    INFO_JSON,
    META_DIRNAME,
)

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Step:
    """一个待执行的步骤及其「已完成」判据。"""

    number: int
    title: str
    cmd: List[str]
    # 全部存在才算该步骤已完成；产物目录存在但缺少标记文件时仍会重跑。
    done_markers: Tuple[Path, ...]
    # 第一步没有 --force：重复运行本来就是合并/覆盖，不需要额外开关。
    forceable: bool

    @property
    def done(self) -> bool:
        return all(path.is_file() for path in self.done_markers)


def resolve_path(raw: str) -> Path:
    """相对路径按调用者的当前目录解析，绝对路径原样返回。"""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def format_command(cmd: Sequence[str]) -> str:
    """给用户看的命令字符串；Windows 与 POSIX 的引号规则不同。"""
    if sys.platform == "win32":
        return subprocess.list2cmdline(list(cmd))
    return shlex.join(cmd)


def build_steps(
    args: argparse.Namespace,
    input_dir: Path,
    output_base: Path,
    dataset_dir: Path,
    lerobot_dir: Path,
) -> List[Step]:
    """把命令行参数翻译成三个子进程命令。"""
    python = sys.executable
    images_dir = output_base / "images"

    clean_cmd = [
        python,
        str(PROJECT_ROOT / "clean_video.py"),
        "--input",
        str(input_dir),
        "--output",
        str(output_base),
        "--fps",
        str(args.fps),
    ]

    format_cmd = [
        python,
        str(PROJECT_ROOT / "format_dataset.py"),
        "--input",
        str(images_dir),
        "--output",
        str(dataset_dir),
        "--fps",
        str(args.fps),
    ]
    if args.blur_threshold is not None:
        format_cmd += ["--blur-threshold", str(args.blur_threshold)]
    if args.max_episode_seconds is not None:
        format_cmd += ["--max-episode-seconds", str(args.max_episode_seconds)]
    if args.link_mode is not None:
        format_cmd += ["--link-mode", args.link_mode]
    if args.camera is not None:
        format_cmd += ["--camera", args.camera]
    if args.force:
        format_cmd.append("--force")

    export_cmd = [
        python,
        str(PROJECT_ROOT / "export_lerobot.py"),
        "--input",
        str(dataset_dir),
        "--output",
        str(lerobot_dir),
        "--repo-id",
        args.repo_id,
    ]
    # 不传 --fps：第三步沿用第二步 info.json 里的值，避免两者不一致。
    if args.camera is not None:
        export_cmd += ["--camera", args.camera]
    if args.task is not None:
        export_cmd += ["--task", args.task]
    if args.vcodec is not None:
        export_cmd += ["--vcodec", args.vcodec]
    if args.force:
        export_cmd.append("--force")

    return [
        Step(
            number=1,
            title="抽帧并记录模糊分数",
            cmd=clean_cmd,
            done_markers=(
                images_dir / IMAGES_META_FILENAME,
                images_dir / BLUR_SCORES_FILENAME,
            ),
            forceable=False,
        ),
        Step(
            number=2,
            title="生成帧清单",
            cmd=format_cmd,
            done_markers=(dataset_dir / META_DIRNAME / INFO_JSON,),
            forceable=True,
        ),
        Step(
            number=3,
            title="导出 LeRobot 数据集",
            cmd=export_cmd,
            done_markers=(lerobot_dir / META_DIRNAME / INFO_JSON,),
            forceable=True,
        ),
    ]


def select_steps(
    steps: Sequence[Step], skip_existing: bool, force: bool
) -> Tuple[List[Step], List[Step], List[Step]]:
    """返回 (要执行的, 要跳过的, 因产物已存在而被挡下的)。"""
    to_run: List[Step] = []
    skipped: List[Step] = []
    blocked: List[Step] = []
    for step in steps:
        if skip_existing and step.done:
            skipped.append(step)
        elif step.done and step.forceable and not force:
            blocked.append(step)
        else:
            to_run.append(step)
    return to_run, skipped, blocked


def run_step(step: Step) -> int:
    print(f"\n[{step.number}/3] {step.title}")
    print(f"      $ {format_command(step.cmd)}", flush=True)
    started = time.monotonic()
    try:
        result = subprocess.run(step.cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n已中断。已生成的产物保留。", file=sys.stderr)
        return 130
    elapsed = time.monotonic() - started

    if result.returncode != 0:
        print(
            f"      失败（退出码 {result.returncode}）。已生成的产物保留。",
            file=sys.stderr,
        )
        return result.returncode if result.returncode > 0 else 1
    print(f"      完成，耗时 {elapsed:.1f}s")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一条命令跑完三步：抽帧 → 帧清单 → LeRobot 数据集。",
        epilog="相对路径按当前目录解析；任何一步失败都会停下，已生成的产物保留。",
    )
    parser.add_argument("--input", "-i", required=True, help="输入视频目录（递归扫描 mp4）")
    parser.add_argument("--fps", type=int, default=5, help="处理帧率，第一步与第二步共用（默认 5）")
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="第一步输出根目录，图片落在 <output>/images（默认 output）",
    )
    parser.add_argument("--dataset", default="dataset", help="第二步产物目录（默认 dataset）")
    parser.add_argument(
        "--lerobot", default="lerobot_dataset", help="第三步产物目录（默认 lerobot_dataset）"
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=None,
        help="模糊标记阈值，只标记不删帧（默认沿用第二步的 100）",
    )
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=None,
        help="episode 时长上限，0 = 不切（默认沿用第二步的 60）",
    )
    parser.add_argument(
        "--link-mode",
        choices=["auto", "hardlink", "symlink", "copy"],
        default=None,
        help="帧图片落盘方式（默认 auto：硬链接 → 软链接 → 复制）",
    )
    parser.add_argument(
        "--camera", default=None, help="相机名，生成 observation.images.<camera>（默认 front）"
    )
    parser.add_argument("--task", default=None, help="每帧任务标注（默认空串，表示没有标注）")
    parser.add_argument(
        "--vcodec", default=None, help="视频编码器（默认 h264；小尺寸素材别用 auto / libsvtav1）"
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID, help=f"数据集标识（默认 {DEFAULT_REPO_ID}）"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已有产物（第三步会整个目录删掉重建）",
    )
    group.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过产物已完成的步骤（不会处理新增素材；有新增视频时别用它）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印将执行的命令，不做任何事"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if sys.version_info < (3, 12):
        print("需要 Python 3.12 或更高（lerobot 0.6 的要求）。", file=sys.stderr)
        return 1

    input_dir = resolve_path(args.input)
    if not input_dir.is_dir():
        print(f"输入目录不存在：{input_dir}", file=sys.stderr)
        return 1

    output_base = resolve_path(args.output)
    dataset_dir = resolve_path(args.dataset)
    lerobot_dir = resolve_path(args.lerobot)

    steps = build_steps(args, input_dir, output_base, dataset_dir, lerobot_dir)
    to_run, skipped, blocked = select_steps(steps, args.skip_existing, args.force)

    print("=== embodied-video-pipeline ===")
    print(f"输入视频：{input_dir}")
    for step in skipped:
        print(f"[{step.number}/3] 跳过（产物已存在）：{step.title}")

    if args.dry_run:
        for step in to_run:
            print(f"[{step.number}/3] 将执行：{format_command(step.cmd)}")
        for step in blocked:
            print(f"[{step.number}/3] 产物已存在，实际运行需 --force：{step.title}")
        print("\n--dry-run：未执行任何步骤。")
        return 0

    if blocked:
        print("\n以下步骤的产物已存在：", file=sys.stderr)
        for step in blocked:
            print(f"  - 第 {step.number} 步：{step.done_markers[0]}", file=sys.stderr)
        print(
            "请选择：加 --force 重建（会覆盖/删除已有产物），"
            "或加 --skip-existing 跳过已完成步骤。",
            file=sys.stderr,
        )
        return 2

    for step in to_run:
        code = run_step(step)
        if code:
            return code

    print("\n=== 完成 ===")
    print(f"帧图片与模糊分数：{output_base / 'images'}")
    print(f"帧清单（中间产物）：{dataset_dir}")
    print(f"LeRobot 数据集：{lerobot_dir}")
    print("加载方式：")
    print(f"  LeRobotDataset({args.repo_id!r}, root={str(lerobot_dir)!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
