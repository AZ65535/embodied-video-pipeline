"""共享的测试夹具。

两条硬约束：

1. **不依赖 `videos/` 和 `output/`** —— 两者都在 `.gitignore` 里，换台机器克隆下来就没有。
   真实素材只在 `test_real_material.py` 里以 `skip` 的形式覆盖。
2. **所有产物写在 `tmp_path` 下** —— 测试不碰仓库里的任何目录。

这里刻意不 import `clean_video`：它会连带拉起 cv2 / tqdm，
而 `format_dataset` 是纯标准库的，不希望它的测试被拖下水。
需要 `clean_video` 的夹具放在 `test_clean_video.py` 里。
"""

import itertools
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import format_dataset  # noqa: E402  （必须在调整 sys.path 之后导入）


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_jpegs(fixtures_dir: Path) -> list:
    """几个极小的真实 JPEG，尺寸各不相同。"""
    jpegs = sorted(fixtures_dir.glob("tiny_*.jpg"))
    assert jpegs, "tests/fixtures 下缺少 tiny_*.jpg"
    return jpegs


@pytest.fixture
def make_images(tmp_path: Path, tiny_jpegs: list):
    """造一个假的 output/images 目录。

    spec 形如 ``{"1": [0, 6, 12], "2": [0, 5]}``，值是该视频**保留下来的原始帧号**
    —— 中间跳过的号就是被 clean_video.py 判定为模糊而丢弃的帧。

    extra_files 用来塞入干扰文件，值是 bytes 或 str；mixed_sizes=True 时让图片尺寸交错，
    用于测试 info.json 的「多尺寸」告警路径。
    """
    counter = itertools.count()

    def _make(spec: dict, extra_files: dict = None, mixed_sizes: bool = False) -> Path:
        root = tmp_path / f"images{next(counter)}"
        root.mkdir()
        for video, frame_numbers in spec.items():
            for index, frame_no in enumerate(frame_numbers):
                source = tiny_jpegs[index % len(tiny_jpegs)] if mixed_sizes else tiny_jpegs[0]
                shutil.copy2(source, root / f"{video}_frame_{frame_no:06d}.jpg")
        for name, content in (extra_files or {}).items():
            data = content if isinstance(content, bytes) else content.encode("utf-8")
            (root / name).write_bytes(data)
        return root

    return _make


@pytest.fixture
def write_meta():
    """直接手写 `_meta.json`。

    不经过 `clean_video.write_images_meta`，这样 format_dataset 的读取逻辑
    可以独立测试；两者的衔接另有集成测试覆盖。
    """

    def _write(images_dir: Path, fps_by_video: dict, target_fps: int = 5) -> Path:
        payload = {
            "format": 1,
            "tool": "clean_video.py",
            "target_fps": target_fps,
            "blur_threshold": 100.0,
            "videos": {name: {"orig_fps": fps} for name, fps in fps_by_video.items()},
        }
        path = Path(images_dir) / format_dataset.IMAGES_META_FILENAME
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def run_format(monkeypatch):
    """按命令行方式调用 format_dataset.main()，返回退出码。"""

    def _run(*args) -> int:
        monkeypatch.setattr(sys, "argv", ["format_dataset.py", *[str(a) for a in args]])
        return format_dataset.main()

    return _run


@pytest.fixture
def read_jsonl():
    """把 jsonl 读成 list[dict]。"""

    def _read(path: Path) -> list:
        text = Path(path).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    return _read
