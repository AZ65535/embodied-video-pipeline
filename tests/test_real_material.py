"""依赖仓库里真实素材的慢速测试。

`videos/` 和 `output/` 都在 `.gitignore` 里，所以这些用例在别人的机器上会自动跳过 ——
它们的价值是在**你自己的机器上**验证真实数据，而不是进 CI。

默认不运行（`pytest.ini` 里 `addopts = -m "not slow"`），要跑：

    pytest -m slow
"""

from pathlib import Path

import pytest

import format_dataset as fd

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = REPO_ROOT / "videos"
IMAGES_DIR = REPO_ROOT / "output" / "images"

pytestmark = pytest.mark.slow


def frame_entries(video: str) -> list:
    """按原始帧号排序，构造 FrameEntry 序列。"""
    entries = []
    for path in IMAGES_DIR.glob(f"{video}_frame_*.jpg"):
        frame_no = int(path.stem.rsplit("_", 1)[-1])
        entries.append(fd.FrameEntry(src=path, source_frame=frame_no, width=0, height=0))
    return sorted(entries, key=lambda entry: entry.source_frame)


@pytest.mark.skipif(not IMAGES_DIR.is_dir(), reason="output/images 不存在（该目录在 .gitignore 里）")
def test_recorded_fps_agrees_with_actual_frame_numbering():
    """交叉校验：clean_video 记下的帧率对应的步长，应等于从帧号反推的步长。

    format_dataset 内部会对每个 episode 做同样的检查，这里对真实素材独立再跑一遍。
    """
    meta = fd.load_images_meta(IMAGES_DIR)
    fps_by_video = fd.extract_source_fps(meta)
    assert fps_by_video, "output/images/_meta.json 里没有可用的帧率记录"

    target_fps = meta.get("target_fps", 5)
    for video, fps in fps_by_video.items():
        entries = frame_entries(video)
        if len(entries) < 2:
            continue
        observed = fd.infer_frame_step(entries)
        expected = max(1, round(fps / target_fps))
        assert observed == expected, (
            f"{video}: _meta.json 记的是 {fps}fps（步长 {expected}），"
            f"但从帧号反推的步长是 {observed}"
        )


@pytest.mark.skipif(not IMAGES_DIR.is_dir(), reason="output/images 不存在（该目录在 .gitignore 里）")
def test_real_images_build_a_consistent_dataset(run_format, tmp_path, read_jsonl):
    out = tmp_path / "ds"
    assert run_format("--input", IMAGES_DIR, "--output", out) == 0

    frames = read_jsonl(out / "meta" / "frames.jsonl")
    episodes = read_jsonl(out / "meta" / "episodes.jsonl")
    assert frames and episodes

    # 全局帧号连续
    assert [f["index"] for f in frames] == list(range(len(frames)))
    # episode 的 length 之和等于总帧数
    assert sum(e["length"] for e in episodes) == len(frames)

    for episode in episodes:
        rows = [f for f in frames if f["episode_index"] == episode["episode_index"]]
        assert len(rows) == episode["length"]
        # episode 内帧号连续
        assert [r["frame_index"] for r in rows] == list(range(len(rows)))
        # 原始帧号严格递增，说明排序没出错
        assert all(b["source_frame"] > a["source_frame"] for a, b in zip(rows, rows[1:]))
        # 时间戳随之严格递增 —— 空洞不会让时间倒流
        assert all(b["timestamp"] > a["timestamp"] for a, b in zip(rows, rows[1:]))


@pytest.mark.skipif(not VIDEOS_DIR.is_dir(), reason="videos/ 不存在（该目录在 .gitignore 里）")
def test_real_videos_are_decodable():
    """轻量冒烟测试：只开头部并读一帧，不解码整个视频。"""
    import cv2

    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    assert videos, "videos/ 下没有 mp4"

    for path in videos:
        cap = cv2.VideoCapture(str(path))
        try:
            assert cap.isOpened(), f"{path.name} 打不开"
            assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0, f"{path.name} 帧数为 0"
            assert cap.get(cv2.CAP_PROP_FPS) > 0, f"{path.name} 帧率为 0"
            ok, frame = cap.read()
            assert ok and frame is not None, f"{path.name} 读不出第一帧"
            assert frame.shape[0] > 0 and frame.shape[1] > 0
        finally:
            cap.release()
