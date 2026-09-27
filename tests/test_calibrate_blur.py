"""calibrate_blur.py 的测试。

大多数是纯函数单测（分位数、分桶、抽样）。涉及视频的用 cv2.VideoWriter 现场合成，
不依赖仓库里的 videos/。全部是快测试 —— 标定工具不碰 lerobot。
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

import calibrate_blur as cb

WIDTH, HEIGHT = 64, 48
BLOCK = 8


def sharp_frame() -> np.ndarray:
    """高对比度方波条纹。块取得大，是为了经得起 mp4 压缩后依然保持高方差。"""
    frame = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    for y in range(HEIGHT):
        frame[y, :] = 235 if (y // BLOCK) % 2 == 0 else 20
    return frame


def flat_frame() -> np.ndarray:
    """纯色块，Laplacian 方差接近 0。"""
    return np.full((HEIGHT, WIDTH, 3), 128, np.uint8)


@pytest.fixture
def make_video(tmp_path: Path):
    """把一串帧写成 mp4，返回文件路径。

    与 test_clean_video.py 里的同名夹具重复。那个不能挪进 conftest ——
    conftest 刻意不 import cv2，否则 format_dataset 的纯标准库测试会被拖慢。
    """

    def _make(name: str, frames: list, fps: int = 30) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
        )
        if not writer.isOpened():
            pytest.skip("当前 OpenCV 写不出 mp4（缺少编码器）")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        assert path.is_file() and path.stat().st_size > 0, "合成的视频是空文件"
        return path

    return _make


def make_scores(scores: list, indices: list = None) -> cb.VideoScores:
    """手工构造一个 VideoScores，用于测分类与导出逻辑（不依赖视频内容）。"""
    return cb.VideoScores(
        path=Path("fake.mp4"),
        name="1",
        orig_fps=30.0,
        frame_step=6,
        candidates=len(scores),
        frame_indices=indices if indices is not None else [i * 6 for i in range(len(scores))],
        scores=scores,
        elapsed=0.0,
    )


# --------------------------------------------------------------------------
# 均匀抽样
# --------------------------------------------------------------------------


def test_evenly_spaced_returns_everything_when_limit_is_zero():
    items = list(range(10))
    assert cb.evenly_spaced(items, 0) == items


def test_evenly_spaced_returns_everything_when_under_limit():
    items = list(range(5))
    assert cb.evenly_spaced(items, 5) == items
    assert cb.evenly_spaced(items, 99) == items


def test_evenly_spaced_keeps_both_ends():
    """不取前 N 个：素材开头往往是静止镜头，分布与全片差别很大。"""
    picked = cb.evenly_spaced(list(range(100)), 4)
    assert picked[0] == 0
    assert picked[-1] == 99
    assert len(picked) == 4


def test_evenly_spaced_handles_limit_of_one():
    """limit=1 时步长公式会除以零，必须单独处理。"""
    picked = cb.evenly_spaced(list(range(100)), 1)
    assert len(picked) == 1


# --------------------------------------------------------------------------
# 分位数
# --------------------------------------------------------------------------


def test_percentiles_of_empty_is_empty():
    assert cb.percentiles([]) == []


def test_percentiles_cover_the_ends():
    values = [1.0, 2.0, 3.0, 4.0]
    points = dict(cb.percentiles(values))
    assert points[0] == 1.0
    assert points[100] == 4.0


def test_percentiles_interpolate_linearly():
    """[0, 10] 的 p25 应当是 2.5，而不是最近的样本值。"""
    points = dict(cb.percentiles([0.0, 10.0], points=(25,)))
    assert points[25] == pytest.approx(2.5)


def test_percentiles_of_single_value():
    points = dict(cb.percentiles([7.5]))
    assert points[0] == points[50] == points[100] == 7.5


# --------------------------------------------------------------------------
# 直方图与阈值统计
# --------------------------------------------------------------------------


def test_histogram_counts_fall_in_the_right_buckets():
    buckets = dict(cb.histogram([5.0, 50.0, 150.0, 5000.0]))
    assert buckets["0-30"] == 1
    assert buckets["30-100"] == 1
    assert buckets["100-300"] == 1
    assert buckets["3000-10000"] == 1


def test_histogram_bucket_edges_go_to_the_upper_bucket():
    """边界值落在上界那一桶（闭下界开上界），100 属于 100-300。"""
    buckets = dict(cb.histogram([100.0, 300.0]))
    assert buckets["100-300"] == 1
    assert buckets["300-1000"] == 1


def test_histogram_handles_zero_and_huge():
    buckets = dict(cb.histogram([0.0, 1e9]))
    assert buckets["0-30"] == 1
    assert buckets["10000-inf"] == 1


def test_filter_rate_is_a_fraction():
    assert cb.filter_rate([10.0, 20.0, 200.0, 300.0], 100.0) == 0.5


def test_filter_rate_of_empty_is_zero():
    assert cb.filter_rate([], 100.0) == 0.0


def test_filter_rate_excludes_exact_threshold():
    """判定用的是 `< threshold`，恰好等于阈值的帧不算模糊。"""
    assert cb.filter_rate([100.0], 100.0) == 0.0
    assert cb.filter_rate([99.999], 100.0) == 1.0


# --------------------------------------------------------------------------
# 视频打分
# --------------------------------------------------------------------------


def test_score_video_samples_the_same_frames_as_clean_video(make_video):
    """采样帧号必须与 clean_video 一致，否则分布对不上实际过滤行为。"""
    video = make_video("1.mp4", [sharp_frame()] * 24, fps=30)
    result = cb.score_video(video, video.parent, target_fps=5, sample=0)
    assert result is not None
    assert result.frame_step == 6           # round(30 / 5)
    assert result.frame_indices == [0, 6, 12, 18]


def test_score_video_separates_sharp_from_flat(make_video):
    video = make_video("1.mp4", [sharp_frame()] * 12 + [flat_frame()] * 12, fps=30)
    result = cb.score_video(video, video.parent, target_fps=5, sample=0)
    assert result is not None
    assert len(result.scores) == 4
    assert result.scores[0] > 100           # 锐利帧分数高
    assert result.scores[-1] < 100          # 纯色帧分数接近 0


def test_score_video_name_matches_clean_video_prefix(make_video, tmp_path):
    """名字要与 clean_video.py 的 _meta.json 键一致，才能互相对照。"""
    video = make_video("sub/1.mp4", [sharp_frame()] * 12, fps=30)
    result = cb.score_video(video, tmp_path, target_fps=5, sample=0)
    assert result.name == "sub_1"


def test_score_video_returns_none_for_unreadable_file(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    assert cb.score_video(broken, tmp_path, target_fps=5, sample=0) is None


def test_sampling_limits_and_keeps_both_ends(make_video):
    video = make_video("1.mp4", [sharp_frame()] * 60, fps=30)
    result = cb.score_video(video, video.parent, target_fps=5, sample=3)
    assert result.frame_indices[0] == 0
    assert len(result.frame_indices) == 3


# --------------------------------------------------------------------------
# 导出复核帧
# --------------------------------------------------------------------------


def test_export_splits_below_and_critical(make_video, tmp_path):
    """以阈值为界、互不重叠：below 是 < 100，critical 是 [100, 300]。

    重叠的话同一帧会导出两遍（原实现用 [阈值/band, 阈值*band]，跨过了阈值）。
    """
    video = make_video("1.mp4", [sharp_frame()] * 24, fps=30)
    result = make_scores([10.0, 50.0, 200.0, 5000.0], indices=[0, 6, 12, 18])
    result.path = video

    below_dir, critical_dir = tmp_path / "below", tmp_path / "critical"
    written_below, written_critical, all_below, all_critical = cb.export_review_frames(
        result, below_dir, critical_dir, threshold=100.0, band=3.0, export_max=0
    )
    # 10 和 50 在阈值下；200 刚过线；5000 超出临界带（明显清晰，不必复核）
    assert (written_below, written_critical) == (2, 1)
    assert (all_below, all_critical) == (2, 1)
    assert len(list(below_dir.iterdir())) == 2
    assert len(list(critical_dir.iterdir())) == 1


def test_export_categories_do_not_overlap(make_video, tmp_path):
    """同一帧不能既在 below 又在 critical —— 否则复核时会重复看同一张。"""
    video = make_video("1.mp4", [sharp_frame()] * 24, fps=30)
    result = make_scores([50.0, 99.0, 100.0, 101.0], indices=[0, 6, 12, 18])
    result.path = video

    below_dir, critical_dir = tmp_path / "below", tmp_path / "critical"
    cb.export_review_frames(result, below_dir, critical_dir, 100.0, 3.0, export_max=0)

    below_names = {p.name for p in below_dir.iterdir()}
    critical_names = {p.name for p in critical_dir.iterdir()}
    assert below_names and critical_names
    assert not (below_names & critical_names)


def test_export_filename_starts_with_score(make_video, tmp_path):
    """文件名以分数开头，所以按名称排序就是按质量排序 —— 这是翻图的前提。"""
    video = make_video("1.mp4", [sharp_frame()] * 24, fps=30)
    result = make_scores([200.0, 120.0, 150.0], indices=[0, 6, 12])
    result.path = video

    critical_dir = tmp_path / "critical"
    cb.export_review_frames(
        result, tmp_path / "below", critical_dir, 100.0, 3.0, export_max=0
    )
    names = sorted(p.name for p in critical_dir.iterdir())
    scores_in_names = [float(name[3:12]) for name in names]
    assert scores_in_names == sorted(scores_in_names)
    assert scores_in_names == [120.0, 150.0, 200.0]


def test_export_max_caps_each_category_separately(make_video, tmp_path):
    """两类分别截取：否则数量多的一类会把另一类挤掉。

    临界带能覆盖大半个视频，不设上限会导出上千个文件，根本翻不完。
    """
    video = make_video("1.mp4", [sharp_frame()] * 50, fps=30)
    # 前 20 帧落在 below，后 30 帧落在 critical（恰好等于阈值也算 critical）
    result = make_scores([10.0] * 20 + [100.0] * 30, indices=list(range(50)))
    result.path = video

    below_dir, critical_dir = tmp_path / "below", tmp_path / "critical"
    written_below, written_critical, all_below, all_critical = cb.export_review_frames(
        result, below_dir, critical_dir, 100.0, 3.0, export_max=5
    )
    assert (written_below, written_critical) == (5, 5)
    # 总数与实际写出数分别返回：屏幕上要能看出"符合多少、导出了多少"
    assert (all_below, all_critical) == (20, 30)


def test_export_returns_zeros_when_nothing_qualifies(make_video, tmp_path):
    video = make_video("1.mp4", [sharp_frame()] * 12, fps=30)
    result = make_scores([5000.0], indices=[0])     # 既不低于 100，也不在临界带内
    result.path = video

    counts = cb.export_review_frames(
        result, tmp_path / "below", tmp_path / "critical", 100.0, 3.0, export_max=0
    )
    assert counts == (0, 0, 0, 0)
    assert not (tmp_path / "below").exists()        # 没有可导出的就不要建空目录


def test_scores_jsonl_is_written(tmp_path):
    results = [make_scores([10.0, 200.0])]
    path = tmp_path / "scores.jsonl"
    cb.write_scores_jsonl(path, results, thresholds=[100.0])

    import json

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["video"] == "1"
    assert rows[0]["source_frame"] == 0
    assert rows[0]["below"] == {"100": True}
    assert rows[1]["below"] == {"100": False}


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------


def run_calibrate(run_args, monkeypatch) -> int:
    import sys

    monkeypatch.setattr(sys, "argv", ["calibrate_blur.py", *[str(a) for a in run_args]])
    return cb.main()


def test_missing_input_fails(tmp_path, monkeypatch):
    assert run_calibrate(["--input", tmp_path / "nope"], monkeypatch) == 1


def test_empty_input_directory_fails(tmp_path, monkeypatch, caplog):
    empty = tmp_path / "videos"
    empty.mkdir()
    with caplog.at_level(logging.ERROR):
        assert run_calibrate(["--input", empty], monkeypatch) == 1
    assert "没有找到 mp4" in caplog.text


@pytest.mark.parametrize(
    "extra",
    [
        ("--fps", 0),
        ("--sample", -1),
        ("--export-band", 1),
        ("--export-max", -1),
        ("--threshold", 0),
    ],
)
def test_invalid_arguments_fail(tmp_path, monkeypatch, extra):
    assert run_calibrate(["--input", tmp_path, *extra], monkeypatch) == 1


def test_end_to_end_reports_and_exports(make_video, tmp_path, monkeypatch, caplog):
    make_video("1.mp4", [sharp_frame()] * 12 + [flat_frame()] * 12, fps=30)
    export_dir = tmp_path / "review"

    with caplog.at_level(logging.INFO):
        code = run_calibrate(
            [
                "--input",
                tmp_path,
                "--threshold",
                100,
                "--export-dir",
                export_dir,
            ],
            monkeypatch,
        )
    assert code == 0
    assert "方差分布" in caplog.text
    assert "阈值影响" in caplog.text
    assert (export_dir / "scores.jsonl").is_file()
    # 纯色帧必然低于阈值，必须被导出到 below/
    assert list((export_dir / "below").iterdir())


def test_export_not_requested_creates_nothing(make_video, tmp_path, monkeypatch):
    make_video("1.mp4", [sharp_frame()] * 12, fps=30)
    assert run_calibrate(["--input", tmp_path], monkeypatch) == 0
    assert not (tmp_path / "below").exists()
    assert not (tmp_path / "review").exists()
