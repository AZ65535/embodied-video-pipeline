"""calibrate_blur.py 的测试。

纯函数单测（分位数、分桶、截取）加上基于 `_scores.jsonl` 的分析与导出。
本模块不碰视频，所以这里也不需要 cv2 —— 帧文件用 conftest 的小 JPEG 充当。
"""

import json
import logging
import shutil
import sys
from pathlib import Path

import pytest

import calibrate_blur as cb


@pytest.fixture
def make_scores_dir(tmp_path, tiny_jpegs):
    """造一个 clean_video.py 产物目录：图片 + _scores.jsonl + _meta.json。

    spec 形如 ``{"1": {0: 250.0, 6: 42.0}}``（视频名 -> 帧号 -> 分数）。
    图片按 `{视频名}_frame_{帧号:06d}.jpg` 命名，与第一步的产物一致。
    """
    counter = [0]

    def _make(spec: dict, write_images: bool = True, write_meta: bool = True) -> Path:
        root = tmp_path / f"images{len(list(tmp_path.glob('images*')))}"
        root.mkdir()
        rows = []
        for name, frame_scores in spec.items():
            for frame, score in frame_scores.items():
                rows.append({"video": name, "source_frame": frame, "blur_score": score})
                if write_images:
                    shutil.copy2(tiny_jpegs[0], root / f"{name}_frame_{frame:06d}.jpg")
        (root / cb.BLUR_SCORES_FILENAME).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        if write_meta:
            (root / cb.IMAGES_META_FILENAME).write_text(
                json.dumps(
                    {
                        "format": 2,
                        "tool": "clean_video.py",
                        "target_fps": 5,
                        "videos": {name: {"orig_fps": 30.0, "frame_step": 6} for name in spec},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return root

    return _make


def make_scores(scores: list, indices: list = None) -> cb.VideoScores:
    """手工构造一个 VideoScores，用于测分类与导出逻辑。"""
    return cb.VideoScores(
        name="1",
        frame_indices=indices if indices is not None else [i * 6 for i in range(len(scores))],
        scores=scores,
        orig_fps=30.0,
        frame_step=6,
    )


# --------------------------------------------------------------------------
# 均匀抽样
# --------------------------------------------------------------------------


def test_evenly_spaced_returns_everything_when_limit_is_zero():
    items = list(range(10))
    assert cb.evenly_spaced(items, 0) == items


def test_evenly_spaced_returns_everything_when_under_limit():
    items = list(range(5))
    assert cb.evenly_spaced(items, 10) == items


def test_evenly_spaced_keeps_both_ends():
    """不取前 N 个：素材开头的静止镜头和全片分布差别很大，必须覆盖到尾部。"""
    picked = cb.evenly_spaced(list(range(100)), 5)
    assert picked[0] == 0 and picked[-1] == 99
    assert len(picked) == 5


def test_evenly_spaced_handles_limit_of_one():
    assert cb.evenly_spaced(list(range(10)), 1) == [0]


# --------------------------------------------------------------------------
# 分位数与直方图
# --------------------------------------------------------------------------


def test_percentiles_of_empty_is_empty():
    assert cb.percentiles([]) == []


def test_percentiles_cover_the_ends():
    points = dict(cb.percentiles([1.0, 2.0, 3.0, 4.0]))
    assert points[0] == 1.0 and points[100] == 4.0


def test_percentiles_interpolate_linearly():
    points = dict(cb.percentiles([0.0, 10.0]))
    assert points[50] == pytest.approx(5.0)


def test_percentiles_of_single_value():
    assert dict(cb.percentiles([7.5]))[50] == 7.5


def test_histogram_counts_fall_in_the_right_buckets():
    buckets = dict(cb.histogram([10.0, 20.0, 50.0, 200.0, 5000.0]))
    assert buckets["0-30"] == 2      # 10 与 20
    assert buckets["30-100"] == 1    # 50
    assert buckets["100-300"] == 1   # 200
    assert buckets["3000-10000"] == 1


def test_histogram_bucket_edges_go_to_the_upper_bucket():
    """边界值归上一桶（左闭右开），否则 30 会同时落在两个桶的叙述里。"""
    assert dict(cb.histogram([30.0]))["30-100"] == 1
    assert dict(cb.histogram([30.0]))["0-30"] == 0


def test_histogram_handles_zero_and_huge():
    buckets = dict(cb.histogram([0.0, 1e9]))
    assert buckets["0-30"] == 1
    assert buckets["10000+"] == 1


# --------------------------------------------------------------------------
# 阈值口径
# --------------------------------------------------------------------------


def test_filter_rate_is_a_fraction():
    assert cb.filter_rate([10.0, 20.0, 30.0, 40.0], 25.0) == pytest.approx(0.5)


def test_filter_rate_of_empty_is_zero():
    assert cb.filter_rate([], 100.0) == 0.0


def test_filter_rate_excludes_exact_threshold():
    """严格小于：恰好等于阈值算清晰，与 clean_video / format_dataset 口径一致。"""
    assert cb.filter_rate([100.0], 100.0) == 0.0
    assert cb.filter_rate([99.999], 100.0) == 1.0


# --------------------------------------------------------------------------
# 读取第一步的分数
# --------------------------------------------------------------------------


def test_load_scores_groups_by_video_and_sorts_by_frame(make_scores_dir):
    images = make_scores_dir({"1": {12: 300.0, 0: 250.0, 6: 42.0}, "2": {0: 9.0}})
    results = cb.load_scores(images)

    assert [r.name for r in results] == ["1", "2"]
    assert results[0].frame_indices == [0, 6, 12]     # 按帧号排序，不按文件里的行序
    assert results[0].scores == [250.0, 42.0, 300.0]
    assert results[0].orig_fps == 30.0
    assert results[0].frame_step == 6


def test_load_scores_needs_no_cv2():
    """本模块是纯标准库 —— 分数是现成的，不需要再解码。"""
    source = Path(cb.__file__).read_text(encoding="utf-8")
    assert "import cv2" not in source


def test_load_scores_survives_bad_lines(make_scores_dir, caplog):
    images = make_scores_dir({"1": {0: 250.0}})
    path = images / cb.BLUR_SCORES_FILENAME
    path.write_text(path.read_text(encoding="utf-8") + "{ 这不是 json\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        results = cb.load_scores(images)
    assert len(results[0].scores) == 1
    assert "跳过" in caplog.text


def test_load_scores_without_meta_still_works(make_scores_dir):
    """_meta.json 只用于显示原始帧率，缺了不影响分析。"""
    images = make_scores_dir({"1": {0: 250.0}}, write_meta=False)
    results = cb.load_scores(images)
    assert results[0].orig_fps is None
    assert results[0].scores == [250.0]


def test_load_scores_rejects_missing_sidecar(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError, match="clean_video.py"):
        cb.load_scores(tmp_path / "images")


def test_load_scores_rejects_a_sidecar_with_no_rows(make_scores_dir):
    images = make_scores_dir({"1": {}})
    with pytest.raises(ValueError, match="没有任何可用的分数"):
        cb.load_scores(images)


# --------------------------------------------------------------------------
# 导出复核帧
# --------------------------------------------------------------------------


def test_export_splits_below_and_critical(make_scores_dir, tmp_path):
    images = make_scores_dir({"1": {0: 50.0, 6: 100.5, 12: 42.0, 18: 500.0}})
    below, critical, total_below, total_critical = cb.export_review_frames(
        cb.load_scores(images)[0], images, tmp_path / "below", tmp_path / "critical", 100.0, 3.0, 0
    )
    assert (below, total_below) == (2, 2)         # 50 与 42
    assert (critical, total_critical) == (1, 1)   # 100.5；500 超出 band
    assert sorted(p.name for p in (tmp_path / "below").glob("*.jpg"))
    assert len(list((tmp_path / "critical").glob("*.jpg"))) == 1


def test_export_categories_do_not_overlap(make_scores_dir, tmp_path):
    images = make_scores_dir({"1": {i * 6: float(i) for i in range(100, 250)}})
    cb.export_review_frames(
        cb.load_scores(images)[0], images, tmp_path / "below", tmp_path / "critical", 100.0, 3.0, 0
    )
    below = {p.name for p in (tmp_path / "below").glob("*.jpg")}
    critical = {p.name for p in (tmp_path / "critical").glob("*.jpg")}
    assert not (below & critical)


def test_export_filename_starts_with_score(make_scores_dir, tmp_path):
    """文件名以分数开头，按名称排序即按质量排序 —— 翻图的顺序才有意义。"""
    images = make_scores_dir({"1": {0: 42.5}})
    cb.export_review_frames(
        cb.load_scores(images)[0], images, tmp_path / "below", tmp_path / "critical", 100.0, 3.0, 0
    )
    names = [p.name for p in (tmp_path / "below").glob("*.jpg")]
    assert len(names) == 1 and names[0].startswith("var000042.50_")


def test_export_max_caps_each_category_separately(make_scores_dir, tmp_path):
    """分类截取而非合并截取：否则数量多的一类会把另一类挤掉。"""
    spec = {"1": {i * 6: 10.0 for i in range(50)}}          # 50 帧 below
    spec["1"].update({(i + 50) * 6: 150.0 for i in range(20)})  # 20 帧 critical
    images = make_scores_dir(spec)
    below, critical, _, _ = cb.export_review_frames(
        cb.load_scores(images)[0], images, tmp_path / "below", tmp_path / "critical", 100.0, 3.0, 5
    )
    assert (below, critical) == (5, 5)


def test_export_returns_zeros_when_nothing_qualifies(make_scores_dir, tmp_path):
    images = make_scores_dir({"1": {0: 100000.0}})
    assert cb.export_review_frames(
        cb.load_scores(images)[0], images, tmp_path / "below", tmp_path / "critical", 100.0, 3.0, 0
    ) == (0, 0, 0, 0)


def test_export_skips_missing_frames(make_scores_dir, tmp_path, caplog):
    """分数表里有、图片却不在时跳过并告警，不能让整个导出挂掉。"""
    images = make_scores_dir({"1": {0: 42.0, 6: 43.0}}, write_images=False)
    with caplog.at_level(logging.WARNING):
        below, _, total_below, _ = cb.export_review_frames(
            cb.load_scores(images)[0], images, tmp_path / "below", tmp_path / "critical",
            100.0, 3.0, 0,
        )
    assert below == 0 and total_below == 2
    assert "帧文件不存在" in caplog.text


def test_scores_jsonl_is_written(make_scores_dir, tmp_path):
    images = make_scores_dir({"1": {0: 42.0, 6: 250.0}})
    out = tmp_path / "out.jsonl"
    cb.write_scores_jsonl(out, cb.load_scores(images), [100.0])

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["blur_score"] == 42.0
    assert rows[0]["below"] == {"100": True}
    assert rows[1]["below"] == {"100": False}


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------


def run_main(monkeypatch, *args) -> int:
    monkeypatch.setattr(sys, "argv", ["calibrate_blur.py", *[str(a) for a in args]])
    return cb.main()


def test_missing_sidecar_fails(tmp_path, monkeypatch, caplog):
    (tmp_path / "images").mkdir()
    with caplog.at_level(logging.ERROR):
        assert run_main(monkeypatch, "--input", tmp_path / "images") == 1
    assert "clean_video.py" in caplog.text


@pytest.mark.parametrize(
    "extra",
    [["--export-band", "1"], ["--export-band", "0.5"], ["--export-max", "-1"], ["--threshold", "0"]],
)
def test_invalid_arguments_fail(tmp_path, monkeypatch, extra):
    images = tmp_path / "images"
    images.mkdir()
    assert run_main(monkeypatch, "--input", images, *extra) == 1


def test_end_to_end_reports_and_exports(make_scores_dir, tmp_path, monkeypatch, caplog):
    images = make_scores_dir({"1": {0: 42.0, 6: 250.0, 12: 150.0}, "2": {0: 300.0}})
    export = tmp_path / "review"
    with caplog.at_level(logging.INFO):
        rc = run_main(
            monkeypatch, "--input", images, "--export-dir", export, "--threshold", "100"
        )
    assert rc == 0
    text = caplog.text
    assert "方差分布" in text
    assert "阈值影响" in text
    # 共 4 帧（3 + 1），只有 42.0 低于阈值 100 —— 合计 25.0%，视频 1 为 33.3%
    assert "33.3%" in text and "25.0%" in text
    assert (export / "below").is_dir() and (export / "critical").is_dir()
    assert (export / "scores.jsonl").is_file()


def test_export_not_requested_creates_nothing(make_scores_dir, tmp_path, monkeypatch):
    images = make_scores_dir({"1": {0: 42.0}}, write_meta=False)
    assert run_main(monkeypatch, "--input", images) == 0
    assert not list(tmp_path.glob("review*"))
