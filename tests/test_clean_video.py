"""clean_video.py 的测试。

需要 cv2 所以单独成文件 —— format_dataset 是纯标准库的，不该因这里的导入失败一起挂。
视频用 cv2.VideoWriter 现场合成，不依赖仓库里的 videos/。
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

import clean_video

WIDTH, HEIGHT = 64, 48
BLOCK = 8


def sharp_frame() -> np.ndarray:
    """高对比度方波条纹。块取得大，是为了经得起 mp4 压缩后依然保持高方差。"""
    frame = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    for y in range(HEIGHT):
        frame[y, :] = 235 if (y // BLOCK) % 2 == 0 else 20
    return frame


def flat_frame() -> np.ndarray:
    """纯色块，Laplacian 方差接近 0，必然被判为模糊。"""
    return np.full((HEIGHT, WIDTH, 3), 128, np.uint8)


@pytest.fixture
def make_video(tmp_path: Path):
    """把一串帧写成 mp4，返回文件路径。"""

    def _make(name: str, frames: list, fps: int = 30) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)   # 允许传 "sub/1.mp4"
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


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    path = tmp_path / "images"
    path.mkdir()
    return path


# --------------------------------------------------------------------------
# 模糊判定
# --------------------------------------------------------------------------


def test_flat_frame_is_blurry():
    assert clean_video.is_frame_blurry(flat_frame(), 100.0) is True


def test_sharp_frame_is_not_blurry():
    assert clean_video.is_frame_blurry(sharp_frame(), 100.0) is False


def test_threshold_above_sharp_score_flags_it():
    """阈值高到离谱时，再锐的帧也该被判模糊。"""
    assert clean_video.is_frame_blurry(sharp_frame(), 1e9) is True


# --------------------------------------------------------------------------
# 视频扫描
# --------------------------------------------------------------------------


def test_list_videos_matches_each_file_once(tmp_path):
    """回归：历史上 "*.mp4" 与 "*.MP4" 两轮 rglob 会把同一个文件匹配两次、重复处理。"""
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.MP4").write_bytes(b"x")
    (tmp_path / "c.avi").write_bytes(b"x")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "d.mp4").write_bytes(b"x")

    assert sorted(p.name for p in clean_video.list_videos(tmp_path)) == ["a.mp4", "b.MP4", "d.mp4"]


def test_list_videos_rejects_missing_directory():
    with pytest.raises(FileNotFoundError):
        list(clean_video.list_videos(Path("no") / "such" / "dir"))


# --------------------------------------------------------------------------
# 输出名分配
# --------------------------------------------------------------------------


def test_video_output_prefix_keeps_flat_names_unchanged(tmp_path):
    """平铺目录下前缀就是原来的 stem —— 现有产物的命名不受影响。"""
    assert clean_video.video_output_prefix(tmp_path / "1.mp4", tmp_path) == "1"


@pytest.mark.parametrize(
    "relative, expected",
    [
        ("a/1.mp4", "a_1"),
        ("a/b/1.mp4", "a_b_1"),
        ("round1/clip 2.mp4", "round1_clip 2"),
    ],
)
def test_video_output_prefix_includes_subdirectories(tmp_path, relative, expected):
    assert clean_video.video_output_prefix(tmp_path / relative, tmp_path) == expected


def test_assign_output_names_separates_same_stem_in_different_dirs(tmp_path):
    videos = [tmp_path / "a" / "1.mp4", tmp_path / "b" / "1.mp4"]
    names = clean_video.assign_output_names(videos, tmp_path)
    assert names[videos[0]] == "a_1"
    assert names[videos[1]] == "b_1"


def test_assign_output_names_rejects_unavoidable_collision(tmp_path):
    """加了目录前缀仍可能撞车（a_1.mp4 与 a/1.mp4 都得到 a_1），必须报错而不是静默覆盖。"""
    videos = [tmp_path / "a_1.mp4", tmp_path / "a" / "1.mp4"]
    with pytest.raises(ValueError, match="输出名冲突"):
        clean_video.assign_output_names(videos, tmp_path)


# --------------------------------------------------------------------------
# 抽帧主流程
# --------------------------------------------------------------------------


def test_process_video_keeps_every_sharp_frame(make_video, images_dir):
    video = make_video("sharp.mp4", [sharp_frame() for _ in range(30)])
    result = clean_video.process_video(video, images_dir, target_fps=5)
    assert result.sampled > 0
    assert result.video_name == "sharp"
    assert result.orig_fps > 0
    assert len(list(images_dir.glob("*.jpg"))) == result.sampled


def test_process_video_keeps_every_blurry_frame_too(make_video, images_dir):
    """核心行为：模糊帧也照样存盘，否则帧就不在 1/fps 的均匀网格上了。"""
    video = make_video("flat.mp4", [flat_frame() for _ in range(30)])
    result = clean_video.process_video(video, images_dir, target_fps=5)
    assert result.sampled > 0
    assert len(list(images_dir.glob("*.jpg"))) == result.sampled
    # 分数确实很低（确实是「模糊」），只是不影响存盘
    assert max(score for _, score in result.scores) < 100.0


def test_process_video_names_files_by_source_frame_index(make_video, images_dir):
    """文件名里的帧号必须是原始帧号，format_dataset 靠它还原时间轴。"""
    video = make_video("clip.mp4", [sharp_frame() for _ in range(30)])
    clean_video.process_video(video, images_dir, target_fps=5)

    names = sorted(p.name for p in images_dir.glob("*.jpg"))
    assert names, "应当至少抽出一帧"
    # 30fps 采样到 5fps，步长 6：帧号必须是 0, 6, 12, ... 连续无缺口
    for index, filename in enumerate(names):
        assert filename == f"clip_frame_{index * 6:06d}.jpg"


def test_process_video_uses_given_output_name(make_video, images_dir):
    video = make_video("clip.mp4", [sharp_frame() for _ in range(12)])
    result = clean_video.process_video(
        video, images_dir, target_fps=5, output_name="sub_clip"
    )
    assert result.video_name == "sub_clip"
    names = sorted(p.name for p in images_dir.glob("*.jpg"))
    assert names and all(n.startswith("sub_clip_frame_") for n in names)


def test_nested_videos_with_the_same_stem_do_not_overwrite(make_video, tmp_path):
    """回归：videos/a/1.mp4 与 videos/b/1.mp4 以前会写成同一批文件名，并发覆盖彼此。"""
    root = tmp_path / "videos"
    make_video("videos/a/1.mp4", [sharp_frame() for _ in range(12)])
    make_video("videos/b/1.mp4", [sharp_frame() for _ in range(12)])

    videos = sorted(clean_video.list_videos(root))
    assert len(videos) == 2

    names = clean_video.assign_output_names(videos, root)
    assert sorted(names.values()) == ["a_1", "b_1"]

    images_dir = tmp_path / "images"
    for video in videos:
        clean_video.process_video(
            video, images_dir, target_fps=5, output_name=names[video]
        )

    produced = sorted(p.name for p in images_dir.glob("*.jpg"))
    a_files = [n for n in produced if n.startswith("a_1_frame_")]
    b_files = [n for n in produced if n.startswith("b_1_frame_")]
    assert a_files, "a/1.mp4 的帧没写出来"
    assert b_files, "b/1.mp4 的帧没写出来"
    assert len(a_files) + len(b_files) == len(produced)


def test_process_video_scores_every_saved_frame(make_video, images_dir):
    """每张落盘图片都必须有一条分数记录，否则 format_dataset 会拿到 None。"""
    # 周期 12 而不是 2：采样步长是 6，两者同周期的话每帧都会落在同一类上
    frames = [sharp_frame()] * 6 + [flat_frame()] * 6
    video = make_video("mixed.mp4", frames * 5)
    result = clean_video.process_video(video, images_dir, target_fps=5)

    saved = sorted(p.stem for p in images_dir.glob("*.jpg"))
    scored = sorted(f"{result.video_name}_frame_{frame:06d}" for frame, _ in result.scores)
    assert saved == scored

    # 帧号升序，且与落盘图片一一对应
    frame_numbers = [frame for frame, _ in result.scores]
    assert frame_numbers == sorted(frame_numbers)
    # 混合素材：既要有清晰的也要有模糊的，否则这个用例没验证到东西
    assert any(s < 100.0 for _, s in result.scores)
    assert any(s >= 100.0 for _, s in result.scores)


def test_process_video_survives_unopenable_file(tmp_path, images_dir):
    """cv2 打开这个文件时，ffmpeg 会往 stderr 打一句 "moov atom not found" —— 属正常噪音。"""
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"definitely not a video")

    result = clean_video.process_video(broken, images_dir, target_fps=5)
    assert result.sampled == 0
    assert result.scores == []
    assert result.video_name == "broken"
    assert result.orig_fps == 0.0


def test_process_video_creates_output_directory(make_video, tmp_path):
    video = make_video("clip.mp4", [sharp_frame() for _ in range(12)])
    nested = tmp_path / "a" / "b" / "images"          # 事先不存在
    clean_video.process_video(video, nested, target_fps=5)
    assert nested.is_dir()


# --------------------------------------------------------------------------
# _meta.json 的写出
# --------------------------------------------------------------------------


def read_meta(images_dir: Path) -> dict:
    return json.loads(
        (images_dir / clean_video.IMAGES_META_FILENAME).read_text(encoding="utf-8")
    )


def read_scores(images_dir: Path) -> list:
    path = images_dir / clean_video.BLUR_SCORES_FILENAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fake_result(
    name: str,
    sampled: int = 1000,
    elapsed: float = 55.7,
    orig_fps: float = 30.0,
    scores: list = None,
) -> "clean_video.VideoResult":
    """构造一个 VideoResult；scores 默认按 30fps/5fps 的步长 6 造一批清晰帧。"""
    if scores is None:
        scores = [(i * 6, 250.0) for i in range(sampled)]
    return clean_video.VideoResult(sampled, elapsed, name, orig_fps, scores)


def test_write_images_meta_records_rates_and_parameters(images_dir):
    scores = [(i * 6, 300.0 if i % 2 else 50.0) for i in range(1000)]
    clean_video.write_images_meta(images_dir, {"1": 30.0}, 5, [fake_result("1", scores=scores)])

    payload = read_meta(images_dir)
    assert payload["format"] == 2
    assert payload["target_fps"] == 5
    # 阈值概念已移到 format_dataset.py，第一步不再记录
    assert "blur_threshold" not in payload

    entry = payload["videos"]["1"]
    assert entry["orig_fps"] == 30.0
    assert entry["frame_step"] == 6         # 30 / 5
    assert entry["sampled"] == 1000
    assert entry["score_median"] == 175.0   # (50 + 300) / 2
    assert "saved" not in entry and "skipped_blurry" not in entry


def test_write_images_meta_merges_instead_of_clobbering(images_dir):
    """分批跑同一个输出目录时，先前的记录不能被抹掉。"""
    clean_video.write_images_meta(images_dir, {"1": 30.0}, 5, [fake_result("1")])
    clean_video.write_images_meta(images_dir, {"2": 25.0}, 5, [fake_result("2")])

    assert set(read_meta(images_dir)["videos"]) == {"1", "2"}


def test_write_images_meta_skips_videos_that_failed(images_dir):
    """失败视频的 orig_fps 是 0，不能写进去污染既有记录。"""
    clean_video.write_images_meta(
        images_dir, {"1": 30.0, "broken": 0.0}, 5, [fake_result("1"), fake_result("broken")]
    )
    assert set(read_meta(images_dir)["videos"]) == {"1"}


def test_write_images_meta_recovers_from_corrupt_file(images_dir):
    (images_dir / clean_video.IMAGES_META_FILENAME).write_text("{ not json at all", encoding="utf-8")
    clean_video.write_images_meta(images_dir, {"1": 30.0}, 5, [fake_result("1")])
    assert read_meta(images_dir)["videos"]["1"]["orig_fps"] == 30.0


def test_write_images_meta_leaves_no_temp_file(images_dir):
    clean_video.write_images_meta(images_dir, {"1": 30.0}, 5, [fake_result("1")])
    assert list(images_dir.glob("*.tmp")) == []


# --------------------------------------------------------------------------
# _scores.jsonl 的写出
# --------------------------------------------------------------------------


def test_write_blur_scores_records_one_row_per_frame(images_dir):
    scores = [(0, 250.5), (6, 42.0), (12, 99.123456)]
    clean_video.write_blur_scores(images_dir, [fake_result("1", sampled=3, scores=scores)])

    assert read_scores(images_dir) == [
        {"video": "1", "source_frame": 0, "blur_score": 250.5},
        {"video": "1", "source_frame": 6, "blur_score": 42.0},
        {"video": "1", "source_frame": 12, "blur_score": 99.1235},  # 保留 4 位
    ]


def test_write_blur_scores_merges_instead_of_clobbering(images_dir):
    """重跑单个视频时只替换它的行，别的视频保留。"""
    clean_video.write_blur_scores(images_dir, [fake_result("1", sampled=2, scores=[(0, 10.0), (6, 20.0)])])
    clean_video.write_blur_scores(images_dir, [fake_result("2", sampled=1, scores=[(0, 30.0)])])
    assert {row["video"] for row in read_scores(images_dir)} == {"1", "2"}

    # 重跑视频 1，帧数变少 —— 旧的行必须被整段替换而不是累加
    clean_video.write_blur_scores(images_dir, [fake_result("1", sampled=1, scores=[(0, 99.0)])])
    rows = read_scores(images_dir)
    assert [r["blur_score"] for r in rows if r["video"] == "1"] == [99.0]
    assert [r["blur_score"] for r in rows if r["video"] == "2"] == [30.0]


def test_write_blur_scores_skips_videos_that_failed(images_dir):
    clean_video.write_blur_scores(
        images_dir,
        [fake_result("1", sampled=1, scores=[(0, 10.0)]), fake_result("broken", sampled=0, orig_fps=0.0)],
    )
    assert {row["video"] for row in read_scores(images_dir)} == {"1"}


def test_write_blur_scores_recovers_from_corrupt_file(images_dir):
    (images_dir / clean_video.BLUR_SCORES_FILENAME).write_text("{ not json at all\n", encoding="utf-8")
    clean_video.write_blur_scores(images_dir, [fake_result("1", sampled=1, scores=[(0, 10.0)])])
    assert read_scores(images_dir) == [{"video": "1", "source_frame": 0, "blur_score": 10.0}]


def test_write_blur_scores_leaves_no_temp_file(images_dir):
    clean_video.write_blur_scores(images_dir, [fake_result("1", sampled=1, scores=[(0, 10.0)])])
    assert list(images_dir.glob("*.tmp")) == []


def test_blur_score_threshold_does_not_affect_what_is_written(make_video, images_dir):
    """第一步已无阈值参数；全模糊的素材在任意口径下都应当全部落盘。"""
    video = make_video("flat.mp4", [flat_frame() for _ in range(30)])
    result = clean_video.process_video(video, images_dir, target_fps=5)
    clean_video.write_blur_scores(images_dir, [result])

    assert len(read_scores(images_dir)) == len(list(images_dir.glob("*.jpg"))) == result.sampled
    assert all(row["blur_score"] < 100.0 for row in read_scores(images_dir))


# --------------------------------------------------------------------------
# 两个模块的接口契约
# --------------------------------------------------------------------------


def test_meta_written_by_clean_video_is_accepted_by_format_dataset(
    make_images, run_format, tmp_path, caplog
):
    """clean_video 写、format_dataset 读 —— 正常路径不应产生任何告警。"""
    images = make_images({"1": [0, 6, 12]})
    result = fake_result("1", sampled=3, scores=[(0, 300.0), (6, 50.0), (12, 200.0)])
    clean_video.write_images_meta(images, {"1": 30.0}, 5, [result])
    clean_video.write_blur_scores(images, [result])

    out = tmp_path / "ds"
    with caplog.at_level(logging.INFO):
        rc = run_format("--input", images, "--output", out)
    assert rc == 0
    assert "已读取" in caplog.text

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], f"正常路径不该有告警：{warnings}"

    # _scores.jsonl 必须被认出来是元数据，而不是「文件名不符合规范」
    assert "不匹配 0" in caplog.text
