"""format_dataset.py 的测试。

分三段：纯函数单测、端到端集成、以及锁住本次开发中实际踩到的坑的回归测试。
"""

import json
import logging
from pathlib import Path

import pytest

import format_dataset as fd


def entries(*frame_numbers: int) -> list:
    """构造 FrameEntry 序列（宽高在本模块的测试里无关紧要）。"""
    return [
        fd.FrameEntry(src=Path("x.jpg"), source_frame=n, width=1, height=1)
        for n in frame_numbers
    ]


def read_info(out: Path) -> dict:
    return json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 文件名解析
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("1_frame_000006.jpg", ("1", 6)),
        ("video1_frame_000000.jpg", ("video1", 0)),
        ("my_video_frame_000012.jpg", ("my_video", 12)),
        # 贪婪匹配：视频名里含 "_frame_" 时，取最后一个作为分隔
        ("a_frame_b_frame_000003.jpg", ("a_frame_b", 3)),
        ("1_frame_6.JPG", ("1", 6)),
        ("1_frame_6.jpeg", ("1", 6)),
    ],
)
def test_parse_frame_name_accepts(name, expected):
    assert fd.parse_frame_name(Path(name)) == expected


@pytest.mark.parametrize(
    "name",
    [
        "1_frame.jpg",          # 缺帧号
        "frame_000001.jpg",     # 缺视频名
        "1_frame_abc.jpg",      # 帧号不是数字
        "1.mp4",                # 不是图片
        "1_frame_1.png",        # 后缀不支持
        "_meta.json",           # 处理参数文件，不是帧
    ],
)
def test_parse_frame_name_rejects(name):
    assert fd.parse_frame_name(Path(name)) is None


def test_natural_key_orders_numbers_numerically():
    """按字典序 "10" 会排在 "2" 前面，自然序必须修正这一点。"""
    assert sorted(["10", "2", "1"], key=fd.natural_key) == ["1", "2", "10"]


# --------------------------------------------------------------------------
# JPEG 头部解析（纯标准库）
# --------------------------------------------------------------------------


def test_probe_jpeg_reads_real_dimensions(fixtures_dir):
    expected = {"tiny_8x8.jpg": (8, 8), "tiny_16x32.jpg": (16, 32), "tiny_32x16.jpg": (32, 16)}
    for name, size in expected.items():
        assert fd.probe_jpeg(fixtures_dir / name) == size


@pytest.mark.parametrize(
    "payload, reason",
    [
        (b"", "空文件"),
        (b"\xff\xd8", "只有 SOI，后面什么都没有"),
        (b"\xff\xd8\xff\xe0", "段长字段缺失"),
        (b"\xff\xd8\xff\xe0\x00\x01", "段长 1 非法（合法值 >= 2）"),
        (b"\xff\xd8\xff\xe0\x00\x10", "段长超出文件长度"),
        (b"\xff\xd8\xff\xda\x00\x02", "SOS 之前没有 SOF"),
        (b"\xff\xd8\xff\xd9", "直接遇到 EOI"),
        (b"not a jpeg at all", "缺少 SOI"),
        (b"\xff\xd8\xff\xc0\x00\x11\x08\x00", "SOF 段被截断"),
        (b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x00\x00\x00\x00", "SOF 里宽高为 0"),
    ],
)
def test_probe_jpeg_rejects_malformed(tmp_path, payload, reason):
    path = tmp_path / "broken.jpg"
    path.write_bytes(payload)
    assert fd.probe_jpeg(path) is None, reason


def test_probe_jpeg_tolerates_fill_bytes(tmp_path):
    """标记前允许 FF FF ... xx 形式的填充字节。"""
    payload = (
        b"\xff\xd8"
        + b"\xff\xff\xff\xc0"          # 填充 + SOF0
        + b"\x00\x11"                  # 段长 17
        + b"\x08"                      # 精度
        + (32).to_bytes(2, "big")      # 高
        + (64).to_bytes(2, "big")      # 宽
        + b"\x00" * 8                  # 其余字段
    )
    path = tmp_path / "padded.jpg"
    path.write_bytes(payload)
    assert fd.probe_jpeg(path) == (64, 32)


def test_has_eoi_detects_truncation(tmp_path, tiny_jpegs):
    good = tmp_path / "good.jpg"
    good.write_bytes(tiny_jpegs[0].read_bytes())
    assert fd.has_eoi(good) is True

    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(tiny_jpegs[0].read_bytes()[:-2])  # 砍掉 FFD9
    assert fd.has_eoi(truncated) is False


# --------------------------------------------------------------------------
# 帧率反推与 _meta.json 读取
# --------------------------------------------------------------------------


def test_infer_frame_step_basic():
    assert fd.infer_frame_step(entries(0, 6, 12, 18)) == 6


def test_infer_frame_step_survives_gaps():
    """帧 18 被丢，差值是 6,6,12 —— gcd 仍应还原出 6。"""
    assert fd.infer_frame_step(entries(0, 6, 12, 24, 30)) == 6


def test_infer_frame_step_uniform_sampling():
    assert fd.infer_frame_step(entries(0, 1, 2, 3)) == 1


@pytest.mark.parametrize("frame_numbers", [(), (0,)])
def test_infer_frame_step_needs_at_least_two_frames(frame_numbers):
    assert fd.infer_frame_step(entries(*frame_numbers)) is None


def test_infer_frame_step_single_diff_overestimates():
    """已知局限：只有一帧时 gcd 就是那个差值本身。

    这是刻意的行为记录 —— 也正因如此，反推值只当兜底，
    权威来源始终是 _meta.json 里 clean_video.py 记下的值。
    """
    assert fd.infer_frame_step(entries(0, 12)) == 12


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({}, {}),
        ({"videos": None}, {}),
        ({"videos": "bad"}, {}),
        ({"videos": {"1": {"orig_fps": 30.0}}}, {"1": 30.0}),
        ({"videos": {"1": {"orig_fps": 25}}}, {"1": 25.0}),
        ({"videos": {"1": {"orig_fps": 0}}}, {}),
        ({"videos": {"1": {"orig_fps": -5}}}, {}),
        ({"videos": {"1": {"orig_fps": "30"}}}, {}),
        ({"videos": {"1": {"orig_fps": None}}}, {}),
        ({"videos": {"1": {"orig_fps": True}}}, {}),   # bool 是 int 的子类，必须排除
        ({"videos": {"1": "not a dict"}}, {}),
    ],
)
def test_extract_source_fps(payload, expected):
    assert fd.extract_source_fps(payload) == expected


# --------------------------------------------------------------------------
# 端到端：完整生成
# --------------------------------------------------------------------------


def build(images: Path, out: Path, run_format, *extra):
    """跑一次 format_dataset，返回 (退出码, 输出目录)。"""
    rc = run_format("--input", images, "--output", out, *extra)
    return rc, out


def test_full_build_produces_expected_layout(make_images, run_format, tmp_path, read_jsonl):
    images = make_images({"1": [0, 6, 12], "2": [0, 5]})
    out = tmp_path / "ds"
    rc, _ = build(images, out, run_format)
    assert rc == 0

    assert sorted(p.name for p in out.iterdir()) == ["data", "meta", "videos"]
    assert (out / "data" / ".gitkeep").is_file()
    assert (out / "videos" / "chunk-000" / "observation.images.front" / "episode_000000").is_dir()
    assert (out / "videos" / "chunk-000" / "observation.images.front" / "episode_000001").is_dir()

    assert len(list((out / "videos").rglob("*.jpg"))) == 5
    for name in ("info.json", "episodes.jsonl", "frames.jsonl", "tasks.jsonl", "README.md"):
        assert (out / "meta" / name).is_file(), name

    # 没有任务标注，tasks.jsonl 是空文件但必须存在
    assert read_jsonl(out / "meta" / "tasks.jsonl") == []


def test_episode_order_is_natural_not_lexicographic(make_images, run_format, tmp_path, read_jsonl):
    """字典序会把 "10" 排在 "2" 前面。"""
    images = make_images({"10": [0, 6], "2": [0, 6]})
    rc, out = build(images, tmp_path / "ds", run_format)
    assert rc == 0

    episodes = read_jsonl(out / "meta" / "episodes.jsonl")
    assert [e["source_video"] for e in episodes] == ["2", "10"]
    assert [e["episode_index"] for e in episodes] == [0, 1]


def test_frames_keep_source_numbering_and_gaps(make_images, write_meta, run_format, tmp_path, read_jsonl):
    """核心设计：帧文件按位置连续编号，原始帧号与空洞记在 jsonl 里。"""
    images = make_images({"1": [0, 6, 12, 24]})   # 帧 18 被当成模糊帧丢掉了
    write_meta(images, {"1": 30.0})
    rc, out = build(images, tmp_path / "ds", run_format)
    assert rc == 0

    frames = read_jsonl(out / "meta" / "frames.jsonl")
    assert [f["index"] for f in frames] == [0, 1, 2, 3]           # 全局连续
    assert [f["frame_index"] for f in frames] == [0, 1, 2, 3]     # episode 内连续
    assert [f["source_frame"] for f in frames] == [0, 6, 12, 24]  # 原始帧号保留
    # 30fps 下 0.6s 那一帧没了，时间戳必须在 0.4 直接跳到 0.8
    assert [f["timestamp"] for f in frames] == [0.0, 0.2, 0.4, 0.8]


def test_frames_on_disk_follow_positional_naming(make_images, run_format, tmp_path, read_jsonl):
    images = make_images({"1": [0, 6, 24]})
    rc, out = build(images, tmp_path / "ds", run_format)
    assert rc == 0

    frames = read_jsonl(out / "meta" / "frames.jsonl")
    for row in frames:
        path = out / row["path"]
        assert path.is_file()
        assert path.name == f"frame_{row['frame_index']:06d}.jpg"


def test_chunks_size_splits_episodes_across_chunks(make_images, run_format, tmp_path):
    """episode 数超过 chunks_size 时必须分块，且路径里的 chunk 号要跟上。"""
    images = make_images({"1": [0, 6], "2": [0, 6], "3": [0, 6]})
    rc, out = build(images, tmp_path / "ds", run_format, "--chunks-size", 2)
    assert rc == 0

    camera = out / "videos" / "chunk-000" / "observation.images.front"
    assert (camera / "episode_000000").is_dir()
    assert (camera / "episode_000001").is_dir()
    assert (out / "videos" / "chunk-001" / "observation.images.front" / "episode_000002").is_dir()
    assert read_info(out)["total_chunks"] == 2


def test_info_json_core_fields(make_images, run_format, tmp_path):
    images = make_images({"1": [0, 6], "2": [0, 5]})
    rc, out = build(images, tmp_path / "ds", run_format)
    assert rc == 0

    info = read_info(out)
    assert info["codebase_version"] == "v2.0"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 4
    assert info["total_chunks"] == 1
    assert info["chunks_size"] == 1000
    assert info["fps"] == 5
    assert info["splits"] == {"train": "0:2"}
    # 上游此处是 "video"（指向 mp4），我们存 jpg，必须标成 "image"
    assert info["features"]["observation.images.front"]["dtype"] == "image"
    assert info["features"]["observation.images.front"]["shape"][2] == 3


def test_warns_on_mixed_image_sizes(make_images, run_format, tmp_path, caplog):
    """尺寸不一致时取众数并告警，不能默不作声。"""
    images = make_images({"1": [0, 6, 12]}, mixed_sizes=True)
    with caplog.at_level(logging.WARNING):
        rc, out = build(images, tmp_path / "ds", run_format)
    assert rc == 0
    assert "多种图片尺寸" in caplog.text
    assert read_info(out)["features"]["observation.images.front"]["shape"][2] == 3


# --------------------------------------------------------------------------
# 回归测试：本次开发实际踩到的坑
# --------------------------------------------------------------------------


def test_paths_stay_relative_with_absolute_output(make_images, run_format, tmp_path, read_jsonl):
    """回归：--output 传绝对路径时，jsonl 里也必须是相对路径。

    原实现写的是 frame_dir.as_posix()，用相对路径调用时一切正常，
    一旦传绝对路径就会把本机路径写进数据集，产物不可移植。
    """
    images = make_images({"1": [0, 6]})
    out = tmp_path / "ds"          # tmp_path 一定是绝对路径
    assert out.is_absolute()
    rc, _ = build(images, out, run_format)
    assert rc == 0

    for row in read_jsonl(out / "meta" / "frames.jsonl"):
        assert not Path(row["path"]).is_absolute(), row["path"]
        assert row["path"].startswith("videos/"), row["path"]
    for row in read_jsonl(out / "meta" / "episodes.jsonl"):
        assert not Path(row["frame_dir"]).is_absolute(), row["frame_dir"]


def test_fps_prefers_meta_file(make_images, write_meta, run_format, tmp_path):
    images = make_images({"1": [0, 6, 12]})
    write_meta(images, {"1": 30.0})
    rc, out = build(images, tmp_path / "ds", run_format)
    assert rc == 0

    info = read_info(out)
    assert info["extras"]["source_fps_by_video"] == {"1": 30.0}
    assert info["extras"]["source_fps_origin_by_video"] == {"1": fd.FPS_ORIGIN_RECORDED}
    assert info["extras"]["observed_frame_step_by_video"] == {"1": 6}


def test_fps_inferred_when_meta_missing(make_images, run_format, tmp_path, caplog):
    """没有 _meta.json 时从帧号反推，并且必须标明是近似值。"""
    images = make_images({"1": [0, 6, 12, 24]})   # 步长 6
    with caplog.at_level(logging.WARNING):
        rc, out = build(images, tmp_path / "ds", run_format)
    assert rc == 0

    info = read_info(out)
    assert info["extras"]["source_fps_by_video"] == {"1": 30.0}   # 6 * 5
    assert info["extras"]["source_fps_origin_by_video"] == {"1": fd.FPS_ORIGIN_INFERRED}
    assert "反推" in caplog.text

    frames = (out / "meta" / "frames.jsonl").read_text(encoding="utf-8")
    assert '"timestamp": 0.8' in frames      # 24 / 30


def test_fps_falls_back_to_nominal_for_single_frame(make_images, run_format, tmp_path, caplog):
    """只有一帧时无法反推，必须退回名义值并告警说明后果。"""
    images = make_images({"1": [0]})
    with caplog.at_level(logging.WARNING):
        rc, out = build(images, tmp_path / "ds", run_format)
    assert rc == 0

    info = read_info(out)
    assert info["extras"]["source_fps_by_video"] == {"1": None}
    assert info["extras"]["source_fps_origin_by_video"] == {"1": fd.FPS_ORIGIN_NOMINAL}
    assert "偏小" in caplog.text


def test_warns_when_meta_contradicts_frame_numbers(make_images, write_meta, run_format, tmp_path, caplog):
    """_meta.json 说 25fps（步长 5），但帧号全是 6 的倍数 —— 交叉校验必须报出来。"""
    images = make_images({"1": [0, 6, 12, 18]})
    write_meta(images, {"1": 25.0})
    with caplog.at_level(logging.WARNING):
        rc, _ = build(images, tmp_path / "ds", run_format)
    assert rc == 0
    assert "不符" in caplog.text


def test_detects_mixed_frame_rates(make_images, run_format, tmp_path, caplog):
    """各视频步长不同 = 原始帧率不统一，必须主动提示。"""
    images = make_images({"1": [0, 6, 12], "2": [0, 5, 10]})
    with caplog.at_level(logging.INFO):
        rc, _ = build(images, tmp_path / "ds", run_format)
    assert rc == 0
    assert "不统一" in caplog.text


def test_meta_json_is_not_mistaken_for_a_frame(make_images, write_meta, run_format, tmp_path, caplog):
    images = make_images({"1": [0, 6]})
    write_meta(images, {"1": 30.0})
    with caplog.at_level(logging.INFO):     # 汇总行是 INFO 级别
        rc, _ = build(images, tmp_path / "ds", run_format)
    assert rc == 0
    assert "不匹配 0" in caplog.text       # _meta.json 不该被计入「文件名不符合规范」


# --------------------------------------------------------------------------
# 异常输入与幂等
# --------------------------------------------------------------------------


def test_bad_files_are_skipped_and_counted(
    make_images, run_format, tmp_path, tiny_jpegs, read_jsonl, caplog
):
    images = make_images(
        {"1": [0, 6]},
        extra_files={
            "readme.txt": b"not a frame",                    # 文件名不匹配
            "1_frame_0.jpg": tiny_jpegs[0].read_bytes(),     # 与 1_frame_000000.jpg 同帧号
            "1_frame_000099.jpg": b"garbage not a jpeg",     # 内容损坏
        },
    )
    with caplog.at_level(logging.INFO):     # 汇总行是 INFO 级别
        rc, out = build(images, tmp_path / "ds", run_format)
    assert rc == 0
    assert "不匹配 1，重复 1，损坏 1" in caplog.text

    frames = read_jsonl(out / "meta" / "frames.jsonl")
    assert len(frames) == 2          # 只有正常的两帧进了数据集


def test_refuses_to_overwrite_without_force(make_images, run_format, tmp_path):
    images = make_images({"1": [0, 6]})
    out = tmp_path / "ds"
    assert build(images, out, run_format)[0] == 0
    assert build(images, out, run_format)[0] == 1     # 第二次拒绝


def test_force_rerun_reuses_existing_files(make_images, run_format, tmp_path, caplog):
    images = make_images({"1": [0, 6]})
    out = tmp_path / "ds"
    assert build(images, out, run_format)[0] == 0
    with caplog.at_level(logging.INFO):
        rc, _ = build(images, out, run_format, "--force")
    assert rc == 0
    assert "已存在=2" in caplog.text
    assert "链接=0" in caplog.text
    assert "复制=0" in caplog.text


def test_dry_run_creates_nothing(make_images, run_format, tmp_path):
    images = make_images({"1": [0, 6]})
    out = tmp_path / "ds"
    assert build(images, out, run_format, "--dry-run")[0] == 0
    assert not out.exists()


def test_empty_input_directory_fails(tmp_path, run_format):
    images = tmp_path / "images"
    images.mkdir()
    assert build(images, tmp_path / "ds", run_format)[0] == 1


def test_missing_input_directory_fails(tmp_path, run_format):
    assert build(tmp_path / "nope", tmp_path / "ds", run_format)[0] == 1


def test_invalid_chunks_size_fails(make_images, run_format, tmp_path):
    images = make_images({"1": [0, 6]})
    assert build(images, tmp_path / "ds", run_format, "--chunks-size", 0)[0] == 1


# --------------------------------------------------------------------------
# 落盘方式
# --------------------------------------------------------------------------


def test_copy_mode_creates_independent_files(make_images, run_format, tmp_path):
    images = make_images({"1": [0]})
    rc, out = build(images, tmp_path / "ds", run_format, "--link-mode", "copy")
    assert rc == 0

    produced = next((out / "videos").rglob("*.jpg"))
    source = images / "1_frame_000000.jpg"
    assert not produced.samefile(source)
    assert produced.stat().st_ino != source.stat().st_ino


def test_hardlink_mode_shares_the_same_file(make_images, run_format, tmp_path):
    images = make_images({"1": [0]})
    rc, out = build(images, tmp_path / "ds", run_format, "--link-mode", "hardlink")
    assert rc == 0

    produced = next((out / "videos").rglob("*.jpg"))
    source = images / "1_frame_000000.jpg"
    assert produced.samefile(source)
    assert produced.stat().st_nlink == 2


def test_auto_mode_resolves_to_a_usable_mode(make_images, run_format, tmp_path, caplog):
    images = make_images({"1": [0]})
    with caplog.at_level(logging.INFO):
        rc, _ = build(images, tmp_path / "ds", run_format)
    assert rc == 0
    assert "链接方式自动判定为" in caplog.text
