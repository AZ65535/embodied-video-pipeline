"""export_lerobot.py 的测试。

分两层，因为 import lerobot 会连带拉起 torch（好几秒）：

1. **快测试**（默认跑）：不碰 lerobot 的纯逻辑，秒级。
2. **慢测试**（`-m slow`）：真的调 lerobot 写一遍再读回来，锁住官方 API 的用法
   —— 比如忘了 finalize() 会让 parquet 没有 footer。
"""

import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

import export_lerobot as ex
import pipeline_common as pc


@pytest.fixture
def run_export(monkeypatch):
    """按命令行方式调用 export_lerobot.main()，返回退出码。"""

    def _run(*args) -> int:
        monkeypatch.setattr(sys, "argv", ["export_lerobot.py", *[str(a) for a in args]])
        return ex.main()

    return _run


@pytest.fixture
def built_dataset(make_images, write_meta, write_scores, run_format, tmp_path):
    """先用第二步造一个 dataset/ 目录，供第三步消费。

    默认每帧都是清晰帧（分数 250）。要用模糊帧就传 scores=。
    """
    counter = itertools.count()

    def _build(spec: dict, scores: dict = None, **make_kwargs) -> Path:
        images = make_images(spec, **make_kwargs)
        # 原始帧率要与帧号间隔自洽，否则 format_dataset 会报「不符」
        write_meta(images, {video: _implied_fps(frames) for video, frames in spec.items()})
        write_scores(
            images,
            scores
            if scores is not None
            else {video: {frame: 250.0 for frame in frames} for video, frames in spec.items()},
        )
        out = tmp_path / f"ds{next(counter)}"
        assert run_format("--input", images, "--output", out) == 0
        return out

    return _build


def _implied_fps(frame_numbers: list, target_fps: int = 5) -> float:
    """从帧号间隔反推出自洽的原始帧率（单帧时无从推断，给个名义值）。"""
    if len(frame_numbers) < 2:
        return 30.0
    return float((frame_numbers[1] - frame_numbers[0]) * target_fps)


def episodes_of(dataset_dir: Path):
    """把 dataset/ 里的两份清单读进来并分组。"""
    return ex.group_episodes(
        pc.read_jsonl(dataset_dir / "meta" / ex.FRAMES_JSONL),
        pc.read_jsonl(dataset_dir / "meta" / ex.EPISODES_JSONL),
    )


def image_root_of(dataset_dir: Path) -> Path:
    """从 info.json 解析第一步图片目录。"""
    return ex.resolve_image_root(dataset_dir, ex.load_info(dataset_dir))


def break_one_frame(dataset_dir: Path) -> None:
    """弄丢一个帧文件，把流程挡在落盘之前。"""
    next(image_root_of(dataset_dir).rglob("*.jpg")).unlink()


@pytest.fixture
def write_actions():
    """往 dataset/meta/actions/ 写动作侧车。

    spec 形如 ``{"1": 40}``（源视频 → 动作行数）。**第 i 行的每个分量都填 i**，
    于是「取到了第几行」可以直接从数值读出来 —— 这正是这组测试要抓的东西：
    join 错了（比如按抽帧后的位置取而不是 source_frame）不会报错，只会静默配错。
    """

    def _write(dataset_dir: Path, spec: dict, dim: int = 8) -> Path:
        out = dataset_dir / "meta" / "actions"
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in spec.items():
            action = np.tile(
                np.arange(rows, dtype=np.float32).reshape(-1, 1), (1, dim)
            )
            np.savez_compressed(out / f"{name}.npz", action=action, state=action + 100.0)
        return out

    return _write


@pytest.fixture
def fixture_hw(tiny_jpegs) -> tuple:
    """make_images 默认用的那张图的 (height, width)。

    不硬编码尺寸：夹具按字母序排，顺序一变硬编码就会静默失准。
    probe_jpeg 返回的是 (width, height)，注意反过来。
    """
    width, height = pc.probe_jpeg(tiny_jpegs[0])
    return height, width


def read_info(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))


def write_frames_jsonl(dataset_dir: Path, rows: list) -> None:
    path = dataset_dir / "meta" / "frames.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# 读清单
# --------------------------------------------------------------------------


def test_read_jsonl_reads_rows(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert pc.read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="缺少文件"):
        pc.read_jsonl(tmp_path / "nope.jsonl")


def test_read_jsonl_reports_line_number_on_bad_json(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        pc.read_jsonl(path)


# --------------------------------------------------------------------------
# 特征组装
# --------------------------------------------------------------------------


def test_resolve_camera_shape(built_dataset, fixture_hw):
    dataset_dir = built_dataset({"1": [0, 6]})
    assert (
        ex.resolve_camera_shape(read_info(dataset_dir), "observation.images.front") == fixture_hw
    )


def test_resolve_camera_shape_rejects_unknown_camera(built_dataset):
    dataset_dir = built_dataset({"1": [0]})
    with pytest.raises(ValueError, match="没有相机特征"):
        ex.resolve_camera_shape(read_info(dataset_dir), "observation.images.wrist")


def test_build_features_declares_custom_columns():
    """自定义特征必须声明，否则 add_frame 会因为「多余字段」报错。"""
    features = ex.build_features("observation.images.front", 1080, 1920)
    # "video" 而不是 "image"：帧编码成 mp4，不嵌进 parquet。标错的话
    # streaming_encoding 不会生效，也会退回那条会撞 2 GB arrow 限制的路径。
    assert features["observation.images.front"]["dtype"] == "video"
    assert features["observation.images.front"]["shape"] == (1080, 1920, 3)
    # (1,) 是元组不是列表：lerobot 靠 `shape == (1,)` 判定成标量列
    assert features[ex.SOURCE_FRAME_KEY]["shape"] == (1,)
    assert features[ex.SOURCE_FRAME_KEY]["dtype"] == "int64"
    assert features[ex.BLUR_SCORE_KEY]["shape"] == (1,)
    assert features[ex.BLUR_SCORE_KEY]["dtype"] == "float32"
    assert sorted(features) == [
        ex.BLUR_SCORE_KEY,
        "observation.images.front",
        ex.SOURCE_FRAME_KEY,
    ]


def test_build_features_declares_action_and_state_when_asked():
    """带动作的数据集才声明这两列 —— 「有没有动作」是事实，不是选项。"""
    plain = ex.build_features("observation.images.front", 480, 640)
    assert ex.ACTION_KEY not in plain and ex.STATE_KEY not in plain

    with_actions = ex.build_features("observation.images.front", 480, 640, action_dim=8)
    # shape 是 (8,)（元组）不是 (8, 1)：lerobot 按声明逐项比对 add_frame 的数组
    assert with_actions[ex.ACTION_KEY] == {
        "dtype": "float32",
        "shape": (8,),
        "names": None,
    }
    assert with_actions[ex.STATE_KEY]["shape"] == (8,)
    assert with_actions[ex.STATE_KEY]["dtype"] == "float32"


def test_action_dim_requires_uniform_dimensions():
    """lerobot 只声明一份 shape，所以各源视频的维度必须一致。"""
    eight = np.zeros((10, 8), dtype=np.float32)
    seven = np.zeros((10, 7), dtype=np.float32)
    assert ex.action_dim_of({"x": (eight, eight)}) == 8
    assert ex.action_dim_of(None) is None
    with pytest.raises(ValueError, match="维度不一致"):
        ex.action_dim_of({"x": (eight, eight), "y": (seven, seven)})


def test_action_dim_rejects_non_2d():
    flat = np.zeros((10,), dtype=np.float32)
    with pytest.raises(ValueError, match="不是二维"):
        ex.action_dim_of({"x": (flat, flat)})


# --------------------------------------------------------------------------
# 分组与时间轴
# --------------------------------------------------------------------------


def frames_of(*episodes_and_counts, fps=30.0, blur_score=250.0):
    """造一批帧清单行；默认是均匀网格（步长 6，对应 30fps 采样到 5fps）。"""
    rows = []
    index = 0
    for episode_index, count in episodes_and_counts:
        for position in range(count):
            rows.append(
                {
                    "index": index,
                    "episode_index": episode_index,
                    "frame_index": position,
                    "source_frame": position * 6,
                    "blur_score": blur_score,
                    "timestamp": round(position * 6 / fps, 6),
                    "path": f"videos/x/episode_{episode_index:06d}/frame_{position:06d}.jpg",
                }
            )
            index += 1
    return rows


def test_group_episodes_orders_by_index_not_insertion():
    """loader 里的顺序必须稳定，不能受 jsonl 行序影响。"""
    rows = frames_of((1, 2), (0, 2))
    episodes = [{"episode_index": 1, "source_video": "b"}, {"episode_index": 0, "source_video": "a"}]
    grouped = ex.group_episodes(rows, episodes)
    assert [e.episode_index for e in grouped] == [0, 1]
    assert [e.source_video for e in grouped] == ["a", "b"]


def test_group_episodes_sorts_frames_within_episode():
    rows = frames_of((0, 3))
    rows.reverse()   # 故意打乱
    episodes = [{"episode_index": 0, "source_video": "a"}]
    grouped = ex.group_episodes(rows, episodes)
    assert [f["frame_index"] for f in grouped[0].frames] == [0, 1, 2]


def test_group_episodes_rejects_orphan_frame():
    rows = frames_of((7, 1))
    episodes = [{"episode_index": 0, "source_video": "a"}]
    with pytest.raises(ValueError, match="在 episodes.jsonl 中不存在"):
        ex.group_episodes(rows, episodes)


def test_group_episodes_rejects_empty_episode():
    rows = frames_of((0, 1))
    episodes = [{"episode_index": 0}, {"episode_index": 1}]
    with pytest.raises(ValueError, match="一帧都没有"):
        ex.group_episodes(rows, episodes)


# --------------------------------------------------------------------------
# 时间轴：帧是否落在均匀网格上
# --------------------------------------------------------------------------


def grid_episode(fps, source_frames, episode_index=0):
    rows = [
        {
            "episode_index": episode_index,
            "frame_index": position,
            "source_frame": source_frame,
            "blur_score": 250.0,
            "path": f"videos/x/frame_{position:06d}.jpg",
        }
        for position, source_frame in enumerate(source_frames)
    ]
    return ex.EpisodeFrames(
        episode_index=episode_index,
        episode="v1",
        source_video="v1",
        cameras=["front"],
        source_fps=fps,
        frames_by_camera={"front": rows},
        length=len(rows),
    )


class FakeEncoder:
    def __init__(self, dropped=None):
        self._dropped_frames = dropped if dropped is not None else {}


def fake_dataset(encoder):
    """按 lerobot 0.6 的挂载方式造假 dataset：编码器挂在 writer 上。

    0.4.x 是 dataset._streaming_encoder，0.6 起 create() 把它交给 DatasetWriter，
    这个路径变过一次，升级时检查静默降级成了告警（下面第三条用例就是那次事故）。
    """
    writer = type("Writer", (), {"_streaming_encoder": encoder})()
    return type("Dataset", (), {"writer": writer})()


def test_encoder_drop_check_passes_when_nothing_was_dropped(caplog):
    dataset = fake_dataset(FakeEncoder())
    with caplog.at_level(logging.WARNING):
        ex.check_encoder_drops(dataset, "episode 0", logging.getLogger("t"))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_encoder_drop_check_fails_loudly_on_dropped_frames():
    """丢帧意味着视频比 parquet 少几帧，帧网格不再均匀 —— 必须炸，不能只 warning。"""
    dataset = fake_dataset(FakeEncoder({"observation.images.front": 3}))
    with pytest.raises(RuntimeError, match="丢了 3 帧"):
        ex.check_encoder_drops(dataset, "episode 0", logging.getLogger("t"))


def test_encoder_drop_check_warns_when_lerobot_changed(caplog):
    """读不到计数时告警说明「没能核对」，而不是假装核对过了。

    用「连 writer 都没有」来模拟挂载位置又变了 —— 这正是升级到 0.6 时发生的事，
    当时 12 个 episode 全部静默降级成告警（导出照常成功，防护悄悄没了）。
    """
    dataset = type("Dataset", (), {})()
    with caplog.at_level(logging.WARNING):
        ex.check_encoder_drops(dataset, "episode 0", logging.getLogger("t"))
    assert "无法核对" in caplog.text


def test_grid_check_passes_on_a_uniform_grid():
    """正常产物：帧号按步长连续，原始帧率是处理帧率的整数倍。"""
    assert ex.check_frame_grid([grid_episode(30.0, [0, 6, 12, 18])], fps=5) == []
    assert ex.check_frame_grid([grid_episode(25.0, [0, 5, 10, 15])], fps=5) == []


def test_grid_check_tolerates_a_trailing_fps_tail():
    """30.00005 这类带尾数的原始帧率不能误报 —— 偏差远小于一帧。"""
    assert ex.check_frame_grid([grid_episode(30.00005, [0, 6, 12])], fps=5) == []


def test_grid_check_flags_non_contiguous_frame_numbers():
    """帧号有缺口时 lerobot 会按 frame_index 取到错误的时间点。"""
    warnings = ex.check_frame_grid([grid_episode(30.0, [0, 6, 12, 24])], fps=5)
    assert len(warnings) == 1
    assert "帧号间隔不均匀" in warnings[0]
    assert "delta_timestamps" in warnings[0]


def test_grid_check_flags_a_fractional_fps_ratio():
    """24fps 采样到 5fps：步长取 5，但 5/24 秒 ≠ 1/5 秒，时间戳会累积偏差。"""
    warnings = ex.check_frame_grid([grid_episode(24.0, [0, 5, 10, 15])], fps=5)
    assert len(warnings) == 1
    assert "不是处理帧率" in warnings[0]
    assert "8.3 ms" in warnings[0]      # |1/5 - 5/24| * 1000


def test_grid_check_flags_a_step_that_contradicts_the_recorded_fps():
    """帧均匀，但间隔与记录的原始帧率对不上 —— 说明记录和素材不是同一批。"""
    warnings = ex.check_frame_grid([grid_episode(30.0, [0, 5, 10])], fps=5)
    assert len(warnings) == 1
    assert "帧号间隔是 5" in warnings[0]
    assert "应为 6" in warnings[0]


def test_grid_check_skips_the_gap_check_without_source_frame():
    """source_frame 可缺省，缺了就不能拿默认值硬凑间隔 —— 那会误报。"""
    rows = [
        {"episode_index": 0, "frame_index": i, "blur_score": 250.0, "path": f"f{i}.jpg"}
        for i in range(3)
    ]
    episode = ex.EpisodeFrames(
        episode_index=0,
        episode="v1",
        source_video="v1",
        cameras=["front"],
        source_fps=30.0,
        frames_by_camera={"front": rows},
        length=len(rows),
    )
    assert ex.check_frame_grid([episode], fps=5) == []


def test_grid_check_flags_a_missing_source_fps():
    warnings = ex.check_frame_grid([grid_episode(None, [0, 6, 12])], fps=5)
    assert len(warnings) == 1
    assert "没有可用的原始帧率" in warnings[0]


def test_a_single_frame_episode_is_vacuous():
    """只有一帧时无从判断间隔，不该因此报警。"""
    assert ex.check_frame_grid([grid_episode(30.0, [0])], fps=5) == []


# --------------------------------------------------------------------------
# 路径还原与落盘前校验
# --------------------------------------------------------------------------


def test_frame_path_is_relative_to_dataset_dir(tmp_path):
    assert ex.frame_path(tmp_path, {"path": "videos/a/b.jpg"}) == tmp_path / "videos" / "a" / "b.jpg"


def test_frame_path_rejects_absolute_path(tmp_path):
    """回归防线：format_dataset 刻意写相对路径，绝对路径说明产物不可移植。"""
    with pytest.raises(ValueError, match="必须是相对路径"):
        ex.frame_path(tmp_path, {"path": str(tmp_path / "x.jpg")})


def test_load_actions_returns_none_without_sidecar(built_dataset):
    """没有侧车目录 = 这个数据集不带动作，一切照旧。"""
    assert ex.load_actions(built_dataset({"1": [0]})) is None


def test_load_actions_reads_sidecar(built_dataset, write_actions):
    dataset_dir = built_dataset({"1": [0, 6]})
    write_actions(dataset_dir, {"1": 20})
    loaded = ex.load_actions(dataset_dir)
    assert set(loaded) == {"1"}
    action, state = loaded["1"]
    assert action.shape == (20, 8)
    # 夹具把第 i 行填成 i，所以「取到第几行」能从数值读出来
    assert action[3].tolist() == [3.0] * 8
    assert state[3].tolist() == [103.0] * 8


def test_load_actions_rejects_mismatched_arrays(built_dataset):
    """action 与 state 行数不一致说明侧车坏了，不能猜。"""
    dataset_dir = built_dataset({"1": [0]})
    out = dataset_dir / "meta" / "actions"
    out.mkdir(parents=True)
    np.savez_compressed(
        out / "1.npz",
        action=np.zeros((5, 8), dtype=np.float32),
        state=np.zeros((6, 8), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="形状不一致"):
        ex.load_actions(dataset_dir)


def test_load_actions_reports_missing_array(built_dataset):
    dataset_dir = built_dataset({"1": [0]})
    out = dataset_dir / "meta" / "actions"
    out.mkdir(parents=True)
    np.savez_compressed(out / "1.npz", action=np.zeros((5, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="缺少数组"):
        ex.load_actions(dataset_dir)


def test_precheck_requires_actions_for_every_source_video(built_dataset, fixture_hw, write_actions):
    """声明了动作就必须每帧都有，缺一个源视频也要拦下 —— 不填默认值。"""
    dataset_dir = built_dataset({"1": [0, 6], "2": [0, 6]})
    write_actions(dataset_dir, {"1": 20})  # 只给 1，2 没有
    actions = ex.load_actions(dataset_dir)
    _, problems = ex.precheck_frames(
        episodes_of(dataset_dir), image_root_of(dataset_dir), *fixture_hw, actions=actions
    )
    assert problems and all("没有动作文件" in p for p in problems)


def test_precheck_detects_frame_beyond_action_range(built_dataset, fixture_hw, write_actions):
    """帧号超出动作行数 = 动作文件与视频不是同一批，不能猜。"""
    dataset_dir = built_dataset({"1": [0, 6, 12]})  # 最大帧号 12
    write_actions(dataset_dir, {"1": 3})  # 动作只有 3 行
    actions = ex.load_actions(dataset_dir)
    _, problems = ex.precheck_frames(
        episodes_of(dataset_dir), image_root_of(dataset_dir), *fixture_hw, actions=actions
    )
    assert problems and any("超出动作范围" in p for p in problems)


def test_precheck_passes_when_actions_cover_every_frame(built_dataset, fixture_hw, write_actions):
    dataset_dir = built_dataset({"1": [0, 6, 12]})
    write_actions(dataset_dir, {"1": 40})
    actions = ex.load_actions(dataset_dir)
    total, problems = ex.precheck_frames(
        episodes_of(dataset_dir), image_root_of(dataset_dir), *fixture_hw, actions=actions
    )
    assert (total, problems) == (3, [])


def test_precheck_passes_on_good_frames(built_dataset, fixture_hw):
    dataset_dir = built_dataset({"1": [0, 6, 12]})
    height, width = fixture_hw
    total, problems = ex.precheck_frames(
        episodes_of(dataset_dir), image_root_of(dataset_dir), height, width
    )
    assert (total, problems) == (3, [])


def test_precheck_detects_missing_file(built_dataset, fixture_hw):
    dataset_dir = built_dataset({"1": [0, 6]})
    break_one_frame(dataset_dir)
    height, width = fixture_hw
    total, problems = ex.precheck_frames(
        episodes_of(dataset_dir), image_root_of(dataset_dir), height, width
    )
    assert total == 2
    assert len(problems) == 1 and "帧文件不存在" in problems[0]


def test_precheck_detects_size_mismatch(built_dataset, fixture_hw):
    """尺寸不一致必须在这里拦下 —— 放进 lerobot 会在写到一半时炸，留下半成品。"""
    dataset_dir = built_dataset({"1": [0, 6, 12]}, mixed_sizes=True)
    height, width = fixture_hw
    _, problems = ex.precheck_frames(
        episodes_of(dataset_dir), image_root_of(dataset_dir), height, width
    )
    assert problems and all("尺寸不符" in p for p in problems)


# --------------------------------------------------------------------------
# 命令行与幂等
# --------------------------------------------------------------------------


def test_missing_input_fails(tmp_path, run_export):
    assert run_export("--input", tmp_path / "nope", "--output", tmp_path / "out") == 1


def test_empty_input_dir_fails(tmp_path, run_export):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_export("--input", empty, "--output", tmp_path / "out") == 1


def test_refuses_to_overwrite_without_force(built_dataset, run_export, tmp_path, caplog):
    dataset_dir = built_dataset({"1": [0, 6]})
    out = tmp_path / "lr"
    out.mkdir()
    (out / "marker.txt").write_text("existing", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        assert run_export("--input", dataset_dir, "--output", out) == 1
    assert "请加 --force" in caplog.text
    # 拒绝时必须原封不动，不能碰用户已有的目录
    assert (out / "marker.txt").read_text(encoding="utf-8") == "existing"


def test_bad_input_produces_no_output(built_dataset, run_export, tmp_path):
    """帧文件缺失时必须在创建数据集之前就退出，不留下半成品目录。"""
    dataset_dir = built_dataset({"1": [0, 6]})
    break_one_frame(dataset_dir)
    out = tmp_path / "lr"
    assert run_export("--input", dataset_dir, "--output", out) == 1
    assert not out.exists()


def test_force_with_bad_input_keeps_existing_output(built_dataset, run_export, tmp_path, caplog):
    """回归：--force 会整个删掉旧输出，所以帧校验必须跑在删除之前。

    原先顺序是先删后校验，输入有问题时用户会先丢掉上一版数据集再收到失败。
    """
    dataset_dir = built_dataset({"1": [0, 6]})
    break_one_frame(dataset_dir)
    out = tmp_path / "lr"
    out.mkdir()
    (out / "precious.txt").write_text("previous dataset", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        assert run_export("--input", dataset_dir, "--output", out, "--force") == 1
    assert (out / "precious.txt").read_text(encoding="utf-8") == "previous dataset"


def test_fps_mismatch_warns(built_dataset, run_export, tmp_path, caplog):
    """帧率不一致要告警。

    同样弄丢一个帧文件把流程挡住，避免真的调用 lerobot（那要好几秒）。
    """
    dataset_dir = built_dataset({"1": [0, 6]})
    break_one_frame(dataset_dir)
    with caplog.at_level(logging.WARNING):
        assert run_export("--input", dataset_dir, "--output", tmp_path / "lr", "--fps", 10) == 1
    assert "不一致" in caplog.text


# --------------------------------------------------------------------------
# 端到端（慢）
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_is_loadable_by_lerobot(built_dataset, run_export, tmp_path, caplog, fixture_hw):
    """真的写一遍再读回来。最容易漏的是 finalize()：不调用就不写 parquet footer，
    文件看起来在、实际加载不了。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_dir = built_dataset(
        {"1": [0, 6, 12, 18], "2": [0, 5]},
        scores={"1": {0: 250.0, 6: 42.0, 12: 99.9, 18: 300.0}, "2": {0: 180.0, 5: 250.0}},
    )
    out = tmp_path / "lr"
    assert run_export("--input", dataset_dir, "--output", out) == 0

    ds = LeRobotDataset("local/embodied-videos", root=out)
    assert len(ds) == 6
    assert ds.num_episodes == 2

    # 原始帧号必须原样活下来
    assert [int(ds[i]["source_frame"]) for i in range(6)] == [0, 6, 12, 18, 0, 5]
    # 模糊分数原样活下来，下游据此自行筛选
    assert [float(ds[i]["blur_score"]) for i in range(6)] == pytest.approx(
        [250.0, 42.0, 99.9, 300.0, 180.0, 250.0], abs=1e-4
    )
    # 帧按 1/fps 等间隔，timestamp 就是 frame_index / fps
    assert [float(ds[i]["timestamp"]) for i in range(4)] == pytest.approx(
        [0.0, 0.2, 0.4, 0.6], abs=1e-6
    )
    height, width = fixture_hw
    assert tuple(ds[0]["observation.images.front"].shape) == (3, height, width)


@pytest.mark.slow
def test_delta_timestamps_lands_on_the_requested_times(built_dataset, run_export, tmp_path):
    """整个改动的目的：帧落在均匀网格上，时序窗口才取得到正确的时间点。

    改这个设计之前，帧因为丢模糊帧而不均匀，delta_timestamps 会静默取到别的时刻。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_dir = built_dataset({"1": [0, 6, 12, 18, 24]})
    out = tmp_path / "lr"
    assert run_export("--input", dataset_dir, "--output", out) == 0

    # 注意 timestamp 不能放进 delta_timestamps：视频型数据集解码时要读标量
    # item["timestamp"]，开了窗口就变成多元素张量，lerobot 会在这里崩。
    # 这不影响用 source_frame 验证窗口取位。
    windowed = LeRobotDataset(
        "local/embodied-videos",
        root=out,
        delta_timestamps={
            "observation.images.front": [-0.2, 0.0, 0.2],
            "source_frame": [-0.2, 0.0, 0.2],
        },
    )
    frame = windowed[2]
    # 请求「当前帧前后各 0.2 秒」（fps=5，即前后各一帧）——
    # 取到的正是相邻帧。帧不均匀时这里会取到别的帧。
    assert frame["source_frame"].tolist() == [6, 12, 18]
    # 三帧都是真实取到的，没有靠填充补齐
    assert frame["source_frame_is_pad"].tolist() == [False, False, False]
    # 单帧字段仍在，说明窗口只作用于列进 delta_timestamps 的那些键
    assert float(frame["timestamp"]) == pytest.approx(0.4, abs=1e-6)


@pytest.mark.slow
def test_end_to_end_with_actions_lands_on_the_right_frames(
    built_dataset, run_export, tmp_path, write_actions
):
    """带动作写一遍再读回来，逐帧核对数值。

    join 错了不会报错，只会让模型学到错的东西 —— 所以这里不满足于「有 action 列」，
    而是看**每一帧拿到的是不是它自己那一帧的动作**。夹具把侧车的第 i 行填成 i，
    于是每帧的 action 应当恰好等于它的 source_frame。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_dir = built_dataset({"1": [0, 6, 12]})
    write_actions(dataset_dir, {"1": 40})
    out = tmp_path / "lr"
    assert run_export("--input", dataset_dir, "--output", out) == 0

    ds = LeRobotDataset("local/embodied-videos", root=out)
    source_frames = [int(ds[i]["source_frame"]) for i in range(3)]
    assert source_frames == [0, 6, 12]
    for index, source_frame in enumerate(source_frames):
        assert ds[index]["action"].numpy() == pytest.approx([source_frame] * 8)
        assert ds[index]["observation.state"].numpy() == pytest.approx(
            [source_frame + 100.0] * 8
        )


@pytest.mark.slow
def test_export_without_actions_stays_action_free(built_dataset, run_export, tmp_path):
    """没有侧车时不能凭空多出 action 列 —— 旧行为要原样保留。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_dir = built_dataset({"1": [0, 6]})
    out = tmp_path / "lr"
    assert run_export("--input", dataset_dir, "--output", out) == 0

    ds = LeRobotDataset("local/embodied-videos", root=out)
    assert ex.ACTION_KEY not in ds.meta.features
    assert ex.STATE_KEY not in ds.meta.features


@pytest.mark.slow
def test_force_rebuild_is_clean(built_dataset, run_export, tmp_path):
    """--force 是整目录重建，上一轮的残渣不能留在里面。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_dir = built_dataset({"1": [0, 6]})
    out = tmp_path / "lr"
    assert run_export("--input", dataset_dir, "--output", out) == 0

    (out / "stale.txt").write_text("from previous run", encoding="utf-8")
    assert run_export("--input", dataset_dir, "--output", out, "--force") == 0
    assert not (out / "stale.txt").exists()
    assert len(LeRobotDataset("local/embodied-videos", root=out)) == 2


# --------------------------------------------------------------------------
# 多相机：相机解析、分组、预检
# --------------------------------------------------------------------------


def multi_episode(
    episode_index=0,
    episode="ep1",
    cameras=("a", "b"),
    source_frames=(0, 6),
    source_video="a",
):
    frames_by_camera = {}
    for camera in cameras:
        frames_by_camera[camera] = [
            {
                "episode_index": episode_index,
                "frame_index": position,
                "source_frame": source_frame,
                "timestamp": round(position * 6 / 30.0, 6),
                "camera": camera,
                "blur_score": 250.0,
                "path": f"{episode}_{camera}_{position}.jpg",
            }
            for position, source_frame in enumerate(source_frames)
        ]
    return ex.EpisodeFrames(
        episode_index=episode_index,
        episode=episode,
        source_video=source_video,
        cameras=list(cameras),
        source_fps=30.0,
        frames_by_camera=frames_by_camera,
        length=len(source_frames),
    )


def test_resolve_cameras_uses_declared_order_and_shapes():
    info = {
        "features": {
            "observation.images.b": {"shape": [10, 20, 3]},
            "observation.images.a": {"shape": [30, 40, 3]},
        },
        "extras": {"cameras": ["b", "a"]},
    }
    cameras = ex.resolve_cameras(info)
    assert [camera.name for camera in cameras] == ["b", "a"]
    assert [(camera.height, camera.width) for camera in cameras] == [(10, 20), (30, 40)]


def test_resolve_cameras_falls_back_to_feature_keys():
    info = {
        "features": {
            "observation.images.cam10": {"shape": [8, 8, 3]},
            "observation.images.cam2": {"shape": [8, 8, 3]},
        }
    }
    assert [camera.name for camera in ex.resolve_cameras(info)] == ["cam2", "cam10"]


def test_resolve_cameras_rejects_camera_override_for_multi_camera_info():
    info = {
        "features": {
            "observation.images.a": {"shape": [8, 8, 3]},
            "observation.images.b": {"shape": [8, 8, 3]},
        },
        "extras": {"cameras": ["a", "b"]},
    }
    with pytest.raises(ValueError, match="多个相机"):
        ex.resolve_cameras(info, override="a")


def test_build_features_multi_camera_has_blur_vector():
    cameras = [
        ex.CameraSpec("a", "observation.images.a", 48, 64),
        ex.CameraSpec("b", "observation.images.b", 24, 32),
    ]
    features = ex.build_features(cameras)
    assert features["observation.images.a"]["shape"] == (48, 64, 3)
    assert features["observation.images.b"]["shape"] == (24, 32, 3)
    assert features[ex.BLUR_SCORE_KEY]["shape"] == (2,)
    assert features[ex.BLUR_SCORE_KEY]["names"] == ["a", "b"]


def test_group_episodes_aggregates_camera_rows_by_position():
    rows = []
    for camera in ("b", "a"):
        for position, source_frame in enumerate((0, 6, 12)):
            rows.append(
                {
                    "episode_index": 0,
                    "frame_index": position,
                    "source_frame": source_frame,
                    "timestamp": position * 0.2,
                    "camera": camera,
                    "blur_score": 250.0,
                    "path": f"x_{camera}_{position}.jpg",
                }
            )
    episodes = [
        {
            "episode_index": 0,
            "episode": "ep1",
            "source_video": "a",
            "cameras": ["a", "b"],
            "length": 3,
            "source_fps": 30.0,
        }
    ]
    grouped = ex.group_episodes(rows, episodes)
    assert len(grouped) == 1
    episode = grouped[0]
    assert episode.episode == "ep1"
    assert episode.cameras == ["a", "b"]
    assert episode.length == 3
    assert [row["source_frame"] for row in episode.frames_by_camera["a"]] == [0, 6, 12]
    assert [row["source_frame"] for row in episode.frames_by_camera["b"]] == [0, 6, 12]


def test_group_episodes_rejects_desynchronized_cameras():
    rows = [
        {
            "episode_index": 0,
            "frame_index": 0,
            "source_frame": 0,
            "camera": "a",
            "blur_score": 1.0,
            "path": "a0.jpg",
        },
        {
            "episode_index": 0,
            "frame_index": 0,
            "source_frame": 3,
            "camera": "b",
            "blur_score": 1.0,
            "path": "b0.jpg",
        },
    ]
    episodes = [{"episode_index": 0, "cameras": ["a", "b"], "length": 1}]
    with pytest.raises(ValueError, match="不同步"):
        ex.group_episodes(rows, episodes)


def test_group_episodes_rejects_missing_declared_camera():
    rows = [
        {
            "episode_index": 0,
            "frame_index": 0,
            "source_frame": 0,
            "camera": "a",
            "blur_score": 1.0,
            "path": "a0.jpg",
        }
    ]
    episodes = [{"episode_index": 0, "cameras": ["a", "b"], "length": 1}]
    with pytest.raises(ValueError, match="缺少.*b"):
        ex.group_episodes(rows, episodes)


def test_group_episodes_legacy_rows_without_camera_use_first_camera():
    rows = [
        {
            "episode_index": 0,
            "frame_index": 0,
            "source_frame": 0,
            "blur_score": 1.0,
            "path": "a0.jpg",
        }
    ]
    grouped = ex.group_episodes(
        rows, [{"episode_index": 0, "source_video": "v", "length": 1}]
    )
    assert grouped[0].cameras == ["front"]
    assert grouped[0].episode == "v"
    assert grouped[0].has_episode_key is False


def test_action_key_for_new_manifest_does_not_fall_back_to_source_video():
    episode = multi_episode(source_video="a")
    actions = {"a": (np.zeros((10, 8), dtype=np.float32),) * 2}
    assert ex.action_key_for(episode, actions) is None
    problem = ex.action_problem(episode, episode.frames[0], actions)
    assert "没有动作文件" in problem


def test_action_key_for_legacy_manifest_falls_back_to_source_video():
    episode = multi_episode(source_video="a")
    episode.has_episode_key = False
    state = np.zeros((10, 8), dtype=np.float32)
    actions = {"a": (state, state)}
    assert ex.action_key_for(episode, actions) == "a"
    assert ex.action_problem(episode, episode.frames[0], actions) is None


def test_precheck_frames_checks_every_camera(tmp_path, tiny_jpegs):
    root = tmp_path / "images"
    root.mkdir()
    width, height = pc.probe_jpeg(tiny_jpegs[0])
    for camera in ("a", "b"):
        (root / f"ep1_{camera}_0.jpg").write_bytes(tiny_jpegs[0].read_bytes())
    episode = multi_episode(source_frames=(0,))
    for camera in ("a", "b"):
        episode.frames_by_camera[camera][0]["path"] = f"ep1_{camera}_0.jpg"
    cameras = [
        ex.CameraSpec("a", "observation.images.a", height, width),
        ex.CameraSpec("b", "observation.images.b", height, width),
    ]
    total, problems = ex.precheck_frames([episode], root, cameras)
    assert (total, problems) == (1, [])

    (root / "ep1_b_0.jpg").unlink()
    total, problems = ex.precheck_frames([episode], root, cameras)
    assert total == 1
    assert problems and "相机 b" in problems[0] and "不存在" in problems[0]


def test_precheck_frames_detects_missing_camera_in_episode(tmp_path, tiny_jpegs):
    root = tmp_path / "images"
    root.mkdir()
    width, height = pc.probe_jpeg(tiny_jpegs[0])
    (root / "ep1_a_0.jpg").write_bytes(tiny_jpegs[0].read_bytes())
    episode = multi_episode(cameras=("a",), source_frames=(0,))
    episode.frames_by_camera["a"][0]["path"] = "ep1_a_0.jpg"
    cameras = [
        ex.CameraSpec("a", "observation.images.a", height, width),
        ex.CameraSpec("b", "observation.images.b", height, width),
    ]
    total, problems = ex.precheck_frames([episode], root, cameras)
    assert total == 0
    assert problems and "缺少 ['b']" in problems[0]
