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

import pytest

import export_lerobot as ex


@pytest.fixture
def run_export(monkeypatch):
    """按命令行方式调用 export_lerobot.main()，返回退出码。"""

    def _run(*args) -> int:
        monkeypatch.setattr(sys, "argv", ["export_lerobot.py", *[str(a) for a in args]])
        return ex.main()

    return _run


@pytest.fixture
def built_dataset(make_images, run_format, tmp_path):
    """先用第二步造一个 dataset/ 目录，供第三步消费。"""
    counter = itertools.count()

    def _build(spec: dict, **make_kwargs) -> Path:
        images = make_images(spec, **make_kwargs)
        out = tmp_path / f"ds{next(counter)}"
        assert run_format("--input", images, "--output", out) == 0
        return out

    return _build


def episodes_of(dataset_dir: Path):
    """把 dataset/ 里的两份清单读进来并分组。"""
    return ex.group_episodes(
        ex.load_jsonl(dataset_dir / "meta" / ex.FRAMES_JSONL),
        ex.load_jsonl(dataset_dir / "meta" / ex.EPISODES_JSONL),
    )


def break_one_frame(dataset_dir: Path) -> None:
    """弄丢一个帧文件，把流程挡在落盘之前。"""
    next((dataset_dir / "videos").rglob("*.jpg")).unlink()


@pytest.fixture
def fixture_hw(tiny_jpegs) -> tuple:
    """make_images 默认用的那张图的 (height, width)。

    不硬编码尺寸：夹具按字母序排，顺序一变硬编码就会静默失准。
    probe_jpeg 返回的是 (width, height)，注意反过来。
    """
    width, height = ex.format_dataset.probe_jpeg(tiny_jpegs[0])
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


def test_load_jsonl_reads_rows(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert ex.load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_load_jsonl_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="请先跑 format_dataset.py"):
        ex.load_jsonl(tmp_path / "nope.jsonl")


def test_load_jsonl_reports_line_number_on_bad_json(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        ex.load_jsonl(path)


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


def test_build_features_declares_source_frame():
    """自定义特征必须声明，否则 add_frame 会因为「多余字段」报错。

    真实时间轴靠改写 lerobot 的 timestamp 列承载（见 inject_real_timestamp），
    所以这里不该再有第二个时间戳列。
    """
    features = ex.build_features("observation.images.front", 1080, 1920)
    assert features["observation.images.front"]["dtype"] == "image"
    assert features["observation.images.front"]["shape"] == (1080, 1920, 3)
    # (1,) 是元组不是列表：lerobot 靠 `shape == (1,)` 判定成标量列
    assert features[ex.SOURCE_FRAME_KEY]["shape"] == (1,)
    assert features[ex.SOURCE_FRAME_KEY]["dtype"] == "int64"
    assert sorted(features) == ["observation.images.front", ex.SOURCE_FRAME_KEY]


# --------------------------------------------------------------------------
# 分组与时间戳
# --------------------------------------------------------------------------


def frames_of(*episodes_and_counts):
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
                    "timestamp": round(position * 0.2, 6),
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


def test_frame_timestamp_prefers_recorded_value():
    """真实时间轴来自 frames.jsonl，不能被名义值覆盖。"""
    assert ex.frame_timestamp({"timestamp": 0.8}, position=1, fps=5) == 0.8


def test_frame_timestamp_falls_back_to_nominal():
    """手工构造的清单可能缺 timestamp，兜底成 position/fps。"""
    assert ex.frame_timestamp({}, position=3, fps=5) == 0.6


def test_frame_timestamp_ignores_bool():
    """bool 是 int 的子类，必须排除，否则 True 会变成 1.0。"""
    assert ex.frame_timestamp({"timestamp": True}, position=2, fps=5) == 0.4


class FakeDataset:
    """只带 episode_buffer 的假数据集，用来单测时间戳注入。"""

    def __init__(self, timestamp=None):
        self.episode_buffer = {} if timestamp is None else {"timestamp": timestamp}

    @classmethod
    def without_buffer(cls):
        return cls.__new__(cls)     # 连 episode_buffer 属性都没有


def test_inject_real_timestamp_overwrites_only_the_last_entry():
    """改的是刚追加的那一帧，前面已经写好的不能被碰。"""
    dataset = FakeDataset(timestamp=[0.0, 0.2])
    ex.inject_real_timestamp(dataset, 0.8)
    assert dataset.episode_buffer["timestamp"] == [0.0, 0.8]


@pytest.mark.parametrize(
    "dataset_factory, expected",
    [
        (FakeDataset.without_buffer, "episode_buffer 结构变了"),
        (lambda: FakeDataset(timestamp={"not": "a list"}), "episode_buffer 结构变了"),
        (lambda: FakeDataset(timestamp=[]), "是空的"),
    ],
)
def test_inject_real_timestamp_fails_loudly(dataset_factory, expected):
    """lerobot 内部结构一变就必须炸，不能静默退回标称时间轴。"""
    with pytest.raises(RuntimeError, match=expected):
        ex.inject_real_timestamp(dataset_factory(), 0.8)


# --------------------------------------------------------------------------
# 路径还原与落盘前校验
# --------------------------------------------------------------------------


def test_frame_path_is_relative_to_dataset_dir(tmp_path):
    assert ex.frame_path(tmp_path, {"path": "videos/a/b.jpg"}) == tmp_path / "videos" / "a" / "b.jpg"


def test_frame_path_rejects_absolute_path(tmp_path):
    """回归防线：format_dataset 刻意写相对路径，绝对路径说明产物不可移植。"""
    with pytest.raises(ValueError, match="必须是相对路径"):
        ex.frame_path(tmp_path, {"path": str(tmp_path / "x.jpg")})


def test_precheck_passes_on_good_frames(built_dataset, fixture_hw):
    dataset_dir = built_dataset({"1": [0, 6, 12]})
    height, width = fixture_hw
    total, problems = ex.precheck_frames(episodes_of(dataset_dir), dataset_dir, height, width)
    assert (total, problems) == (3, [])


def test_precheck_detects_missing_file(built_dataset, fixture_hw):
    dataset_dir = built_dataset({"1": [0, 6]})
    break_one_frame(dataset_dir)
    height, width = fixture_hw
    total, problems = ex.precheck_frames(episodes_of(dataset_dir), dataset_dir, height, width)
    assert total == 2
    assert len(problems) == 1 and "帧文件不存在" in problems[0]


def test_precheck_detects_size_mismatch(built_dataset, fixture_hw):
    """尺寸不一致必须在这里拦下 —— 放进 lerobot 会在写到一半时炸，留下半成品。"""
    dataset_dir = built_dataset({"1": [0, 6, 12]}, mixed_sizes=True)
    height, width = fixture_hw
    _, problems = ex.precheck_frames(episodes_of(dataset_dir), dataset_dir, height, width)
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

    dataset_dir = built_dataset({"1": [0, 6, 12, 24], "2": [0, 5]})
    out = tmp_path / "lr"
    assert run_export("--input", dataset_dir, "--output", out) == 0

    ds = LeRobotDataset("local/embodied-videos", root=out)
    assert len(ds) == 6
    assert ds.num_episodes == 2

    # 原始帧号必须原样活下来
    assert [int(ds[i]["source_frame"]) for i in range(6)] == [0, 6, 12, 24, 0, 5]
    # 帧 18 被当模糊帧丢了，timestamp 必须从 0.4 跳到 0.8。
    # 没有 inject_real_timestamp() 的话这里会是 0.6（frame_index/fps）。
    assert [float(ds[i]["timestamp"]) for i in range(4)] == pytest.approx(
        [0.0, 0.2, 0.4, 0.8], abs=1e-6
    )
    height, width = fixture_hw
    assert tuple(ds[0]["observation.images.front"].shape) == (3, height, width)


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
