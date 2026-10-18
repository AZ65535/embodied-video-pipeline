"""ui_helpers.py 的单元测试（不需要启动 Gradio / 跑真实管线）。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import calibrate_blur as cb
import run_pipeline as rp
import ui_helpers as h


# ---------------------------------------------------------------------------
# 配置与命令规划
# ---------------------------------------------------------------------------


def make_config(tmp_path, **overrides):
    values = dict(
        input_dir=str(tmp_path / "videos"),
        output_base=str(tmp_path / "out"),
        dataset_dir=str(tmp_path / "ds"),
        lerobot_dir=str(tmp_path / "ld"),
    )
    values.update(overrides)
    return h.UiConfig(**values)


def make_step(number, path, title="step", forceable=True):
    return rp.Step(
        number=number,
        title=title,
        cmd=["python", "script.py"],
        done_markers=(path,),
        forceable=forceable,
    )


def test_config_from_form_maps_checkboxes_and_mode():
    config = h.config_from_form(
        "in",
        "out",
        "ds",
        "ld",
        True,
        False,
        True,
        5,
        100,
        60,
        "front",
        "task",
        "h264",
        "local/x",
        "force",
    )
    assert config.selected_steps == (1, 3)
    assert config.mode == "force"
    assert config.fps == 5
    assert config.task == "task"


def test_config_from_form_maps_invalid_numbers_to_validation_errors():
    config = h.config_from_form(
        "in", "out", "ds", "ld", True, True, True,
        None, "bad", None, "front", "", "h264", "local/x", "normal",
    )
    assert config.fps == 0
    assert config.blur_threshold == 0.0
    assert config.max_episode_seconds == -1.0


def test_build_plan_reuses_run_pipeline_commands(tmp_path):
    (tmp_path / "videos").mkdir()
    config = make_config(tmp_path, selected_steps=(1,))
    plan = h.build_plan(config)

    assert [step.number for step in plan.to_run] == [1]
    cmd = plan.to_run[0].cmd
    assert str(tmp_path / "videos") in cmd
    assert str(tmp_path / "out") in cmd
    assert cmd[cmd.index("--fps") + 1] == "5"


def test_force_applies_only_to_steps_2_and_3(tmp_path):
    config = make_config(tmp_path, selected_steps=(1, 2, 3), mode="force")
    plan = h.build_plan(config)

    assert "--force" not in plan.steps[0].cmd
    assert "--force" in plan.steps[1].cmd
    assert "--force" in plan.steps[2].cmd


def test_step_3_does_not_get_fps(tmp_path):
    config = make_config(tmp_path)
    plan = h.build_plan(config)
    assert "--fps" not in plan.steps[2].cmd


def test_normal_mode_blocks_existing_dataset_step(tmp_path):
    (tmp_path / "out" / "images").mkdir(parents=True)
    meta = tmp_path / "ds" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}", encoding="utf-8")

    config = make_config(tmp_path, selected_steps=(2,))
    plan = h.build_plan(config)

    assert [step.number for step in plan.blocked] == [2]
    assert plan.to_run == []
    assert "skip-existing" in h.render_plan_preview(plan)


def test_skip_existing_skips_completed_steps(tmp_path):
    images = tmp_path / "out" / "images"
    images.mkdir(parents=True)
    (images / "_meta.json").write_text("{}", encoding="utf-8")
    (images / "_scores.jsonl").write_text("", encoding="utf-8")
    meta = tmp_path / "ds" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}", encoding="utf-8")

    config = make_config(tmp_path, selected_steps=(1, 2), mode="skip-existing")
    plan = h.build_plan(config)

    assert [step.number for step in plan.skipped] == [1, 2]
    assert plan.to_run == []


def test_step_2_without_images_is_an_error(tmp_path):
    config = make_config(tmp_path, selected_steps=(2,))
    plan = h.build_plan(config)
    assert any("第 2 步需要图片目录" in message for message in plan.errors)


def test_step_3_without_dataset_files_is_an_error(tmp_path):
    config = make_config(tmp_path, selected_steps=(3,))
    plan = h.build_plan(config)
    assert any("第 3 步需要" in message for message in plan.errors)


def test_step_1_and_2_together_do_not_require_existing_images(tmp_path):
    (tmp_path / "videos").mkdir()
    config = make_config(tmp_path, selected_steps=(1, 2))
    plan = h.build_plan(config)

    assert plan.errors == []
    assert [step.number for step in plan.to_run] == [1, 2]


def test_step_2_and_3_together_do_not_require_existing_dataset(tmp_path):
    (tmp_path / "out" / "images").mkdir(parents=True)
    config = make_config(tmp_path, selected_steps=(2, 3))
    plan = h.build_plan(config)

    assert plan.errors == []
    assert [step.number for step in plan.to_run] == [2, 3]


# ---------------------------------------------------------------------------
# 数据集摘要
# ---------------------------------------------------------------------------


def build_dataset(tmp_path, malformed_frame=False):
    dataset = tmp_path / "ds"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    info = {
        "total_episodes": 2,
        "total_frames": 5,
        "fps": 5,
        "features": {
            "observation.images.front": {"shape": [48, 64, 3]},
        },
        "extras": {
            "blur_score_threshold": 100,
            "image_root": "../out/images",
            "camera_key": "observation.images.front",
        },
    }
    (meta / "info.json").write_text(
        json.dumps(info), encoding="utf-8"
    )
    episodes = [
        {
            "episode_index": 0,
            "source_video": "1",
            "length": 3,
            "source_fps_origin": "_meta.json",
        },
        {
            "episode_index": 1,
            "source_video": "2",
            "length": 2,
            "source_fps_origin": "_meta.json",
        },
    ]
    (meta / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in episodes) + "\n", encoding="utf-8"
    )
    frame_lines = [
        json.dumps({"is_blurry": True, "blur_score": 10.0}),
        json.dumps({"is_blurry": False, "blur_score": 200.0}),
        json.dumps({"is_blurry": False, "blur_score": 300.0}),
        json.dumps({"is_blurry": False, "blur_score": None}),
        json.dumps({"is_blurry": False, "blur_score": 400.0}),
    ]
    if malformed_frame:
        frame_lines.append("not json")
    (meta / "frames.jsonl").write_text(
        "\n".join(frame_lines) + "\n", encoding="utf-8"
    )
    return dataset


def test_summary_reads_info_episodes_and_frames(tmp_path):
    dataset = build_dataset(tmp_path)
    paths = h.ResolvedPaths(
        input_dir=tmp_path / "videos",
        output_base=tmp_path / "out",
        images_dir=tmp_path / "out" / "images",
        dataset_dir=dataset,
        lerobot_dir=tmp_path / "ld",
    )
    steps = [
        make_step(1, tmp_path / "out" / "images" / "_meta.json", forceable=False),
        make_step(2, dataset / "meta" / "info.json"),
        make_step(3, tmp_path / "ld" / "meta" / "info.json"),
    ]
    summary = h.summarize_paths(paths, steps)

    assert summary.total_episodes == 2
    assert summary.total_frames == 5
    assert summary.fps == 5
    assert summary.blur_threshold == 100
    assert summary.camera_key == "observation.images.front"
    assert summary.shape == (48, 64)
    assert summary.image_root == "../out/images"
    assert dict(summary.episode_counts) == {"1": 1, "2": 1}
    assert dict(summary.episode_frames) == {"1": 3, "2": 2}
    assert dict(summary.fps_origins) == {"_meta.json": 2}
    assert summary.frames_rows == 5
    assert summary.blurry_frames == 1
    assert summary.missing_score_frames == 1
    assert [artifact.done for artifact in summary.artifacts] == [False, True, False]
    assert "数据集摘要" in h.render_summary(summary)


def test_summary_counts_malformed_frame_lines(tmp_path):
    dataset = build_dataset(tmp_path, malformed_frame=True)
    paths = h.ResolvedPaths(
        input_dir=tmp_path,
        output_base=tmp_path,
        images_dir=tmp_path / "images",
        dataset_dir=dataset,
        lerobot_dir=tmp_path / "ld",
    )
    summary = h.summarize_paths(paths, [])

    assert summary.frames_rows == 5
    assert summary.malformed_frame_lines == 1
    assert any("无法解析" in message for message in summary.warnings)


def test_summary_tolerates_missing_info(tmp_path):
    meta = tmp_path / "ds" / "meta"
    meta.mkdir(parents=True)
    (meta / "episodes.jsonl").write_text(
        json.dumps(
            {
                "source_video": "1",
                "length": 3,
                "source_fps_origin": "nominal",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = h.ResolvedPaths(
        input_dir=tmp_path,
        output_base=tmp_path,
        images_dir=tmp_path / "images",
        dataset_dir=tmp_path / "ds",
        lerobot_dir=tmp_path / "ld",
    )
    summary = h.summarize_paths(paths, [])

    assert summary.info_error
    assert dict(summary.episode_counts) == {"1": 1}
    assert "尚未生成" in h.render_summary(summary) or "无法解析" in h.render_summary(summary)


def test_dir_usage_respects_file_limit(tmp_path):
    root = tmp_path / "many"
    root.mkdir()
    for index in range(5):
        (root / f"{index}.bin").write_bytes(b"x" * 10)

    usage = h.dir_usage(root, max_files=3)

    assert usage.exists
    assert usage.file_count == 3
    assert usage.truncated


# ---------------------------------------------------------------------------
# 模糊复核
# ---------------------------------------------------------------------------


def make_review_data(tmp_path):
    return h.ReviewData(
        images_dir=tmp_path,
        scores_path=tmp_path / "_scores.jsonl",
        results=[
            cb.VideoScores(
                name="1", frame_indices=[0, 1, 2], scores=[50.0, 100.0, 300.0]
            ),
            cb.VideoScores(name="2", frame_indices=[0, 1], scores=[99.0, 100.0]),
        ],
        total_frames=5,
    )


def test_threshold_is_strict_and_critical_band_is_inclusive(tmp_path):
    selection = h.select_review_frames(
        make_review_data(tmp_path), 100, 3.0, h.ALL_VIDEOS, 10
    )

    # 50 和 99 低于阈值；100 不算低于，但算临界；300 是临界上界
    assert selection.total_below == 2
    assert selection.total_critical == 3
    assert selection.total_frames == 5


def test_video_filter_limits_gallery(tmp_path):
    selection = h.select_review_frames(make_review_data(tmp_path), 100, 3.0, "1", 10)

    assert selection.filtered_frames == 3
    assert selection.filtered_below == 1
    assert all(frame.video == "1" for frame in selection.selected)


def test_sampling_covers_min_and_max(tmp_path):
    data = h.ReviewData(
        images_dir=tmp_path,
        scores_path=tmp_path / "_scores.jsonl",
        results=[
            cb.VideoScores(
                name="1", frame_indices=list(range(100)), scores=[float(i) for i in range(100)]
            )
        ],
        total_frames=100,
    )
    selection = h.select_review_frames(data, 50, 3.0, h.ALL_VIDEOS, 8)

    assert selection.selected[0].score == 0.0
    assert selection.selected[-1].score == 99.0
    assert [frame.score for frame in selection.selected] == sorted(
        frame.score for frame in selection.selected
    )


def test_gallery_uses_thumbnail_cache_and_skips_missing(tmp_path, fixtures_dir):
    images = tmp_path / "images"
    images.mkdir()
    source = fixtures_dir / sorted(p.name for p in fixtures_dir.glob("tiny_*.jpg"))[0]
    (images / "1_frame_000000.jpg").write_bytes(source.read_bytes())

    data = h.ReviewData(
        images_dir=images,
        scores_path=images / "_scores.jsonl",
        results=[
            cb.VideoScores(name="1", frame_indices=[0, 1], scores=[50.0, 150.0])
        ],
        total_frames=2,
    )
    selection = h.select_review_frames(data, 100, 3.0, h.ALL_VIDEOS, 10)
    cache = tmp_path / "thumbs"
    gallery, skipped = h.build_review_gallery(selection, cache_dir=cache)

    assert skipped == 1  # 第 1 帧没有对应文件
    assert len(gallery) == 1
    thumb_path = Path(gallery[0][0])
    assert cache in thumb_path.parents
    assert "score" not in thumb_path.name.lower()

    pytest.importorskip("PIL")
    from PIL import Image

    with Image.open(thumb_path) as image:
        assert image.width <= h.THUMB_MAX_WIDTH


def test_load_review_data_missing_scores(tmp_path):
    data = h.load_review_data(tmp_path, use_cache=False)
    assert data.error
    assert "第 1 步" in data.error


def test_load_review_data_warns_for_large_scores(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    (images / "_scores.jsonl").write_text(
        json.dumps({"video": "1", "source_frame": 0, "blur_score": 10.0}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(h, "LARGE_SCORES_WARN_BYTES", 1)

    data = h.load_review_data(images, use_cache=False)

    assert data.error is None
    assert data.warning and "首次加载" in data.warning


def test_config_for_format_rerun_derives_params_from_info(tmp_path):
    dataset = build_dataset(tmp_path)
    fallback = make_config(tmp_path)
    config = h.config_for_format_rerun(fallback, 150, True, dataset)

    assert config.blur_threshold == 150
    assert config.selected_steps == (2,)
    assert config.mode == "force"
    assert config.fps == 5
    assert config.camera == "front"
    assert config.max_episode_seconds == 60.0


def test_config_for_format_rerun_falls_back_without_info(tmp_path):
    fallback = make_config(tmp_path, fps=10, camera="wrist")
    config = h.config_for_format_rerun(
        fallback, 150, False, tmp_path / "missing"
    )

    assert config.fps == 10
    assert config.camera == "wrist"
    assert config.mode == "normal"
    assert config.selected_steps == (2,)


# ---------------------------------------------------------------------------
# 日志切分
# ---------------------------------------------------------------------------


def test_log_decoder_handles_crlf_bare_cr_and_chunk_boundaries():
    decoder = h.LogStreamDecoder()

    assert decoder.feed("hello\r\n") == [h.LogRecord("line", "hello")]
    assert decoder.feed("50%\r") == []
    assert decoder.feed("60%\r") == [h.LogRecord("progress", "50%")]
    assert decoder.feed("\nworld\n") == [
        h.LogRecord("line", "60%"),
        h.LogRecord("line", "world"),
    ]


def test_log_decoder_strips_ansi_and_caps_length():
    decoder = h.LogStreamDecoder(max_line_chars=10)
    records = decoder.feed("\x1b[31mred\x1b[0m\n" + "x" * 30 + "\n")

    assert records[0] == h.LogRecord("line", "red")
    assert all(len(record.text) <= 11 for record in records)


def test_log_buffer_keeps_only_latest_progress():
    buffer = h.LogBuffer()
    buffer.add(h.LogRecord("progress", "one"))
    buffer.add(h.LogRecord("progress", "two"))
    assert "one" not in buffer.render()
    assert "two" in buffer.render()

    buffer.add(h.LogRecord("line", "done"))
    assert "done" in buffer.render()
    assert "two" not in buffer.render()


# ---------------------------------------------------------------------------
# 子进程调度
# ---------------------------------------------------------------------------


class FakeRunner:
    def __init__(self, codes, stop_after=None):
        self._codes = iter(codes)
        self._stop_after = stop_after
        self.started = []
        self.closed = 0
        self.stop_requested = False

    def try_start(self, step):
        self.started.append(step.number)

    def events(self):
        return iter(())

    def wait(self):
        code = next(self._codes)
        if self._stop_after == len(self.started):
            self.stop_requested = True
        return code

    def close(self):
        self.closed += 1


def make_plan(tmp_path, numbers=(1, 2, 3)):
    steps = [make_step(number, tmp_path / f"{number}.marker") for number in numbers]
    config = make_config(tmp_path)
    return h.RunPlan(
        config=config,
        paths=h.resolve_paths(config),
        steps=steps,
        selected=steps,
        to_run=steps,
        skipped=[],
        blocked=[],
    )


def test_iter_run_stops_after_failed_step(tmp_path):
    runner = FakeRunner(codes=[1, 0, 0])
    events = list(h.iter_run(make_plan(tmp_path), runner))

    assert runner.started == [1]
    assert events[-1].kind == "failed"


def test_iter_run_stops_after_stop_request(tmp_path):
    runner = FakeRunner(codes=[1, 0, 0], stop_after=1)
    events = list(h.iter_run(make_plan(tmp_path), runner))

    assert runner.started == [1]
    assert events[-1].kind == "stopped"


def test_iter_run_finishes_all_steps(tmp_path):
    runner = FakeRunner(codes=[0, 0, 0])
    events = list(h.iter_run(make_plan(tmp_path), runner))

    assert runner.started == [1, 2, 3]
    assert events[-1].kind == "finished"
    assert runner.closed == 3


def test_job_runner_streams_merged_output():
    step = rp.Step(
        number=1,
        title="echo",
        cmd=[
            sys.executable,
            "-c",
            "import sys; print('hello'); sys.stderr.write('err\\n')",
        ],
        done_markers=(),
        forceable=False,
    )
    runner = h.JobRunner()
    try:
        runner.try_start(step)
        records = list(runner.events())
        code = runner.wait()
    finally:
        runner.close()

    text = "\n".join(record.text for record in records)
    assert code == 0
    assert "hello" in text
    assert "err" in text


def test_job_runner_rejects_second_job():
    sleeper = rp.Step(
        number=1,
        title="sleep",
        cmd=[sys.executable, "-c", "import time; time.sleep(10)"],
        done_markers=(),
        forceable=False,
    )
    runner = h.JobRunner()
    try:
        runner.try_start(sleeper)
        with pytest.raises(h.JobBusyError):
            runner.try_start(sleeper)
    finally:
        runner.stop()
        runner.wait()
        runner.close()


def test_ui_helpers_does_not_import_cv2_or_gradio():
    code = (
        "import sys, ui_helpers; "
        "print('cv2' in sys.modules, 'gradio' in sys.modules)"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent),
        text=True,
        encoding="utf-8",
    )
    assert output.strip() == "False False"


# ---------------------------------------------------------------------------
# 多相机
# ---------------------------------------------------------------------------


def test_config_from_form_maps_multi_camera_fields():
    config = h.config_from_form(
        "in", "out", "ds", "ld", True, True, True, 5, 100, 60,
        "", "", "h264", "local/x", "normal",
        "dir", "names.json", "droid_autolab", "droid",
    )
    assert config.group_by == "dir"
    assert config.camera_names == "names.json"
    assert config.dataset_id == "droid_autolab"
    assert config.robot_type == "droid"


def test_dir_mode_plan_does_not_require_camera(tmp_path):
    (tmp_path / "videos").mkdir()
    config = make_config(
        tmp_path,
        selected_steps=(1,),
        group_by="dir",
        camera="",
    )
    plan = h.build_plan(config)
    assert not [error for error in plan.errors if "相机名" in error]
    cmd = plan.to_run[0].cmd
    assert cmd[cmd.index("--group-by") + 1] == "dir"
    assert "--camera" not in cmd


def test_dir_mode_with_camera_names_validates_file_exists(tmp_path):
    (tmp_path / "videos").mkdir()
    config = make_config(
        tmp_path,
        selected_steps=(1,),
        group_by="dir",
        camera="front",
        camera_names=str(tmp_path / "missing.json"),
    )
    plan = h.build_plan(config)
    assert any("相机名映射文件不存在" in error for error in plan.errors)


def test_summary_lists_all_cameras_without_false_frame_count_warning(tmp_path):
    dataset = tmp_path / "ds"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    info = {
        "total_episodes": 1,
        "total_frames": 5,
        "fps": 5,
        "features": {
            "observation.images.a": {"shape": [48, 64, 3]},
            "observation.images.b": {"shape": [24, 32, 3]},
        },
        "extras": {
            "group_by": "dir",
            "cameras": ["a", "b"],
            "total_camera_rows": 10,
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "episode": "ep1",
                "source_video": "a",
                "length": 5,
                "source_fps_origin": "_meta.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (meta / "frames.jsonl").write_text(
        "\n".join(
            json.dumps({"is_blurry": False, "blur_score": 200.0})
            for _ in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    paths = h.ResolvedPaths(
        input_dir=tmp_path / "videos",
        output_base=tmp_path / "out",
        images_dir=tmp_path / "out" / "images",
        dataset_dir=dataset,
        lerobot_dir=tmp_path / "ld",
    )
    summary = h.summarize_paths(paths, [])

    assert summary.cameras == [("a", (48, 64)), ("b", (24, 32))]
    assert summary.camera_key == "observation.images.a"
    assert dict(summary.episode_counts) == {"ep1": 1}
    assert not [
        warning for warning in summary.warnings if "相机帧行" in warning
    ]
    rendered = h.render_summary(summary)
    assert "2 路" in rendered
    assert "observation.images.a" in rendered
    assert "observation.images.b" in rendered
