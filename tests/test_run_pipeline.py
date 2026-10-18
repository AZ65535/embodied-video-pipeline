"""run_pipeline.py 的命令组装与调度逻辑。

不真正跑三步管线：这里只验证「参数翻译成什么命令」「哪些步骤该跳过/挡下」，
以及 dry-run 与失败即停的行为。
"""

import run_pipeline as rp


def make_args(tmp_path, *extra):
    return rp.parse_args(
        [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--dataset",
            str(tmp_path / "ds"),
            "--lerobot",
            str(tmp_path / "ld"),
            *extra,
        ]
    )


def steps_for(tmp_path, *extra):
    args = make_args(tmp_path, *extra)
    return args, rp.build_steps(
        args, tmp_path, tmp_path / "out", tmp_path / "ds", tmp_path / "ld"
    )


def value_after(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_build_steps_wires_default_paths(tmp_path):
    args, steps = steps_for(tmp_path)

    assert [step.number for step in steps] == [1, 2, 3]
    assert value_after(steps[0].cmd, "--input") == str(tmp_path)
    assert value_after(steps[0].cmd, "--output") == str(tmp_path / "out")
    assert value_after(steps[0].cmd, "--fps") == "5"
    assert value_after(steps[1].cmd, "--input") == str(tmp_path / "out" / "images")
    assert value_after(steps[1].cmd, "--output") == str(tmp_path / "ds")
    assert value_after(steps[2].cmd, "--input") == str(tmp_path / "ds")
    assert value_after(steps[2].cmd, "--output") == str(tmp_path / "ld")
    assert value_after(steps[2].cmd, "--repo-id") == rp.DEFAULT_REPO_ID
    # 第三步不传 --fps：沿用第二步 info.json，避免两边不一致
    assert "--fps" not in steps[2].cmd


def test_cli_options_are_forwarded(tmp_path):
    args, steps = steps_for(
        tmp_path,
        "--blur-threshold",
        "150",
        "--max-episode-seconds",
        "0",
        "--camera",
        "wrist",
        "--task",
        "pick",
        "--vcodec",
        "h264",
        "--repo-id",
        "local/x",
        "--force",
    )
    clean, fmt, exp = steps[0].cmd, steps[1].cmd, steps[2].cmd

    assert value_after(clean, "--camera") == "wrist"
    assert value_after(fmt, "--blur-threshold") == "150.0"
    assert value_after(fmt, "--max-episode-seconds") == "0.0"
    assert value_after(fmt, "--camera") == "wrist"
    assert "--force" in fmt and "--force" in exp
    assert value_after(exp, "--camera") == "wrist"
    assert value_after(exp, "--task") == "pick"
    assert value_after(exp, "--vcodec") == "h264"
    assert value_after(exp, "--repo-id") == "local/x"


def test_dir_mode_forwards_group_by_and_ignores_camera(tmp_path):
    args, steps = steps_for(
        tmp_path, "--group-by", "dir", "--camera", "front"
    )
    assert value_after(steps[0].cmd, "--group-by") == "dir"
    assert value_after(steps[1].cmd, "--group-by") == "dir"
    assert "--camera" not in steps[1].cmd
    assert "--camera" not in steps[2].cmd


def test_camera_names_is_forwarded_to_steps_1_and_2(tmp_path):
    names = tmp_path / "names.json"
    names.write_text("{}", encoding="utf-8")
    _, steps = steps_for(
        tmp_path, "--group-by", "dir", "--camera-names", str(names)
    )
    expected = str(names.resolve())
    assert value_after(steps[0].cmd, "--camera-names") == expected
    assert value_after(steps[1].cmd, "--camera-names") == expected
    assert "--camera-names" not in steps[2].cmd


def test_robot_type_goes_to_step_3_only(tmp_path):
    _, steps = steps_for(tmp_path, "--robot-type", "droid")
    assert value_after(steps[2].cmd, "--robot-type") == "droid"
    for step in steps[:2]:
        assert "--robot-type" not in step.cmd


def test_skip_existing_skips_completed_steps(tmp_path):
    _, steps = steps_for(tmp_path)
    for path in steps[0].done_markers + steps[1].done_markers:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    to_run, skipped, blocked = rp.select_steps(steps, skip_existing=True, force=False)

    assert [step.number for step in skipped] == [1, 2]
    assert [step.number for step in to_run] == [3]
    assert blocked == []


def test_existing_output_without_force_is_blocked(tmp_path):
    _, steps = steps_for(tmp_path)
    marker = steps[1].done_markers[0]
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")

    to_run, skipped, blocked = rp.select_steps(steps, skip_existing=False, force=False)

    # 第一步没有 --force，重复运行是合并/覆盖，所以照跑；第二步被挡下。
    assert [step.number for step in to_run] == [1, 3]
    assert [step.number for step in blocked] == [2]
    assert skipped == []


def test_main_stops_at_first_failing_step(tmp_path, monkeypatch):
    calls = []

    class Result:
        returncode = 1

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Result()

    monkeypatch.setattr(rp.subprocess, "run", fake_run)

    code = rp.main(
        [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--dataset",
            str(tmp_path / "ds"),
            "--lerobot",
            str(tmp_path / "ld"),
        ]
    )

    assert code == 1
    assert len(calls) == 1


def test_dry_run_prints_commands_without_running(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("--dry-run 不应执行任何子进程")

    monkeypatch.setattr(rp.subprocess, "run", fail)

    code = rp.main(
        [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--dataset",
            str(tmp_path / "ds"),
            "--lerobot",
            str(tmp_path / "ld"),
            "--dry-run",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "clean_video.py" in output
    assert "export_lerobot.py" in output
    assert "--dry-run：未执行任何步骤" in output
    assert not (tmp_path / "out").exists()
