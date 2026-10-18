#!/usr/bin/env python
"""本地 Gradio 可视化界面。

    python ui.py

只做两件事：运行三步管线（实时日志、停止）、复核模糊阈值。
所有实际工作仍由 clean_video.py / format_dataset.py / export_lerobot.py 完成；
本文件只负责表单、事件与展示。

Gradio 是可选依赖：没装时 `python ui.py` 会打印安装提示，不影响原 CLI。
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
# 本机若配置了系统代理，Gradio 对 127.0.0.1 的健康检查会走代理并返回 502。
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - 由 test_ui_app 覆盖
    gr = None
    GRADIO_IMPORT_ERROR = exc

import ui_helpers as h


def _safe_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _preview_fn(*form_values) -> str:
    try:
        config = h.config_from_form(*form_values)
        return h.render_plan_preview(h.build_plan(config))
    except Exception as exc:  # noqa: BLE001 - 界面不显示 traceback
        return f"**参数错误**：`{type(exc).__name__}: {exc}`"


def _summary_fn(*form_values) -> str:
    try:
        config = h.config_from_form(*form_values)
        plan = h.build_plan(config)
        return h.render_summary(h.summarize_paths(plan.paths, plan.steps))
    except Exception as exc:  # noqa: BLE001
        return f"**摘要生成失败**：`{type(exc).__name__}: {exc}`"


def _status_for(event, started: float) -> Tuple[str, bool]:
    step = event.step
    elapsed = time.monotonic() - started
    label = f"第 {step.number}/3 步：{step.title}" if step else "运行中"
    if event.kind == "step_start":
        return f"### 运行中… {label}", True
    if event.kind == "tick":
        return f"### 运行中… {label}（已用时 {elapsed:.1f}s）", True
    if event.kind in {"line", "progress"}:
        return f"### 运行中… {label}（已用时 {elapsed:.1f}s）", True
    if event.kind == "step_done":
        return (
            f"### {label}完成（{event.elapsed:.1f}s，退出码 {event.returncode}）",
            True,
        )
    if event.kind == "failed":
        return f"### 失败：{label}\n\n{event.text}", False
    if event.kind == "stopped":
        return f"### 已停止：{label}\n\n{event.text}", False
    if event.kind == "finished":
        return f"### 完成\n\n总耗时 {event.elapsed:.1f}s", False
    return f"### {label}", True


def _stream_plan(plan: h.RunPlan, buffer: h.LogBuffer, summary_after_step2: bool):
    """把 iter_run 的事件翻译成 (status, log, summary, active) 供两个生成器复用。"""
    started = time.monotonic()
    summary_update = gr.update()
    yield "### 运行中…", buffer.render(), summary_update, True

    try:
        for event in h.iter_run(plan, h.JOB_RUNNER):
            if event.kind == "step_start":
                buffer.add_ui(f"[{event.step.number}/3] {event.step.title}")
                buffer.add_ui(f"$ {event.text}")
            elif event.kind in {"line", "progress"}:
                buffer.add(h.LogRecord(event.kind, event.text))
            elif event.kind == "step_done":
                buffer.add_ui(
                    f"完成（{event.elapsed:.1f}s，退出码 {event.returncode}）"
                )
            elif event.kind in {"failed", "stopped"}:
                buffer.add_ui(event.text)
            elif event.kind == "finished":
                buffer.add_ui(f"全部步骤完成（{event.elapsed:.1f}s）")

            if (
                summary_after_step2
                and event.kind == "step_done"
                and event.step.number == 2
                and event.returncode == 0
            ):
                summary_update = h.render_summary(
                    h.summarize_paths(plan.paths, plan.steps)
                )
            if event.kind == "finished" and summary_after_step2:
                summary_update = h.render_summary(
                    h.summarize_paths(plan.paths, plan.steps)
                )

            status, active = _status_for(event, started)
            yield status, buffer.render(), summary_update, active
    except Exception as exc:  # noqa: BLE001 - 生成器里的异常也要显示出来
        yield (
            f"### 界面异常\n\n`{type(exc).__name__}: {exc}`",
            buffer.render(),
            summary_update,
            False,
        )
    finally:
        h.JOB_RUNNER.close()


def _run_fn(*form_values):
    config = h.config_from_form(*form_values)
    plan = h.build_plan(config)

    if plan.errors or plan.blocked:
        reason = (
            "参数或产物检查未通过，未开始运行。详见下方预览。"
            if plan.errors
            else "已有产物挡下了选中的步骤；请选择 force 或 skip-existing。"
        )
        yield (
            f"### 未运行\n\n{reason}",
            h.render_plan_preview(plan),
            gr.update(),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return
    if not plan.to_run:
        yield (
            "### 没有需要运行的步骤\n\n"
            "所有选定步骤都已完成。需要重做请选择 force 模式。",
            h.render_plan_preview(plan),
            gr.update(),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return

    buffer = h.LogBuffer()
    for status, log_text, summary_update, active in _stream_plan(
        plan, buffer, summary_after_step2=True
    ):
        yield (
            status,
            log_text,
            summary_update,
            gr.update(interactive=not active),
            gr.update(interactive=active),
        )


def _stop_fn():
    message = h.JOB_RUNNER.stop()
    return f"### 停止请求\n\n{message}", gr.update(interactive=False)


def _review_view(output_base, threshold, band, video, sample_size) -> dict:
    images_dir = h.images_dir_for(str(output_base or ""))
    data = h.load_review_data(images_dir)
    if data.error:
        return {
            "path": str(images_dir),
            "names": [],
            "video": h.ALL_VIDEOS,
            "stats": f"**无法读取复核数据**：{data.error}",
            "gallery": [],
            "message": data.error,
        }

    names = h.review_video_names(data)
    if video != h.ALL_VIDEOS and video not in names:
        video = h.ALL_VIDEOS
    selection = h.select_review_frames(
        data,
        _safe_float(threshold, h.DEFAULT_BLUR_THRESHOLD),
        _safe_float(band, 3.0),
        video,
        _safe_int(sample_size, 48),
    )
    gallery, skipped = h.build_review_gallery(selection)
    messages: List[str] = []
    if data.warning:
        messages.append(data.warning)
    if skipped:
        messages.append(f"有 {skipped} 张缩略图生成失败或原图缺失，已跳过。")
    return {
        "path": str(images_dir),
        "names": names,
        "video": video,
        "stats": h.render_review_stats(selection),
        "gallery": gallery,
        "message": "\n\n".join(messages),
    }


def _reload_review(output_base, threshold, band, video, sample_size):
    view = _review_view(output_base, threshold, band, video, sample_size)
    choices = [("全部", h.ALL_VIDEOS)] + [
        (name, name) for name in view["names"]
    ]
    return (
        view["path"],
        gr.update(choices=choices, value=view["video"]),
        view["stats"],
        view["gallery"],
        view["message"],
    )


def _refresh_review(output_base, threshold, band, video, sample_size):
    view = _review_view(output_base, threshold, band, video, sample_size)
    return view["stats"], view["gallery"], view["message"]


def _sync_review_path(output_base):
    path = str(h.images_dir_for(str(output_base or "")))
    return (
        path,
        [],
        "",
        "路径已变化。点击「刷新复核数据」或切到复核页重新加载。",
    )


def _rerun_fn(
    input_dir,
    output_base,
    dataset_dir,
    lerobot_dir,
    fps,
    blur_threshold,
    max_episode_seconds,
    link_mode,
    camera,
    task,
    vcodec,
    repo_id,
    threshold,
    force,
):
    fallback = h.config_from_form(
        input_dir,
        output_base,
        dataset_dir,
        lerobot_dir,
        True,
        True,
        True,
        fps,
        blur_threshold,
        max_episode_seconds,
        link_mode,
        camera,
        task,
        vcodec,
        repo_id,
        "normal",
    )
    config = h.config_for_format_rerun(
        fallback,
        _safe_float(threshold, h.DEFAULT_BLUR_THRESHOLD),
        bool(force),
        h.rp.resolve_path(str(dataset_dir or "")),
    )
    plan = h.build_plan(config)

    if plan.errors or plan.blocked or not plan.to_run:
        reason = (
            "参数或产物检查未通过，未执行。"
            if plan.errors
            else "dataset 已存在或没有可运行的步骤；需要覆盖请勾选 --force。"
        )
        yield (
            f"### 未执行\n\n{reason}",
            h.render_plan_preview(plan),
            gr.update(),
        )
        return

    buffer = h.LogBuffer()
    for status, log_text, summary_update, active in _stream_plan(
        plan, buffer, summary_after_step2=True
    ):
        yield status, log_text, summary_update


def build_ui():
    """构建界面但不启动服务；测试只调用这里。"""
    if gr is None:
        raise RuntimeError("Gradio 未安装")

    with gr.Blocks(title="embodied-video-pipeline") as demo:
        gr.Markdown(
            "# embodied-video-pipeline\n"
            "本地控制台：运行三步管线，并复核模糊阈值。"
        )

        with gr.Tabs():
            with gr.Tab("运行管线") as run_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 路径")
                        input_dir = gr.Textbox(label="输入视频目录", value="videos")
                        output_base = gr.Textbox(
                            label="第 1 步输出根目录",
                            value="output",
                            info="图片落在 <output>/images",
                        )
                        dataset_dir = gr.Textbox(
                            label="dataset 目录", value="dataset"
                        )
                        lerobot_dir = gr.Textbox(
                            label="lerobot 目录", value="lerobot_dataset"
                        )

                        gr.Markdown("### 步骤")
                        step1 = gr.Checkbox(
                            label="1. 抽帧 + 记录模糊分数", value=True
                        )
                        step2 = gr.Checkbox(label="2. 生成帧清单", value=True)
                        step3 = gr.Checkbox(
                            label="3. 导出 LeRobot 数据集", value=True
                        )

                        mode = gr.Radio(
                            label="已有产物时",
                            choices=[
                                ("正常（产物存在时停下）", "normal"),
                                ("force（覆盖/重建）", "force"),
                                ("skip-existing（跳过已完成）", "skip-existing"),
                            ],
                            value="normal",
                        )

                        with gr.Accordion("参数", open=False):
                            fps = gr.Number(label="fps", value=5, precision=0)
                            blur_threshold = gr.Number(
                                label="模糊阈值", value=h.DEFAULT_BLUR_THRESHOLD
                            )
                            max_episode_seconds = gr.Number(
                                label="episode 时长上限（秒）", value=60
                            )
                            link_mode = gr.Dropdown(
                                label="落盘方式",
                                choices=["auto", "hardlink", "symlink", "copy"],
                                value="auto",
                            )
                            camera = gr.Textbox(
                                label="相机名", value=h.DEFAULT_CAMERA
                            )
                            task = gr.Textbox(label="任务标注", value="")
                            vcodec = gr.Textbox(
                                label="视频编码器",
                                value="h264",
                                info="小尺寸素材不要用 auto / libsvtav1",
                            )
                            repo_id = gr.Textbox(
                                label="repo-id", value=h.DEFAULT_REPO_ID
                            )

                    with gr.Column(scale=1):
                        preview = gr.Textbox(
                            label="命令预览",
                            lines=14,
                            interactive=False,
                        )
                        with gr.Row():
                            run_btn = gr.Button("运行选定步骤", variant="primary")
                            stop_btn = gr.Button(
                                "停止", interactive=False, variant="stop"
                            )
                        status = gr.Markdown("### 空闲")
                        log = gr.Textbox(
                            label="日志",
                            lines=20,
                            interactive=False,
                            autoscroll=True,
                        )

                gr.Markdown("### 数据集摘要")
                summary = gr.Markdown("（尚未读取）")
                summary_btn = gr.Button("刷新摘要")

            with gr.Tab("模糊复核") as review_tab:
                gr.Markdown(
                    "读取第 1 步产物中的 `_scores.jsonl`。"
                    "帧只被标记，不会被删除。"
                )
                review_path = gr.Textbox(
                    label="复核数据目录（跟随第 1 步输出根目录）",
                    interactive=False,
                )
                refresh_review_btn = gr.Button("刷新复核数据")

                with gr.Row():
                    threshold = gr.Slider(
                        label="阈值",
                        minimum=1,
                        maximum=10000,
                        value=h.DEFAULT_BLUR_THRESHOLD,
                        step=1,
                    )
                    band = gr.Slider(
                        label="临界带倍数",
                        minimum=1.1,
                        maximum=10,
                        value=3.0,
                        step=0.1,
                    )
                    video_filter = gr.Dropdown(
                        label="源视频",
                        choices=[("全部", h.ALL_VIDEOS)],
                        value=h.ALL_VIDEOS,
                    )
                    sample_size = gr.Slider(
                        label="图库采样数",
                        minimum=8,
                        maximum=200,
                        value=48,
                        step=8,
                    )

                review_stats = gr.Markdown()
                review_gallery = gr.Gallery(
                    label="复核帧（缩略图）",
                    columns=6,
                    object_fit="contain",
                    allow_preview=True,
                )
                review_msg = gr.Markdown()

                with gr.Accordion("用当前阈值重新生成帧清单", open=False):
                    gr.Markdown(
                        "重跑第 2 步只改 `frames.jsonl` 的 `is_blurry` 标记，"
                        "不会删除任何帧图片。dataset 已存在时必须勾选 `--force`；"
                        "`--force` 不会清理旧的多余 episode 目录。"
                    )
                    rerun_force = gr.Checkbox(
                        label="允许覆盖已有 dataset（--force）", value=False
                    )
                    rerun_btn = gr.Button("重新生成帧清单")
                    rerun_status = gr.Markdown()
                    rerun_log = gr.Textbox(
                        label="重跑日志",
                        lines=12,
                        interactive=False,
                        autoscroll=True,
                    )

        form_inputs = [
            input_dir,
            output_base,
            dataset_dir,
            lerobot_dir,
            step1,
            step2,
            step3,
            fps,
            blur_threshold,
            max_episode_seconds,
            link_mode,
            camera,
            task,
            vcodec,
            repo_id,
            mode,
        ]
        review_inputs = [output_base, threshold, band, video_filter, sample_size]
        rerun_inputs = [
            input_dir,
            output_base,
            dataset_dir,
            lerobot_dir,
            fps,
            blur_threshold,
            max_episode_seconds,
            link_mode,
            camera,
            task,
            vcodec,
            repo_id,
            threshold,
            rerun_force,
        ]

        for component in form_inputs:
            component.change(
                _preview_fn,
                inputs=form_inputs,
                outputs=preview,
                queue=False,
            )

        run_btn.click(
            _run_fn,
            inputs=form_inputs,
            outputs=[status, log, summary, run_btn, stop_btn],
            concurrency_limit=1,
        )
        stop_btn.click(
            _stop_fn,
            inputs=None,
            outputs=[status, stop_btn],
            queue=False,
        )
        summary_btn.click(
            _summary_fn,
            inputs=form_inputs,
            outputs=summary,
            queue=True,
        )

        output_base.change(
            _sync_review_path,
            inputs=[output_base],
            outputs=[review_path, review_gallery, review_stats, review_msg],
            queue=False,
        )
        review_tab.select(
            _reload_review,
            inputs=review_inputs,
            outputs=[
                review_path,
                video_filter,
                review_stats,
                review_gallery,
                review_msg,
            ],
            queue=True,
        )
        refresh_review_btn.click(
            _reload_review,
            inputs=review_inputs,
            outputs=[
                review_path,
                video_filter,
                review_stats,
                review_gallery,
                review_msg,
            ],
            queue=True,
        )
        for component in (threshold, band, sample_size):
            component.release(
                _refresh_review,
                inputs=review_inputs,
                outputs=[review_stats, review_gallery, review_msg],
                queue=True,
            )
        video_filter.change(
            _refresh_review,
            inputs=review_inputs,
            outputs=[review_stats, review_gallery, review_msg],
            queue=True,
        )

        rerun_btn.click(
            _rerun_fn,
            inputs=rerun_inputs,
            outputs=[rerun_status, rerun_log, summary],
            concurrency_limit=1,
        )

        demo.load(_preview_fn, inputs=form_inputs, outputs=preview)
        demo.load(_summary_fn, inputs=form_inputs, outputs=summary)
        demo.load(
            lambda value: str(h.images_dir_for(str(value or ""))),
            inputs=[output_base],
            outputs=[review_path],
        )

    return demo


def main() -> int:
    if gr is None:
        print(
            "Gradio 未安装（可选依赖）。请运行：\n"
            "    python -m pip install -r requirements-ui.txt\n"
            "如果 pip 镜像不可用：\n"
            "    python -m pip install -r requirements-ui.txt -i https://pypi.org/simple/\n"
            "核心三步管线不受影响，仍可用 run_pipeline.py。",
            file=sys.stderr,
        )
        return 1

    h.THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_ui()
    demo.launch(
        server_name=h.UI_HOST,
        share=False,
        allowed_paths=[str(h.THUMB_CACHE_DIR)],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
