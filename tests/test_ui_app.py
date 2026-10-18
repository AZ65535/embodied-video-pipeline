"""ui.py 的冒烟测试；不启动 Gradio 服务。"""

import pytest

import ui


def test_main_without_gradio_prints_install_hint(monkeypatch, capsys):
    monkeypatch.setattr(ui, "gr", None)

    code = ui.main()

    assert code == 1
    assert "requirements-ui.txt" in capsys.readouterr().err


def test_build_ui_smoke():
    gr = pytest.importorskip("gradio")
    demo = ui.build_ui()
    assert isinstance(demo, gr.Blocks)
