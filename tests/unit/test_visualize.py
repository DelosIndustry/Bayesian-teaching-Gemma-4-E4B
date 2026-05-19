"""Visualization behavior tests."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.quant_visualize import save_figures, write_captions_md


def _figure():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [0.2, 0.4])
    return fig


def test_save_figures_atomic_ordering_blocks_all_writes_when_any_none(tmp_path):
    """Feature: quantization-and-sample-efficiency, Property 7: Figure ordering and cross-domain guard"""
    save_figures([("a", _figure()), ("b", None), ("c", _figure())], tmp_path)

    assert list(tmp_path.glob("*.pdf")) == []
    assert list(tmp_path.glob("*.png")) == []


def test_save_figures_success_writes_pdf_and_png(tmp_path):
    save_figures([("a", _figure()), ("b", _figure())], tmp_path)

    assert (tmp_path / "a.pdf").exists()
    assert (tmp_path / "a.png").exists()
    assert (tmp_path / "b.pdf").exists()
    assert (tmp_path / "b.png").exists()


def test_captions_headers_match_figure_stems(tmp_path):
    save_figures([("a", _figure()), ("b", _figure())], tmp_path)
    captions = {"a": "첫 번째 그림 캡션", "b": "두 번째 그림 캡션"}
    write_captions_md(tmp_path / "captions.md", captions)

    header_stems = {
        line.removeprefix("## ").strip()
        for line in (tmp_path / "captions.md").read_text().splitlines()
        if line.startswith("## ")
    }
    figure_stems = {p.stem for p in tmp_path.glob("*.png")}
    assert header_stems == figure_stems


def test_cross_domain_figure_is_absent_unless_requested(tmp_path):
    save_figures([("learning_curves", _figure())], tmp_path)

    assert not (tmp_path / "cross_domain_robustness.pdf").exists()
    assert not (tmp_path / "cross_domain_robustness.png").exists()
