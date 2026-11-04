import pytest

pytest.importorskip("matplotlib")

from PIL import Image

from viz.figures import FIGURES, main


def test_all_figures_render(tmp_path):
    paths = main(tmp_path)
    assert [p.name for p in paths] == list(FIGURES)
    assert all(p.stat().st_size > 10_000 for p in paths)
    # every figure is 16:9 at 4800 x 2700 px so it drops straight into a slide (01 is the retired first pipeline)
    assert all(Image.open(p).size == (4800, 2700) for p in paths if p.name != "01_pipeline.png")
