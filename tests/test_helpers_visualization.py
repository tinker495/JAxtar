from types import SimpleNamespace

import imageio
import numpy as np

from helpers import visualization


def test_solution_animation_loops_after_holding_final_frame(monkeypatch, tmp_path):
    saved = {}
    cv2 = SimpleNamespace(
        COLOR_BGR2RGB=0,
        cvtColor=lambda image, _conversion: image,
        imwrite=lambda _path, _image: True,
    )
    state = SimpleNamespace(
        img=lambda **_kwargs: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    steps = [SimpleNamespace(state=state, cost=0, dist=0) for _ in range(3)]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(visualization, "_require_cv2", lambda: cv2)
    monkeypatch.setattr(
        imageio,
        "mimsave",
        lambda path, images, **kwargs: saved.update(path=path, images=images, kwargs=kwargs),
    )

    visualization.save_solution_animation_and_frames(
        path_steps=steps,
        puzzle_name="puzzle",
        solve_config=None,
        max_animation_time=10,
    )

    assert saved["kwargs"] == {"duration": [250, 250, 3000], "loop": 0}
