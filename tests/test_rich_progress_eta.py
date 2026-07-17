from helpers.rich_progress import RichProgressBar


def test_eta_survives_step_intervals_longer_than_rich_default_window():
    bar = RichProgressBar(total=10)
    now = [0.0]
    bar.progress.get_time = lambda: now[0]  # must patch before add_task; Task captures it
    task_id = bar.progress.add_task("", total=10)
    for _ in range(2):
        now[0] += 60.0  # 60s/step, beyond rich's default 30s speed window
        bar.progress.update(task_id, advance=1)
    remaining = bar.progress.tasks[0].time_remaining
    assert remaining is not None  # None is exactly the bug: window emptied
    assert 479 <= remaining <= 481  # ~8 steps * 60s
