import time


from confopt.utils.tracking import RuntimeTracker


def test_runtime_tracker__return_runtime():
    dummy_tracker = RuntimeTracker()
    sleep_time = 2
    time.sleep(sleep_time)
    time_elapsed = dummy_tracker.return_runtime()
    assert sleep_time - 1 < round(time_elapsed) < sleep_time + 1


def test_runtime_tracker__pause_runtime():
    dummy_tracker = RuntimeTracker()
    dummy_tracker.pause_runtime()
    sleep_time = 2
    time.sleep(sleep_time)
    dummy_tracker.resume_runtime()
    time_elapsed = dummy_tracker.return_runtime()
    assert time_elapsed < 1
