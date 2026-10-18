"""Startup stages run in order and preserve the five-second splash minimum."""
import heapq
from types import SimpleNamespace
from unittest.mock import Mock

from src.desktop.startup import prepare_workspace, show_workspace


class EventLoop:
    def __init__(self):
        self.now = 0
        self.events = []
        self.visible_at = None

    def after(self, milliseconds, callback):
        heapq.heappush(self.events, (self.now + milliseconds / 1000, callback))

    def deiconify(self):
        self.visible_at = self.now

    def run(self):
        while self.events:
            self.now, callback = heapq.heappop(self.events)
            callback()


def test_stages_run_before_reveal_and_wait_for_five_seconds(monkeypatch):
    root = EventLoop()
    monkeypatch.setattr('src.desktop.startup.time.monotonic', lambda: root.now)
    splash = SimpleNamespace(visible_since=0, stage=Mock(), destroy=Mock())
    tasks, ready, failures = [], Mock(), []
    prepare_workspace(root, splash, [
        ('Read preferences', 25, lambda: tasks.append('preferences')),
        ('Build controls', 55, lambda: tasks.append('controls')),
    ], lambda: show_workspace(root, splash, ready), failures.append)
    assert not tasks and root.visible_at is None
    root.run()
    assert tasks == ['preferences', 'controls'] and not failures
    assert 5 <= root.visible_at < 5.01
    assert [call.args[0] for call in splash.stage.call_args_list] == [
        'Read preferences', 'Build controls', 'Ready — opening your workspace…']
    ready.assert_called_once()
    splash.destroy.assert_called_once()


def test_slow_startup_does_not_add_another_five_seconds(monkeypatch):
    root = EventLoop()
    root.now = 7
    monkeypatch.setattr('src.desktop.startup.time.monotonic', lambda: root.now)
    show_workspace(root, SimpleNamespace(visible_since=0, destroy=Mock()), Mock())
    root.run()
    assert root.visible_at == 7


def test_startup_failure_stops_later_stages(monkeypatch):
    root = EventLoop()
    monkeypatch.setattr('src.desktop.startup.time.monotonic', lambda: root.now)
    failures, done, later = [], Mock(), Mock()
    def broken():
        raise OSError('Cannot initialize workspace')
    prepare_workspace(root, None, [('Read', 25, broken), ('Build', 55, later)], done, failures.append)
    root.run()
    assert len(failures) == 1 and isinstance(failures[0], OSError)
    done.assert_not_called()
    later.assert_not_called()
