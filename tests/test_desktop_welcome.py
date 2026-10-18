"""First-run guidance remembers completion and remains replayable."""
import json

from src.desktop.welcome import CARDS, Introduction, Preferences


def test_preferences_survive_restart_and_preserve_other_settings(tmp_path):
    path = tmp_path / 'preferences.json'
    path.write_text(json.dumps({'other': 'unchanged'}))
    preferences = Preferences(path)
    assert not preferences.completed()
    preferences.complete()
    assert Preferences(path).completed()
    assert json.loads(path.read_text())['other'] == 'unchanged'


def test_malformed_preferences_allow_first_run(tmp_path):
    path = tmp_path / 'preferences.json'
    for text in ('broken', '[]', 'null'):
        path.write_text(text)
        preferences = Preferences(path)
        assert not preferences.completed()
        preferences.complete()
        assert preferences.completed()


def test_tour_back_next_finish_and_replay(desktop_tk, tmp_path):
    preferences = Preferences(tmp_path / 'preferences.json')
    tour = Introduction(desktop_tk, preferences)
    try:
        tour.next()
        tour.back()
        assert tour.index == 0 and not preferences.completed()
        for _ in CARDS:
            tour.next()
        assert preferences.completed()
        assert not tour.window.winfo_exists()
        replay = Introduction(desktop_tk, preferences)
        assert replay.index == 0
        replay.dismiss()
    finally:
        if tour.window.winfo_exists():
            tour.window.destroy()


def test_skip_remembers_dismissal(desktop_tk, tmp_path):
    preferences = Preferences(tmp_path / 'preferences.json')
    tour = Introduction(desktop_tk, preferences)
    tour.dismiss()
    assert preferences.completed()
    assert not tour.window.winfo_exists()
