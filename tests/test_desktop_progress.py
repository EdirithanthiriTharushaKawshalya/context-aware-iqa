"""Exercise actual Tk polling and rows while analysis is still running."""
import time
import tkinter as tk

import pytest

from src.desktop.analysis import AnalysisSession
from src.desktop.app import DesktopApp
from src.desktop.project import Photo, Project


def first_result_then_wait(connection, data):
    photo = dict(data['photos'][0], status='Keep', score=87.0)
    connection.send(('photo', photo))
    connection.send(('stage', 'Assessing two.jpg…'))
    time.sleep(30)


def completed_but_worker_lingers(connection, data):
    connection.send(('stage', 'Assessing one.jpg…'))
    connection.send(('photo', dict(data['photos'][0], status='Keep', score=87.0)))
    connection.send(('photo', dict(data['photos'][1], status='Review', score=32.0)))
    connection.send(('done', 'Analysis complete.'))
    time.sleep(30)  # Reproduce slow ML-library shutdown after sending results.


def incomplete_done(connection, data):
    connection.send(('done', 'Analysis complete.'))
    time.sleep(30)


@pytest.fixture
def app(tmp_path, monkeypatch, desktop_tk):
    root = tk.Toplevel(desktop_tk)
    root.withdraw()
    application = DesktopApp(root)
    source = tmp_path / 'source'
    source.mkdir()
    application.project = Project(str(source), photos=[Photo('one.jpg', 1, 1), Photo('two.jpg', 1, 1)])
    application.project_file = tmp_path / 'project.iqa.json'
    application.refresh()
    application.dialogs = []
    monkeypatch.setattr('src.desktop.app.messagebox.showerror', lambda title, message: application.dialogs.append((title, message)))
    yield application
    if application.session:
        application.session.close()
    for callback in desktop_tk.tk.splitlist(desktop_tk.tk.call('after', 'info')):
        root.after_cancel(callback)
    root.destroy()


def pump_until(app, condition, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.root.update()
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(f'GUI did not reach expected state: {app.status.get()}')


def begin(app, target):
    app.busy = True
    app.session = AnalysisSession(app.project, target=target)
    return app.session


def test_rows_scores_filters_and_progress_refresh_before_worker_finishes(app):
    session = begin(app, first_result_then_wait)
    pump_until(app, lambda: app.table.item('0', 'values')[0] == 'Keep')
    assert session.alive and app.busy
    assert app.table.item('0', 'values') == ('Keep', '87.0', 'No')
    assert app.table.item('1', 'values') == ('Pending', '—', 'No')
    assert float(app.bar['value']) == 1 and float(app.bar['maximum']) == 2
    assert '1/2 processed' in app.status.get()
    app.view.set('Keep')
    app.refresh()
    assert app.table.get_children() == ('0',)
    app.view.set('Review')
    app.refresh()
    assert app.table.get_children() == ()


def test_done_message_finishes_gui_without_waiting_for_process_shutdown(app):
    session = begin(app, completed_but_worker_lingers)
    pump_until(app, lambda: not app.busy)
    assert not session.alive
    assert app.session is None
    assert app.table.item('0', 'values')[0:2] == ('Keep', '87.0')
    assert app.table.item('1', 'values')[0:2] == ('Review', '32.0')
    assert float(app.bar['value']) == 2
    assert 'Analysis complete.' in app.status.get()
    assert not app.dialogs


def test_pending_files_cannot_be_reported_as_completed(app):
    begin(app, incomplete_done)
    pump_until(app, lambda: not app.busy)
    assert len(app.dialogs) == 1
    assert '2 photos still pending' in app.dialogs[0][1]
    assert float(app.bar['value']) == 0
    assert app.table.item('0', 'values')[0] == 'Pending'
