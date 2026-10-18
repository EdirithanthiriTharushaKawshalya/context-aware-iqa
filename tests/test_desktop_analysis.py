"""Real-process regression checks: pause must not wait for blocked inference."""
import time
from dataclasses import asdict

from src.desktop.analysis import AnalysisSession
from src.desktop.project import Photo, Project


def blocked_startup(connection, data):
    connection.send(('stage', 'Initializing slow model'))
    time.sleep(30)


def blocked_second_photo(connection, data):
    photo = Photo(**data['photos'][0])
    photo.status, photo.score = 'Keep', 90
    connection.send(('photo', asdict(photo)))
    connection.send(('stage', 'Assessing second photo'))
    time.sleep(30)


def complete_pending(connection, data):
    for record in data['photos']:
        photo = Photo(**record)
        if photo.status == 'Pending':
            photo.status, photo.score = 'Keep', 85
            connection.send(('photo', asdict(photo)))
    connection.send(('done', 'Analysis complete.'))
    connection.close()


def wait_for(session, event):
    messages = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        messages.extend(session.drain())
        if any(kind == event for kind, detail in messages):
            return messages
        time.sleep(0.02)
    raise AssertionError(f'Worker did not emit {event}: {messages}')


def stop_promptly(session):
    start = time.monotonic()
    session.pause()
    while session.alive and time.monotonic() - start < 2:
        time.sleep(0.01)
    assert not session.alive, 'Pause waited for blocked inference'
    assert time.monotonic() - start < 2


def test_pause_during_model_startup(tmp_path):
    session = AnalysisSession(Project(str(tmp_path)), target=blocked_startup)
    try:
        wait_for(session, 'stage')
        stop_promptly(session)
        assert session.cancelled
    finally:
        session.close()


def test_pause_preserves_completed_photo_and_resume_retries_only_pending(tmp_path):
    project = Project(str(tmp_path), photos=[Photo('one.jpg', 1, 1), Photo('two.jpg', 1, 1)])
    session = AnalysisSession(project, target=blocked_second_photo)
    try:
        messages = wait_for(session, 'stage')
        stop_promptly(session)
        messages.extend(session.drain())
        for event, detail in messages:
            if event == 'photo':
                project.photos[0] = Photo(**detail)
        project.save(tmp_path / 'session.iqa.json')
    finally:
        session.close()
    restored = Project.load(tmp_path / 'session.iqa.json')
    assert [photo.status for photo in restored.photos] == ['Keep', 'Pending']
    resumed = AnalysisSession(restored, target=complete_pending)
    try:
        messages = wait_for(resumed, 'done')
        assert [detail['relative'] for event, detail in messages if event == 'photo'] == ['two.jpg']
    finally:
        resumed.close()


def test_real_worker_skips_model_loading_when_nothing_is_pending(tmp_path):
    session = AnalysisSession(Project(str(tmp_path)))
    try:
        messages = wait_for(session, 'done')
        assert not any(event == 'stage' and 'models' in detail for event, detail in messages)
        assert not any(event == 'error' for event, detail in messages)
    finally:
        session.close()
