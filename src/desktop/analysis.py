"""Killable inference worker. Only the desktop parent writes project files."""
import multiprocessing
import threading
from dataclasses import asdict

from .project import Photo, Project


def assess_project(connection, data):
    """Spawn entry point: no Tk objects, filesystem writes, or shared models."""
    try:
        data['photos'] = [Photo(**photo) for photo in data['photos']]
        project = Project(**data)
        connection.send(('stage', 'Scanning for new or changed photos…'))
        project.scan()
        connection.send(('inventory', [asdict(photo) for photo in project.photos]))
        if any(photo.status == 'Pending' and not photo.recycled for photo in project.photos):
            connection.send(('stage', 'Loading assessment engine…'))
            from src.pipeline import Phase4Pipeline, PipelineConfig
            connection.send(('stage', 'Initializing models. First use may download weights; Pause can cancel startup.'))
            pipeline = Phase4Pipeline()
            config = PipelineConfig(shot_type=project.shot_type, motion_intent=project.motion_intent)
            project.analyze(
                pipeline, config, threading.Event(),
                lambda photo: connection.send(('photo', asdict(photo))),
                before_photo=lambda photo: connection.send(('stage', f'Assessing {photo.relative}…')),
            )
        connection.send(('done', 'Analysis complete.'))
    except Exception as exc:
        connection.send(('error', str(exc)))
    finally:
        connection.close()


class AnalysisSession:
    """Parent-side process owner; stopping never waits for model inference."""
    def __init__(self, project, target=assess_project):
        context = multiprocessing.get_context('spawn')
        self.receiver, sender = context.Pipe(duplex=False)
        self.process = context.Process(target=target, args=(sender, asdict(project)), daemon=True)
        self.cancelled = False
        self.eof = False
        try:
            self.process.start()
        except BaseException:
            self.receiver.close()
            sender.close()
            raise
        sender.close()

    def drain(self):
        messages = []
        while not self.eof:
            try:
                if not self.receiver.poll():
                    break
                messages.append(self.receiver.recv())
            except (EOFError, OSError):
                self.eof = True
        return messages

    def pause(self):
        self.cancelled = True
        if self.process.is_alive():
            self.process.terminate()

    @property
    def alive(self):
        return self.process.is_alive()

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=0.2)
        self.receiver.close()
