"""Private recovery checkpoints, separate from the user's finished project file."""
import os
from pathlib import Path
from uuid import uuid4


class SessionStore:
    def __init__(self, directory=None):
        if directory is None:
            base = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local' / 'share'))
            directory = base / 'ContextAwareIQA' / 'Sessions'
        self.directory = Path(directory)

    def create(self, project):
        source = Path(project.source).resolve()
        directory = self.directory.resolve()
        if directory.is_relative_to(source):
            raise ValueError('Choose a photo folder that does not contain the application recovery directory.')
        destination = directory / f'{uuid4().hex}.iqa.json'
        project.save(destination)
        return destination

    def unfinished(self):
        from .project import Project
        if not self.directory.exists():
            return []
        entries = []
        paths = sorted(self.directory.glob('*.iqa.json'), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in paths:
            try:
                project = Project.load(path)
                if project.finished_at is None:
                    entries.append((path, project))
                    if len(entries) == 30:
                        break
            except (OSError, ValueError, TypeError, KeyError):
                continue
        return entries
