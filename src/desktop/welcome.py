"""First-run guidance with explicit dismissal and a replayable tour."""
import json
import os
from pathlib import Path
from .tour import Introduction

CARDS = (
    ('Bring your photographs in',
     'Choose your photo folder here. Set an optional final count, or leave it blank for all suitable photos. Assessment settings are tucked away until you need them.',
     'Opening a folder leaves your originals in place.'),
    ('Give every moment a closer look',
     'Analyze photos checks subject detail, exposure, and context. Pause whenever you need to; completed assessments are preserved for your next session.',
     'Keep and Review are recommendations. You stay in control.'),
    ('Make the selection yours',
     'Browse your photographs here. Filter Keep, Review, or Chosen; select a photo to inspect it on the right. Add favorites and protect essential moments.',
     'Use the Photo and View menus for more review tools.'),
    ('Finish with confidence',
     'Review & finish folder previews your final selection and any proposed removals. Confirm the exact cleanup list, or save your project without moving photos.',
     'Replay this tour from Help. Change your theme in Edit > Preferences.'),
)


class Preferences:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local/share')) / 'ContextAwareIQA/preferences.json'

    def completed(self):
        try:
            return json.loads(self.path.read_text(encoding='utf-8')).get('introduction_complete') is True
        except (OSError, ValueError, AttributeError):
            return False

    def theme(self):
        try:
            value = json.loads(self.path.read_text(encoding='utf-8')).get('theme')
            return value if value in ('light', 'dark') else 'light'
        except (OSError, ValueError, AttributeError):
            return 'light'

    def complete(self):
        self.save(introduction_complete=True)

    def save(self, **settings):
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        data.update(settings)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
        temporary.replace(self.path)
