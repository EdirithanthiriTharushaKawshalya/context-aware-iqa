"""Launch with python desktop_app.py. The Streamlit website is independent."""
from multiprocessing import freeze_support
import sys

if __name__ == '__main__':
    freeze_support()
    try:
        from src.desktop.app import main
        main(smoke_test='--smoke-test' in sys.argv)
    except Exception:
        if '--smoke-test' in sys.argv:
            import tempfile
            import traceback
            from pathlib import Path
            Path(tempfile.gettempdir(), 'context-aware-iqa-startup.log').write_text(
                traceback.format_exc(), encoding='utf-8')
            raise SystemExit(1)
        raise
