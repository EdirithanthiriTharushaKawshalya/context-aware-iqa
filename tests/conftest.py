"""Use one Tk interpreter, matching the application's window lifecycle."""
import tkinter as tk

import pytest


@pytest.fixture(scope='session')
def desktop_tk():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()
