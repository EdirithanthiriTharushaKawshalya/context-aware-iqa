"""Windows Recycle Bin operation with no permanent-delete fallback."""
import sys


def recycle_file(path):
    if sys.platform != 'win32':
        raise OSError('Source-folder cleanup is supported on Windows in this release.')
    import pythoncom
    import pywintypes
    from win32com.server.exception import COMException
    from win32com.shell import shell, shellcon
    from send2trash.win.IFileOperationProgressSink import FileOperationProgressSink

    class RecycleOnlySink(FileOperationProgressSink):
        def __init__(self):
            super().__init__()
            self.recycled = False

        def PreDeleteItem(self, flags, item):
            if not flags & shellcon.TSF_DELETE_RECYCLE_IF_POSSIBLE:
                raise COMException('This file cannot be recycled. Cleanup was stopped.', scode=-2147467259)
            return 0

        def PostDeleteItem(self, flags, item, result, newly_created):
            self.recycled = result >= 0 and newly_created is not None

    pythoncom.CoInitialize()
    try:
        sink = RecycleOnlySink()
        operation = pythoncom.CoCreateInstance(shell.CLSID_FileOperation, None,
                                               pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IFileOperation)
        # Recycle, record undo, fail on error, and operate on the exact file only.
        operation.SetOperationFlags(shellcon.FOF_SILENT | shellcon.FOF_NOERRORUI |
                                    shellcon.FOF_NOCONFIRMATION | shellcon.FOF_NO_CONNECTED_ELEMENTS |
                                    shellcon.FOFX_EARLYFAILURE | 0x20000000 | 0x00080000)
        item = shell.SHCreateItemFromParsingName(str(path), None, shell.IID_IShellItem)
        operation.DeleteItem(item, pythoncom.WrapObject(sink, shell.IID_IFileOperationProgressSink))
        result = operation.PerformOperations()
        if result or operation.GetAnyOperationsAborted() or not sink.recycled:
            raise OSError('Windows could not confirm moving this file to the Recycle Bin. Cleanup stopped.')
    except pywintypes.com_error as exc:
        raise OSError(f'Recycle Bin operation failed: {exc.strerror}') from exc
    finally:
        pythoncom.CoUninitialize()
