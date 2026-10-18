# Build from the project root: python -m PyInstaller desktop.spec
from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=[('src/images/icon.png', 'src/images')],
    hiddenimports=collect_submodules('torchvision'),
    hookspath=[],
    runtime_hooks=[],
    excludes=['streamlit', 'notebook', 'pytest', 'transformers', 'datasets',
              'tensorflow', 'jax', 'jaxlib', 'diffusers', 'accelerate',
              'bitsandbytes', 'timm', 'IPython', 'pyarrow'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name='ContextAwareIQA', icon='src/images/icon.ico', debug=False, strip=False, upx=False, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='ContextAwareIQA')
