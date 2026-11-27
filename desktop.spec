# Build from the project root: python -m PyInstaller desktop.spec
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs, copy_metadata
from src.training_data import IMPLEMENTATION_FILES

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=collect_dynamic_libs('rawpy'),
    # Retain the implementation evidence used to validate trained models.
    datas=[('src/images/icon.png', 'src/images'), ('src/images/splash-mirrored-dusk.png', 'src/images')] + [('src/' + name, 'src') for name in IMPLEMENTATION_FILES] + copy_metadata('rawpy') + copy_metadata('ExifRead'),
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
