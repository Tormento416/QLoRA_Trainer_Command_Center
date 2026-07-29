# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

# These libraries do heavy dynamic/conditional importing internally
# (lazy submodule loading, plugin-style registration) that PyInstaller's
# static analyzer can miss. collect_all() grabs their submodules, data
# files, and binaries explicitly so the frozen exe doesn't throw
# ModuleNotFoundError at runtime for something only imported conditionally.
# Depends on: Finetune/qlora_trainer.py existing at build time.
# Downstream: Analysis() below consumes these three lists directly.
datas = []
binaries = []
hiddenimports = ['Finetune.qlora_trainer']

for pkg in ['transformers', 'peft', 'bitsandbytes', 'datasets', 'accelerate', 'huggingface_hub']:
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

import os, huggingface_hub
hf_tmpl = os.path.join(os.path.dirname(huggingface_hub.__file__), 'templates')
if os.path.exists(hf_tmpl):
    datas.append((hf_tmpl, 'huggingface_hub/templates'))

a = Analysis(
    ['rez_slm_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Rez_SLM_Control_Center',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Rez_SLM_Control_Center',
)