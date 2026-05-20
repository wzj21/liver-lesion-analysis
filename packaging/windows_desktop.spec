# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path.cwd()


block_cipher = None


a = Analysis(
    ["scripts/run_desktop_app.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        ("configs/software_inference.yaml", "configs"),
        ("configs/global_config.yaml", "configs"),
        ("docs/REAL_WORLD_SOFTWARE_GUIDE.md", "docs"),
    ],
    hiddenimports=[
        "src.software.gui_app",
        "src.software.inference_app",
        "src.software.clinical_rules",
        "src.software.llm_advisor",
        "src.pipeline",
        "src.preprocessing.pipeline",
        "src.models.stage1_liver_seg",
        "src.models.stage2_det_cls_deform_seg",
        "src.models.stage3_temporal_cls",
        "src.models.stage4_activity",
        "SimpleITK",
        "nibabel",
        "numpy",
        "yaml",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "jupyter",
        "notebook",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LiverLesionAI",
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
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="LiverLesionAI",
)
