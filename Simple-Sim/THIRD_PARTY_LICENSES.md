# Third-Party Licenses (Simple-Sim)

Simple-Sim depends on third-party software packages. Those packages are **not**
covered by this repository's proprietary license; they remain under their own
respective licenses.

This file is a placeholder for future production/release use. Before any
distribution, fill in **version + license + required notices** for each
dependency actually shipped.

## Runtime / Dev Dependencies (from `requirements.txt`)

- numpy
- opencv-python
- torch
- torchvision
- scikit-learn
- PyYAML
- tqdm
- pytest
- pillow

## Vendored Files

- three.js 0.170.0 (MIT, Copyright 2010-2024 three.js authors): `web/board3d/vendor/three/`
  (`build/three.module.js`, `OrbitControls`, `TransformControls`, `GLTFLoader`, `BufferGeometryUtils`), full
  license text in `web/board3d/vendor/three/LICENSE`. Used by the 3D-Board-Renderer viewer (`tools/board3d_viewer.py`).

## How To Update For A Release

1. Freeze exact versions (for example by generating a lock file or a pinned
   `requirements.txt` with `==` versions).
2. Generate a license report for the frozen environment.
3. Copy any required license texts / notices into this file (or into separate
   files under `Simple-Sim/third_party/`).

Notes:
- Some dependencies pull in transitive dependencies; those must be included too.
- License fields in package metadata can be incomplete; verify against the
  upstream repository license files when preparing a public release.
