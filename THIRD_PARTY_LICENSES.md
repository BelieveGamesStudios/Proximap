# Third-Party Licenses

Proximap utilizes several third-party open-source libraries and toolchains. Below is a list of the components, their licenses, and links to their repositories.

## 1. OpenMVS
* **License**: GNU Affero General Public License v3 (AGPL v3)
* **Description**: Multi-View Stereo library used for dense point cloud generation, mesh reconstruction, mesh refinement, and texturing.
* **Usage**: Native binaries executed as subprocesses.
* **Project Page**: [https://github.com/cdcseacave/openMVS](https://github.com/cdcseacave/openMVS)

## 2. COLMAP
* **License**: New BSD License (3-clause BSD)
* **Description**: Structure-from-Motion (SfM) toolchain used for camera registration and sparse point cloud generation.
* **Usage**: Native binaries executed as subprocesses.
* **Project Page**: [https://colmap.github.io/](https://colmap.github.io/)

## 3. PySide6 (Qt)
* **License**: GNU Lesser General Public License v3 (LGPL v3)
* **Description**: Python bindings for the Qt GUI framework, used to build the desktop frontend interface.
* **Usage**: Dynamic import in Python.
* **Project Page**: [https://wiki.qt.io/Qt_for_Python](https://wiki.qt.io/Qt_for_Python)

## 4. NumPy
* **License**: BSD License
* **Description**: Scientific computing library.
* **Usage**: Python dependency.
* **Project Page**: [https://numpy.org/](https://numpy.org/)

## 5. Pillow (PIL)
* **License**: HPND License (Historical Permission Notice and Disclaimer)
* **Description**: Image processing library used to read image metadata (EXIF data).
* **Usage**: Python dependency.
* **Project Page**: [https://python-pillow.org/](https://python-pillow.org/)

---

## Compliance Notes for Open-Sourcing

* **Copyleft Compliance**: Because **OpenMVS** is licensed under **AGPL v3**, if you distribute Proximap as a packaged tool (including the OpenMVS binaries), the combined application is subject to AGPL/GPL copyleft requirements. Open-sourcing your Python GUI under **GPL v3** (as currently configured in the `LICENSE` file) satisfies this condition.
* **Attribution**: The licenses above require including their copyright notices in binary distributions. Be sure to bundle this third-party licenses document alongside your compiled releases.
