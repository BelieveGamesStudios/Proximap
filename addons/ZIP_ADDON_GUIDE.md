# Developer Guide: Packaging Proximap Add-ons as ZIP Archives

Proximap supports modular add-on packages distributed as `.zip` archives. This allows developers to include Python scripts, AI/ML model weights (e.g. `.onnx`, `.pt`), configuration files, and custom assets in a self-contained bundle.

---

## 1. Directory & ZIP Structure

An add-on ZIP package can be structured in one of two standard layouts:

### Layout A: Root Folder Package (Recommended)
```text
my_ai_tool.zip/
└── my_ai_tool/
    ├── __init__.py           <-- Addon entry point (contains ProximapAddon class)
    ├── model_runner.py       <-- Model inference code
    └── models/
        └── mesh_simplifier.onnx  <-- AI Model weight file
```

### Layout B: Flat Package
```text
my_ai_tool.zip/
├── __init__.py               <-- Addon entry point
├── model_runner.py
└── models/
    └── mesh_simplifier.onnx
```

---

## 2. Add-on Code Structure (`__init__.py`)

```python
import os
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from addons.addon_base import ProximapAddon

class MyAIModelAddon(ProximapAddon):
    addon_id = "my_ai_tool"
    addon_name = "AI Mesh Optimizer"
    addon_version = "1.0.0"
    addon_description = "Uses ONNX model weights to optimize mesh geometry."
    addon_author = "Dev Team"
    addon_category = "Mesh Editor"
    dependencies = ["onnxruntime", "trimesh"]

    def register(self, mesh_editor):
        super().register(mesh_editor)
        # Determine path to bundled ONNX model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, "models", "mesh_simplifier.onnx")
        print(f"[MY_AI_TOOL] Loaded model from: {self.model_path}")

    def get_panel_widget(self, parent=None):
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)
        
        lbl = QLabel("AI Mesh Optimizer Panel", panel)
        btn = QPushButton("Run Model Optimization", panel)
        btn.clicked.connect(self._run_model)
        
        layout.addWidget(lbl)
        layout.addWidget(btn)
        return panel

    def _run_model(self):
        if self.mesh_editor and hasattr(self.mesh_editor, "viewport"):
            print("Executing model inference...")

# Mandatory addon export
addon = MyAIModelAddon()
```

---

## 3. How Users Install Your Add-on ZIP

1. In Proximap, switch to the **Mesh Editor** tab.
2. Go to **Edit → Preferences — Add-ons...**.
3. Click **Install Add-on...** and choose your `my_ai_tool.zip` file.
4. Proximap will extract the archive to `~/.proximap/addons/my_ai_tool/` and enable the add-on instantly.
