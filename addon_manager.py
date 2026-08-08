import os
import sys
import json
import zipfile
import shutil
import importlib.util
import inspect
from typing import Dict, List, Type, Optional
from PySide6.QtCore import QObject, Signal

from addons.addon_base import ProximapAddon

def get_user_addons_dir() -> str:
    user_home = os.path.expanduser("~")
    user_addons = os.path.join(user_home, ".proximap", "addons")
    os.makedirs(user_addons, exist_ok=True)
    return user_addons

def get_builtin_addons_dir() -> str:
    if getattr(sys, 'frozen', False):
        builtin_dir = os.path.join(os.path.dirname(sys.executable), "addons")
        if not os.path.exists(builtin_dir):
            meipass = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            builtin_dir = os.path.join(meipass, "addons")
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        builtin_dir = os.path.join(base_dir, "addons")
    os.makedirs(builtin_dir, exist_ok=True)
    return builtin_dir

def get_addon_prefs_file() -> str:
    user_home = os.path.expanduser("~")
    prefs_dir = os.path.join(user_home, ".proximap")
    os.makedirs(prefs_dir, exist_ok=True)
    return os.path.join(prefs_dir, "addon_prefs.json")


class AddonManager(QObject):
    addon_state_changed = Signal(str, bool)  # addon_id, is_enabled

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.discovered_addons: Dict[str, Type[ProximapAddon]] = {}
        self.active_instances: Dict[str, ProximapAddon] = {}
        self.enabled_states: Dict[str, bool] = {}
        self._load_prefs()

    def _load_prefs(self):
        prefs_file = get_addon_prefs_file()
        if os.path.exists(prefs_file):
            try:
                with open(prefs_file, "r", encoding="utf-8") as f:
                    self.enabled_states = json.load(f)
            except Exception as e:
                print(f"[ADDONS] Warning: Failed to read addon preferences: {e}")
                self.enabled_states = {}
        else:
            # Default state: mesh_inspector enabled by default
            self.enabled_states = {"mesh_inspector": True}

    def save_prefs(self):
        prefs_file = get_addon_prefs_file()
        try:
            with open(prefs_file, "w", encoding="utf-8") as f:
                json.dump(self.enabled_states, f, indent=2)
        except Exception as e:
            print(f"[ADDONS] Error saving addon preferences: {e}")

    @staticmethod
    def _load_addon_from_spec(mod_name: str, filepath: str, discovered_dict: dict):
        try:
            spec = importlib.util.spec_from_file_location(mod_name, filepath)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, ProximapAddon) and obj is not ProximapAddon:
                        addon_id = getattr(obj, "addon_id", None)
                        if addon_id:
                            discovered_dict[addon_id] = obj
        except Exception as e:
            print(f"[ADDONS] Failed to load addon module from {filepath}: {e}")

    @staticmethod
    def discover_addons() -> List[Type[ProximapAddon]]:
        """Scans built-in and user add-on directories (files and package folders) for valid ProximapAddon subclasses."""
        search_dirs = [get_builtin_addons_dir(), get_user_addons_dir()]
        discovered = {}

        for sdir in search_dirs:
            if not os.path.exists(sdir):
                continue

            if sdir not in sys.path:
                sys.path.insert(0, sdir)

            for item in os.listdir(sdir):
                if item.startswith("_") or item.startswith("."):
                    continue

                item_path = os.path.join(sdir, item)

                # Case 1: Standalone .py file
                if os.path.isfile(item_path) and item.endswith(".py") and item != "addon_base.py":
                    mod_name = f"proximap_addon_{os.path.splitext(item)[0]}"
                    AddonManager._load_addon_from_spec(mod_name, item_path, discovered)

                # Case 2: Extracted package directory
                elif os.path.isdir(item_path):
                    if item_path not in sys.path:
                        sys.path.insert(0, item_path)

                    candidates = []
                    init_py = os.path.join(item_path, "__init__.py")
                    main_py = os.path.join(item_path, "main.py")
                    if os.path.isfile(init_py):
                        candidates.append(init_py)
                    elif os.path.isfile(main_py):
                        candidates.append(main_py)
                    else:
                        for sub_f in os.listdir(item_path):
                            if sub_f.endswith(".py") and not sub_f.startswith("_"):
                                candidates.append(os.path.join(item_path, sub_f))

                    for cand in candidates:
                        mod_name = f"proximap_addon_pkg_{item}_{os.path.splitext(os.path.basename(cand))[0]}"
                        AddonManager._load_addon_from_spec(mod_name, cand, discovered)

        return list(discovered.values())

    def refresh_discovery(self):
        classes = AddonManager.discover_addons()
        self.discovered_addons = {cls.addon_id: cls for cls in classes}

    def install_addon_from_file(self, file_path: str) -> tuple[bool, str]:
        """Installs a .zip archive or single .py add-on into the user add-ons directory."""
        if not os.path.exists(file_path):
            return False, "File does not exist."

        dest_dir = get_user_addons_dir()
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".py":
                filename = os.path.basename(file_path)
                target_path = os.path.join(dest_dir, filename)
                shutil.copy2(file_path, target_path)
                self.refresh_discovery()
                return True, f"Add-on '{filename}' installed successfully!"

            elif ext == ".zip":
                stem = os.path.splitext(os.path.basename(file_path))[0]
                extract_target = os.path.join(dest_dir, stem)

                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    namelist = zip_ref.namelist()
                    first_parts = {p.split('/')[0] for p in namelist if p and not p.startswith('__MACOSX')}
                    
                    if len(first_parts) == 1 and list(first_parts)[0]:
                        zip_ref.extractall(dest_dir)
                    else:
                        os.makedirs(extract_target, exist_ok=True)
                        zip_ref.extractall(extract_target)

                self.refresh_discovery()
                return True, f"Add-on package '{stem}' extracted and installed successfully!"

            else:
                return False, f"Unsupported file type '{ext}'. Please select a .zip package or .py script."

        except Exception as e:
            import traceback
            err_msg = str(e)
            traceback.print_exc()
            return False, f"Failed to install add-on: {err_msg}"


    def initialize_addons(self, main_window):
        self.main_window = main_window
        self.refresh_discovery()

        for addon_id, cls in self.discovered_addons.items():
            # Check enabled state, default to enabled for mesh_inspector if missing
            should_enable = self.enabled_states.get(addon_id, (addon_id == "mesh_inspector"))
            if should_enable:
                self.enable_addon(addon_id)

    def is_enabled(self, addon_id: str) -> bool:
        return addon_id in self.active_instances and self.active_instances[addon_id].is_enabled

    def enable_addon(self, addon_id: str) -> bool:
        if addon_id not in self.discovered_addons:
            print(f"[ADDONS] Cannot enable unknown addon: {addon_id}")
            return False

        if self.is_enabled(addon_id):
            return True

        try:
            cls = self.discovered_addons[addon_id]
            instance = cls()
            instance.register(self.main_window)
            self.active_instances[addon_id] = instance
            self.enabled_states[addon_id] = True
            self.save_prefs()
            self.addon_state_changed.emit(addon_id, True)
            print(f"[ADDONS] Enabled addon: '{instance.addon_name}' ({addon_id})")
            return True
        except Exception as e:
            print(f"[ADDONS] Error enabling addon '{addon_id}': {e}")
            import traceback
            traceback.print_exc()
            return False

    def disable_addon(self, addon_id: str) -> bool:
        if not self.is_enabled(addon_id):
            self.enabled_states[addon_id] = False
            self.save_prefs()
            self.addon_state_changed.emit(addon_id, False)
            return True

        try:
            instance = self.active_instances[addon_id]
            instance.unregister(self.main_window)
            del self.active_instances[addon_id]
            self.enabled_states[addon_id] = False
            self.save_prefs()
            self.addon_state_changed.emit(addon_id, False)
            print(f"[ADDONS] Disabled addon: '{instance.addon_name}' ({addon_id})")
            return True
        except Exception as e:
            print(f"[ADDONS] Error disabling addon '{addon_id}': {e}")
            return False

    def toggle_addon(self, addon_id: str) -> bool:
        if self.is_enabled(addon_id):
            return self.disable_addon(addon_id)
        else:
            return self.enable_addon(addon_id)

    def get_enabled_instances(self) -> List[ProximapAddon]:
        return list(self.active_instances.values())
