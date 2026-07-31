"""
Base class definition for all Proximap Add-ons.
Add-ons must inherit from ProximapAddon and define class-level metadata.
Proximap Add-ons extend the functionality of the Mesh Editor workspace.
"""

from typing import Optional, List
from PySide6.QtWidgets import QWidget

class ProximapAddon:
    """Base class every Proximap add-on must inherit from."""
    addon_id: str = "base_addon"
    addon_name: str = "Base Add-on"
    addon_version: str = "1.0.0"
    addon_description: str = "Base add-on description"
    addon_author: str = "Unknown"
    addon_category: str = "Mesh Editor"
    dependencies: List[str] = []  # List of python package names required by this addon

    def __init__(self):
        self.is_enabled = False
        self.mesh_editor = None

    def register(self, mesh_editor) -> None:
        """
        Called when the add-on is enabled by the user or at application launch.
        Use this method to attach UI elements, hooks, or event listeners to the Mesh Editor.
        """
        self.mesh_editor = mesh_editor
        self.is_enabled = True

    def unregister(self, mesh_editor) -> None:
        """
        Called when the add-on is disabled by the user or at application shutdown.
        Use this method to detach UI elements and clean up resources from the Mesh Editor.
        """
        self.mesh_editor = None
        self.is_enabled = False

    def get_panel_widget(self, parent: Optional[QWidget] = None) -> Optional[QWidget]:
        """
        Optional: Return a QWidget to be embedded into the Mesh Editor sidebar's Add-on section.
        Return None if this add-on does not contribute a sidebar panel.
        """
        return None
