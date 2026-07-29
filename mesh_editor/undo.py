from abc import ABC, abstractmethod
from collections import deque
import numpy as np


class BaseCommand(ABC):
    """Abstract Base Class for all Undoable operations.
    
    Addons and extensions can inherit from BaseCommand to create custom undoable actions.
    """
    label: str = "Command"

    @abstractmethod
    def execute(self):
        """Execute the command for the first time."""
        pass

    @abstractmethod
    def undo(self):
        """Revert the action performed by execute()."""
        pass

    @abstractmethod
    def redo(self):
        """Re-apply the action after an undo. Subclasses must implement."""
        pass


class TransformCommand(BaseCommand):
    """Captures gizmo transform edits across one or multiple objects."""
    label = "Transform"

    def __init__(self, states: dict, viewport=None, operation: str = None):
        """
        states dict format:
        { obj: (pos_before, rot_before, scale_before, pos_after, rot_after, scale_after) }
        operation: optional gizmo operation type ('translate', 'rotate', 'scale')
        """
        self.states = states
        self.viewport = viewport
        # Set a descriptive label based on the gizmo operation
        if operation == "translate":
            self.label = "Translate"
        elif operation == "rotate":
            self.label = "Rotate"
        elif operation == "scale":
            self.label = "Scale"
        else:
            self.label = "Transform"

    def execute(self):
        # Initial execution happens during interactive drag, so execute() reapplies 'after' state
        self.redo()

    def undo(self):
        for obj, (pos_b, rot_b, scale_b, _, _, _) in self.states.items():
            obj.position = np.copy(pos_b)
            obj.rotation = np.copy(rot_b)
            obj.scale = np.copy(scale_b)
        if self.viewport:
            self.viewport.update()

    def redo(self):
        for obj, (_, _, _, pos_a, rot_a, scale_a) in self.states.items():
            obj.position = np.copy(pos_a)
            obj.rotation = np.copy(rot_a)
            obj.scale = np.copy(scale_a)
        if self.viewport:
            self.viewport.update()


class SpinboxTransformCommand(BaseCommand):
    """Captures explicit numerical transform edits from sidebar spinboxes."""
    
    def __init__(self, obj, transform_type: str, before_val: np.ndarray, after_val: np.ndarray, viewport=None):
        """
        transform_type: 'position' | 'rotation' | 'scale'
        """
        self.obj = obj
        self.transform_type = transform_type
        self.before_val = np.copy(before_val)
        self.after_val = np.copy(after_val)
        self.viewport = viewport
        self.label = f"Change {transform_type.capitalize()}"

    def execute(self):
        self.redo()

    def undo(self):
        setattr(self.obj, self.transform_type, np.copy(self.before_val))
        if self.viewport:
            self.viewport.update()

    def redo(self):
        setattr(self.obj, self.transform_type, np.copy(self.after_val))
        if self.viewport:
            self.viewport.update()


class DeleteCommand(BaseCommand):
    """Captures object deletion from scene."""
    label = "Delete Object"

    def __init__(self, objects: list, viewport=None, on_scene_changed_cb=None):
        self.objects = list(objects)
        self.viewport = viewport
        self.on_scene_changed_cb = on_scene_changed_cb
        # Store tuples of (object, original_index_in_scene)
        self.indexed_objects = []

    def execute(self):
        if not self.viewport:
            return
        scene = self.viewport.scene
        self.indexed_objects = []
        for obj in self.objects:
            if obj in scene.objects:
                idx = scene.objects.index(obj)
                self.indexed_objects.append((obj, idx))
                scene.remove_object(obj)
        
        # Clear selection if deleted objects were selected
        scene.selected_objects = [o for o in scene.selected_objects if o not in self.objects]
        if scene.active_object in self.objects:
            scene.active_object = scene.selected_objects[-1] if scene.selected_objects else None

        if self.on_scene_changed_cb:
            self.on_scene_changed_cb()
        if self.viewport:
            self.viewport.selection_changed.emit(scene.active_object)
            self.viewport.update()

    def undo(self):
        if not self.viewport:
            return
        scene = self.viewport.scene
        # Re-insert objects at original indices
        for obj, idx in sorted(self.indexed_objects, key=lambda x: x[1]):
            if obj not in scene.objects:
                idx_clamped = min(idx, len(scene.objects))
                scene.objects.insert(idx_clamped, obj)
        
        scene.selected_objects = list(self.objects)
        scene.active_object = self.objects[-1] if self.objects else None

        if self.on_scene_changed_cb:
            self.on_scene_changed_cb()
        if self.viewport:
            self.viewport.selection_changed.emit(scene.active_object)
            self.viewport.update()

    def redo(self):
        self.execute()


class AddObjectCommand(BaseCommand):
    """Captures mesh import or creation into the scene."""
    label = "Add Object"

    def __init__(self, objects: list, viewport=None, on_scene_changed_cb=None):
        self.objects = list(objects)
        self.viewport = viewport
        self.on_scene_changed_cb = on_scene_changed_cb

    def execute(self):
        if not self.viewport:
            return
        scene = self.viewport.scene
        for obj in self.objects:
            if obj not in scene.objects:
                scene.add_object(obj)
        scene.selected_objects = list(self.objects)
        scene.active_object = self.objects[-1] if self.objects else None

        if self.on_scene_changed_cb:
            self.on_scene_changed_cb()
        if self.viewport:
            self.viewport.selection_changed.emit(scene.active_object)
            self.viewport.update()

    def undo(self):
        if not self.viewport:
            return
        scene = self.viewport.scene
        for obj in self.objects:
            if obj in scene.objects:
                scene.remove_object(obj)
        scene.selected_objects = [o for o in scene.selected_objects if o not in self.objects]
        if scene.active_object in self.objects:
            scene.active_object = scene.selected_objects[-1] if scene.selected_objects else None

        if self.on_scene_changed_cb:
            self.on_scene_changed_cb()
        if self.viewport:
            self.viewport.selection_changed.emit(scene.active_object)
            self.viewport.update()

    def redo(self):
        self.execute()


class UndoStack:
    """Unity-style undo/redo stack manager.
    
    Exposes push, undo, redo, and state query methods. Max history size defaults to 30.
    """

    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.undo_stack: deque[BaseCommand] = deque(maxlen=max_size)
        self.redo_stack: deque[BaseCommand] = deque()

    def push(self, cmd: BaseCommand):
        """Pushes a executed command onto the stack and clears the redo history."""
        self.undo_stack.append(cmd)
        self.redo_stack.clear()

    def undo(self) -> str | None:
        """Undoes the top command. Returns command label or None if stack empty."""
        if not self.undo_stack:
            return None
        cmd = self.undo_stack.pop()
        cmd.undo()
        self.redo_stack.append(cmd)
        return cmd.label

    def redo(self) -> str | None:
        """Redoes the top command. Returns command label or None if stack empty."""
        if not self.redo_stack:
            return None
        cmd = self.redo_stack.pop()
        cmd.redo()
        self.undo_stack.append(cmd)
        return cmd.label

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def undo_label(self) -> str:
        return self.undo_stack[-1].label if self.undo_stack else ""

    def redo_label(self) -> str:
        return self.redo_stack[-1].label if self.redo_stack else ""

    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
