import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from mesh_cleanup import Open3DBackend, TrimeshBackend, NotSupportedError

ROOT = Path(__file__).resolve().parents[1]


class CleanupSafetyTests(unittest.TestCase):
    def test_unsupported_repair_does_not_report_success_or_write_output(self):
        for backend, getter in [(Open3DBackend(), '_get_o3d'),
                                (TrimeshBackend(), '_get_tm')]:
            engine = Mock()
            setattr(backend, getter, lambda: engine)
            with self.subTest(backend=backend.name):
                with self.assertRaisesRegex(NotSupportedError, 'Non-manifold repair'):
                    backend.cleanup('input.ply', 'output.ply', cleanup_params={
                        'repair_nonmanifold': True, 'remove_duplicates': False,
                        'enable_reduction': False, 'close_holes': False})
                self.assertEqual(engine.mock_calls, [])

    def test_trimesh_duplicate_removal_supports_current_api(self):
        backend = TrimeshBackend()
        engine = Mock()
        mesh = engine.load.return_value
        mesh.vertices = [0, 1, 2]
        mesh.faces = [[0, 1, 2]]
        mesh.unique_faces.return_value = [True]
        mesh.export.side_effect = lambda path, **kw: Path(path).write_text('ply')
        backend._get_tm = lambda: engine
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(backend.cleanup('input.ply', directory + '/out.ply',
                                          cleanup_params={'remove_duplicates': True}))
        mesh.update_faces.assert_called_once_with([True])
        mesh.remove_duplicate_faces.assert_not_called()

    def test_busy_clears_all_shared_backup_revert_buttons(self):
        # Execute the real method without requiring a graphical Qt session.
        tree = ast.parse((ROOT / 'viewport_tool_system.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'MeshCleanupToolWindow')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'set_busy')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<set_busy>', 'exec'), namespace)
        window = Mock()
        namespace['set_busy'](window, True)
        for name in ['set_revert_enabled', 'set_holes_revert_enabled',
                     'set_nonmanifold_revert_enabled', 'set_duplicates_revert_enabled']:
            getattr(window, name).assert_called_once_with(False)
        window.apply_nonmanifold_btn.setEnabled.assert_called_once_with(False)
        window.apply_duplicates_btn.setEnabled.assert_called_once_with(False)


if __name__ == '__main__':
    unittest.main()
