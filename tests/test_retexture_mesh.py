import unittest

import numpy as np

from main_window import _weld_exact_duplicate_vertices


class RetextureMeshTests(unittest.TestCase):
    def test_welds_expanded_obj_corners_and_restores_adjacency(self):
        points = np.array(
            [
                [0, 0, 0], [1, 0, 0], [1, 1, 0],
                [0, 0, 0], [1, 1, 0], [0, 1, 0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        colors = np.arange(18, dtype=np.uint8).reshape(6, 3)

        welded_points, welded_faces, welded_colors = _weld_exact_duplicate_vertices(
            points, faces, colors
        )

        self.assertEqual(len(welded_points), 4)
        self.assertEqual(welded_faces.shape, (2, 3))
        self.assertEqual(len(set(welded_faces[0]) & set(welded_faces[1])), 2)
        self.assertEqual(len(welded_colors), 4)

    def test_drops_faces_that_become_degenerate_after_welding(self):
        points = np.array(
            [[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        _, welded_faces, _ = _weld_exact_duplicate_vertices(points, faces)

        self.assertEqual(welded_faces.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
