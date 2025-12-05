"""
Unit tests for simplex_grid.core module
"""

import unittest
import numpy as np
from simplex_grid import simplex_grid
from simplex_grid.core import fact, choose


class TestSimplexGrid(unittest.TestCase):
    """Test cases for simplex_grid function"""

    def test_simplex_constraint(self):
        """Test that all points satisfy the simplex constraint (sum to 1)"""
        grid = simplex_grid(m=3, r=5)
        for point in grid:
            self.assertTrue(
                np.isclose(np.sum(point), 1.0),
                f"Point {point.T} does not sum to 1"
            )

    def test_non_negative(self):
        """Test that all points have non-negative coordinates"""
        grid = simplex_grid(m=3, r=5)
        for point in grid:
            self.assertTrue(
                np.all(point >= 0),
                f"Point {point.T} has negative coordinates"
            )

    def test_point_shape(self):
        """Test that points have correct shape"""
        m = 4
        grid = simplex_grid(m=m, r=3)
        for point in grid:
            self.assertEqual(point.shape, (m, 1),
                           f"Point has shape {point.shape}, expected ({m}, 1)")

    def test_vertices_r2(self):
        """Test that r=2 returns only vertices"""
        m = 3
        grid = simplex_grid(m=m, r=2)
        self.assertEqual(len(grid), m,
                        f"r=2 should return {m} vertices, got {len(grid)}")

        # Check that we have the standard basis vectors
        grid_sorted = sorted([tuple(p.flatten()) for p in grid])
        expected = sorted([
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0)
        ])
        for got, exp in zip(grid_sorted, expected):
            np.testing.assert_array_almost_equal(got, exp)

    def test_grid_size(self):
        """Test that grid size matches expected binomial coefficient"""
        # For m=3, r=5, we expect C(3+5-2, 3-1) = C(6, 2) = 15 points
        grid = simplex_grid(m=3, r=5)
        expected_size = int(choose(3 + 5 - 2, 3 - 1))
        self.assertEqual(len(grid), expected_size,
                        f"Expected {expected_size} points, got {len(grid)}")

    def test_small_simplex(self):
        """Test a small 2D simplex"""
        grid = simplex_grid(m=2, r=3)
        # Should have 3 points: [1,0], [0.5,0.5], [0,1]
        self.assertEqual(len(grid), 3)

    def test_uniform_spacing(self):
        """Test that grid points are uniformly spaced"""
        grid = simplex_grid(m=2, r=4)
        # For 2D simplex with r=4, coordinates should be multiples of 1/3
        for point in grid:
            for coord in point.flatten():
                # Check that coord * 3 is close to an integer
                self.assertTrue(
                    np.isclose(coord * 3, round(coord * 3)),
                    f"Coordinate {coord} is not a multiple of 1/3"
                )


class TestHelperFunctions(unittest.TestCase):
    """Test cases for helper functions"""

    def test_factorial_base_cases(self):
        """Test factorial base cases"""
        self.assertEqual(fact(0), 1)
        self.assertEqual(fact(1), 1)

    def test_factorial_small_values(self):
        """Test factorial for small values"""
        self.assertEqual(fact(2), 2)
        self.assertEqual(fact(3), 6)
        self.assertEqual(fact(4), 24)
        self.assertEqual(fact(5), 120)

    def test_choose_basic(self):
        """Test binomial coefficient basic cases"""
        self.assertEqual(choose(5, 2), 10)
        self.assertEqual(choose(6, 2), 15)
        self.assertEqual(choose(4, 2), 6)

    def test_choose_invalid_input(self):
        """Test that choose raises ValueError for invalid input"""
        with self.assertRaises(ValueError):
            choose(2, 5)  # m must be > n


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_minimum_dimension(self):
        """Test minimum dimension (2D simplex)"""
        grid = simplex_grid(m=2, r=2)
        self.assertEqual(len(grid), 2)
        self.assertTrue(all(np.isclose(np.sum(p), 1.0) for p in grid))

    def test_minimum_resolution(self):
        """Test minimum resolution (r=2)"""
        grid = simplex_grid(m=4, r=2)
        self.assertEqual(len(grid), 4)
        # Should be the 4 vertices
        self.assertTrue(all(np.isclose(np.sum(p), 1.0) for p in grid))

    def test_larger_dimension(self):
        """Test higher dimensional simplex"""
        grid = simplex_grid(m=5, r=3)
        self.assertTrue(len(grid) > 0)
        self.assertTrue(all(np.isclose(np.sum(p), 1.0) for p in grid))
        self.assertTrue(all(p.shape == (5, 1) for p in grid))


if __name__ == "__main__":
    unittest.main()
