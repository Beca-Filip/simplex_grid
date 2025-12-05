"""
Example usage of the simplex_grid package
"""

from simplex_grid import simplex_grid
import numpy as np

def main():
    print("=== simplex_grid Example ===\n")

    # Example 1: Simple 3D simplex
    print("Example 1: 3D simplex with resolution 3")
    grid = simplex_grid(m=3, r=3)
    print(f"  Number of points: {len(grid)}")
    print(f"  First point:\n{grid[0]}")
    print(f"  Sum of first point: {np.sum(grid[0])}\n")

    # Example 2: Different resolutions
    print("Example 2: Comparing different resolutions")
    for r in [2, 3, 5, 10]:
        grid = simplex_grid(m=3, r=r)
        print(f"  r={r}: {len(grid)} points")
    print()

    # Example 3: Verify simplex constraint
    print("Example 3: Verify all points satisfy simplex constraint")
    grid = simplex_grid(m=4, r=5)
    all_valid = all(
        np.isclose(np.sum(point), 1.0) and np.all(point >= 0)
        for point in grid
    )
    print(f"  All points valid: {all_valid}")
    print(f"  Total points generated: {len(grid)}\n")

    print("=== All examples completed successfully! ===")

if __name__ == "__main__":
    main()
