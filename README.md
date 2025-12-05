# simplex_grid

Generate uniform grids of points on probability simplexes.

## Installation

```bash
pip install simplex_grid
```

## Usage

```python
from simplex_grid import simplex_grid
import numpy as np

# Generate a grid on a 3-dimensional simplex with resolution 3
grid = simplex_grid(m=3, r=3)

# Print the number of points
print(f"Number of points: {len(grid)}")

# Each point is a numpy array with shape (m, 1)
print(f"First point:\n{grid[0]}")

# Verify that each point sums to 1
for point in grid:
    assert np.isclose(np.sum(point), 1.0)
```

## API Reference

### `simplex_grid(m, r)`

Generate a uniform grid of points on the (m-1)-dimensional probability simplex.

**Parameters:**
- `m` (int): Dimension of the simplex (number of objectives). The simplex will be (m-1)-dimensional embedded in R^m.
- `r` (int): Resolution parameter controlling grid density. The grid will have (r-1) divisions along each dimension, resulting in approximately C(m+r-2, m-1) points.

**Returns:**
- `list of ndarray`: List of points on the simplex. Each point is an (m, 1) array with non-negative entries that sum to 1.

**Notes:**
- The simplex is defined as: {w in R^m : w_i >= 0, sum(w_i) = 1}
- For r=2, returns only the vertices of the simplex
- For r=3, adds midpoints of edges
- Higher r values create denser grids

## Examples

### Example 1: Simple 2D simplex

```python
from simplex_grid import simplex_grid

# Generate points on a 2D simplex (triangle)
grid = simplex_grid(m=2, r=5)
print(f"Generated {len(grid)} points")

# Visualize the points
import matplotlib.pyplot as plt
points = [p.flatten() for p in grid]
plt.scatter([p[0] for p in points], [p[1] for p in points])
plt.xlabel('w_0')
plt.ylabel('w_1')
plt.title('Points on 2D Simplex')
plt.show()
```

### Example 2: 3D simplex for multi-objective optimization

```python
from simplex_grid import simplex_grid
import numpy as np

# Generate weight vectors for 3-objective optimization
m = 3  # number of objectives
r = 11  # resolution (higher = more points)

weights = simplex_grid(m=m, r=r)
print(f"Generated {len(weights)} weight vectors")

# Use in multi-objective optimization
for w in weights:
    # Each w is a weight vector with shape (3, 1)
    # Use it to combine multiple objectives
    # combined_objective = w[0] * obj1 + w[1] * obj2 + w[2] * obj3
    pass
```

### Example 3: Check density

```python
from simplex_grid import simplex_grid

# Compare different resolutions
for r in [2, 3, 5, 10]:
    grid = simplex_grid(m=3, r=r)
    print(f"r={r}: {len(grid)} points")
```

Output:
```
r=2: 3 points
r=3: 6 points
r=5: 15 points
r=10: 66 points
```

## Mathematical Background

The function generates a uniform grid on the probability simplex defined as:

$$\Delta^{m-1} = \{w \in \mathbb{R}^m : w_i \geq 0, \sum_{i=1}^m w_i = 1\}$$

The algorithm works by:
1. Generating all integer vectors j where each component satisfies 0 ≤ j_i ≤ r-1 and Σj_i = r-1
2. Normalizing each vector by dividing by (r-1) to get points on the simplex

The number of points generated is approximately:

$$\binom{m+r-2}{m-1} = \frac{(m+r-2)!}{(m-1)!(r-1)!}$$

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
