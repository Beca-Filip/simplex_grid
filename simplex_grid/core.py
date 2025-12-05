import numpy as np

def simplex_grid(m, r):
    """
    Generate a uniform grid of points on the (m-1)-dimensional probability simplex.

    Parameters
    ----------
    m : int
        Dimension of the simplex (number of objectives). The simplex will be
        (m-1)-dimensional embedded in R^m.
    r : int
        Resolution parameter controlling grid density. The grid will have
        (r-1) divisions along each dimension, resulting in approximately
        C(m+r-2, m-1) points.

    Returns
    -------
    list of ndarray
        List of points on the simplex. Each point is an (m, 1) array with
        non-negative entries that sum to 1.

    Notes
    -----
    The simplex is defined as: {w in R^m : w_i >= 0, sum(w_i) = 1}.
    For r=2, returns only the vertices of the simplex.
    For r=3, adds midpoints of edges.
    Higher r values create denser grids.

    Examples
    --------
    >>> grid = simplex_grid(3, 3)  # 3D simplex with resolution 3
    >>> len(grid)
    6
    """
    d = 0
    j = np.zeros((m, 1))
    sigma = 0
    grid = []
    grid = simplex_grid_rec(d, m, r, j, sigma, grid)
    return grid


def simplex_grid_rec(d, m, r, j, sigma, grid):
    """
    Recursive helper function for generating simplex grid points.

    Parameters
    ----------
    d : int
        Current dimension being processed (0 to m-1).
    m : int
        Total number of dimensions (simplex dimension).
    r : int
        Resolution parameter.
    j : ndarray
        Current point being constructed, shape (m, 1).
    sigma : int
        Sum of coordinates assigned so far.
    grid : list
        Accumulator for grid points.

    Returns
    -------
    list of ndarray
        Updated grid with new points added.

    Notes
    -----
    This function recursively generates integer lattice points that satisfy
    the constraint sum(j_i) = r-1, then normalizes by (r-1) to get simplex
    points. The algorithm uses a combinatorial enumeration strategy.
    """
    if d >= m - 1:
        j[d] = (r-1) - sigma
        grid.append(j / (r-1))
        return grid

    for i in range(r - sigma):
        j[d] = i
        sigmai = sigma + i
        grid = simplex_grid_rec(d+1, m, r, j, sigmai, grid)
    return grid


def fact(n):
    """
    Compute factorial of n.

    Parameters
    ----------
    n : int
        Non-negative integer.

    Returns
    -------
    int
        Factorial of n.
    """
    if n > 1:
        return fact(n-1) * n
    return 1


def choose(m, n):
    """
    Compute binomial coefficient C(m, n).

    Parameters
    ----------
    m : int
        Number of items.
    n : int
        Number of selections.

    Returns
    -------
    float
        Binomial coefficient.

    Raises
    ------
    ValueError
        If m is not greater than n.
    """
    if m > n:
        return fact(m) / (fact(m-n) * fact(n))
    else:
        raise ValueError("m must be > n.")