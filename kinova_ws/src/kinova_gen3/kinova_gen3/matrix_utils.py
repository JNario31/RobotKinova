import numpy as np

def compute_transformation(pixel_pts, world_pts):
    """
    Compute a full 2D affine transformation (6 parameters).
    Requires at least 3 non-collinear points.
    
    pixel_pts: Nx2 array/list of (x_pixel, y_pixel)
    world_pts: Nx2 array/list of (x_world, y_world)
    
    Returns: 2x3 matrix:
        [[a, b, tx],
         [c, d, ty]]
    
    Where:
        x_world = a*x_pixel + b*y_pixel + tx
        y_world = c*x_pixel + d*y_pixel + ty
    """
    pixel_pts = np.array(pixel_pts, dtype=float)
    world_pts = np.array(world_pts, dtype=float)
    
    N = len(pixel_pts)
    if N < 3:
        raise ValueError("Need at least 3 points for affine transformation")
    
    # Build the A matrix for least squares: [x_p, y_p, 1]
    A = np.column_stack([pixel_pts, np.ones(N)])
    
    # Solve for x: x_world = A @ [a, b, tx]
    params_x = np.linalg.lstsq(A, world_pts[:, 0], rcond=None)[0]
    
    # Solve for y: y_world = A @ [c, d, ty]
    params_y = np.linalg.lstsq(A, world_pts[:, 1], rcond=None)[0]
    
    # Construct the 2x3 transformation matrix
    matrix = np.array([
        [params_x[0], params_x[1], params_x[2]],
        [params_y[0], params_y[1], params_y[2]]
    ])
    
    return matrix

def apply_transformation(A, pixel_point):
    """
    Apply the 2x3 transform A to a single pixel point (xp, yp).
    """
    xp, yp = pixel_point
    vec = np.array([xp, yp, 1.0])
    return A @ vec  # (x_world, y_world)