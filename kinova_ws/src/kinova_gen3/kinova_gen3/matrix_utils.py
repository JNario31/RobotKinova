import numpy as np

def compute_transformation(pixel_pts, world_pts):
    """
    Compute a constrained transform:
        x_world = sx * x_pixel + tx
        y_world = sy * y_pixel + ty

    pixel_pts: Nx2 array/list of (x_pixel, y_pixel)
    world_pts: Nx2 array/list of (x_world, y_world)

    Returns: 2x3 matrix:
        [[sx, 0,  tx],
         [0,  sy, ty]]
    """
    pixel_pts = np.array(pixel_pts, dtype=float)
    world_pts = np.array(world_pts, dtype=float)

    xp = pixel_pts[:, 0]
    yp = pixel_pts[:, 1]
    xw = world_pts[:, 0]
    yw = world_pts[:, 1]

    # Solve x_world = sx * x_pixel + tx  (least squares)
    A_x = np.vstack([xp, np.ones_like(xp)]).T
    sx, tx = np.linalg.lstsq(A_x, xw, rcond=None)[0]

    # Solve y_world = sy * y_pixel + ty  (least squares)
    A_y = np.vstack([yp, np.ones_like(yp)]).T
    sy, ty = np.linalg.lstsq(A_y, yw, rcond=None)[0]

    A = np.array([
        [sx, 0.0, tx],
        [0.0, sy, ty]
    ])

    return A


def apply_transformation(A, pixel_point):
    """
    Apply the 2x3 transform A to a single pixel point (xp, yp).
    """
    xp, yp = pixel_point
    vec = np.array([xp, yp, 1.0])
    return A @ vec  # (x_world, y_world)