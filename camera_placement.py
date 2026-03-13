"""
Standalone camera placement algorithms extracted from stablegen/render_tools.py.
No Blender dependencies — only math and numpy required.

Input: numpy arrays of face normals, areas, and/or vertex positions.
Output: list of (x, y, z) unit direction vectors for camera placement.
"""

import math
import numpy as np


def fibonacci_sphere_points(n):
    """Generate *n* approximately evenly-spaced unit vectors on a sphere
    using a Fibonacci spiral.  Returns list of (x, y, z) tuples."""
    points = []
    golden_ratio = (1 + math.sqrt(5)) / 2
    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden_ratio
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        points.append((x, y, z))
    return points


def kmeans_on_sphere(directions, weights, k, max_iter=50):
    """Spherical K-means: cluster unit vectors weighted by area.
    Returns (k, 3) numpy array of cluster-centre unit vectors.

    Parameters
    ----------
    directions : (N, 3) numpy array of unit vectors (face normals)
    weights    : (N,) numpy array of weights (face areas)
    k          : number of clusters / cameras
    """
    n_pts = len(directions)
    if n_pts == 0 or k == 0:
        return np.zeros((max(k, 1), 3))
    k = min(k, n_pts)
    rng = np.random.default_rng(42)
    probs = weights / weights.sum()
    indices = rng.choice(n_pts, size=k, replace=False, p=probs)
    centers = directions[indices].copy()
    for _ in range(max_iter):
        dots = directions @ centers.T
        labels = np.argmax(dots, axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if mask.any():
                ws = (directions[mask] * weights[mask, np.newaxis]).sum(axis=0)
                nrm = np.linalg.norm(ws)
                new_centers[j] = ws / nrm if nrm > 0 else centers[j]
            else:
                new_centers[j] = centers[j]
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers
    return centers


def compute_pca_axes(verts):
    """Return the 3 principal axes of a (N, 3) vertex array (rows of a
    3x3 array, sorted by descending eigenvalue).

    Parameters
    ----------
    verts : (N, 3) numpy array of vertex positions
    """
    mean = verts.mean(axis=0)
    centered = verts - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx].T  # rows = principal axes


def greedy_coverage_directions(normals, areas, max_cameras=12,
                               coverage_target=0.95, n_candidates=200,
                               existing_dirs=None):
    """Greedy set-cover: iteratively pick the camera direction that adds the
    most newly-visible surface area (back-face culling only, no occlusion,
    for speed).  Returns (selected_directions, final_coverage_fraction).

    Parameters
    ----------
    normals          : (N, 3) numpy array of unit face normals
    areas            : (N,) numpy array of face areas
    max_cameras      : hard upper limit on cameras placed
    coverage_target  : stop early when this fraction of area is covered (0–1)
    n_candidates     : number of sphere sample directions to evaluate
    existing_dirs    : list of (x,y,z) directions already covered (pre-seed)
    """
    total_area = areas.sum()
    if total_area <= 0:
        return [], 0.0

    candidates = np.array(fibonacci_sphere_points(n_candidates))

    # visibility[i, j] = True if face i faces toward candidate j
    # cos(75°) ≈ 0.26 – ignore near-grazing faces that won't texture well
    visibility = normals @ candidates.T > 0.26  # (n_faces, n_candidates)

    covered = np.zeros(len(areas), dtype=bool)
    if existing_dirs:
        for edir in existing_dirs:
            covered |= normals @ np.asarray(edir, dtype=float) > 0.26
    selected = []

    for _ in range(max_cameras):
        uncovered = ~covered
        new_vis = visibility & uncovered[:, np.newaxis]
        new_areas = (new_vis * areas[:, np.newaxis]).sum(axis=0)

        best_idx = int(np.argmax(new_areas))
        if new_areas[best_idx] < total_area * 0.005:  # < 0.5% new coverage
            break

        selected.append(candidates[best_idx].copy())
        covered |= visibility[:, best_idx]

        coverage = float(areas[covered].sum() / total_area)
        if coverage >= coverage_target:
            break

    final_coverage = float(areas[covered].sum() / total_area) if covered.any() else 0.0
    return selected, final_coverage


def orbit_ring_directions(n, elevation_deg=30.0, start_angle_deg=0.0):
    """Place *n* cameras evenly around the Z axis at a fixed elevation.
    Equivalent to Blender's Orbit Ring strategy.

    Parameters
    ----------
    n               : number of cameras
    elevation_deg   : camera elevation above the XY plane (-90..90)
    start_angle_deg : azimuth of the first camera (degrees)

    Returns (n, 3) numpy array of unit direction vectors pointing
    *toward* the origin (i.e. the camera looks inward).
    """
    el = math.radians(elevation_deg)
    dirs = []
    for i in range(n):
        az = math.radians(start_angle_deg + 360.0 * i / n)
        x = math.cos(el) * math.cos(az)
        y = math.cos(el) * math.sin(az)
        z = math.sin(el)
        dirs.append([x, y, z])
    return np.array(dirs)


def fan_arc_directions(n, forward=(1.0, 0.0, 0.0), fan_angle_deg=90.0,
                       elevation_deg=0.0):
    """Spread *n* cameras in an arc centred on *forward* within *fan_angle_deg*.
    Equivalent to Blender's Fan Arc strategy.

    Parameters
    ----------
    n               : number of cameras
    forward         : (x, y, z) reference/centre direction of the arc
    fan_angle_deg   : total angular width of the arc (degrees)
    elevation_deg   : tilt all cameras up/down from the horizontal plane

    Returns (n, 3) numpy array of unit direction vectors.
    """
    fwd = np.asarray(forward, dtype=float)
    fwd /= np.linalg.norm(fwd) + 1e-12

    # Build a right-vector perpendicular to fwd in the XY plane
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:          # fwd is nearly vertical
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)

    fan_rad = math.radians(fan_angle_deg)
    el_rad  = math.radians(elevation_deg)

    dirs = []
    for i in range(n):
        t = (i / max(n - 1, 1)) - 0.5        # -0.5 .. +0.5
        az = t * fan_rad
        # Rotate fwd around up by az, then tilt by elevation
        d = fwd * math.cos(az) + right * math.sin(az)
        d = d * math.cos(el_rad) + up * math.sin(el_rad)
        d /= np.linalg.norm(d) + 1e-12
        dirs.append(d)
    return np.array(dirs)


def visibility_weighted_directions(normals, areas, k, n_candidates=200,
                                   balance=0.5):
    """K-means placement weighted by per-face back-face visibility count.
    Approximates Blender's Interactive Visibility-Weighted strategy
    (back-face culling only — no BVH ray occlusion).

    Parameters
    ----------
    normals      : (N, 3) unit face normals
    areas        : (N,) face areas
    k            : number of cameras
    n_candidates : sphere samples used to estimate visibility fraction
    balance      : 0.0 = full vis-weighting, 1.0 = uniform (area only)

    Returns (k, 3) numpy array of cluster-centre unit vectors.
    """
    candidates = np.array(fibonacci_sphere_points(n_candidates))
    # vis_count[i] = number of candidate directions that see face i
    vis_count = (normals @ candidates.T > 0.26).sum(axis=1).astype(float)
    vis_fraction = vis_count / (n_candidates + 1e-12)

    # weight = area × (vis_fraction + balance × (1 − vis_fraction))
    weighted_areas = areas * (vis_fraction + balance * (1.0 - vis_fraction))
    # Exclude faces never seen from any direction
    visible = vis_count > 0
    return kmeans_on_sphere(normals[visible], weighted_areas[visible], k=k)


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Hemisphere: 8 evenly distributed directions
    dirs = fibonacci_sphere_points(8)
    print("Hemisphere directions:")
    for d in dirs:
        print(f"  {d}")

    # Synthetic mesh: random normals + areas
    rng = np.random.default_rng(0)
    normals = rng.standard_normal((500, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    areas = rng.uniform(0.1, 1.0, 500)

    # Normal-Weighted K-means: 6 cameras
    directions = kmeans_on_sphere(normals, areas, k=6)
    print(f"\nK-means camera directions (k=6):\n{directions}")

    # PCA axes from random verts
    verts = rng.standard_normal((200, 3))
    verts[:, 0] *= 3  # stretch along X
    axes = compute_pca_axes(verts)
    print(f"\nPCA axes:\n{axes}")

    # Greedy coverage
    selected, coverage = greedy_coverage_directions(normals, areas)
    print(f"\nGreedy coverage: {len(selected)} cameras, {coverage:.1%} covered")
