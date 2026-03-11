"""
Camera placement CLI — standalone, no Blender required.

Usage:
    python camera_placement_cli.py model.obj <mode> [options]

Modes (matching StableGen's 7 strategies):
    1  Orbit Ring              — requires Blender (skipped)
    2  Fan Arc                 — requires Blender (skipped)
    3  Hemisphere              — Fibonacci sphere distribution
    4  PCA Axes                — cameras along principal mesh axes
    5  Normal-Weighted K-means — cluster face normals by area weight
    6  Greedy Coverage         — iterative max-coverage selection
    7  Interactive Visibility  — requires Blender (skipped)

Options:
    --cameras N       Number of cameras / K  (default: 6)
    --coverage F      Greedy coverage target 0.0-1.0  (default: 0.95)
    --output FORMAT   Output format: pretty | csv | json  (default: pretty)
    --out FILE        Write output to file instead of stdout

Examples:
    python camera_placement_cli.py mesh.obj 5
    python camera_placement_cli.py mesh.obj 6 --cameras 8 --coverage 0.90
    python camera_placement_cli.py mesh.obj 3 --cameras 12 --output json --out cams.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np

from camera_placement import (
    fibonacci_sphere_points,
    kmeans_on_sphere,
    compute_pca_axes,
    greedy_coverage_directions,
)

# ── OBJ loader (same as gui) ───────────────────────────────────────────────────

def load_obj(path):
    verts = []
    faces = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])

    if not verts or not faces:
        raise ValueError(f"No geometry found in {path}")

    V = np.array(verts, dtype=float)
    F = np.clip(np.array(faces, dtype=int), 0, len(V) - 1)

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area_vec = np.linalg.norm(cross, axis=1)
    areas = area_vec * 0.5

    valid = areas > 1e-12
    cross, areas, area_vec = cross[valid], areas[valid], area_vec[valid]
    normals = cross / area_vec[:, np.newaxis]

    return normals, areas, V

# ── Strategies ─────────────────────────────────────────────────────────────────

BLENDER_ONLY = {
    1: "Orbit Ring",
    2: "Fan Arc",
    7: "Interactive Visibility-Weighted",
}

def strategy_hemisphere(normals, areas, verts, n_cams, coverage):
    dirs = np.array(fibonacci_sphere_points(n_cams))
    return dirs, f"Hemisphere — {len(dirs)} directions"

def strategy_pca(normals, areas, verts, n_cams, coverage):
    axes = compute_pca_axes(verts)
    dirs = np.vstack([axes, -axes])[:n_cams]
    return dirs, f"PCA Axes — {len(dirs)} directions (max 6)"

def strategy_kmeans(normals, areas, verts, n_cams, coverage):
    dirs = kmeans_on_sphere(normals, areas, k=n_cams)
    return dirs, f"Normal-Weighted K-means — {len(dirs)} cameras"

def strategy_greedy(normals, areas, verts, n_cams, coverage):
    selected, frac = greedy_coverage_directions(
        normals, areas, max_cameras=n_cams, coverage_target=coverage
    )
    dirs = np.array(selected) if selected else np.zeros((1, 3))
    return dirs, f"Greedy Coverage — {len(dirs)} cameras, {frac:.1%} covered"

STRATEGIES = {
    3: ("Hemisphere",              strategy_hemisphere),
    4: ("PCA Axes",                strategy_pca),
    5: ("Normal-Weighted K-means", strategy_kmeans),
    6: ("Greedy Coverage",         strategy_greedy),
}

# ── Output formatters ──────────────────────────────────────────────────────────

def fmt_pretty(dirs, label, mesh_info):
    lines = []
    lines.append(f"Strategy : {label}")
    lines.append(f"Mesh     : {mesh_info['file']}  "
                 f"({mesh_info['faces']} faces, {mesh_info['verts']} verts)")
    lines.append(f"Cameras  : {len(dirs)}")
    lines.append("")
    lines.append(f"  {'#':>3}   {'X':>9}   {'Y':>9}   {'Z':>9}   "
                 f"{'Az(°)':>7}   {'El(°)':>7}")
    lines.append("  " + "-" * 58)
    for i, d in enumerate(dirs):
        d = d / (np.linalg.norm(d) + 1e-12)
        az = math.degrees(math.atan2(d[1], d[0]))
        el = math.degrees(math.asin(float(np.clip(d[2], -1, 1))))
        lines.append(f"  {i+1:>3}   {d[0]:>+9.4f}   {d[1]:>+9.4f}   {d[2]:>+9.4f}"
                     f"   {az:>+7.1f}   {el:>+7.1f}")
    return "\n".join(lines)

def fmt_csv(dirs, label, mesh_info):
    lines = ["index,x,y,z,azimuth_deg,elevation_deg"]
    for i, d in enumerate(dirs):
        d = d / (np.linalg.norm(d) + 1e-12)
        az = math.degrees(math.atan2(d[1], d[0]))
        el = math.degrees(math.asin(float(np.clip(d[2], -1, 1))))
        lines.append(f"{i+1},{d[0]:.6f},{d[1]:.6f},{d[2]:.6f},{az:.4f},{el:.4f}")
    return "\n".join(lines)

def fmt_json(dirs, label, mesh_info):
    cameras = []
    for i, d in enumerate(dirs):
        d = d / (np.linalg.norm(d) + 1e-12)
        az = math.degrees(math.atan2(d[1], d[0]))
        el = math.degrees(math.asin(float(np.clip(d[2], -1, 1))))
        cameras.append({
            "index": i + 1,
            "direction": [round(float(d[0]), 6), round(float(d[1]), 6), round(float(d[2]), 6)],
            "azimuth_deg": round(az, 4),
            "elevation_deg": round(el, 4),
        })
    return json.dumps({
        "strategy": label,
        "mesh": mesh_info,
        "cameras": cameras,
    }, indent=2)

FORMATTERS = {
    "pretty": fmt_pretty,
    "csv":    fmt_csv,
    "json":   fmt_json,
}

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Camera placement strategies — standalone (no Blender)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("obj",    help="Path to OBJ file")
    parser.add_argument("mode",   type=int, choices=range(1, 8), metavar="MODE",
                        help="Strategy 1-7 (3-6 available standalone)")
    parser.add_argument("--cameras",  type=int,   default=6,    metavar="N",
                        help="Number of cameras / K  (default: 6)")
    parser.add_argument("--coverage", type=float, default=0.95, metavar="F",
                        help="Greedy coverage target 0-1  (default: 0.95)")
    parser.add_argument("--output",   choices=["pretty", "csv", "json"],
                        default="pretty", help="Output format  (default: pretty)")
    parser.add_argument("--out",      metavar="FILE",
                        help="Write result to file instead of stdout")
    args = parser.parse_args()

    # Blender-only modes
    if args.mode in BLENDER_ONLY:
        name = BLENDER_ONLY[args.mode]
        print(f"Mode {args.mode} ({name}) requires Blender and cannot run standalone.",
              file=sys.stderr)
        print(f"Available standalone modes: 3 (Hemisphere), 4 (PCA), 5 (K-means), 6 (Greedy)",
              file=sys.stderr)
        sys.exit(1)

    # Load OBJ
    if not os.path.isfile(args.obj):
        print(f"File not found: {args.obj}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.obj} ...", file=sys.stderr)
    try:
        normals, areas, verts = load_obj(args.obj)
    except Exception as e:
        print(f"OBJ parse error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(normals)} faces, {len(verts)} verts", file=sys.stderr)

    mesh_info = {
        "file":  os.path.basename(args.obj),
        "faces": len(normals),
        "verts": len(verts),
    }

    # Run strategy
    name, fn = STRATEGIES[args.mode]
    print(f"Running mode {args.mode}: {name} ...", file=sys.stderr)
    dirs, label = fn(normals, areas, verts, args.cameras, args.coverage)

    # Format
    output = FORMATTERS[args.output](dirs, label, mesh_info)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output + "\n")
        print(f"Written to {args.out}", file=sys.stderr)
    else:
        print(output)

if __name__ == "__main__":
    main()
