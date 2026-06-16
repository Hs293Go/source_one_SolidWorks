"""Reconstruct our own 5" racer from the SourceOne meshes — physics first.

Stage A: extract the four motor-mount positions from sourceone.stl (real arm geometry).
Stage B: place a documented component mass budget at those positions + the frame's own
         mesh-derived inertia, and reduce to a single rigid body: total mass, COM, and
         the inertia tensor about the COM. These replace the cited Agilicious tensor and
         the hand-picked arm in quadrotor.py.

Body convention is +x forward, +y left, +z up (FLU). A 5" frame is longer front-to-back,
so the larger extracted arm maps to the forward axis (the STL is permuted to match).

Run: env -u PYTHONPATH uv run python other/source_one_SolidWorks/scripts/build_racer.py
(author_racer_usd.py turns these numbers + the Gazebo_sim meshes into racer.usd.)
"""

from pathlib import Path
import struct

import numpy as np

# The SourceOne source meshes (stl + prop daes) live alongside this vendored build.
ASSETS = Path(__file__).resolve().parents[1] / "Gazebo_sim"
MM = 1e-3  # the STL is in millimetres; we work in metres


def load_stl(path: Path) -> np.ndarray:
    """Return (n,3,3) triangle vertices from a binary STL, in millimetres."""
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    raw = np.frombuffer(data[84 : 84 + 50 * n], dtype=np.uint8).reshape(n, 50)
    return raw[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)


def mesh_mass_properties(tri: np.ndarray):
    """Volume, centroid, inertia (about centroid, density 1) of a closed mesh.

    Signed-tetrahedron method (each triangle + origin). V in mm^3, com in mm, inertia in
    mm^5 (density 1) — scale by rho = mass/V for kg*mm^2.
    """
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    d = np.einsum("ij,ij->i", a, np.cross(b, c))  # 6*signed tet volume
    vol = d.sum() / 6.0
    com = (d[:, None] * (a + b + c) / 4).sum(0) / 6.0 / vol
    m0 = (np.ones((3, 3)) + np.eye(3)) / 120.0
    second = np.zeros((3, 3))
    for i in range(len(tri)):
        j = np.stack([a[i], b[i], c[i]], axis=1)
        second += d[i] * (j @ m0 @ j.T)
    i_o = np.trace(second) * np.eye(3) - second  # inertia about origin
    i_c = i_o - vol * ((com @ com) * np.eye(3) - np.outer(com, com))  # -> centroid
    return vol, com, i_c


def extract_rotor_xy(verts: np.ndarray) -> tuple[float, float]:
    """Arm-tip centroids of the low-z (arm-plate) slice -> (arm |x|, |y|) in mm."""
    low = verts[verts[:, 2] < 12]
    arms = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            q = low[(np.sign(low[:, 0]) == sx) & (np.sign(low[:, 1]) == sy)]
            r = np.hypot(q[:, 0], q[:, 1])
            arms.append(q[r >= np.quantile(r, 0.97)].mean(0))
    arms = np.array(arms)
    return abs(arms[:, 0]).mean(), abs(arms[:, 1]).mean()


def box_inertia(m: float, dims_mm) -> np.ndarray:
    """Solid-box inertia tensor (kg*m^2) about its own centre, dims in mm."""
    lx, ly, lz = (np.array(dims_mm) * MM) ** 2
    return m / 12.0 * np.diag([ly + lz, lx + lz, lx + ly])


def derive_rigid_body():
    """Return (mass kg, com_mm body, inertia kg*m^2 body, arm_fwd_mm, arm_lat_mm)."""
    tri = load_stl(ASSETS / "sourceone.stl")
    verts = np.unique(tri.reshape(-1, 3), axis=0)
    a_x, a_y = extract_rotor_xy(verts)  # STL-x, STL-y arm magnitudes (mm)
    # +x is forward; a 5" is longer front-to-back, so the larger arm -> forward. Permute
    # the STL so its long axis becomes body-x (P swaps x,y when forward is STL-y).
    fwd_is_y = a_y > a_x
    arm_fwd, arm_lat = (a_y, a_x) if fwd_is_y else (a_x, a_y)
    perm = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]]) if fwd_is_y else np.eye(3)

    vol, fcom_stl, i_c_stl = mesh_mass_properties(tri)
    m_frame = 0.045  # kg, SourceOne carbon frame
    i_frame = perm @ ((m_frame / vol) * i_c_stl * MM**2) @ perm.T  # kg*m^2, body axes
    fcom = perm @ fcom_stl  # mm, body axes

    # Documented 5" build budget: (name, mass kg, pos mm body, box dims mm | None,
    # self-inertia kg*m^2 | None). Motors/props at the extracted arm; battery on the top
    # plate (long axis fore-aft), camera at the nose.
    rot = [
        (arm_fwd, arm_lat),
        (arm_fwd, -arm_lat),
        (-arm_fwd, arm_lat),
        (-arm_fwd, -arm_lat),
    ]
    comps = [("frame", m_frame, tuple(fcom), None, i_frame)]
    comps += [
        (f"motor{i}", 0.033, (x, y, 16.0), None, None) for i, (x, y) in enumerate(rot)
    ]
    comps += [
        (f"prop{i}", 0.0050, (x, y, 30.0), None, None) for i, (x, y) in enumerate(rot)
    ]
    comps += [
        ("battery", 0.190, (0.0, 0.0, 44.0), (72, 34, 28), None),
        ("stack", 0.030, (0.0, 0.0, 16.0), (30, 30, 16), None),
        ("camera", 0.030, (arm_fwd + 12, 0.0, 30.0), (24, 20, 20), None),
        ("vtx_rx_wire", 0.025, (-40.0, 0.0, 24.0), (30, 30, 12), None),
    ]

    mass = sum(c[1] for c in comps)
    com = sum((c[1] * np.array(c[2]) for c in comps), np.zeros(3)) / mass  # mm
    com_m = com * MM
    inertia = np.zeros((3, 3))
    for _name, m, pos, box, i_self in comps:
        d = np.array(pos) * MM - com_m
        inertia += m * ((d @ d) * np.eye(3) - np.outer(d, d))  # parallel axis
        if i_self is not None:
            inertia += i_self
        elif box is not None:
            inertia += box_inertia(m, box)
    return mass, com, inertia, arm_fwd, arm_lat


def main():
    """Derive the rigid body and print the numbers destined for quadrotor.py."""
    mass, com, inertia, arm_fwd, arm_lat = derive_rigid_body()
    evals = np.linalg.eigvalsh(inertia)
    print(
        f"[A] arm: forward={arm_fwd:.1f} mm  lateral={arm_lat:.1f} mm  "
        f"(wheelbase tip {2 * np.hypot(arm_fwd, arm_lat):.0f} mm)"
    )
    print(f"[B] total mass     : {mass * 1000:.0f} g")
    print(f"[B] COM (body)     : ({com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}) mm")
    print("[B] inertia about COM (kg*m^2):")
    for row in inertia:
        print("      " + "  ".join(f"{v: .3e}" for v in row))
    print(
        f"[B] diag(roll,pitch,yaw): "
        f"({inertia[0, 0]:.2e}, {inertia[1, 1]:.2e}, {inertia[2, 2]:.2e})"
    )
    print(f"[B] principal ratios: {np.round(evals / evals.min(), 2)}")
    print("[cmp] cited Agilicious: m=0.75  I=(2.5e-3, 2.1e-3, 4.3e-3)")


if __name__ == "__main__":
    main()
