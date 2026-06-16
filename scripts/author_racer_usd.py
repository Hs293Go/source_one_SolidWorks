"""Author racer.usd from the SourceOne meshes (Gazebo_sim) — our own 5" racer.

A PhysX articulation: /racer/body (the SourceOne frame STL, carrying the derived
mass/inertia from build_racer.py) + four nulled prop links FIXED to it (so the body is
rigid, 0 dof). The two prop .dae meshes sit at the extracted rotor positions (visual,
spin-ready). Writes racer.usd next to the source meshes in Gazebo_sim; the package
vendors a copy at src/isaacrace/assets/racer.usd. Needs the Kit runtime for pxr:

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
        uv run python other/source_one_SolidWorks/scripts/author_racer_usd.py
"""

import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET  # noqa: S405

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_racer import ASSETS, MM, derive_rigid_body, load_stl

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402

OUT = ASSETS / "racer.usd"  # next to the source meshes in Gazebo_sim
RZ_NEG90 = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # body_x = stl_y (long = fwd)


def load_dae(path: Path):
    """COLLADA polylist -> (points Nx3 mm, faceVertexCounts, faceVertexIndices)."""
    txt = re.sub(r'\sxmlns="[^"]+"', "", path.read_text(), count=1)
    mesh = ET.fromstring(txt).find(".//mesh")  # noqa: S314 (local vendored asset)
    pts = None
    for src in mesh.findall("source"):
        fa = src.find("float_array")
        if fa is not None and "positions" in (src.get("id") or ""):
            pts = np.array(fa.text.split(), dtype=np.float64).reshape(-1, 3)
    poly = mesh.find("polylist")
    inputs = poly.findall("input")
    stride = max(int(i.get("offset")) for i in inputs) + 1
    voff = next(int(i.get("offset")) for i in inputs if i.get("semantic") == "VERTEX")
    counts = np.array(poly.find("vcount").text.split(), dtype=np.int32)
    flat = np.array(poly.find("p").text.split(), dtype=np.int32).reshape(-1, stride)
    return pts, counts, flat[:, voff]


def add_mesh(stage, path, points, counts, indices, color):
    """Author a UsdGeom.Mesh from numpy points/topology with a flat display colour."""
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts.astype(np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices.astype(np.int32)))
    m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return m


def main():
    """Build racer.usd from the SourceOne meshes + derived physics, and verify it."""
    mass, _com, inertia, arm_fwd, arm_lat = derive_rigid_body()
    fwd, lat = arm_fwd * MM, arm_lat * MM

    # body frame mesh: STL is mm, Z-up, long axis = its y -> rotate so long axis is +x.
    tri = load_stl(ASSETS / "sourceone.stl")
    body_pts = (tri.reshape(-1, 3) @ RZ_NEG90.T) * MM  # (3n,3) m, body frame
    nf = len(body_pts) // 3
    body_counts = np.full(nf, 3, dtype=np.int32)
    body_idx = np.arange(3 * nf, dtype=np.int32)

    # props: the .dae's <unit meter="1"> is wrong — coords are mm (127 mm span = a 5"
    # prop), so scale by MM like the STL. The disc already lies in xy with thrust along
    # +z (span x=127, chord y=14, airfoil z=6.5), so no reorientation is needed.
    props = {}
    for spin in ("ccw", "cw"):
        p, c, idx = load_dae(ASSETS / f"sourceone_prop_{spin}.dae")
        p *= MM  # mm -> m
        p[:, :2] -= p[:, :2].mean(0)  # centre hub in xy
        props[spin] = (p, c, idx)

    stage = Usd.Stage.CreateNew(str(OUT))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    racer = UsdGeom.Xform.Define(stage, "/racer")
    stage.SetDefaultPrim(racer.GetPrim())
    # Robot/SingleArticulation needs an articulation: root one here (/body + the fixed
    # prop links below form a valid, rigid, 0-dof articulation).
    UsdPhysics.ArticulationRootAPI.Apply(racer.GetPrim())

    body = UsdGeom.Xform.Define(stage, "/racer/body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    # Zero damping: PhysX defaults angular damping to 0.05, which would corrupt the
    # kinematic set-angular-velocity loop (the faithful OQCRL model has no damping).
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
    px.CreateLinearDampingAttr(0.0)
    px.CreateAngularDampingAttr(0.0)
    mapi = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mapi.CreateMassAttr(float(mass))
    mapi.CreateDiagonalInertiaAttr(
        Gf.Vec3f(float(inertia[0, 0]), float(inertia[1, 1]), float(inertia[2, 2]))
    )
    # COM at the body origin (the prop plane): classical applies thrust at z=0, so a
    # coplanar COM avoids a spurious thrust-offset tilt torque (the validated iris
    # convention). Inertia is the derived tensor.
    mapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mapi.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    add_mesh(stage, "/racer/body/frame", body_pts, body_counts, body_idx,
             (.07, .07, .08))

    # rotor order matches quadrotor._ROTOR_POS_FLU: 0 back-right, 1 front-right, 2
    # back-left, 3 front-left; spin ccw,ccw,cw,cw (quadrotor._rotor_spin). Each rotor is
    # a nulled link FIXED to /body (rigid, 0 dof). The prop spin is purely visual.
    UsdGeom.Scope.Define(stage, "/racer/joints")
    rotors = [(-fwd, -lat, "ccw"), (fwd, -lat, "ccw"),
              (-fwd, lat, "cw"), (fwd, lat, "cw")]
    for i, (x, y, spin) in enumerate(rotors):
        rx = UsdGeom.Xform.Define(stage, f"/racer/rotor{i}")
        rx.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), 0.030))
        UsdPhysics.RigidBodyAPI.Apply(rx.GetPrim())
        rm = UsdPhysics.MassAPI.Apply(rx.GetPrim())
        rm.CreateMassAttr(1e-4)
        rm.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-7, 1e-7, 1e-7))
        p, c, idx = props[spin]
        add_mesh(stage, f"/racer/rotor{i}/prop", p, c, idx, (.04, .04, .04))
        jt = UsdPhysics.FixedJoint.Define(stage, f"/racer/joints/rotor{i}")
        jt.CreateBody0Rel().SetTargets(["/racer/body"])
        jt.CreateBody1Rel().SetTargets([f"/racer/rotor{i}"])
        jt.CreateLocalPos0Attr(Gf.Vec3f(float(x), float(y), 0.030))
        jt.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

    stage.GetRootLayer().Save()

    # re-open fresh and verify the asset resolves (Isaac swallows stdout, so assert)
    chk = Usd.Stage.Open(str(OUT))
    bm = UsdPhysics.MassAPI(chk.GetPrimAtPath("/racer/body")).GetMassAttr().Get()
    ok = all(chk.GetPrimAtPath(f"/racer/rotor{i}/prop").IsValid() for i in range(4))
    default = chk.GetDefaultPrim().GetPath()
    if not (default == "/racer" and abs(bm - 0.472) < 1e-3 and ok):
        raise RuntimeError(f"racer.usd verify failed: {default=} {bm=} {ok=}")
    _app.close()


if __name__ == "__main__":
    main()
