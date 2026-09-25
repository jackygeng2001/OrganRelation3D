"""Bounded NIfTI-1 single-file header inspection; never loads voxel arrays.

Only scalar 3D .nii/.nii.gz files are supported. Unknown geometry is an error,
not an identity-affine fallback. See docs/metadata_audit.md for conventions.
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import math
from pathlib import Path
import struct


class HeaderError(ValueError):
    """Unsupported or invalid metadata (not a voxel integrity verdict)."""


SCALAR_TYPES = {2: ("uint8", 8), 4: ("int16", 16), 8: ("int32", 32),
                16: ("float32", 32), 64: ("float64", 64), 256: ("int8", 8),
                512: ("uint16", 16), 768: ("uint32", 32),
                1024: ("int64", 64), 1280: ("uint64", 64)}
UNIT_MM = {1: 1000.0, 2: 1.0, 3: 0.001}


def transform(affine: list[list[float]], xyz: tuple[float, ...]) -> list[float]:
    return [sum(affine[r][a] * xyz[a] for a in range(3)) + affine[r][3]
            for r in range(3)]


def corners(shape: list[int], *, boundary: bool = False):
    """Voxel-center corners, or outer voxel-cell boundary corners."""
    limits = [(-0.5, n - 0.5) if boundary else (0.0, float(n - 1)) for n in shape]
    return itertools.product(*limits)


def max_corner_distance(a: list[list[float]], b: list[list[float]],
                        shape: list[int]) -> float:
    # Boundaries also detect a basis-vector difference on a singleton axis.
    return max(math.dist(transform(a, p), transform(b, p))
               for p in corners(shape, boundary=True))


def _spacing(affine):
    return [math.sqrt(sum(affine[r][c] ** 2 for r in range(3))) for c in range(3)]


def _validate_affine(affine, name):
    if not all(math.isfinite(v) for row in affine for v in row):
        raise HeaderError(f"{name}: non-finite affine")
    a, b, c = affine
    det = (a[0] * (b[1]*c[2] - b[2]*c[1])
           - a[1] * (b[0]*c[2] - b[2]*c[0])
           + a[2] * (b[0]*c[1] - b[1]*c[0]))
    scales = _spacing(affine)
    if min(scales) <= 0 or abs(det) / math.prod(scales) < 1e-8:
        raise HeaderError(f"{name}: singular affine")


def parse_header(raw: bytes, *, geometry_tolerance_mm: float = 0.001,
                 axis_alignment_tolerance: float = 0.0001) -> dict:
    if len(raw) != 352:
        raise HeaderError(f"need 352 header bytes, got {len(raw)}")
    if struct.unpack_from("<i", raw)[0] == 348:
        endian = "<"
    elif struct.unpack_from(">i", raw)[0] == 348:
        endian = ">"
    else:
        raise HeaderError("not a NIfTI-1 header (NIfTI-2/Analyze unsupported)")
    if raw[344:348] != b"n+1\0":
        raise HeaderError("only single-file NIfTI-1 n+1 magic is supported")

    def unpack(fmt, offset):
        return struct.unpack_from(endian + fmt, raw, offset)

    dim = unpack("8h", 40)
    if dim[0] != 3 or any(n <= 0 for n in dim[1:4]):
        raise HeaderError(f"expected scalar 3D dimensions, got {dim}")
    shape = list(dim[1:4])
    datatype, bitpix = unpack("2h", 70)
    if datatype not in SCALAR_TYPES or bitpix != SCALAR_TYPES[datatype][1]:
        raise HeaderError(f"unsupported datatype/bitpix: {datatype}/{bitpix}")
    pixdim = unpack("8f", 76)
    if not all(math.isfinite(s) and s > 0 for s in pixdim[1:4]):
        raise HeaderError("invalid spatial pixdim")
    offset, slope, intercept = unpack("3f", 108)
    if not math.isfinite(offset) or offset < 352 or offset != int(offset):
        raise HeaderError(f"invalid voxel offset {offset}")
    spatial_unit = raw[123] & 7
    if spatial_unit not in UNIT_MM:
        raise HeaderError(f"unknown/unsupported spatial unit {spatial_unit}; cannot assume mm")
    factor = UNIT_MM[spatial_unit]
    spacing_mm = [x * factor for x in pixdim[1:4]]
    notices = []
    if slope == 0:
        effective_slope, effective_intercept = 1.0, 0.0
    elif not math.isfinite(slope):
        effective_slope, effective_intercept = 1.0, 0.0
        notices.append("nonfinite_scaling_assumed_identity")
    elif not math.isfinite(intercept):
        raise HeaderError("finite nonzero slope with nonfinite intercept")
    else:
        effective_slope, effective_intercept = slope, intercept

    qcode, scode = unpack("2h", 252)
    if qcode < 0 or scode < 0:
        raise HeaderError("negative transform code")
    qaffine = None
    if qcode > 0:
        b, c, d, ox, oy, oz = unpack("6f", 256)
        if not all(math.isfinite(v) for v in (b, c, d, ox, oy, oz)):
            raise HeaderError("nonfinite quaternion/offset")
        norm = b*b + c*c + d*d
        if norm > 1.0 + 1e-5:
            raise HeaderError("invalid quaternion norm")
        if 1.0 - norm < 1e-7:
            scale = math.sqrt(norm)
            b, c, d = b/scale, c/scale, d/scale
            a = 0.0
        else:
            a = math.sqrt(1.0 - norm)
        qfac = pixdim[0]
        if qfac not in (-1.0, 0.0, 1.0):
            raise HeaderError(f"invalid qfac {qfac}")
        signed_scales = list(spacing_mm)
        signed_scales[2] *= -1.0 if qfac < 0 else 1.0
        rotation = [
            [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
            [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
            [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-c*c-b*b],
        ]
        qaffine = [[rotation[r][col]*signed_scales[col] for col in range(3)]
                   + [(ox, oy, oz)[r]*factor] for r in range(3)]
        _validate_affine(qaffine, "qform")
    saffine = None
    if scode > 0:
        saffine = [[x*factor for x in unpack("4f", off)] for off in (280, 296, 312)]
        _validate_affine(saffine, "sform")
    if saffine is None and qaffine is None:
        raise HeaderError("no coded qform/sform; geometry cannot be verified")
    affine = saffine if saffine is not None else qaffine
    source = "sform" if saffine is not None else "qform"
    affine_spacing = _spacing(affine)
    if max(abs(a-b) for a, b in zip(spacing_mm, affine_spacing)) > geometry_tolerance_mm:
        notices.append("pixdim_affine_spacing_disagree")
    qs_distance = None
    if saffine is not None and qaffine is not None:
        qs_distance = max_corner_distance(saffine, qaffine, shape)
        if qs_distance > geometry_tolerance_mm:
            notices.append("qform_sform_disagree")

    # Report exact cardinal reordering only; oblique/sheared files are not
    # silently interpreted as already resampled RAS volumes.
    directions = [[affine[r][c]/affine_spacing[c] for c in range(3)] for r in range(3)]
    assignment = max(itertools.permutations(range(3)),
                     key=lambda p: sum(abs(directions[p[c]][c]) for c in range(3)))
    aligned = all(abs(abs(directions[assignment[c]][c])-1) <= axis_alignment_tolerance
                  and all(abs(directions[r][c]) <= axis_alignment_tolerance
                          for r in range(3) if r != assignment[c]) for c in range(3))
    axis_codes = "".join(("RL", "AP", "SI")[assignment[c]][
        0 if directions[assignment[c]][c] > 0 else 1] for c in range(3))
    ras_order = [assignment.index(r) for r in range(3)] if aligned else None
    if not aligned:
        notices.append("oblique_or_sheared_grid")
    boundary_world = [transform(affine, p) for p in corners(shape, boundary=True)]
    world_min = [min(p[r] for p in boundary_world) for r in range(3)]
    world_max = [max(p[r] for p in boundary_world) for r in range(3)]
    return {
        "shape_native": shape, "spacing_pixdim_mm": spacing_mm,
        "spacing_affine_mm": affine_spacing,
        "center_span_native_mm": [(n-1)*s for n, s in zip(shape, affine_spacing)],
        "cell_coverage_native_mm": [n*s for n, s in zip(shape, affine_spacing)],
        "world_boundary_min_mm": world_min, "world_boundary_max_mm": world_max,
        "world_aabb_extent_mm": [hi-lo for lo, hi in zip(world_min, world_max)],
        "axis_codes": axis_codes, "axis_aligned": aligned,
        "shape_ras": [shape[c] for c in ras_order] if aligned else None,
        "spacing_ras_mm": [affine_spacing[c] for c in ras_order] if aligned else None,
        "coverage_ras_mm": [shape[c]*affine_spacing[c] for c in ras_order] if aligned else None,
        "voxel_count": math.prod(shape),
        "anisotropy_ratio": max(affine_spacing)/min(affine_spacing),
        "datatype": SCALAR_TYPES[datatype][0], "bitpix": bitpix,
        "estimated_raw_voxel_bytes": math.prod(shape)*bitpix//8,
        "spatial_unit_code": spatial_unit, "qform_code": qcode, "sform_code": scode,
        "affine_source": source, "affine_mm": affine,
        "qform_sform_max_corner_distance_mm": qs_distance,
        "scaling_raw": [x if math.isfinite(x) else None for x in (slope, intercept)],
        "scaling_effective": [effective_slope, effective_intercept],
        "voxel_offset": int(offset), "endianness": "little" if endian == "<" else "big",
        "header_sha256": hashlib.sha256(raw).hexdigest(), "notices": notices,
    }


def read_header(path: Path, **kwargs) -> dict:
    path = Path(path)
    if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        raise HeaderError("expected .nii or .nii.gz")
    before = path.stat()
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as stream:
        raw = stream.read(352)
    result = parse_header(raw, **kwargs)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise HeaderError("file changed while reading header")
    result.update(file_size_bytes=after.st_size, mtime_ns=after.st_mtime_ns,
                  requested_uncompressed_bytes=352)
    return result


def compare_geometry(image: dict, label: dict, tolerance_mm: float = 0.001) -> dict:
    shape_equal = image["shape_native"] == label["shape_native"]
    spacing_delta = max(abs(a-b) for a, b in zip(image["spacing_affine_mm"],
                                                label["spacing_affine_mm"]))
    # Compare mapping of corresponding array indices, not only bounding boxes.
    distance = max_corner_distance(image["affine_mm"], label["affine_mm"],
                                   image["shape_native"])
    return {"shape_equal": shape_equal, "max_spacing_delta_mm": spacing_delta,
            "max_corner_distance_mm": distance,
            "consistent": shape_equal and spacing_delta <= tolerance_mm and distance <= tolerance_mm}
