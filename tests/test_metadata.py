"""Synthetic NIfTI headers, analytic geometry and audit integration tests."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from organ_relation.data.nifti_header import HeaderError, compare_geometry, parse_header, read_header
from organ_relation.data.ct_stats import audit, quantile, select_training
from organ_relation.provenance import git_state

ROOT = Path(__file__).resolve().parents[1]


def header(*, endian="<", shape=(2, 3, 4), spacing=(2., 3., 4.), qcode=1,
           scode=0, unit=2, quaternion=(0., 0., 0.), qfac=1.,
           offset=(10., 20., 30.), slope=1., intercept=0., affine=None):
    raw = bytearray(352)
    def put(fmt, pos, *values):
        struct.pack_into(endian+fmt, raw, pos, *values)
    put("i", 0, 348)
    put("8h", 40, 3, *shape, 1, 1, 1, 1)
    put("2h", 70, 4, 16)
    put("8f", 76, qfac, *spacing, 0., 0., 0., 0.)
    put("3f", 108, 352., slope, intercept)
    raw[123] = unit
    put("2h", 252, qcode, scode)
    put("6f", 256, *quaternion, *offset)
    if affine is None:
        affine = [[spacing[0], 0., 0., offset[0]],
                  [0., spacing[1], 0., offset[1]],
                  [0., 0., spacing[2], offset[2]]]
    for pos, row in zip((280, 296, 312), affine):
        put("4f", pos, *row)
    raw[344:348] = b"n+1\0"
    return bytes(raw)


class HeaderTests(unittest.TestCase):
    def test_little_and_big_endian(self):
        a, b = (parse_header(header(endian=e)) for e in ("<", ">"))
        self.assertEqual(a["shape_native"], [2, 3, 4])
        self.assertEqual(a["affine_mm"], b["affine_mm"])
        self.assertEqual(b["endianness"], "big")

    def test_analytic_coverage_and_world_bounds(self):
        h = parse_header(header())
        self.assertEqual(h["cell_coverage_native_mm"], [4., 9., 16.])
        self.assertEqual(h["center_span_native_mm"], [2., 6., 12.])
        self.assertEqual(h["world_boundary_min_mm"], [9., 18.5, 28.])
        self.assertEqual(h["world_boundary_max_mm"], [13., 27.5, 44.])

    def test_qform_only_negative_qfac(self):
        h = parse_header(header(quaternion=(0., 1., 0.), qfac=-1.))
        self.assertEqual(h["affine_source"], "qform")
        self.assertEqual(h["affine_mm"], [[-2., 0., 0., 10.], [0., 3., 0., 20.], [0., 0., 4., 30.]])
        self.assertEqual(h["axis_codes"], "LAS")

    def test_sform_precedence_and_disagreement(self):
        h = parse_header(header(scode=1, affine=[[2,0,0,12], [0,3,0,20], [0,0,4,30]]))
        self.assertEqual(h["affine_source"], "sform")
        self.assertAlmostEqual(h["qform_sform_max_corner_distance_mm"], 2.)
        self.assertIn("qform_sform_disagree", h["notices"])

    def test_sform_vs_label_qform_equivalence(self):
        image = parse_header(header(scode=1))
        label = parse_header(header(scode=0))
        self.assertTrue(compare_geometry(image, label)["consistent"])

    def test_meter_and_micron_units(self):
        for unit, value in ((1, .001), (3, 1000.)):
            h = parse_header(header(unit=unit, spacing=(value,)*3, offset=(0.,)*3))
            for actual in h["spacing_affine_mm"]:
                self.assertAlmostEqual(actual, 1., places=6)

    def test_unknown_unit_rejected(self):
        with self.assertRaises(HeaderError):
            parse_header(header(unit=0))

    def test_missing_transform_rejected(self):
        with self.assertRaises(HeaderError):
            parse_header(header(qcode=0, scode=0))

    def test_invalid_declared_sform_not_silently_replaced(self):
        with self.assertRaises(HeaderError):
            parse_header(header(scode=1, affine=[[0,0,0,0]]*3))

    def test_oblique_not_reported_as_resampled_ras(self):
        h = parse_header(header(quaternion=(0., 0., math.sin(math.pi/8))))
        self.assertFalse(h["axis_aligned"])
        self.assertIsNone(h["shape_ras"])
        self.assertIn("oblique_or_sheared_grid", h["notices"])

    def test_permuted_cardinal_axes(self):
        h = parse_header(header(qcode=0, scode=1, spacing=(3.,2.,4.),
                               affine=[[0,2,0,0],[-3,0,0,0],[0,0,4,0]]))
        self.assertEqual(h["axis_codes"], "PRS")
        self.assertEqual(h["shape_ras"], [3,2,4])
        self.assertEqual(h["spacing_ras_mm"], [2.,3.,4.])

    def test_translation_mismatch(self):
        a = parse_header(header())
        b = parse_header(header(offset=(10.01,20.,30.)))
        self.assertFalse(compare_geometry(a,b)["consistent"])

    def test_axis_flip_same_world_bounds_is_mismatch(self):
        a = parse_header(header(offset=(0.,0.,0.)))
        b = parse_header(header(qcode=0, scode=1,
                               affine=[[-2,0,0,2],[0,3,0,0],[0,0,4,0]]))
        self.assertEqual(a["world_boundary_min_mm"], b["world_boundary_min_mm"])
        self.assertFalse(compare_geometry(a,b)["consistent"])

    def test_singleton_axis_spacing_mismatch(self):
        a = parse_header(header(shape=(1,3,4)))
        b = parse_header(header(shape=(1,3,4), spacing=(4.,3.,4.)))
        self.assertEqual(a["center_span_native_mm"][0], 0.)
        self.assertFalse(compare_geometry(a,b)["consistent"])

    def test_zero_slope_means_identity(self):
        self.assertEqual(parse_header(header(slope=0., intercept=99.))["scaling_effective"], [1.,0.])

    def test_nonfinite_scaling_not_written_as_json_nan(self):
        h = parse_header(header(slope=float("nan"), intercept=float("nan")))
        json.dumps(h, allow_nan=False)
        self.assertIn("nonfinite_scaling_assumed_identity", h["notices"])

    def test_invalid_magic_truncation_and_nifti2(self):
        for raw in (header()[:348], bytes(352), header()[:344]+b"ni1\0"+bytes(4)):
            with self.assertRaises(HeaderError):
                parse_header(raw)

    def test_4d_and_bad_spacing_rejected(self):
        raw = bytearray(header())
        struct.pack_into("<h", raw, 40, 4)
        for data in (bytes(raw), header(spacing=(0.,3.,4.))):
            with self.assertRaises(HeaderError):
                parse_header(data)

    def test_gzip_and_plain_header_only(self):
        # Intentionally no voxel payload: metadata reading succeeds without
        # claiming the complete NIfTI volume is intact.
        with tempfile.TemporaryDirectory() as td:
            for name in ("test.nii", "test.nii.gz"):
                p = Path(td)/name
                p.write_bytes(gzip.compress(header()) if name.endswith("gz") else header())
                h = read_header(p)
                self.assertEqual(h["requested_uncompressed_bytes"], 352)
                self.assertEqual(h["voxel_count"], 24)

    def test_reader_requests_only_352_bytes(self):
        class BoundedReader:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size):
                if size != 352: raise AssertionError("unbounded voxel read")
                return header()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"case.nii.gz"
            p.write_bytes(b"dummy")
            with patch("organ_relation.data.nifti_header.gzip.open", return_value=BoundedReader()):
                self.assertEqual(read_header(p)["shape_native"], [2,3,4])


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.data = self.base/"dataset"
        for directory in ("imagesTr", "labelsTr"):
            (self.data/directory).mkdir(parents=True)
        self.entries = []
        for i in (1, 4, 501):
            name = f"amos_{i:04d}.nii.gz"
            entry = {"image": f"./imagesTr/{name}", "label": f"./labelsTr/{name}"}
            self.entries.append(entry)
            for relative in entry.values():
                (self.data/relative).write_bytes(gzip.compress(header()))
        self.save_manifest()
        self.config_path = ROOT/"configs/ct_stats.json"
        self.config = json.loads(self.config_path.read_text())

    def tearDown(self):
        self.temp.cleanup()

    def save_manifest(self):
        (self.data/"dataset.json").write_text(json.dumps({"training": self.entries}), encoding="utf-8")

    def test_selection_reports_and_data_unchanged(self):
        hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.data.rglob("*") if p.is_file()}
        result = audit(self.data, self.config_path, self.base/"output")
        self.assertEqual(result["summary"]["selected_ct_count"], 2)
        self.assertEqual(result["summary"]["excluded_training_count"], 1)
        self.assertEqual(result["summary"]["geometry_consistent_count"], 2)
        self.assertEqual(result["summary"]["error_count"], 0)
        for p, digest in hashes.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)
        for name in ("metadata.json", "cases.csv", "anomalies.csv", "report.md"):
            self.assertTrue((self.base/"output"/name).is_file())

    def test_missing_label_is_recorded_and_other_cases_continue(self):
        (self.data/self.entries[0]["label"]).unlink()
        result = audit(self.data, self.config_path, self.base/"output")
        self.assertEqual(result["summary"]["error_count"], 1)
        self.assertEqual(result["summary"]["geometry_consistent_count"], 1)

    def test_geometry_failure_is_error(self):
        (self.data/self.entries[0]["label"]).write_bytes(gzip.compress(header(offset=(99.,20.,30.))))
        result = audit(self.data, self.config_path, self.base/"output")
        self.assertEqual(result["summary"]["anomaly_code_counts"], {"image_label_geometry_mismatch":1})

    def test_zlib_corruption_is_reported(self):
        with patch("organ_relation.data.ct_stats.read_header", side_effect=zlib.error("bad compressed block")):
            result = audit(self.data, self.config_path, self.base/"output")
        self.assertEqual(result["summary"]["error_count"], 4)
        self.assertEqual(result["summary"]["readable_pair_count"], 0)

    def test_shape_mismatch_is_reported(self):
        (self.data/self.entries[0]["label"]).write_bytes(gzip.compress(header(shape=(2,3,5))))
        result = audit(self.data, self.config_path, self.base/"output")
        self.assertFalse(result["cases"][0]["geometry"]["shape_equal"])
        self.assertEqual(result["summary"]["geometry_consistent_count"], 1)

    def test_source_output_and_existing_output_rejected(self):
        for output in (self.data/"newreport", self.base):
            with self.assertRaises(ValueError):
                audit(self.data, self.config_path, output)
        self.assertFalse((self.data/"newreport").exists())

    def test_duplicate_case_rejected(self):
        self.entries.append(self.entries[0])
        self.save_manifest()
        with self.assertRaises(ValueError):
            select_training(self.data,self.config)

    def test_path_escape_rejected(self):
        self.entries[0]["image"] = "../../amos_0001.nii.gz"
        self.save_manifest()
        with self.assertRaises(ValueError):
            select_training(self.data,self.config)

    def test_id500_not_selected(self):
        name = "amos_0500.nii.gz"
        self.entries.append({"image":f"imagesTr/{name}", "label":f"labelsTr/{name}"})
        self.save_manifest()
        selected, excluded, _ = select_training(self.data,self.config)
        self.assertEqual(len(selected),2)
        self.assertIn("amos_0500", [x["case_id"] for x in excluded])

    def test_quantiles(self):
        self.assertEqual(quantile([1.,2.,3.,4.], .5), 2.5)
        self.assertAlmostEqual(quantile([1.,2.,3.,4.], .9), 3.7)

    def test_git_provenance_explicit_utf8_for_chinese_path(self):
        with patch("organ_relation.provenance.subprocess.check_output",
                   side_effect=[str(ROOT), "abc123", ""]) as run:
            self.assertEqual(git_state(), {"commit":"abc123", "dirty":False})
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["encoding"], "utf-8")

    def test_parent_repository_not_claimed_as_project_commit(self):
        with patch("organ_relation.provenance.subprocess.check_output", return_value=str(ROOT.parent)):
            self.assertIsNone(git_state()["commit"])


if __name__ == "__main__":
    unittest.main()
