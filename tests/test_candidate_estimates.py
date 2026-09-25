"""Analytic complete-FOV and padding tests without any image payload."""

from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from organ_relation.data.candidate_estimates import estimate_case, estimate_dataset, run

ROOT = Path(__file__).resolve().parents[1]


def case(shape=(10,20,30), spacing=(1.,1.,5.)):
    coverage = [n*s for n,s in zip(shape,spacing)]
    return {"case_id":"amos_0001", "image":"nonexistent.nii.gz", "label":"nonexistent_label.nii.gz",
            "geometry":{"consistent":True}, "image_header":{
                "axis_aligned":True, "shape_ras":list(shape), "spacing_ras_mm":list(spacing),
                "coverage_ras_mm":coverage, "world_boundary_min_mm":[0.,0.,0.],
                "world_boundary_max_mm":coverage, "voxel_count":math.prod(shape),
                "anisotropy_ratio":max(spacing)/min(spacing)}}


class EstimateTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/"configs/preprocessing_candidates.json").read_text())
        self.config["expected_training_ct_count"] = 1
        self.metadata = {"schema_version":1,"cases":[case()],
                         "provenance":{"config":{"split":"training"},"git":{"commit":"source"},
                                       "manifest_sha256":"source_manifest"}}

    def test_analytic_grid_and_axis_order(self):
        r = estimate_case(case(),{"id":"B","spacing_ras_mm":[2.,2.,3.]},[16,32])
        self.assertEqual(r["shape_dhw"],[50,10,5])
        self.assertEqual(r["voxel_count"],2500)
        self.assertEqual(r["rounding_extra_extent_ras_mm"],[0.,0.,0.])
        self.assertEqual(r["estimated_first_voxel_center_ras_mm"],[1.,1.,1.5])

    def test_rounding_preserves_both_scan_ends(self):
        source = case(shape=(11,21,31),spacing=(1.,1.,5.))
        r = estimate_case(source,{"id":"B","spacing_ras_mm":[2.,2.,3.]},[16,32])
        self.assertEqual(r["shape_dhw"],[52,11,6])
        for a, t in enumerate((2.,2.,3.)):
            self.assertLessEqual(r["estimated_output_boundary_min_ras_mm"][a],0.)
            self.assertGreaterEqual(r["estimated_output_boundary_max_ras_mm"][a],source["image_header"]["coverage_ras_mm"][a])
            self.assertGreaterEqual(r["rounding_extra_extent_ras_mm"][a],0.)
            self.assertLess(r["rounding_extra_extent_ras_mm"][a],t)

    def test_padding_counts_and_two_denominators(self):
        r = estimate_case(case(),{"id":"B","spacing_ras_mm":[2.,2.,3.]},[16,32])
        p = r["padding"]["16"]
        self.assertEqual(p["shape_dhw"],[64,16,16])
        self.assertEqual(p["voxel_count"],16384)
        self.assertAlmostEqual(p["overhead_percent"],100*(16384/2500-1))
        self.assertAlmostEqual(p["padded_fraction_percent"],100*(1-2500/16384))

    def test_no_padding_when_already_divisible(self):
        r = estimate_case(case(shape=(32,32,32),spacing=(2.,2.,3.)),
                          {"id":"B","spacing_ras_mm":[2.,2.,3.]},[16,32])
        self.assertEqual(r["padding"]["32"]["overhead_percent"],0.)

    def test_single_tensor_bytes_not_training_memory(self):
        r = estimate_case(case(),{"id":"B","spacing_ras_mm":[2.,2.,3.]},[16,32])
        self.assertAlmostEqual(r["single_16channel_fp32_gib"],2500*16*4/1024**3)

    def test_sampling_direction_is_not_created_information(self):
        for spacing, expected in ((5.,"upsample"),(3.,"approximately_unchanged"),(1.25,"downsample")):
            r = estimate_case(case(spacing=(1.,1.,spacing)),{"id":"B","spacing_ras_mm":[2.,2.,3.]},[16,32])
            self.assertEqual(r["z_sampling_action"],expected)

    def test_oblique_and_invalid_spacing_rejected(self):
        source = case(); source["image_header"]["axis_aligned"] = False
        with self.assertRaises(ValueError):
            estimate_case(source,self.config["candidates"][0],[16,32])
        with self.assertRaises(ValueError):
            estimate_case(case(),{"id":"X","spacing_ras_mm":[0.,2.,3.]},[16,32])

    def test_non_training_and_failed_geometry_rejected(self):
        source = deepcopy(self.metadata); source["provenance"]["config"]["split"] = "validation"
        with self.assertRaises(ValueError): estimate_dataset(source,self.config)
        source = deepcopy(self.metadata); source["cases"][0]["geometry"]["consistent"] = False
        with self.assertRaises(ValueError): estimate_dataset(source,self.config)

    def test_duplicate_cases_and_final_config_rejected(self):
        source = deepcopy(self.metadata); source["cases"] *= 2
        config = deepcopy(self.config); config["expected_training_ct_count"] = 2
        with self.assertRaises(ValueError): estimate_dataset(source,config)
        config = deepcopy(self.config); config["status"] = "frozen"
        with self.assertRaises(ValueError): estimate_dataset(self.metadata,config)

    def test_full_pipeline_needs_no_nifti_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); mp = root/"metadata.json"; cp = root/"config.json"
            mp.write_text(json.dumps(self.metadata)); cp.write_text(json.dumps(self.config))
            result = run(mp,cp,root/"output")
            self.assertEqual(len(result["cases"]),3)
            self.assertEqual(result["provenance"]["nifti_files_opened"],0)
            self.assertFalse(result["provenance"]["resampling_performed"])
            with (root/"output/cases.csv").open(encoding="utf-8-sig",newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))),3)
            with self.assertRaises(ValueError): run(mp,cp,root/"output")

    def test_output_inside_original_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data = root/"data"; data.mkdir()
            source = deepcopy(self.metadata); source["provenance"]["data_root"] = str(data)
            mp = root/"metadata.json"; cp = root/"config.json"
            mp.write_text(json.dumps(source)); cp.write_text(json.dumps(self.config))
            with self.assertRaises(ValueError): run(mp,cp,data/"output")
            self.assertFalse((data/"output").exists())


if __name__ == "__main__":
    unittest.main()
