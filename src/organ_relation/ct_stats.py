"""Read-only AMOS training CT metadata audit with local review reports."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import zlib

from .nifti_header import HeaderError, compare_geometry, read_header

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("manifest paths must be relative strings")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes data root")
    return result


def select_training(root: Path, config: dict) -> tuple[list[dict], list[dict], list[dict]]:
    if config.get("schema_version") != 1 or config.get("split") != "training":
        raise ValueError("this audit supports schema 1 and training split only")
    if config.get("header_bytes") != 352:
        raise ValueError("NIfTI-1 metadata audit must request exactly 352 bytes")
    if not 1 <= config["ct_id_min"] <= config["ct_id_max"] < 500:
        raise ValueError("training CT selector must stay within IDs 1..499")
    for key in ("geometry_tolerance_mm", "axis_alignment_tolerance"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    manifest = json.loads(contained_path(root, config["manifest"]).read_text(encoding="utf-8"))
    entries = manifest.get("training")
    if not isinstance(entries, list):
        raise ValueError("manifest training must be a list")
    selected, excluded, issues = [], [], []
    seen = set()
    for entry in entries:
        image_path = contained_path(root, entry["image"])
        label_path = contained_path(root, entry["label"])
        match = re.fullmatch(r"amos_(\d{4})\.nii(?:\.gz)?", image_path.name)
        if not match or image_path.name != label_path.name:
            raise ValueError(f"invalid image/label naming pair: {entry}")
        if image_path.parent != contained_path(root, config["image_directory"]):
            raise ValueError("training image outside configured training directory")
        if label_path.parent != contained_path(root, config["label_directory"]):
            raise ValueError("training label outside configured training directory")
        case_id = image_path.name.split(".")[0]
        if case_id in seen:
            raise ValueError(f"duplicate training case: {case_id}")
        seen.add(case_id)
        record = {"case_id": case_id, "image": entry["image"], "label": entry["label"]}
        case_number = int(match.group(1))
        if config["ct_id_min"] <= case_number <= config["ct_id_max"]:
            selected.append(record)
        else:
            excluded.append({**record, "reason": "outside configured training CT ID interval"})
    for field, folder in (("image", "image_directory"), ("label", "label_directory")):
        expected = {Path(x[field]).name for x in entries}
        directory = contained_path(root, config[folder])
        actual = {x.name for x in directory.iterdir()
                  if x.is_file() and (x.name.endswith(".nii") or x.name.endswith(".nii.gz"))}
        for name in sorted(actual - expected):
            issues.append({"case_id": name.split(".")[0], "severity": "warning",
                           "code": "unlisted_training_file", "detail": f"{config[folder]}/{name}"})
    if not selected:
        raise ValueError("no training CT selected")
    return sorted(selected, key=lambda x: x["case_id"]), excluded, issues


def quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values)-1)*probability
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high]-values[low])*(position-low)


def distribution(values: list[float]) -> dict:
    return {"count": len(values), "min": min(values), "p50": quantile(values, .5),
            "p90": quantile(values, .9), "p95": quantile(values, .95), "max": max(values)}


def summarize(records: list[dict], anomalies: list[dict], excluded_count: int) -> dict:
    readable = [r for r in records if "image_header" in r]
    paired = [r for r in readable if "label_header" in r]
    distributions = {}
    for field in ("shape_native", "spacing_affine_mm", "cell_coverage_native_mm",
                  "center_span_native_mm", "shape_ras", "spacing_ras_mm", "coverage_ras_mm"):
        group = [r["image_header"][field] for r in readable if r["image_header"][field] is not None]
        if group:
            distributions[field] = [distribution([v[a] for v in group]) for a in range(3)]
    for field in ("voxel_count", "anisotropy_ratio", "estimated_raw_voxel_bytes"):
        if readable:
            distributions[field] = distribution([r["image_header"][field] for r in readable])
    representatives = {}
    rank_functions = {
        "largest_native_voxel_count": lambda h: h["voxel_count"],
        "largest_physical_cell_volume": lambda h: math.prod(h["cell_coverage_native_mm"]),
        "largest_axis_spacing": lambda h: max(h["spacing_affine_mm"]),
        "highest_anisotropy": lambda h: h["anisotropy_ratio"],
        "longest_physical_axis": lambda h: max(h["cell_coverage_native_mm"]),
        "smallest_axis_spacing": lambda h: -min(h["spacing_affine_mm"]),
    }
    for name, key in rank_functions.items():
        representatives[name] = [r["case_id"] for r in sorted(
            readable, key=lambda r: (-key(r["image_header"]), r["case_id"]))[:5]]
    return {
        "selected_ct_count": len(records), "excluded_training_count": excluded_count,
        "readable_image_count": len(readable), "readable_pair_count": len(paired),
        "geometry_consistent_count": sum(r.get("geometry", {}).get("consistent", False) for r in records),
        "error_count": sum(a["severity"] == "error" for a in anomalies),
        "warning_count": sum(a["severity"] == "warning" for a in anomalies),
        "cases_with_errors": sorted({a["case_id"] for a in anomalies if a["severity"] == "error"}),
        "cases_with_warnings": sorted({a["case_id"] for a in anomalies if a["severity"] == "warning"}),
        "anomaly_code_counts": dict(sorted(Counter(a["code"] for a in anomalies).items())),
        "image_axis_code_counts": dict(Counter(r["image_header"]["axis_codes"] for r in readable)),
        "image_scaling_counts": dict(Counter(str(r["image_header"]["scaling_effective"]) for r in readable)),
        "image_form_counts": dict(Counter(f"q={r['image_header']['qform_code']},s={r['image_header']['sform_code']}" for r in readable)),
        "label_form_counts": dict(Counter(f"q={r['label_header']['qform_code']},s={r['label_header']['sform_code']}" for r in paired)),
        "max_pair_corner_distance_mm": max((r["geometry"]["max_corner_distance_mm"] for r in paired), default=None),
        "distributions": distributions, "representative_case_ids": representatives,
        "representative_note": "Metadata extremes only; candidate-resampled sizes and small-organ risks not evaluated.",
    }


def git_state() -> dict:
    def run(*args):
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args],
                                       stderr=subprocess.DEVNULL, text=True,
                                       encoding="utf-8").strip()
    try:
        if Path(run("rev-parse", "--show-toplevel")).resolve() != PROJECT_ROOT:
            return {"commit": None, "reason": "not an independent project repository"}
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "reason": "no available project commit"}


def code_hashes() -> dict:
    files = sorted([*PROJECT_ROOT.glob("src/**/*.py"), *PROJECT_ROOT.glob("scripts/*.py")])
    return {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in files}


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(summary: dict, provenance: dict) -> str:
    lines = ["# 训练 CT 元数据统计报告", "",
             f"生成时间：{provenance['utc_time']}；设备：{provenance['platform']}；Python {provenance['python']}。", "",
             "仅按配置从 training 清单选择 CT，逐文件请求 352 字节解压头信息；不载入体素数组、不修改数据。", "",
             f"- 选择 CT：{summary['selected_ct_count']} 例；排除其他训练条目：{summary['excluded_training_count']}。",
             f"- 可读图像：{summary['readable_image_count']}；可读图像标签对：{summary['readable_pair_count']}。",
             f"- 头信息几何一致：{summary['geometry_consistent_count']} 对。",
             f"- 错误：{summary['error_count']} 条；警告：{summary['warning_count']} 条。",
             f"- 配对最大体素边界角点偏差：{summary['max_pair_corner_distance_mm']} mm。", "",
             "## 尺寸与 spacing 分布", "",
             "native 轴为 NIfTI 存储的 i,j,k，不直接称 D,H,W。RAS 列只对轴对齐病例做轴置换统计，不实际翻转、重采样，也不冻结模型方向。", "",
             "cell coverage=n*spacing 是体素单元边界覆盖长度；center span=(n-1)*spacing 是首末体素中心距离。斜切网格的世界包围盒另存病例 JSON。", "",
             "| 指标 | 轴 | n | 最小 | 中位数 | P90 | P95 | 最大 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, ds in summary["distributions"].items():
        items = enumerate(ds) if isinstance(ds, list) else [(None, ds)]
        for axis, value in items:
            label = "—" if axis is None else (("R", "A", "S")[axis] if "ras" in name else ("i", "j", "k")[axis])
            vals = " | ".join(f"{value[k]:.6g}" for k in ("min", "p50", "p90", "p95", "max"))
            lines.append(f"| {name} | {label} | {value['count']} | {vals} |")
    if summary["distributions"]:
        native_shape = summary["distributions"]["shape_native"]
        native_spacing = summary["distributions"]["spacing_affine_mm"]
        lines += ["", "快速阅读（原生存储轴）：",
                  f"- i/j 尺寸范围分别为 {native_shape[0]['min']}–{native_shape[0]['max']} / {native_shape[1]['min']}–{native_shape[1]['max']}；k 轴 {native_shape[2]['min']}–{native_shape[2]['max']}，中位数 {native_shape[2]['p50']:.6g}。",
                  f"- k 轴 spacing 范围 {native_spacing[2]['min']:.6g}–{native_spacing[2]['max']:.6g} mm，中位数 {native_spacing[2]['p50']:.6g} mm。",
                  f"- 最大体素数 {summary['distributions']['voxel_count']['max']:,}；最高各向异性比 {summary['distributions']['anisotropy_ratio']['max']:.6g}。"]
    lines += ["", "## 空间与强度注意事项", "",
              f"- 图像方向计数：`{summary['image_axis_code_counts']}`。",
              f"- 图像 qform/sform：`{summary['image_form_counts']}`。",
              f"- 标签 qform/sform：`{summary['label_form_counts']}`。",
              f"- 图像有效强度缩放 slope/intercept：`{summary['image_scaling_counts']}`。",
              "- 强度缩放必须逐文件遵循元数据，不应统一对所有病例再次减去 1024；本次没有读取实际 HU 值。",
              "- 标签 sform_code=0 但 qform 有效并不等于错位；比较有效仿射而不是原始 srow。",
              "- 体素数和各向异性极值是待考察风险，不是数据损坏阈值。", "",
              "## 异常与待核查病例", "",
              f"异常分类计数：`{summary['anomaly_code_counts']}`。",
              f"错误病例：`{summary['cases_with_errors']}`。",
              f"警告病例：`{summary['cases_with_warnings']}`。",
              "完整清单见 anomalies.csv；有效但互相不同的 qform/sform 保留为警告，需审查坐标空间含义。", "",
              "## 后续候选测试的元数据代表病例", ""]
    lines += [f"- {key}: {', '.join(ids)}" for key, ids in summary["representative_case_ids"].items()]
    lines += ["", "这些只覆盖元数据极值，不能替代候选重采样后最大输入及小器官风险病例筛选。", "",
              "## 已做和未做", "",
              "已做：清单筛选/文件配对、NIfTI-1 头解析、空间单位换算、有效仿射与几何一致性、统计分布。",
              "未做：图像强度分布、标签值域/器官体积、像素级对齐、完整 gzip CRC/体素载荷校验、重采样保真度、图像质量评估、神经网络、GPU 显存或训练。",
              "头信息正常不能证明体素内容正确或压缩文件完整；RAS 重排统计不等于数据预处理完成。",
              "本报告不冻结输入 spacing/尺寸，也不预设小器官合格阈值。", "",
              "## 复现", "", f"Git：`{provenance['git']}`。",
              f"数据清单 SHA256：`{provenance['manifest_sha256']}`。",
              "配置、代码哈希、选中病例清单哈希、设备和每文件头哈希见 JSON；绝对数据路径只保留在被 Git 忽略的本地报告。", ""]
    return "\n".join(lines)


def audit(data_root: Path, config_path: Path, output_dir: Path) -> dict:
    root, output = data_root.resolve(), output_dir.resolve()
    if output.is_relative_to(root):
        raise ValueError("output directory must be outside the original dataset")
    if output.exists():
        raise ValueError("output directory already exists; choose a fresh run directory")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    selected, excluded, anomalies = select_training(root, config)
    manifest_path = contained_path(root, config["manifest"])
    manifest_digest = sha256(manifest_path)
    records = []
    for entry in selected:
        record = dict(entry)
        for role in ("image", "label"):
            try:
                header = read_header(contained_path(root, entry[role]),
                                     geometry_tolerance_mm=config["geometry_tolerance_mm"],
                                     axis_alignment_tolerance=config["axis_alignment_tolerance"])
                record[role+"_header"] = header
                for notice in header["notices"]:
                    anomalies.append({"case_id": entry["case_id"], "severity": "warning",
                                      "code": notice, "detail": role})
                if role == "label" and (header["datatype"].startswith("float")
                                        or header["scaling_effective"] != [1.0, 0.0]):
                    anomalies.append({"case_id": entry["case_id"], "severity": "warning",
                                      "code": "label_datatype_or_scaling_review", "detail": role})
            except (OSError, EOFError, zlib.error, HeaderError) as exc:
                anomalies.append({"case_id": entry["case_id"], "severity": "error",
                                  "code": "header_read_error", "detail": f"{role}: {exc}"})
        if "image_header" in record and "label_header" in record:
            record["geometry"] = compare_geometry(record["image_header"], record["label_header"],
                                                   config["geometry_tolerance_mm"])
            if not record["geometry"]["consistent"]:
                anomalies.append({"case_id": entry["case_id"], "severity": "error",
                                  "code": "image_label_geometry_mismatch",
                                  "detail": json.dumps(record["geometry"])})
        records.append(record)
    if sha256(manifest_path) != manifest_digest:
        raise ValueError("manifest changed during audit")
    summary = summarize(records, anomalies, len(excluded))
    provenance = {
        "utc_time": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python": platform.python_version(), "data_root": str(root), "config": config,
        "manifest_sha256": manifest_digest, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "selected_manifest_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
        "source_sha256": code_hashes(), "git": git_state(),
        "read_scope": "352 uncompressed bytes requested per selected file; no voxel array reads",
    }
    output.mkdir(parents=True, exist_ok=False)
    payload = {"schema_version": 1, "provenance": provenance, "summary": summary,
               "excluded_training": excluded, "cases": records, "anomalies": anomalies}
    (output/"metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    write_csv(output/"anomalies.csv", ["case_id", "severity", "code", "detail"], anomalies)
    fields = ["case_id", "image", "label", "status", "shape_native", "spacing_mm",
              "cell_coverage_mm", "shape_ras", "spacing_ras_mm", "axis_codes", "voxel_count",
              "anisotropy_ratio", "image_affine_source", "label_affine_source", "max_corner_distance_mm"]
    rows = []
    for r in records:
        ih, lh, geo = r.get("image_header", {}), r.get("label_header", {}), r.get("geometry", {})
        row = {key: r[key] for key in ("case_id", "image", "label")}
        row.update(status="consistent" if geo.get("consistent") else "error",
                   shape_native=ih.get("shape_native"), spacing_mm=ih.get("spacing_affine_mm"),
                   cell_coverage_mm=ih.get("cell_coverage_native_mm"), shape_ras=ih.get("shape_ras"),
                   spacing_ras_mm=ih.get("spacing_ras_mm"), axis_codes=ih.get("axis_codes"),
                   voxel_count=ih.get("voxel_count"), anisotropy_ratio=ih.get("anisotropy_ratio"),
                   image_affine_source=ih.get("affine_source"), label_affine_source=lh.get("affine_source"),
                   max_corner_distance_mm=geo.get("max_corner_distance_mm"))
        rows.append(row)
    write_csv(output/"cases.csv", fields, rows)
    (output/"report.md").write_text(markdown_report(summary, provenance), encoding="utf-8")
    return payload


def main(argv=None) -> int:
    # Windows redirected stdout may otherwise encode Chinese paths as GBK.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"configs"/"ct_stats.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory outside dataset; never overwrites an old report")
    args = parser.parse_args(argv)
    try:
        result = audit(args.data_root, args.config, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")
    print(json.dumps({k: v for k, v in result["summary"].items() if k.endswith("count")}, ensure_ascii=True))
    print(f"Report: {args.output_dir.resolve() / 'report.md'}")
    return 2 if result["summary"]["error_count"] else 0
