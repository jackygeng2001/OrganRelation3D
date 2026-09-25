"""Full-FOV grid estimates from an existing metadata JSON; no NIfTI reads."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

from ..provenance import PROJECT_ROOT, code_hashes, git_state, sha256
from .ct_stats import distribution, write_csv


def estimate_case(case: dict, candidate: dict, multiples: list[int]) -> dict:
    h = case["image_header"]
    if h.get("axis_aligned") is not True:
        raise ValueError(f"{case['case_id']}: cardinal RAS metadata required; no oblique-grid approximation")
    shape, spacing = h["shape_ras"], h["spacing_ras_mm"]
    target = candidate["spacing_ras_mm"]
    if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("invalid source shape")
    if len(spacing) != 3 or len(target) != 3 or not all(
            math.isfinite(x) and x > 0 for x in [*spacing, *target]):
        raise ValueError("invalid source or target spacing")
    lengths = [n*s for n, s in zip(shape, spacing)]
    if any(abs(a-b) > 1e-5 for a, b in zip(lengths, h["coverage_ras_mm"])):
        raise ValueError("inconsistent source coverage")
    # Native float header precision is retained: a tiny extent excess can add
    # one voxel. Do not round down and silently discard physical coverage.
    output_shape = [math.ceil(length/t) for length, t in zip(lengths, target)]
    delta = [n*t-length for n, t, length in zip(output_shape, target, lengths)]
    voxels = math.prod(output_shape)
    boundary_min = h["world_boundary_min_mm"]
    boundary_max = h["world_boundary_max_mm"]
    if any(abs((hi-lo)-length) > 1e-3 for lo, hi, length in zip(boundary_min, boundary_max, lengths)):
        raise ValueError("world boundary extent disagrees with axis-aligned coverage")
    out_min = [lo-extra/2 for lo, extra in zip(boundary_min, delta)]
    out_max = [hi+extra/2 for hi, extra in zip(boundary_max, delta)]
    pads = {}
    for multiple in multiples:
        if type(multiple) is not int or multiple <= 0:
            raise ValueError("padding multiples must be positive integers")
        padded = [((n+multiple-1)//multiple)*multiple for n in output_shape]
        padded_voxels = math.prod(padded)
        pads[str(multiple)] = {
            "shape_dhw": padded[::-1], "added_voxels_dhw": [p-n for p,n in zip(padded,output_shape)][::-1],
            "voxel_count": padded_voxels, "added_voxel_count": padded_voxels-voxels,
            "overhead_percent": 100*(padded_voxels/voxels-1),
            "padded_fraction_percent": 100*(1-voxels/padded_voxels),
            "added_extent_ras_mm": [(p-n)*t for p,n,t in zip(padded,output_shape,target)],
            "single_16channel_fp32_gib": padded_voxels*16*4/(1024**3),
        }
    # Tolerance only classifies nearly identical spacing in descriptive counts;
    # it never changes the shape or grid calculation.
    z_ratio = spacing[2]/target[2]
    z_action = "approximately_unchanged" if math.isclose(z_ratio, 1, rel_tol=1e-6) else (
        "upsample" if z_ratio > 1 else "downsample")
    return {
        "case_id": case["case_id"], "candidate_id": candidate["id"],
        "source_shape_dhw": shape[::-1], "source_spacing_ras_mm": spacing,
        "source_coverage_ras_mm": lengths, "source_voxel_count": math.prod(shape),
        "target_spacing_ras_mm": target, "shape_dhw": output_shape[::-1],
        "voxel_count": voxels, "voxel_ratio_to_source": voxels/math.prod(shape),
        "axis_sample_density_ratio_ras": [s/t for s,t in zip(spacing,target)],
        "z_sampling_action": z_action,
        "rounding_extra_extent_ras_mm": delta,
        "rounding_extra_physical_volume_percent": 100*(math.prod([n*t for n,t in zip(output_shape,target)])/math.prod(lengths)-1),
        "estimated_output_boundary_min_ras_mm": out_min,
        "estimated_output_boundary_max_ras_mm": out_max,
        "estimated_first_voxel_center_ras_mm": [lo+t/2 for lo,t in zip(out_min,target)],
        "single_16channel_fp32_gib": voxels*16*4/(1024**3), "padding": pads,
    }


def summarize_candidate(rows: list[dict], multiples: list[int]) -> dict:
    dist = distribution([r["voxel_count"] for r in rows])
    typical = min(rows, key=lambda r: (abs(r["voxel_count"]-dist["p50"]), r["case_id"]))
    maximum = max(rows, key=lambda r: r["voxel_count"])
    summary = {
        "case_count": len(rows), "voxel_count": dist,
        "shape_dhw": [distribution([r["shape_dhw"][axis] for r in rows]) for axis in range(3)],
        "typical_case_id": typical["case_id"], "typical_shape_dhw": typical["shape_dhw"],
        "maximum_case_id": maximum["case_id"], "maximum_shape_dhw": maximum["shape_dhw"],
        "z_sampling_action_counts": {action:sum(r["z_sampling_action"]==action for r in rows)
                                     for action in ("upsample","approximately_unchanged","downsample")},
        "rounding_extra_volume_percent": distribution([r["rounding_extra_physical_volume_percent"] for r in rows]),
        "z_density_ratio": distribution([r["axis_sample_density_ratio_ras"][2] for r in rows]),
        "padding": {},
    }
    for m in multiples:
        k = str(m)
        worst_pad = max(rows, key=lambda r:r["padding"][k]["overhead_percent"])
        largest = max(rows, key=lambda r:r["padding"][k]["voxel_count"])
        summary["padding"][k] = {
            "overhead_percent": distribution([r["padding"][k]["overhead_percent"] for r in rows]),
            "voxel_count": distribution([r["padding"][k]["voxel_count"] for r in rows]),
            "max_overhead_case_id": worst_pad["case_id"],
            "maximum_padded_case_id": largest["case_id"],
            "maximum_padded_shape_dhw": largest["padding"][k]["shape_dhw"],
            "max_single_16channel_fp32_gib": largest["padding"][k]["single_16channel_fp32_gib"],
        }
    return summary


def estimate_dataset(metadata: dict, config: dict) -> dict:
    if metadata.get("schema_version") != 1 or config.get("schema_version") != 1:
        raise ValueError("schema version must be 1")
    if config.get("status") != "exploratory_not_frozen" or config.get("estimate_grid") != "ceil_cell_coverage_centered_extension":
        raise ValueError("only explicitly exploratory coverage-preserving grid estimates are supported")
    cases = metadata["cases"]
    if len(cases) != config["expected_training_ct_count"] or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("unexpected case count or duplicate case IDs")
    if metadata["provenance"]["config"]["split"] != "training":
        raise ValueError("training metadata only")
    if any(not c.get("geometry", {}).get("consistent") for c in cases):
        raise ValueError("all image-label pairs must first pass geometry audit")
    candidates = config["candidates"]
    if not candidates or len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique and nonempty")
    multiples = config["hypothetical_padding_multiples"]
    if 16 not in multiples or 32 not in multiples:
        raise ValueError("report requires hypothetical 16 and 32 sensitivity scenarios")
    rows, summaries = [], {}
    for candidate in candidates:
        candidate_rows = [estimate_case(c, candidate, multiples) for c in cases]
        rows.extend(candidate_rows)
        summaries[candidate["id"]] = summarize_candidate(candidate_rows, multiples)
    # Same comparison cases in all candidates, selected from metadata, no labels.
    largest_native = max(cases, key=lambda c:c["image_header"]["voxel_count"])["case_id"]
    longest_scan = max(cases, key=lambda c:c["image_header"]["coverage_ras_mm"][2])["case_id"]
    most_anisotropic = max(cases, key=lambda c:c["image_header"]["anisotropy_ratio"])["case_id"]
    representatives = {"typical": summaries[candidates[0]["id"]]["typical_case_id"],
                       "largest_target": summaries[candidates[0]["id"]]["maximum_case_id"],
                       "largest_native": largest_native, "longest_scan": longest_scan,
                       "highest_anisotropy": most_anisotropic}
    source_z_counts = dict(sorted(Counter(str(round(c["image_header"]["spacing_ras_mm"][2],4))
                                          for c in cases).items()))
    return {"config": config, "summaries": summaries, "cases": rows, "representatives": representatives,
            "source_z_spacing_counts_rounded_mm": source_z_counts}


def shape_text(shape):
    return "×".join(str(n) for n in shape)


def render_report(result: dict) -> str:
    lines = ["# 完整 CT 候选网格元数据估算", "",
             "状态：候选研究，未冻结预处理。只读取已有训练元数据 JSON，未打开 CT/标签体素、未重采样、未测试 GPU。", "",
             "## 估算约定", "",
             "spacing 顺序为 R,A,S（平面两轴、层间）；尺寸展示 D=S,H=A,W=R，仅为报告约定。",
             "以原体素单元边界覆盖 L=n*s 估算新尺寸 n'=ceil(L/t)，保持候选 spacing t，网格中心不变。每轴向两端各扩展 (n'*t-L)/2，不裁掉任何原始覆盖。总扩展小于一个目标体素；实际边界取值/插值规则仍待确认。",
             "不舍弃头信息浮点精度；接近整除边界时微小差异也可能增加一格。此估算网格不是已选定的预处理实现。",
             "假设各轴补齐到 16/32 倍数只用于敏感性分析，不确定骨干步幅、padding 位置/值或有效区域定义。padding 开销=(补齐体素/未补齐体素-1)*100%。", "",
             f"## 候选对照（{result['config']['expected_training_ct_count']} 例逐例计算后汇总）", "",
             "| 候选 | spacing mm | 典型尺寸 | 最大体素病例尺寸 | 体素 M 中位/P95/最大 | pad16 开销 中位/最大 | pad32 开销 中位/最大 |",
             "|---|---|---|---|---|---|---|"]
    for c in result["config"]["candidates"]:
        s = result["summaries"][c["id"]]; v = s["voxel_count"]
        p16, p32 = [s["padding"][m]["overhead_percent"] for m in ("16","32")]
        lines.append(f"| {c['id']} | {shape_text(c['spacing_ras_mm'])} | {shape_text(s['typical_shape_dhw'])} | {shape_text(s['maximum_shape_dhw'])} | {v['p50']/1e6:.2f}/{v['p95']/1e6:.2f}/{v['max']/1e6:.2f} | {p16['p50']:.1f}%/{p16['max']:.1f}% | {p32['p50']:.1f}%/{p32['max']:.1f}% |")
    lines += ["", "典型病例为体素数最接近中位数的真实病例；最大尺寸来自最大体素病例，不是把三个轴各自最大值拼成虚构病例。", "",
              "## 全部病例与典型/极端病例", "",
              f"逐病例结果共 {len(result['cases'])} 行见 cases.csv；全部轴尺寸、分位数、grid 边界及 padding 细节见 estimates.json。", "",
              "| 选择依据 | 病例 | 原 D×H×W | 原 spacing R,A,S | A 尺寸 | B 尺寸 | C 尺寸 |",
              "|---|---|---|---|---|---|---|"]
    lookup = {(r["case_id"],r["candidate_id"]):r for r in result["cases"]}
    for reason, case_id in result["representatives"].items():
        rows = [lookup[(case_id,c["id"])] for c in result["config"]["candidates"]]
        lines.append(f"| {reason} | {case_id} | {shape_text(rows[0]['source_shape_dhw'])} | {shape_text([round(x,5) for x in rows[0]['source_spacing_ras_mm']])} | " + " | ".join(shape_text(r["shape_dhw"]) for r in rows)+" |")
    lines += ["", "## 假设性张量内存量，不是训练峰值", "",
              "只估算 batch=1 的单个全输入分辨率 16 通道 FP32 张量：4*16*体素数/2^30 GiB。没有包含其他 logits/概率副本、骨干激活、skip、梯度、优化器、算子工作区及分配器保留。不能据此判断 24GB 能否训练。", "",
              "| 候选 | 最大 pad16 病例 | 最大 pad16 尺寸 | 最大 pad16 体素 M | 单个 16 通道 FP32 GiB | pad16 最大比例病例 |",
              "|---|---|---|---:|---:|---|"]
    for c in result["config"]["candidates"]:
        p = result["summaries"][c["id"]]["padding"]["16"]
        lines.append(f"| {c['id']} | {p['maximum_padded_case_id']} | {shape_text(p['maximum_padded_shape_dhw'])} | {p['voxel_count']['max']/1e6:.2f} | {p['max_single_16channel_fp32_gib']:.2f} | {p['max_overhead_case_id']} |")
    lines += ["", "## 层间采样变化与风险", ""]
    for c in result["config"]["candidates"]:
        s = result["summaries"][c["id"]]
        lines.append(f"- {c['id']}：层间采样分类 {s['z_sampling_action_counts']}；采样密度比（原 spacing/目标 spacing）范围 {s['z_density_ratio']['min']:.3g}–{s['z_density_ratio']['max']:.3g}。")
    lines += ["", f"原层间 spacing 分组（仅显示时四舍五入到 4 位）：`{result['source_z_spacing_counts_rounded_mm']}`。",
              "采样密度比大于 1 表示插值加密，不增加真实解剖信息；小于 1 表示降采样，存在细节合并风险。尺寸估算不能证明小器官保真。平面/层间的插值、抗混叠、标签重采样细节需下一阶段验证，当前三候选的具体取舍见 docs/preprocessing_candidates.md。", "",
              "## padding 与方法定义尚未决定", "",
              "补入位置改变 N 和归一化坐标；即使新增区域器官概率为零，相对体积 M/N 和质心坐标也可能改变。若把 padding 标签视作背景，CE 的体素分母和监督范围也改变，soft Dice 还会受新增预测影响。",
              "只在损失中排除 padding 不能消除建图统计的变化；若同时修改节点统计的有效域，则必须明确特征层 mask、N、坐标范围、边界混合等新规则，这属于需确认的方法决定。",
              "可比较允许不整除尺寸的骨干，减少显式输入 padding；其逐级尺寸/skip 对齐仍需设计和测试。也可讨论完整 FOV 内微调实际 spacing 使尺寸整除，但这是不同于本表的输入网格方案，未采用。", "",
              "## 下一步", "",
              "先审查三候选和网格/padding 语义；再对少量训练病例做实际 CT/标签重采样与原空间往返，评估小器官消失/体积/边界、强度与纹理风险。阈值未设，先报告实际结果。网络建立后在 AMD 上测完整双分支前后向显存，不裁扫描两端、不局部建图。", "",
              "## 来源", "", f"元数据 SHA256：`{result['provenance']['metadata_sha256']}`。",
              f"Git：`{result['provenance']['git']}`。",
              "配置、来源统计 commit/清单哈希和当前代码哈希见 estimates.json；生成的病例级文件默认不提交 Git。", ""]
    return "\n".join(lines)


def run(metadata_path: Path, config_path: Path, output_dir: Path) -> dict:
    output = output_dir.resolve()
    if output.exists():
        raise ValueError("use a fresh output directory; existing reports are never overwritten")
    raw = metadata_path.read_bytes()
    metadata = json.loads(raw)
    data_root_text = metadata["provenance"].get("data_root")
    if data_root_text and Path(data_root_text).is_absolute() and output.is_relative_to(Path(data_root_text).resolve()):
        raise ValueError("output must stay outside original data directory")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = estimate_dataset(metadata, config)
    result["provenance"] = {
        "utc_time": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python": platform.python_version(), "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "source_audit_git": metadata["provenance"]["git"],
        "source_manifest_sha256": metadata["provenance"]["manifest_sha256"],
        "config_sha256": sha256(config_path), "source_sha256": code_hashes(), "git": git_state(),
        "nifti_files_opened": 0, "resampling_performed": False, "gpu_tested": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output/"estimates.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    rows = []
    for r in result["cases"]:
        item = {key:r[key] for key in ("case_id","candidate_id","shape_dhw","voxel_count",
                                      "source_shape_dhw","source_spacing_ras_mm","source_coverage_ras_mm",
                                      "target_spacing_ras_mm","z_sampling_action","rounding_extra_extent_ras_mm")}
        for m in ("16","32"):
            p = r["padding"][m]
            for key in ("shape_dhw","voxel_count","overhead_percent"):
                item[f"pad{m}_{key}"] = p[key]
        rows.append(item)
    write_csv(output/"cases.csv", list(rows[0]), rows)
    (output/"report.md").write_text(render_report(result), encoding="utf-8")
    return result


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"configs/preprocessing_candidates.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args.metadata,args.config,args.output_dir)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        parser.exit(2,f"Estimation failed: {exc}\n")
    print(f"Estimated {len(result['cases'])} case-candidate combinations; NIfTI reads=0.")
    print(args.output_dir.resolve()/"report.md")
    return 0
