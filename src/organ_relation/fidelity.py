"""CPU pilot resampling and fidelity measurements; never model predictions.

Uses physical affines (voxel centers), a full-coverage target grid, no network
padding or validity masks. All volume writes, if any, belong in ignored reports.
"""

from __future__ import annotations

import gzip
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .nifti_header import compare_geometry, read_header

LABELS = ["background", "spleen", "right kidney", "left kidney", "gallbladder",
          "esophagus", "liver", "stomach", "aorta", "inferior vena cava", "pancreas",
          "right adrenal", "left adrenal", "duodenum", "bladder", "prostate/uterus"]


def affine4(header: dict) -> np.ndarray:
    a = np.eye(4, dtype=np.float64)
    a[:3] = header["affine_mm"]
    return a


def build_target_grid(header: dict, target_spacing) -> tuple:
    if not header["axis_aligned"]:
        raise ValueError("pilot restricted to audited cardinal grids")
    t = np.asarray(target_spacing, dtype=np.float64)
    if t.shape != (3,) or not np.all(np.isfinite(t) & (t > 0)):
        raise ValueError("target spacing must be three positive finite values")
    lo = np.asarray(header["world_boundary_min_mm"])
    hi = np.asarray(header["world_boundary_max_mm"])
    lengths = hi-lo
    shape = tuple(int(math.ceil(length/s)) for length, s in zip(lengths, t))
    extension = np.array(shape)*t-lengths
    out_lo, out_hi = lo-extension/2, hi+extension/2
    a = np.eye(4, dtype=np.float64)
    a[:3,:3] = np.diag(t)
    a[:3,3] = out_lo+t/2
    info = {"shape_xyz":list(shape),"spacing_ras_mm":t.tolist(),
            "affine_ras_mm":a.tolist(),"original_boundary_min_ras_mm":lo.tolist(),
            "original_boundary_max_ras_mm":hi.tolist(),"target_boundary_min_ras_mm":out_lo.tolist(),
            "target_boundary_max_ras_mm":out_hi.tolist(),"center_ras_mm":((lo+hi)/2).tolist(),
            "rounding_extension_ras_mm":extension.tolist(),"network_padding":False}
    if not np.all(out_lo <= lo+1e-8) or not np.all(out_hi >= hi-1e-8):
        raise AssertionError("target does not cover complete source boundaries")
    return shape, a, info


def antialias_sigmas(source_affine, target_affine):
    mapping = np.linalg.solve(np.asarray(source_affine), np.asarray(target_affine))[:3,:3]
    # This pilot's fixed grids are cardinal; each input axis receives one output
    # axis, possibly flipped/permuted. Rotation/shear needs a separate protocol.
    if not np.all(np.count_nonzero(np.abs(mapping)>1e-7,axis=1)==1):
        raise ValueError("Gaussian pilot prefilter requires cardinal axis mapping")
    ratios = np.max(np.abs(mapping),axis=1)
    return np.maximum((ratios-1)/2,0)


def resample(array, source_affine, target_shape, target_affine, order,
             antialias=False, truncate=3.0):
    if order not in (0,1) or (order == 0 and antialias):
        raise ValueError("labels: nearest/no filter; CT: linear/optional prefilter")
    mapping = np.linalg.solve(np.asarray(source_affine),np.asarray(target_affine))
    source = array
    if antialias:
        sigmas = antialias_sigmas(source_affine,target_affine)
        if np.any(sigmas > 1e-8):
            source = ndi.gaussian_filter(array,sigma=sigmas,mode="nearest",
                                         truncate=truncate,output=np.float32)
    return ndi.affine_transform(source,mapping[:3,:3],offset=mapping[:3,3],
                                output_shape=tuple(target_shape),order=order,
                                mode="nearest",prefilter=False,
                                output=np.float32 if order==1 else array.dtype)


def chunks(shape, depth=8):
    for start in range(0,shape[2],depth):
        yield (slice(None),slice(None),slice(start,min(start+depth,shape[2])))


def counts(array, chunk_slices=8):
    total = np.zeros(16,dtype=np.int64)
    for sl in chunks(array.shape,chunk_slices):
        part = array[sl]
        if part.min() < 0 or part.max() > 15:
            raise ValueError("illegal label outside 0..15")
        total += np.bincount(part.ravel().astype(np.int64),minlength=16)
    return total


def load_pair(image_path: Path, label_path: Path):
    ih, lh = read_header(image_path), read_header(label_path)
    geometry = compare_geometry(ih,lh)
    if not geometry["consistent"]:
        raise ValueError("image/label geometry mismatch")
    im, la = nib.load(str(image_path)), nib.load(str(label_path))
    if not np.allclose(im.affine,affine4(ih),atol=1e-5,rtol=0):
        raise ValueError("independent NIfTI image affine implementations disagree")
    if not np.allclose(la.affine,affine4(lh),atol=1e-5,rtol=0):
        raise ValueError("independent NIfTI label affine implementations disagree")
    if not np.allclose([im.dataobj.slope,im.dataobj.inter],ih["scaling_effective"]):
        raise ValueError("scaling implementations disagree")
    ct = im.get_fdata(dtype=np.float32,caching="unchanged")
    raw_label = np.asanyarray(la.dataobj)
    if (not np.all(np.isfinite(raw_label)) or raw_label.min()<0 or raw_label.max()>15
            or not np.all(raw_label == np.rint(raw_label))):
        raise ValueError("expected integer labels in 0..15; refusing truncation")
    labels = raw_label.astype(np.uint8)
    del raw_label, im, la
    for sl in chunks(ct.shape):
        if not np.all(np.isfinite(ct[sl])):
            raise ValueError("nonfinite CT voxels")
    # Independently decode a small raw sample, so applying scaling twice fails.
    opener = gzip.open if image_path.name.endswith(".gz") else open
    dtype = np.dtype(ih["datatype"]).newbyteorder("<" if ih["endianness"]=="little" else ">")
    with opener(image_path,"rb") as stream:
        stream.seek(ih["voxel_offset"])
        raw = np.frombuffer(stream.read(min(8,ct.shape[0])*dtype.itemsize),dtype=dtype)
    expected = raw.astype(np.float32)*ih["scaling_effective"][0]+ih["scaling_effective"][1]
    error = float(np.max(np.abs(ct[:8,0,0]-expected)))
    if error > 1e-4:
        raise ValueError("raw sample scaling check failed")
    diagnostic = {"image_label_geometry":geometry,"nibabel_affine_crosscheck":True,
                  "raw_scaling_sample_max_error_hu":error,
                  "scaling_effective":ih["scaling_effective"],
                  "source_label_counts":counts(labels).tolist()}
    return ct, labels, ih, lh, diagnostic


def label_metrics(original, target, restored, original_affine, target_affine,
                  chunk_slices=8):
    if original.shape != restored.shape:
        raise ValueError("roundtrip must return to original label grid")
    confusion = np.zeros((16,16),dtype=np.int64)
    for sl in chunks(original.shape,chunk_slices):
        a,b=original[sl],restored[sl]
        if a.max()>15 or b.max()>15 or a.min()<0 or b.min()<0:
            raise ValueError("illegal labels in roundtrip")
        indices = (a.astype(np.uint16)*16+b).ravel().astype(np.int64)
        confusion += np.bincount(indices,minlength=256).reshape(16,16)
    orig_count, rt_count = confusion.sum(axis=1),confusion.sum(axis=0)
    tar_count = counts(target,chunk_slices)
    vs = abs(float(np.linalg.det(np.asarray(original_affine)[:3,:3])))
    vt = abs(float(np.linalg.det(np.asarray(target_affine)[:3,:3])))
    rows=[]
    for i in range(1,16):
        n, nt, nr = int(orig_count[i]),int(tar_count[i]),int(rt_count[i])
        if n==0 and (nt or nr):
            raise AssertionError("nearest-neighbor introduced an absent class")
        rows.append({"label":i,"organ":LABELS[i],"source_present":n>0,
                     "source_voxels":n,"target_voxels":nt,"roundtrip_voxels":nr,
                     "source_volume_mm3":n*vs,"target_volume_mm3":nt*vt,"roundtrip_volume_mm3":nr*vs,
                     "roundtrip_dice":2*int(confusion[i,i])/(n+nr) if n else None,
                     "target_volume_change_percent":100*(nt*vt/(n*vs)-1) if n else None,
                     "roundtrip_volume_change_percent":100*(nr/n-1) if n else None,
                     "disappeared_in_target":bool(n and not nt),
                     "disappeared_in_roundtrip":bool(n and not nr),
                     "source_absent":n==0})
    return rows


def ct_metrics(original, restored, labels, spacing, ring_mm=(2.,5.),chunk_slices=8):
    n=np.zeros(16,np.int64); sums=np.zeros((5,16),np.float64)
    gradient_sum=np.zeros((2,3),np.float64); gradient_n=np.zeros(3,np.int64)
    for sl in chunks(original.shape,chunk_slices):
        a=original[sl];b=restored[sl]; lab=labels[sl]
        ids=lab.ravel().astype(np.int64)
        n+=np.bincount(ids,minlength=16)
        for j,arr in enumerate((a,b,a*a,b*b,np.abs(a-b))):
            sums[j]+=np.bincount(ids,weights=arr.ravel(),minlength=16)
        # Include next z plane at chunk boundary; each pair counted once.
        stop=min(sl[2].stop+1,original.shape[2])
        block=(slice(None),slice(None),slice(sl[2].start,stop))
        for axis in range(3):
            aa,bb,ll=(original[block],restored[block],labels[block]) if axis==2 else (a,b,lab)
            left=[slice(None)]*3;right=[slice(None)]*3
            left[axis]=slice(None,-1);right[axis]=slice(1,None)
            mask=(ll[tuple(left)]>0)&(ll[tuple(right)]>0)
            gradient_n[axis]+=int(mask.sum())
            for side,arr in enumerate((aa,bb)):
                diff=np.abs(np.diff(arr,axis=axis))/spacing[axis]
                gradient_sum[side,axis]+=float(diff[mask].sum(dtype=np.float64))
    regions=ndi.find_objects(labels,max_label=15)
    organs=[]
    for i in range(1,16):
        if not n[i]:
            organs.append({"label":i,"source_present":False});continue
        mean0,mean1=sums[0,i]/n[i],sums[1,i]/n[i]
        var0=max(0.,sums[2,i]/n[i]-mean0**2);var1=max(0.,sums[3,i]/n[i]-mean1**2)
        item={"label":i,"organ":LABELS[i],"source_present":True,
              "source_mean_hu":mean0,"roundtrip_mean_hu":mean1,
              "source_std_hu":math.sqrt(var0),"roundtrip_std_hu":math.sqrt(var1),
              "std_ratio":math.sqrt(var1/var0) if var0>1e-10 else None,
              "mae_hu":sums[4,i]/n[i]}
        bbox=regions[i-1]
        slices=tuple(slice(max(0,s.start-math.ceil(ring_mm[1]/spacing[a])-1),
                           min(original.shape[a],s.stop+math.ceil(ring_mm[1]/spacing[a])+1))
                     for a,s in enumerate(bbox))
        local=labels[slices];dist=ndi.distance_transform_edt(local!=i,sampling=spacing)
        ring=(dist>ring_mm[0])&(dist<=ring_mm[1])&(local==0)
        item["contrast_ring_voxels"]=int(ring.sum())
        if ring.any():
            contrast0=mean0-float(original[slices][ring].mean(dtype=np.float64))
            contrast1=mean1-float(restored[slices][ring].mean(dtype=np.float64))
            item.update(source_local_contrast_hu=contrast0,roundtrip_local_contrast_hu=contrast1,
                        local_contrast_abs_ratio=abs(contrast1/contrast0) if abs(contrast0)>1e-6 else None)
        organs.append(item)
    fg=int(n[1:].sum())
    ratio=[float(gradient_sum[1,a]/gradient_sum[0,a]) if gradient_sum[0,a]>0 else None for a in range(3)]
    return {"global_mae_hu":float(sums[4].sum()/n.sum()),
            "original_foreground_mae_hu":float(sums[4,1:].sum()/fg) if fg else None,
            "foreground_gradient_abs_ratio_native_axes":ratio,
            "gradient_pair_count":gradient_n.tolist()},organs
