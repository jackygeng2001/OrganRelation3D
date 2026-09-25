"""Analytic physical-coordinate tests before any real-volume pilot."""
from pathlib import Path
import sys
import tempfile
import unittest
import json
from unittest.mock import patch
import contextlib
import io
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
try:
    import numpy as np
    import nibabel as nib
    from organ_relation.data.fidelity import (affine4, build_target_grid, resample,
        antialias_sigmas, load_pair, label_metrics, ct_metrics)
    from organ_relation.data.nifti_header import read_header
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,"isolated CPU resampling dependencies required")
class FidelityTests(unittest.TestCase):
    def test_identity_ct_and_labels(self):
        rng=np.random.default_rng(5)
        a=np.diag([-0.7,0.7,5.,1.]);a[:3,3]=[180,-210,44]
        for order,x in [(1,rng.normal(size=(9,10,7)).astype(np.float32)),
                        (0,rng.integers(0,16,(9,10,7),dtype=np.uint8))]:
            np.testing.assert_allclose(resample(x,a,x.shape,a,order),x,atol=1e-6)

    def test_physical_ramp_translation_and_linear(self):
        shape=(12,13,14);i=np.indices(shape)
        x=(2*i[0]+6*i[1]+12*i[2]+10).astype(np.float32)
        a=np.diag([2.,3.,4.,1.]);b=a.copy();b[:3,3]=[1.,1.5,2.]
        y=resample(x,a,(10,11,12),b,1)
        expected=x[:10,:11,:12]+10
        np.testing.assert_allclose(y,expected,atol=1e-5)

    def test_flip_and_axis_permutation(self):
        x=np.arange(4*5*6,dtype=np.float32).reshape(4,5,6)
        a=np.diag([-2.,3.,4.,1.]);a[0,3]=6
        b=np.diag([2.,3.,4.,1.])
        np.testing.assert_array_equal(resample(x,a,x.shape,b,1),x[::-1])
        p=np.eye(4);p[:3,:3]=[[0,0,1],[0,1,0],[1,0,0]]
        np.testing.assert_array_equal(resample(x,np.eye(4),(6,5,4),p,0),x.transpose(2,1,0))

    def test_nearest_labels_known_block_and_boundary(self):
        x=np.zeros((8,8,8),np.uint8);x[2:6,2:6,2:6]=11
        b=np.diag([2.,2.,2.,1.]);b[:3,3]=.5
        y=resample(x,np.eye(4),(4,4,4),b,0)
        self.assertEqual(int((y==11).sum()),8)
        self.assertEqual(set(np.unique(y)),{0,11})
        rt=resample(y,b,x.shape,np.eye(4),0)
        m=label_metrics(x,y,rt,np.eye(4),b)[10]
        self.assertEqual(m['roundtrip_dice'],1.)
        self.assertAlmostEqual(m['target_volume_change_percent'],0.)
        edge=np.ones((3,3,3),np.float32)*17;b=np.eye(4);b[:3,3]=-10
        np.testing.assert_array_equal(resample(edge,np.eye(4),(2,2,2),b,1),17)

    def test_disappearance_and_absence(self):
        x=np.zeros((5,5,5),np.uint8);x[1,1,1]=11
        b=np.diag([3.,3.,3.,1.]);y=resample(x,np.eye(4),(2,2,2),b,0)
        rt=resample(y,b,x.shape,np.eye(4),0)
        rows=label_metrics(x,y,rt,np.eye(4),b)
        self.assertTrue(rows[10]['disappeared_in_target'])
        self.assertEqual(rows[10]['roundtrip_dice'],0.)
        self.assertIsNone(rows[11]['roundtrip_dice'])
        self.assertTrue(rows[11]['source_absent'])
        y[0,0,0]=16
        with self.assertRaises(ValueError):label_metrics(x,y,rt,np.eye(4),b)

    def test_antialias_constant_and_axis_sigma(self):
        a=np.diag([-0.5,1.,5.,1.]);b=np.diag([2.,2.,3.,1.])
        np.testing.assert_allclose(antialias_sigmas(a,b),[1.5,.5,0])
        x=np.ones((9,10,11),np.float32)*-1024
        np.testing.assert_allclose(resample(x,a,(5,5,5),b,1,True),-1024)
        with self.assertRaises(ValueError):resample(x,a,(2,2,2),b,0,True)

    def test_grid_center_full_coverage_scaling_and_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            ip,lp=Path(tmp)/'im.nii.gz',Path(tmp)/'la.nii.gz'
            a=np.diag([-.75,.75,5.,1.]);a[:3,3]=[120,-130,40]
            raw=np.arange(4*5*6,dtype=np.int16).reshape((4,5,6))
            lab=(raw%16).astype(np.uint8)
            im=nib.Nifti1Image(raw,a);im.header.set_xyzt_units('mm');im.header.set_slope_inter(2.,-1024.)
            la=nib.Nifti1Image(lab,a);la.header.set_xyzt_units('mm')
            nib.save(im,ip);nib.save(la,lp)
            ct,labels,h,_,diag=load_pair(ip,lp)
            np.testing.assert_array_equal(ct,raw*2-1024)
            np.testing.assert_array_equal(labels,lab)
            self.assertEqual(diag['raw_scaling_sample_max_error_hu'],0.)
            shape,b,info=build_target_grid(h,[2,2,3])
            center0=a@np.r_[((np.array(raw.shape)-1)/2),1]
            center1=b@np.r_[((np.array(shape)-1)/2),1]
            np.testing.assert_allclose(center0,center1)
            self.assertTrue(np.all(np.array(info['rounding_extension_ras_mm'])>=0))
            self.assertTrue(np.all(np.array(info['rounding_extension_ras_mm'])<[2,2,3]))
            bad=a.copy();bad[0,3]+=1;la.set_sform(bad);nib.save(la,lp)
            with self.assertRaises(ValueError):load_pair(ip,lp)

    def test_ct_metrics_identity_and_chunk_boundary(self):
        x=np.indices((9,10,11)).sum(axis=0).astype(np.float32)
        lab=np.zeros(x.shape,np.uint8);lab[2:7,2:8,2:9]=11
        x[lab==11]+=100
        g,o=ct_metrics(x,x,lab,[1,1,1],chunk_slices=3)
        self.assertEqual(g['global_mae_hu'],0)
        np.testing.assert_allclose(g['foreground_gradient_abs_ratio_native_axes'],[1,1,1])
        g2,_=ct_metrics(x,x,lab,[1,1,1],chunk_slices=20)
        self.assertEqual(g['gradient_pair_count'],g2['gradient_pair_count'])
        self.assertAlmostEqual(o[10]['local_contrast_abs_ratio'],1)

    def test_label_scaling_is_applied_and_fractional_labels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ip,lp=Path(tmp)/'ct.nii.gz',Path(tmp)/'label.nii.gz'
            raw=np.arange(8*9*7,dtype=np.int16).reshape((8,9,7))
            im=nib.Nifti1Image(raw,np.eye(4));im.header.set_xyzt_units('mm');nib.save(im,ip)
            stored=(raw%8).astype(np.uint8)
            label=nib.Nifti1Image(stored,np.eye(4));label.header.set_xyzt_units('mm')
            label.header.set_slope_inter(2.,0.);nib.save(label,lp)
            _,scaled,_,_,_=load_pair(ip,lp)
            np.testing.assert_array_equal(scaled,stored*2)
            label.header.set_slope_inter(.5,0.);nib.save(label,lp)
            with self.assertRaises(ValueError):load_pair(ip,lp)

    def test_sequential_pilot_end_to_end_and_source_unchanged(self):
        from organ_relation.data.fidelity_pilot import main
        root=Path(__file__).resolve().parents[1]
        config=json.loads((root/'configs/fidelity_pilot.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);data=base/'data';data.mkdir();records=[];selection=[];before={}
            for i in range(8):
                cid=f'synthetic_{i}';a=np.diag([-1.,1.,2.,1.])
                lab=np.zeros((12,13,9),np.uint8);lab[2:7,3:9,2:7]=11;lab[8:11,3:9,2:7]=12
                raw=(np.indices(lab.shape).sum(axis=0)*2+lab*10).astype(np.int16)
                paths=[]
                for name,arr in [('image',raw),('label',lab)]:
                    p=data/f'{cid}_{name}.nii.gz';n=nib.Nifti1Image(arr,a)
                    n.header.set_xyzt_units('mm')
                    if name=='image':n.header.set_slope_inter(1,-1024)
                    nib.save(n,p);paths.append(p);before[p.name]=p.read_bytes()
                records.append({'case_id':cid,'image':paths[0].name,'label':paths[1].name,
                                'image_header':read_header(paths[0]),'label_header':read_header(paths[1])})
                selection.append({'id':cid,'reason':'synthetic integration'})
            config['cases']=selection;cp=base/'config.json';mp=base/'metadata.json';out=base/'reports'
            cp.write_text(json.dumps(config),encoding='utf-8');mp.write_text(json.dumps({'cases':records}),encoding='utf-8')
            argv=['pilot','--data-root',str(data),'--metadata',str(mp),'--config',str(cp),'--output-dir',str(out)]
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(),0)
                with self.assertRaises(ValueError):main()
            result=json.loads((out/'results.json').read_text(encoding='utf-8'))
            self.assertEqual(len(result['cases']),8)
            self.assertEqual(len(result['organ_summary']),45)
            self.assertEqual(len(list((out/'visuals').glob('*.png'))),48)
            import importlib.util
            spec=importlib.util.spec_from_file_location('fidelity_summary',root/'scripts/summarize_fidelity.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.assertTrue(module.summarize(out).is_file())
            transforms=json.loads((out/'spatial_transforms.json').read_text())['transforms']
            self.assertEqual(len(transforms),24)
            for row in transforms:
                np.testing.assert_allclose(np.array(row['target_to_original_image_index'])@np.array(row['original_image_to_target_index']),np.eye(4),atol=1e-10)
            for name,content in before.items():self.assertEqual((data/name).read_bytes(),content)

if __name__=='__main__':unittest.main()
