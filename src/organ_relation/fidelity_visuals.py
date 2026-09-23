"""Review images only: label-derived viewports never change input volumes."""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


def view_specs(labels,spacing):
    if labels.ndim!=3:raise ValueError('3D required')
    specs={}
    for name,organ in [('overview',None),('right_adrenal',11),('left_adrenal',12)]:
        mask=labels>0 if organ is None else labels==organ
        if not mask.any():continue
        box=ndi.find_objects(mask.astype(np.uint8))[0]
        views=[]
        for axis,title in [(2,'Axial'),(1,'Coronal'),(0,'Sagittal')]:
            axes=tuple(i for i in range(3) if i!=axis)
            index=int(np.argmax(mask.sum(axis=axes)))
            # Anatomical review zoom only, not a model/resampling crop.
            ranges=[]
            for d in axes:
                margin=int(np.ceil(12/spacing[d])) if organ else 0
                ranges.append(slice(max(0,box[d].start-margin),min(labels.shape[d],box[d].stop+margin)) if organ else slice(0,labels.shape[d]))
            views.append({'axis':axis,'index':index,'ranges':ranges,'title':title,'organ':organ})
        specs[name]=views
    return specs


def extract(array,view):
    selector=[slice(None)]*3;selector[view['axis']]=view['index']
    return array[tuple(selector)][tuple(view['ranges'])].T[::-1,:].copy()


def capture(ct,labels,specs):
    return {name:[(extract(ct,v),extract(labels,v)) for v in views] for name,views in specs.items()}


def contour(lab,organ):
    # Separate class contours (not only outer foreground border).
    result=np.zeros(lab.shape,bool)
    for label in ([organ] if organ else range(1,16)):
        mask=lab==label
        result|=mask & ~ndi.binary_erosion(mask)
    return result


def save_review(out,case_id,specs,captures,spacing,affine,window):
    paths=[];w,h=330,300
    for name,views in specs.items():
        canvas=Image.new('RGB',(4*w,3*h+80),'#15191f');draw=ImageDraw.Draw(canvas)
        draw.text((12,8),f'{case_id} / {name} / original-space matched planes; display HU {window}',fill='white')
        draw.text((12,27),'Green: original label boundary. Magenta: roundtrip. White: overlap. No input crop/padding.',fill='white')
        draw.text((12,44),'Native LAS: axial top A/left R; coronal top S/left R; sagittal top S/left P.',fill='white')
        for col,candidate in enumerate(['original','A','B','C']):
            for row,v in enumerate(views):
                ct,lab=captures[candidate][name][row]
                orig=captures['original'][name][row][1]
                gray=np.clip((ct-window[0])/(window[1]-window[0])*255,0,255).astype(np.uint8)
                rgb=np.repeat(gray[:,:,None],3,axis=2)
                c0=contour(orig,v['organ']);c1=contour(lab,v['organ'])
                rgb[c0]=[20,255,90]
                if candidate!='original':
                    rgb[c1]=[255,45,230];rgb[c0&c1]=[255,255,255]
                axes=[d for d in range(3) if d!=v['axis']]
                physical_w=rgb.shape[1]*spacing[axes[0]];physical_h=rgb.shape[0]*spacing[axes[1]]
                scale=min((w-12)/physical_w,(h-42)/physical_h)
                size=(max(1,round(physical_w*scale)),max(1,round(physical_h*scale)))
                tile=Image.fromarray(rgb).resize(size,Image.Resampling.NEAREST)
                x=col*w+(w-size[0])//2;y=80+row*h+30+(h-42-size[1])//2
                canvas.paste(tile,(x,y))
                pos=float(affine[v['axis'],v['axis']]*v['index']+affine[v['axis'],3])
                draw.text((col*w+8,80+row*h+6),f'{candidate} {v["title"]} plane={pos:.2f}mm',fill='white')
        path=out/f'{case_id}_{name}.png';canvas.save(path);paths.append(str(path.name))
    return paths


def save_target_review(out,case_id,candidate,ct,labels,spacing,window):
    # Target RAS -> LAS view only, no interpolation or volume mutation.
    ct,labels=ct[::-1],labels[::-1]
    specs=view_specs(labels,spacing);w,h=360,300
    canvas=Image.new('RGB',(3*w,len(specs)*h+55),'#15191f');draw=ImageDraw.Draw(canvas)
    draw.text((10,8),f'{case_id} {candidate}: DIRECT target CT + target label contours; spacing={spacing}',fill='white')
    draw.text((10,28),'LAS views. Display ROI only. No roundtrip interpolation in these panels.',fill='white')
    for row,(name,views) in enumerate(specs.items()):
        for col,v in enumerate(views):
            x,lab=extract(ct,v),extract(labels,v)
            gray=np.clip((x-window[0])/(window[1]-window[0])*255,0,255).astype(np.uint8)
            rgb=np.repeat(gray[:,:,None],3,axis=2);rgb[contour(lab,v['organ'])]=[20,255,90]
            axes=[d for d in range(3) if d!=v['axis']]
            pw=rgb.shape[1]*spacing[axes[0]];ph=rgb.shape[0]*spacing[axes[1]]
            scale=min((w-12)/pw,(h-35)/ph)
            tile=Image.fromarray(rgb).resize((max(1,round(pw*scale)),max(1,round(ph*scale))),Image.Resampling.NEAREST)
            canvas.paste(tile,(col*w+(w-tile.width)//2,55+row*h+30+(h-35-tile.height)//2))
            draw.text((col*w+8,55+row*h+6),f'{name} {v["title"]}',fill='white')
    path=out/f'{case_id}_{candidate}_target.png';canvas.save(path)
    return path.name
