"""Three overlapping-region observations for the existing resumable engine."""
import torch
from organ_relation.losses import REGION_NAMES, region_targets

PROTOCOL = 'brats_regions_threshold_both_empty_null_v1'


def region_dice(logits, label, threshold):
    if not 0 < threshold < 1:
        raise ValueError('threshold must be inside (0,1)')
    target = region_targets(label)[0]
    if logits.shape != (1,3,*label.shape[1:]) or not torch.isfinite(logits).all():
        raise ValueError('single-case finite three-region logits required')
    predicted = logits.sigmoid()[0] >= threshold
    regions = []
    for i,name in enumerate(REGION_NAMES):
        p,t = predicted[i],target[i]
        count,gt,tp = int(p.sum()),int(t.sum()),int((p&t).sum())
        regions.append(dict(label=i+1,name=name,predicted_voxels=count,gt_voxels=gt,true_positive=tp,
                            false_positive_voxels=count-tp,dice=2*tp/(count+gt) if count+gt else None))
    valid = [r['dice'] for r in regions if r['dice'] is not None]
    return dict(protocol=PROTOCOL,threshold=threshold,organs=regions,mean_dice=sum(valid)/len(valid) if valid else None,
                hierarchy_violations=dict(ET_outside_TC=int((predicted[2]&~predicted[1]).sum()),
                                          TC_outside_WT=int((predicted[1]&~predicted[0]).sum())))


def summarize_regions(cases):
    means = [c['mean_dice'] for c in cases if c['mean_dice'] is not None]
    regions = []
    for i,name in enumerate(REGION_NAMES):
        values = [c['organs'][i]['dice'] for c in cases if c['organs'][i]['dice'] is not None]
        regions.append(dict(label=i+1,name=name,valid_cases=len(values),mean_dice=sum(values)/len(values) if values else None))
    return dict(protocol=PROTOCOL,cases=len(cases),valid_cases=len(means),mean_case_dice=sum(means)/len(means) if means else None,
                organs=regions,hd95=None,hd95_status='not_official_evaluation_protocol')


def region_diagnostics(output, label, criterion, threshold):
    result = criterion(output.coarse_logits,output.final_logits,label,getattr(output,'containment_logits',None))
    def stats(branch):
        return dict(bce=branch.ce.item(),dice_loss=branch.dice_loss.item(),loss=branch.segmentation.item(),
                    soft_dice=branch.dice_per_class[0].tolist())
    coarse = stats(result.coarse)
    coarse['hard'] = region_dice(criterion.align_coarse(output.coarse_logits,label),label,threshold)
    diagnostic = dict(total_loss=result.total.item(),final=stats(result.final),coarse=coarse)
    if getattr(result, 'hierarchy', None) is not None:  # Only when lambda_hier is configured.
        diagnostic['hierarchy'] = result.hierarchy.item()
    if getattr(result, 'containment', None) is not None:  # 仅在配置了 lambda_contain 时；另记预测的包含 logit
        diagnostic['containment'] = result.containment.item()
        diagnostic['containment_logits'] = output.containment_logits[0].tolist()
    return diagnostic


def region_scalars(row):
    if row['phase'] == 'train_epoch':
        summary = row['epoch_metrics']
        values = {'TrainEpoch/Total_Loss': summary['total_loss']}
        for branch in ('final', 'coarse'):
            values[f'TrainEpoch/{branch}/Loss'] = summary[branch]['loss']
            for name, value in zip(REGION_NAMES, summary[branch]['soft_per_region']):
                values[f'TrainEpoch/{branch}/SoftDice/{name}'] = value
        if 'hierarchy' in summary:
            values['TrainEpoch/final/Hierarchy'] = summary['hierarchy']
        return values
    if row['phase'] == 'train':
        values = {'Train/Total_Loss':row['total_loss'],'Optimizer/LR':row['lr'],'System/Step_Time':row['step_seconds']}
        for branch in ('final','coarse'):
            for key in ('bce','dice_loss','segmentation'):
                values[f'Train/{branch}/{key}'] = row[branch][key]
        for name,v in zip(REGION_NAMES,row['soft_dice_per_organ']):
            values['Train/SoftDice/'+name] = v
        if 'hierarchy' in row:
            values['Train/final/hierarchy'] = row['hierarchy']
        return values
    if row['phase'] in ('train_monitor','internal_dev'):
        prefix = 'Monitor' if row['phase']=='train_monitor' else 'Val'
        values = {prefix+'/HardDice_Mean':row['metrics']['mean_case_dice']}
        for region in row['metrics']['organs']:
            values[prefix+'/HardDice/'+region['name']] = region['mean_dice']
        cases = list(row['region_diagnostic_cases'].values())
        for branch in ('final','coarse'):
            for key in ('bce','dice_loss','loss'):
                values[f'{prefix}/{branch}/{key}'] = sum(c[branch][key] for c in cases)/len(cases)
            for i,name in enumerate(REGION_NAMES):
                values[f'{prefix}/{branch}/SoftDice/{name}'] = sum(c[branch]['soft_dice'][i] for c in cases)/len(cases)
        if all('hierarchy' in c for c in cases):
            values[prefix+'/final/hierarchy'] = sum(c['hierarchy'] for c in cases)/len(cases)
        if row.get('early_stopping'):
            values[prefix+'/EarlyStop/NonImprovingValidations'] = row['early_stopping']['no_improvement_count']
            values[prefix+'/EarlyStop/BestDice'] = row['early_stopping']['best_metric']
        return {k:v for k,v in values.items() if v is not None}
    return {}


def region_training_branches(loss):
    """Reuse scalar loss outputs, without an additional full-volume monitor pass."""
    result = dict(total_loss=loss.total.detach().item(), **{name: dict(
        loss=getattr(loss, name).segmentation.detach().item(),
        soft_per_region=getattr(loss, name).dice_per_class.detach().cpu()[0].tolist())
        for name in ('final', 'coarse')})
    if getattr(loss, 'hierarchy', None) is not None:
        result['hierarchy'] = loss.hierarchy.detach().item()
    if getattr(loss, 'containment', None) is not None:
        result['containment'] = loss.containment.detach().item()
    return result


def region_validation_branches(record):
    diagnostic = record['region_diagnostic']
    result = dict(total_loss=diagnostic['total_loss'], **{name: dict(
        loss=diagnostic[name]['loss'], soft_per_region=diagnostic[name]['soft_dice'],
        hard=record if name == 'final' else diagnostic[name]['hard'])
        for name in ('final', 'coarse')})
    if 'hierarchy' in diagnostic:
        result['hierarchy'] = diagnostic['hierarchy']
    if 'containment' in diagnostic:
        result['containment'] = diagnostic['containment']
    return result


def summarize_region_branches(cases):
    result = dict(total_loss=sum(c['total_loss'] for c in cases)/len(cases))
    if all('hierarchy' in c for c in cases):
        result['hierarchy'] = sum(c['hierarchy'] for c in cases)/len(cases)
    if all('containment' in c for c in cases):
        result['containment'] = sum(c['containment'] for c in cases)/len(cases)
    for name in ('final', 'coarse'):
        values = [c[name] for c in cases]
        soft = [sum(v['soft_per_region'][i] for v in values)/len(values) for i in range(3)]
        result[name] = dict(loss=sum(v['loss'] for v in values)/len(values),
                            soft_per_region=soft, soft_mean=sum(soft)/3)
        if 'hard' in values[0]:
            result[name]['hard'] = summarize_regions([v['hard'] for v in values])
    return result
