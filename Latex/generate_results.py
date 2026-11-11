"""Regenerate LaTeX tables/text from the notebook's measured result files.

Run from the repository root: python Latex/generate_results.py
This reads saved measurements only; it never trains a model or re-evaluates test data.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SOURCE, OUT = ROOT / 'report/results', ROOT / 'Latex'

def read(name):
    return json.loads((SOURCE / name).read_text())

def tex(text):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                    '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(text))

def write(name, text):
    (OUT / name).write_text(text.strip() + '\n')

def table(caption, columns, header, rows):
    return ('\\begin{table}[htbp]\\centering\\small\n\\caption{' + caption + '}\n'
            '\\begin{tabular}{@{}' + columns + '@{}}\\toprule\n' + ' & '.join(header) + r'\\\midrule' + '\n'
            + '\n'.join(' & '.join(row) + r'\\' for row in rows)
            + '\n\\bottomrule\\end{tabular}\\end{table}\n')

if __name__ == '__main__':
    runs = read('experiments.json')
    final = read('final_evaluation.json')
    prep = read('preprocessing.json')
    robustness = read('occlusion_experiment.json')
    assert [r['run_id'] for r in runs] == ['01', '02', '03', '04']
    assert all(len(r['history']) == 10 and r['epochs'] == 10 for r in runs)
    winner = max(runs, key=lambda r: (r['validation']['macro_f1'], -r['validation']['loss']))
    assert winner['run_id'] == final['run_id']
    assert winner['best_epoch'] == final['checkpoint_epoch']
    assert final['test_inference_passes'] == 1
    pred = pd.read_csv(SOURCE / 'test_predictions.csv')
    matrix = np.loadtxt(SOURCE / 'test_confusion_matrix.csv', delimiter=',', dtype=int)
    assert len(pred) == 10000 and matrix.sum() == 10000
    assert np.all(matrix.sum(axis=1) == 1000)
    np.testing.assert_array_equal(matrix, confusion_matrix(pred.true, pred.predicted, labels=list(range(10))))
    assert abs(accuracy_score(pred.true, pred.predicted) - final['test']['accuracy']) < 1e-12
    assert abs(matrix.trace() / matrix.sum() - final['test']['accuracy']) < 1e-12
    for average in ('macro', 'weighted'):
        p, r, f, _ = precision_recall_fscore_support(pred.true, pred.predicted, average=average, zero_division=0)
        for key, value in zip(('precision', 'recall', 'f1'), (p, r, f)):
            assert abs(value - final['test'][f'{average}_{key}']) < 1e-12
    for run in runs:
        chosen = run['history'][run['best_epoch'] - 1]
        assert abs(chosen['val_macro_f1'] - run['validation']['macro_f1']) < 1e-12
        expected = max(run['history'], key=lambda row: (row['val_macro_f1'], -row['val_loss']))
        assert expected['epoch'] == run['best_epoch']
    print('Verified: 40 completed training epochs, validation-based winner, 10,000 saved test predictions and all aggregate metrics')
    numbers = {
        'WinningRun': final['run_id'], 'WinningEpoch': str(final['checkpoint_epoch']),
        'WinningName': tex(final['name']), 'WinningParameters': f"{final['parameters']:,}",
        'BaselineParameters': f"{runs[0]['parameters']:,}",
        'TestAccuracy': f"{100 * final['test']['accuracy']:.2f}",
        'TestMacroFone': f"{final['test']['macro_f1']:.4f}",
        'ValidationMacroFone': f"{final['validation']['macro_f1']:.4f}",
        'TrainingMean': r'$(' + ', '.join(f'{x:.6f}' for x in prep['mean']) + ')$',
        'TrainingStd': r'$(' + ', '.join(f'{x:.6f}' for x in prep['std']) + ')$'}
    write('results.tex', '\n'.join(r'\newcommand{' + chr(92) + k + '}{' + v + '}' for k, v in numbers.items()))
    rows = []
    for run in runs:
        rows.append([run['run_id'], f"{run['parameters']:,}", str(run['best_epoch']),
                     f"{100 * run['validation']['accuracy']:.2f}", f"{run['validation']['loss']:.4f}",
                     f"{run['validation']['macro_f1']:.4f}", f"{run['seconds']:.1f}"])
    write('comparison_table.tex', table(
        'Measured comparison. Accuracy and loss are from each run\'s macro-F1-selected checkpoint, not independent best values. Wall time includes training and validation.',
        'crrrrrr', ['Run', 'Parameters', 'Epoch', r'Val. acc. (\%)', 'Val. loss', 'Val. macro F1', 'Time (s)'], rows))
    discussion = [f"Run {winner['run_id']} has the highest validation macro F1 "
                  f"({winner['validation']['macro_f1']:.4f}) under the fixed protocol."]
    base = runs[0]
    for run in runs[1:]:
        delta = run['validation']['macro_f1'] - base['validation']['macro_f1']
        discussion.append(f"Run {run['run_id']} changes macro F1 by ${delta:+.4f}$ versus the baseline "
                          f"and uses {run['parameters']:,} parameters.")
    ds = runs[3]
    reduction = 1 - ds['parameters'] / base['parameters']
    failures = [r for r in runs[1:] if r['validation']['macro_f1'] <= base['validation']['macro_f1']]
    if failures:
        r = failures[0]
        discussion.append(f"In particular, run {r['run_id']} did not improve baseline validation macro F1. "
                          "This is a measured unsuccessful change under this budget, not proof that the architecture can never work.")
    else:
        discussion.append("All alternatives improved on baseline in this seed; there is no observed failed architecture improvement to invent.")
    discussion.append(f"The separable model reduces trainable parameters by {100*reduction:.1f}\\%. "
                      "Its accuracy and measured time must be considered together with that saving. "
                      "No multi-seed statistical significance is established by these comparisons.")
    write('comparison_discussion.tex', '\n\n'.join(discussion))
    text=[]
    for run in runs:
        h=run['history']; last=h[-1]; min_loss=min(row['val_loss'] for row in h)
        text.append(f"\\textbf{{Run {run['run_id']}.}} At epoch 10, training/validation loss is "
                    f"{last['train_loss']:.4f}/{last['val_loss']:.4f} and accuracy is "
                    f"{100*last['train_accuracy']:.2f}\\%/{100*last['val_accuracy']:.2f}\\%. "
                    f"The best observed validation loss is {min_loss:.4f}. "
                    + ("A higher final validation loss indicates some late deterioration; checkpointing avoids selecting that last epoch automatically."
                       if last['val_loss'] > min_loss + 0.02 else
                       "There is no clear late validation-loss increase exceeding 0.02; this short run does not establish full convergence."))
    write('curve_discussion.tex', '\n\n'.join(text))
    config=final['config']; rows=[]; size=32; channels=3
    for i,b in enumerate(config['conv_blocks'],1):
        out=b['out_channels']; k=b['kernel_size']; stride=b['stride']; pad=b['padding']
        size=(size+2*pad-k)//stride+1
        if b.get('pool'): size=size//b['pool']
        kind='Standard' if b['conv_type']=='standard' else 'Depthwise + pointwise'
        rows.append([str(i), f'{channels} $\\to$ {out}', kind, f'{k}/{stride}/{pad}',
                     f'({out},{size},{size})'])
        channels=out
    write('architecture_table.tex', table('Selected convolution blocks. Output shapes omit the batch dimension. Each block has batch normalization and max pooling by two; activation is ' + tex(config['activation']) + '.',
          'cllll', ['Block', 'Channels', 'Type', 'Kernel/stride/padding', 'Output $(C,H,W)$'], rows))
    features=read('selected_feature_stats.json')
    bfeatures=read('baseline_feature_stats.json')
    first=[r for r in features if r['class']=='airplane']
    shape=' $\\to$ '.join('$('+','.join(map(str,r['shape'][1:]))+')$' for r in first)
    feature_text=f"The selected model's block-output shapes are {shape}. "
    constant=sum(r['near_constant_channels'] for r in features)
    feature_text+=f"Across the two selected-model inputs and all blocks, {constant} near-constant channel responses are recorded (a response count, not a count of distinct filters). "
    feature_text+=("In the baseline airplane image, first-block channels 0, 2, 3 and 4 emphasize a narrow vertical central structure, "
                   "consistent with a local contrast or edge response. Channels 1, 5 and 7 distribute responses more broadly, "
                   "including image borders; channel 6 contains more scattered local responses. "
                   "For the cat image the strong regions move toward upper-left and lower image structures, "
                   "which shows that these are input-dependent filters rather than fixed class masks. "
                   "Baseline late channels 0, 2 and 6 emphasize middle/lower coarse cells for the airplane, "
                   "while channels 1, 4 and 7 show distinct upper/lower spatial responses. "
                   "These are visual descriptions, not evidence that any particular channel detects an airplane, a cat or a named body part.")
    write('feature_discussion.tex',feature_text)
    summary=[]
    for key,label in [('accuracy','Accuracy'),('macro_precision','Macro precision'),('macro_recall','Macro recall'),
                      ('macro_f1','Macro F1'),('weighted_precision','Weighted precision'),
                      ('weighted_recall','Weighted recall'),('weighted_f1','Weighted F1')]:
        summary.append([label,f"{final['test'][key]:.4f}"])
    tables=table('Final aggregate test metrics, all on the same 10,000 predictions.','lr',['Metric','Value'],summary)
    classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    rows=[]
    for name in classes:
        r=final['classification_report'][name]
        rows.append([name,f"{r['precision']:.4f}",f"{r['recall']:.4f}",f"{r['f1-score']:.4f}",str(int(r['support']))])
    tables+=table('Per-class final test precision, recall and F1.','lrrrr',['Class','Precision','Recall','F1','Support'],rows)
    write('test_tables.tex',tables)
    ranked=sorted(classes,key=lambda n:final['classification_report'][n]['f1-score'])
    off=matrix.copy();np.fill_diagonal(off,0)
    a,b=np.unravel_index(off.argmax(),off.shape)
    write('test_discussion.tex',f"By test F1, the easiest class is {ranked[-1]} "
          f"({final['classification_report'][ranked[-1]]['f1-score']:.4f}) and the hardest is {ranked[0]} "
          f"({final['classification_report'][ranked[0]]['f1-score']:.4f}). "
          f"The largest directed confusion is {classes[a]} predicted as {classes[b]} ({off.max()} examples). "
          "The confusion matrix supplies the specific errors hidden by the overall accuracy. "
          "Class-wise results are descriptive outcomes of the frozen model, not a further model-selection input.")
    write('occlusion_discussion.tex',f"Clean validation accuracy changes from {100*robustness['clean']['accuracy']:.2f}\\% "
          f"to {100*robustness['occluded']['accuracy']:.2f}\\% under occlusion, a decrease of "
          f"{100*robustness['accuracy_drop']:.2f} percentage points. Macro F1 changes from "
          f"{robustness['clean']['macro_f1']:.4f} to {robustness['occluded']['macro_f1']:.4f} "
          f"(difference ${-robustness['macro_f1_drop']:+.4f}$).")
    changed=max(runs[1:],key=lambda r:abs(r['validation']['macro_f1']-base['validation']['macro_f1']))
    write('reflection.tex',f"Among these interventions, run {changed['run_id']} has the largest absolute validation macro-F1 "
          f"change versus baseline (${changed['validation']['macro_f1']-base['validation']['macro_f1']:+.4f}$). "
          f"The deeper model (run 02) scores {runs[1]['validation']['macro_f1']:.4f}, compared with "
          f"{base['validation']['macro_f1']:.4f} for baseline. This single comparison does not imply that "
          "increasing depth or width always improves validation performance; depth and the extra output width "
          "were changed together, and width alone was not isolated. The histories show any late "
          "validation deterioration directly; the final test evaluation does not diagnose which additional architecture should be tried.")
    print('Generated LaTeX measurements, tables and discussion from saved results')
