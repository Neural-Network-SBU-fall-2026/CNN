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
        'Validation results at each run\'s selected checkpoint. Time includes training and validation.',
        'crrrrrr', ['Run', 'Parameters', 'Epoch', r'Val. acc. (\%)', 'Val. loss', 'Val. macro F1', 'Time (s)'], rows))
    base, deeper, gelu, separable = runs
    base_f1 = base['validation']['macro_f1']
    reduction = 1 - separable['parameters'] / base['parameters']
    write('comparison_discussion.tex',
          f"Run {winner['run_id']} led on validation macro F1 at {winner['validation']['macro_f1']:.4f}. "
          f"The fourth block took the score from {base_f1:.4f} to {deeper['validation']['macro_f1']:.4f} "
          f"with {deeper['parameters']:,} parameters. The switch to GELU kept the baseline's parameter "
          f"count and scored {gelu['validation']['macro_f1']:.4f}.\n\n"
          f"The separable model used {100 * reduction:.1f}\\% fewer parameters. Its validation macro F1 "
          f"was {separable['validation']['macro_f1']:.4f}, compared with {base_f1:.4f} for the baseline. "
          f"The separable run took {separable['seconds']:.1f} seconds and the baseline took "
          f"{base['seconds']:.1f} seconds.")

    first_epoch, last_epoch = base['history'][0], base['history'][-1]
    final_loss_minima = sum(
        run['history'][-1]['val_loss'] == min(epoch['val_loss'] for epoch in run['history'])
        for run in runs)
    if final_loss_minima == len(runs):
        loss_summary = ("All four runs had their lowest validation loss at the final epoch. "
                        "The curves give little evidence of sustained overfitting within ten epochs.")
    else:
        loss_summary = (f"Validation loss was lowest at the final epoch in {final_loss_minima} of the four runs. "
                        "A sustained rise in validation loss alongside falling training loss would suggest overfitting.")
    write('curve_discussion.tex',
          f"Baseline training loss went from {first_epoch['train_loss']:.4f} to {last_epoch['train_loss']:.4f}. "
          f"Validation loss went from {first_epoch['val_loss']:.4f} to {last_epoch['val_loss']:.4f}. "
          f"The baseline checkpoint comes from epoch {base['best_epoch']}, where validation macro F1 "
          "was highest. Loss and macro F1 track different aspects of the predictions.\n\n"
          + loss_summary)
    config=final['config']; rows=[]; size=32; channels=3
    for i,b in enumerate(config['conv_blocks'],1):
        out=b['out_channels']; k=b['kernel_size']; stride=b['stride']; pad=b['padding']
        size=(size+2*pad-k)//stride+1
        if b.get('pool'): size=size//b['pool']
        kind='Standard' if b['conv_type']=='standard' else 'Depthwise + pointwise'
        rows.append([str(i), f'{channels} $\\to$ {out}', kind, f'{k}/{stride}/{pad}',
                     f'({out},{size},{size})'])
        channels=out
    activation = {'relu': 'ReLU', 'gelu': 'GELU'}[config['activation']]
    write('architecture_table.tex', table(
        'Convolution blocks in the selected model. Output shapes omit the batch dimension. '
        f'Each block uses batch normalization, {activation} and max pooling by two.',
        'cllll', ['Block', 'Channels', 'Type', 'Kernel/stride/padding', 'Output $(C,H,W)$'], rows))
    features=read('selected_feature_stats.json')
    first=[r for r in features if r['class']=='airplane']
    shape=' $\\to$ '.join('$('+','.join(map(str,r['shape'][1:]))+')$' for r in first)
    constant=sum(r['near_constant_channels'] for r in features)
    write('feature_discussion.tex',
          "In the baseline's first block, channels 0, 2, 3 and 4 pick out a narrow vertical band "
          "near the middle of the airplane image. Channel 1 also responds along the borders. "
          "For the cat, strong responses appear farther toward the upper left and the bottom. "
          "By the last baseline block, the airplane's channel 0 has its strongest cells near "
          "the middle and bottom of a coarse grid. Fine boundaries are harder to trace at that resolution.\n\n"
          f"The selected model's block outputs have shapes {shape}. "
          f"There are {constant} near-constant responses, counted over both images and all blocks.")
    summary=[]
    for key,label in [('accuracy','Accuracy'),('macro_precision','Macro precision'),('macro_recall','Macro recall'),
                      ('macro_f1','Macro F1'),('weighted_precision','Weighted precision'),
                      ('weighted_recall','Weighted recall'),('weighted_f1','Weighted F1')]:
        summary.append([label,f"{final['test'][key]:.4f}"])
    tables=table('Aggregate metrics for the 10,000 test examples.','lr',['Metric','Value'],summary)
    classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    rows=[]
    for name in classes:
        r=final['classification_report'][name]
        rows.append([name,f"{r['precision']:.4f}",f"{r['recall']:.4f}",f"{r['f1-score']:.4f}",str(int(r['support']))])
    tables+=table('Test precision, recall and F1 by class.','lrrrr',['Class','Precision','Recall','F1','Support'],rows)
    write('test_tables.tex',tables)
    ranked=sorted(classes,key=lambda n:final['classification_report'][n]['f1-score'])
    off=matrix.copy();np.fill_diagonal(off,0)
    a,b=np.unravel_index(off.argmax(),off.shape)
    write('test_discussion.tex',
          f"The highest test F1 belongs to {ranked[-1]} ({final['classification_report'][ranked[-1]]['f1-score']:.4f}). "
          f"The lowest belongs to {ranked[0]} ({final['classification_report'][ranked[0]]['f1-score']:.4f}). "
          f"The largest off-diagonal cell is {classes[a]} classified as {classes[b]}, "
          f"accounting for {off[a, b]} of the {matrix[a].sum():,} true {classes[a]} examples.")
    write('occlusion_discussion.tex',
          f"Covering the centre moved validation accuracy from {100 * robustness['clean']['accuracy']:.2f}\\% "
          f"to {100 * robustness['occluded']['accuracy']:.2f}\\%, a change of "
          f"${-100 * robustness['accuracy_drop']:+.2f}$ percentage points. Macro F1 moved from "
          f"{robustness['clean']['macro_f1']:.4f} to {robustness['occluded']['macro_f1']:.4f}.")
    changed = max(runs[1:], key=lambda run: abs(run['validation']['macro_f1'] - base_f1))
    delta = changed['validation']['macro_f1'] - base_f1
    write('reflection.tex',
          f"Run {changed['run_id']} moved validation macro F1 the most (${delta:+.4f}$ relative to the baseline). "
          "Run 02 adds a block, increases the final channel count to 96 and halves the spatial grid once more. "
          "The comparison measures the effect of that whole block.")
    print('Generated LaTeX measurements, tables and discussion from saved results')
