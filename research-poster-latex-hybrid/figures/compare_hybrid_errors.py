"""Compare saved full inverse-PINN runs with mechanistic or neural regulation.

No training or ODE solving. Parameter comparisons use the common unknown set;
function comparisons use the same observed-input grid for both fitted models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / 'research-poster-latex-hybrid/data/hybrid_comparison.json'
REGIMES = ['Normal', 'Early Adenoma', 'Advanced Adenoma', 'Severe APC Loss']
SHORT = ['Normal', 'Early adenoma', 'Advanced adenoma', 'Severe APC loss']
COLORS = ['#64748B', '#1F77B4']
LABELS = ['Mechanistic', 'Hybrid (neural regulation)']
SPECS = {
    'bm_myc': {
        'title': 'β-catenin → MYC',
        'host': [('aM', 'MYC basal production'), ('epsM', 'MYC timescale')],
        'replaced': ['etaBM', 'kappaBM'],
        'regulator': 'β-catenin',
        'hill_exponent': 2,
        'expected_counts': (48, 43, 136),
    },
    'ra_h5': {
        'title': 'RA → HOXA5',
        'host': [('a5', 'HOXA5 basal production'), ('eps5', 'HOXA5 timescale'),
                 ('etaM', 'MYC inhibition strength'), ('kappaM', 'MYC inhibition scale')],
        'replaced': ['etaR', 'kappaR'],
        'regulator': 'Retinoic acid',
        'hill_exponent': 1,
        'expected_counts': (51, 53, 136),
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbol(name):
    for prefix, latex in [('eps', r'\varepsilon'), ('theta', r'\theta'),
                          ('eta', r'\eta'), ('kappa', r'\kappa'),
                          ('lambda', r'\lambda'), ('rho', r'\rho'),
                          ('delta', r'\delta'), ('a', 'a')]:
        if name.startswith(prefix):
            return '$' + latex + '_{' + name[len(prefix):] + '}$'
    return '$' + name + '$'


def activation(key, x, parameters):
    spec = SPECS[key]
    amplitude, scale = (parameters[name] for name in spec['replaced'])
    n = spec['hill_exponent']
    return amplitude * x**n / (scale**n + x**n)


def relative_error(estimate, truth):
    if truth == 0:
        raise ValueError('Relative parameter error requires nonzero truth.')
    return 100 * abs(estimate - truth) / abs(truth)


def collect(data, keys, repo, cached_only):
    parameters, functions, audit = [], [], {}
    for key in keys:
        records = {row['regime']: row for row in data[key]}
        assert set(records) == set(REGIMES)
        counts = [0, 0, 0]
        for regime in REGIMES:
            row = records[regime]
            control, hybrid = row['control'], row['hybrid']
            common = sorted(set(control['recovered']) & set(hybrid['recovered']))
            assert common == row['common_parameters']
            assert set(control['recovered']) - set(hybrid['recovered']) == set(SPECS[key]['replaced'])
            assert not (set(hybrid['recovered']) - set(control['recovered']))
            assert all(control['true'][p] == hybrid['true'][p] for p in common)
            counts[2] += len(common)
            for side, folder in [('control', 'control_source'), ('hybrid', 'source')]:
                if not cached_only:
                    path = repo / row[folder] / (regime.replace(' ', '_') + '_recovered.json')
                    original = json.loads(path.read_text())
                    assert all(original[name] == row[side][name] for name in ['recovered', 'true'])
                    audit[str(path.relative_to(repo))] = digest(path)
            if not cached_only:
                path = repo / row['source'] / (regime.replace(' ', '_') + '_term.json')
                original = json.loads(path.read_text())
                assert original == row['term']
                audit[str(path.relative_to(repo))] = digest(path)
            for p in control['recovered']:
                if p not in common:
                    continue
                truth = control['true'][p]
                a, b = (relative_error(model['recovered'][p], truth) for model in [control, hybrid])
                counts[0] += a < 10
                counts[1] += b < 10
                parameters.append({
                    'relationship': key, 'regime': regime, 'parameter': p, 'truth': truth,
                    'mechanistic_estimate': control['recovered'][p],
                    'hybrid_estimate': hybrid['recovered'][p],
                    'mechanistic_error_percent': a, 'hybrid_error_percent': b,
                    'hybrid_minus_mechanistic_percentage_points': b - a,
                    'in_replaced_term_equation': p in dict(SPECS[key]['host']),
                })
            term = row['term']
            x, truth, learned = (np.asarray(term[p], dtype=float) for p in ['grid', 'truth', 'learned'])
            assert x.ndim == 1 and len(x) > 1 and np.all(np.diff(x) > 0)
            np.testing.assert_allclose(np.diff(x), np.diff(x)[0], rtol=1e-10)
            np.testing.assert_allclose([x[0], x[-1]], [term['r_lo'], term['r_hi']])
            np.testing.assert_allclose(activation(key, x, control['true']), truth, rtol=1e-10)
            fitted = activation(key, x, control['recovered'])
            denominator = np.sqrt(np.mean(truth**2))
            control_error = 100 * np.sqrt(np.mean((fitted - truth)**2)) / denominator
            hybrid_error = 100 * np.sqrt(np.mean((learned - truth)**2)) / denominator
            np.testing.assert_allclose(hybrid_error, 100 * term['nrmse'], rtol=1e-10)
            functions.append({
                'relationship': key, 'regime': regime,
                'mechanistic_nrmse_percent': float(control_error),
                'hybrid_nrmse_percent': float(hybrid_error),
                'input_min': float(x[0]), 'input_max': float(x[-1]), 'grid_points': len(x),
                'hill_exponent': SPECS[key]['hill_exponent'],
            })
        assert tuple(counts) == SPECS[key]['expected_counts'], (key, counts)
    return parameters, functions, audit


def save(fig, out, name):
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(out / f'{name}.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def legend(fig, y=.93):
    handles = [Line2D([], [], color=c, lw=9, solid_capstyle='butt', label=label)
               for c, label in zip(COLORS, LABELS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, y), ncol=2,
               frameon=False, handlelength=1.6, columnspacing=2)


def paired_bars(ax, control, hybrid, title, xlabel, threshold=False):
    y = np.arange(4)
    largest = max(max(control), max(hybrid))
    offset = max(largest, 1) * .015
    for values, shift, color in [(control, -.19, COLORS[0]), (hybrid, .19, COLORS[1])]:
        ax.barh(y + shift, values, height=.31, color=color)
        for position, value in zip(y + shift, values):
            ax.text(value + offset, position, f'{value:.2f}', va='center', ha='left', fontsize=13)
    ax.set_yticks(y, SHORT)
    ax.set_ylim(3.6, -.6)
    ax.set_xlim(0, max(largest, 1) * 1.28)
    ax.set_title(title, fontsize=19, loc='left', pad=14)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.tick_params(axis='y', length=0, pad=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    if threshold:
        ax.axvline(10, color='#AAA', lw=1, ls=':', zorder=0)
    ax.grid(axis='x', color='#E8E8E8', lw=.7)
    ax.set_axisbelow(True)


def host_parameters(parameters, key, out):
    spec = SPECS[key]
    rows = 1 if len(spec['host']) == 2 else 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 6.7 if rows == 1 else 11.5), squeeze=False)
    fig.subplots_adjust(left=.135, right=.975, bottom=.14 if rows == 1 else .085,
                        top=.75 if rows == 1 else .85, wspace=.65, hspace=.48)
    fig.suptitle(spec['title'] + ' · parameter recovery', fontsize=25, y=.985)
    legend(fig, .92 if rows == 1 else .955)
    for ax, (parameter, description) in zip(axes.flat, spec['host']):
        records = [next(r for r in parameters if r['relationship'] == key and
                        r['regime'] == regime and r['parameter'] == parameter) for regime in REGIMES]
        paired_bars(ax, [r['mechanistic_error_percent'] for r in records],
                    [r['hybrid_error_percent'] for r in records],
                    symbol(parameter) + '  ' + description, 'Parameter error (%)', threshold=True)
    save(fig, out, key + '_parameter_errors')


def function_errors(functions, keys, out):
    fig, axes = plt.subplots(1, len(keys), figsize=(8 * len(keys), 6.7), squeeze=False)
    fig.subplots_adjust(left=.135, right=.975, bottom=.14, top=.75, wspace=.65)
    fig.suptitle('Regulatory function recovery', fontsize=25, y=.985)
    legend(fig, .92)
    for ax, key in zip(axes.flat, keys):
        rows = [next(r for r in functions if r['relationship'] == key and r['regime'] == regime)
                for regime in REGIMES]
        paired_bars(ax, [r['mechanistic_nrmse_percent'] for r in rows],
                    [r['hybrid_nrmse_percent'] for r in rows], SPECS[key]['title'],
                    'Function error (NRMSE, %)')
    save(fig, out, 'regulatory_function_errors')


def poster_comparison(data, out):
    """Four poster panels: function and basal-production error for both terms."""
    keys = ['bm_myc', 'ra_h5']
    parameters, functions, _ = collect(data, keys, REPO, cached_only=True)
    basal = {'bm_myc': 'aM', 'ra_h5': 'a5'}
    selected = []
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 6.4), sharex=True, sharey='row')
    fig.subplots_adjust(left=.105, right=.985, bottom=.09, top=.83,
                        wspace=.17, hspace=.58)
    handles = [Line2D([], [], color=color, lw=12, solid_capstyle='butt', label=label)
               for color, label in zip(COLORS, LABELS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, 1.02),
               ncol=2, frameon=False, fontsize=22, handlelength=1.8, columnspacing=2)
    x = np.arange(len(REGIMES))
    for column, key in enumerate(keys):
        function_rows = [next(r for r in functions if r['relationship'] == key and r['regime'] == regime)
                         for regime in REGIMES]
        parameter_rows = [next(r for r in parameters if r['relationship'] == key and
                              r['regime'] == regime and r['parameter'] == basal[key])
                          for regime in REGIMES]
        selected.extend(parameter_rows)
        panels = [
            (function_rows, 'nrmse', SPECS[key]['title']),
            (parameter_rows, 'error', ('MYC' if key == 'bm_myc' else 'HOXA5') +
             ' basal production ' + symbol(basal[key])),
        ]
        for row, (records, metric, title) in enumerate(panels):
            ax = axes[row, column]
            for side, shift, color in [('mechanistic', -.22, COLORS[0]), ('hybrid', .22, COLORS[1])]:
                values = [record[f'{side}_{metric}_percent'] for record in records]
                ax.bar(x + shift, values, width=.38, color=color, zorder=3)
                for position, value in zip(x + shift, values):
                    ax.annotate(f'{value:.1f}', (position, value), xytext=(0, 5),
                                textcoords='offset points', ha='center', va='bottom', fontsize=18)
            ax.set_title(title, fontsize=24, pad=8)
            ax.set_xticks(x, ['Normal', 'Early', 'Advanced', 'Severe'])
            ax.set_xlim(-.65, 3.65)
            ax.set_ylim(0, 32 if row == 0 else 150)
            ax.set_yticks([0, 10, 20, 30] if row == 0 else [0, 50, 100, 150])
            ax.tick_params(axis='both', labelsize=20, length=3)
            ax.tick_params(axis='x', labelbottom=True, pad=7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#E8E8E8', lw=.7, zorder=0)
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel('NRMSE (%)', fontsize=22, labelpad=10)
    axes[1, 0].set_ylabel('Error (%)', fontsize=22, labelpad=10)
    save(fig, out, 'hybrid-errors')
    (out / 'hybrid_error_audit.json').write_text(json.dumps({
        'source': 'data/hybrid_comparison.json',
        'fit_type': 'Saved full multi-condition inverse PINNs',
        'function_scoring': 'Same observed-input grid for mechanistic and hybrid functions',
        'parameter_selection': 'Basal production in each replaced-term equation',
        'function_comparisons': functions,
        'basal_parameter_comparisons': selected,
        'display_decimal_places': 1,
    }, indent=2) + '\n')


def all_parameters(parameters, key, out):
    first = [r for r in parameters if r['relationship'] == key and r['regime'] == REGIMES[0]]
    names = [r['parameter'] for r in first]
    values = [r[f'{model}_error_percent'] for r in parameters if r['relationship'] == key
              for model in ['mechanistic', 'hybrid']]
    if min(values) <= 0:
        raise ValueError('A zero error requires a different display; no log-scale flooring is applied.')
    lower, upper = 10**np.floor(np.log10(min(values))), 10**np.ceil(np.log10(max(values)))
    fig, axes = plt.subplots(1, 4, figsize=(20, 13), sharey=True)
    fig.subplots_adjust(left=.10, right=.975, bottom=.065, top=.865, wspace=.16)
    fig.suptitle(SPECS[key]['title'] + f' · all {len(names)} shared parameters', fontsize=25, y=.985)
    handles = [Line2D([], [], color=c, marker=marker, ls='', markersize=8, label=label)
               for c, marker, label in zip(COLORS, ['o', 'D'], LABELS)]
    handles.append(Line2D([], [], color='#999', ls=':', label='10% error'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .952), ncol=3, frameon=False)
    for ax, regime, title in zip(axes, REGIMES, SHORT):
        rows = {r['parameter']: r for r in parameters if r['relationship'] == key and r['regime'] == regime}
        a = np.array([rows[p]['mechanistic_error_percent'] for p in names])
        b = np.array([rows[p]['hybrid_error_percent'] for p in names])
        y = np.arange(len(names))
        ax.hlines(y, np.minimum(a, b), np.maximum(a, b), color='#CBD5E1', lw=1.1)
        ax.scatter(a, y, color=COLORS[0], s=32, marker='o', zorder=3)
        ax.scatter(b, y, color=COLORS[1], s=28, marker='D', zorder=4)
        ax.axvline(10, color='#999', ls=':', lw=1)
        ax.set_xscale('log')
        ax.set_xlim(lower, upper)
        ax.set_ylim(len(names) - .3, -.7)
        ax.set_yticks(y, [symbol(p) for p in names])
        ax.tick_params(axis='y', length=0, pad=8, labelsize=15)
        ax.set_title(title, fontsize=18, pad=13)
        ax.set_xlabel('Parameter error (%) · log scale', fontsize=13, labelpad=11)
        ax.grid(axis='x', which='major', color='#ECECEC', lw=.7)
        ax.minorticks_off()
        ax.spines[['top', 'right', 'left']].set_visible(False)
    save(fig, out, key + '_all_parameters')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def page(out, keys):
    def card(name, title, caption):
        links = ' · '.join(f'<a href="{name}.{ext}">{ext.upper()}</a>' for ext in ['pdf', 'svg', 'png'])
        return (f'<article><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p>'
                f'<p class="links">{links}</p><a href="{name}.pdf">'
                f'<img src="{name}.png" alt="{html.escape(title)}"></a></article>')
    cards = card('regulatory_function_errors', 'Error in the regulatory relationship',
                 'Both fitted functions are scored against truth over the same observed input range.')
    for key in keys:
        cards += card(key + '_parameter_errors', SPECS[key]['title'] + ': remaining equation parameters',
                      'Absolute relative parameter error. Lower is better; dotted line: 10%.')
    cards += '<details><summary>All shared parameters</summary>'
    for key in keys:
        cards += card(key + '_all_parameters', SPECS[key]['title'] + ': all 34 shared parameters',
                      'Each row compares the same parameter. Logarithmic error axis.')
    cards += '</details>'
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mechanistic vs hybrid models</title>
<style>body{font:18px/1.5 system-ui,sans-serif;color:#152a3a;background:#faf8f6;max-width:1280px;margin:36px auto;padding:0 24px}h1,h2,a{color:#78253b}h1{font-size:32px}h2{font-size:24px}article{background:white;padding:24px;margin:30px 0;border:1px solid #e3ded8}img{display:block;width:100%;height:auto}.links{font-size:16px}summary{cursor:pointer;font-size:22px;font-weight:600}footer{margin:32px 0;font-size:16px}</style>
<nav><a href="../">Poster previews</a></nav><h1>Mechanistic vs hybrid models</h1>
<p>Saved full inverse-PINN runs. Both models use PINNs; the hybrid also learns the selected biological regulatory relationship with a neural network.</p>
<p><a href="hybrid_error_graphs.zip">Download all graphs and data</a> · <a href="parameter_errors.csv">Parameter errors CSV</a> · <a href="function_errors.csv">Function errors CSV</a></p>
''' + cards + '''<footer>Simulated data with known truth. Parameter comparisons use the 34 unknowns shared by each model pair. The two replaced Hill-law parameters are excluded. Function NRMSE = 100 × RMSE / RMS(truth). Figures show saved selected fits, without uncertainty bars. Errors describe parameter and function recovery, not trajectory prediction.<br><a href="README.md">Methods and sources</a> · <a href="summary.json">Numerical summary</a> · <a href="manifest.json">Provenance</a></footer></html>'''
    (out / 'index.html').write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--terms', nargs='+', choices=list(SPECS), default=list(SPECS))
    parser.add_argument('--repo-root', type=Path, default=REPO)
    parser.add_argument('--cached-only', action='store_true', help='Use the bundled input without checking original run files.')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.input.read_text())
    parameters, functions, audit = collect(data, args.terms, args.repo_root, args.cached_only)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15, 'axes.labelsize': 15,
                         'xtick.labelsize': 13, 'ytick.labelsize': 14, 'text.color': '#152A3A',
                         'axes.labelcolor': '#152A3A', 'axes.edgecolor': '#CBD5E1',
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'mathtext.fontset': 'stixsans'})
    function_errors(functions, args.terms, out)
    for key in args.terms:
        host_parameters(parameters, key, out)
        all_parameters(parameters, key, out)
        print('Generated', key, flush=True)
    write_csv(out / 'parameter_errors.csv', parameters)
    write_csv(out / 'function_errors.csv', functions)
    summary = {}
    for key in args.terms:
        rows = [r for r in parameters if r['relationship'] == key]
        summary[key] = {
            'common_parameter_regime_pairs': len(rows),
            'mechanistic_under_10_percent': sum(r['mechanistic_error_percent'] < 10 for r in rows),
            'hybrid_under_10_percent': sum(r['hybrid_error_percent'] < 10 for r in rows),
            'function_errors': [r for r in functions if r['relationship'] == key],
        }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    shutil.copy2(args.input, out / 'input_hybrid_comparison.json')
    shutil.copy2(Path(__file__), out / 'compare_hybrid_errors.py')
    (out / 'README.md').write_text(
        '# Mechanistic versus hybrid errors\n\n'
        'These graphs use saved full multi-condition inverse-PINN fits from July 2026. '
        'No intervention screens, training, ODE solves or posterior samples are used.\n\n'
        'Mechanistic control: `PINN-hybrid-ude/runs/20260726_195830_control/`.\n'
        'MYC hybrid: `PINN-hybrid-ude/runs/20260727_012448_bm_myc/`.\n'
        'HOXA5 hybrid: `PINN-hybrid-ude/runs/20260726_214316_ra_h5/`.\n\n'
        'Parameter error = 100 * abs(estimate - truth) / abs(truth). Compare only '
        'the intersection of recovered parameter names; each pair has 34. '
        'Both estimates use the same true value. The focused plots show every '
        'remaining parameter in the equation containing the neural term. '
        'The detailed plots and CSV include all shared parameters.\n\n'
        'Function error = 100 * RMS(fitted function - true function) / RMS(true function). '
        'The mechanistic function is evaluated using the control run\'s estimated Hill parameters. '
        'The hybrid uses its saved learned function. Both use exactly the hybrid\'s saved '
        'uniform grid on observed regulator inputs, with no extrapolation. The MYC law '
        'is etaBM*b^2/(kappaBM^2+b^2); the HOXA5 law is etaR*r/(kappaR+r). '
        'These match `src/wnt_pinn/model/hybrid_rhs.py` and the saved true-function grids.\n\n'
        'The selected fits are not independent replicates; no uncertainty bars or '
        'statistical-significance claims are made. Function errors are on observed input '
        'support, not held-out predictions. Parameter, function and trajectory errors '
        'are distinct; trajectory error is not evaluated here.\n\n'
        'To regenerate from this download (Python, NumPy and Matplotlib):\n\n'
        '```bash\npython compare_hybrid_errors.py --input input_hybrid_comparison.json '
        '--out new_output --cached-only\n```\n')
    page(out, args.terms)
    manifest = {
        'purpose': 'Post-processing of saved full hybrid/control fits',
        'input_sha256': digest(args.input), 'script_sha256': digest(Path(__file__)),
        'original_run_files_checked': len(audit), 'original_run_sha256': audit,
        'relationships': args.terms, 'new_training': False, 'new_ode_solves': False,
        'parameter_rows': len(parameters), 'function_comparisons': len(functions),
        'output_sha256': {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(out / 'hybrid_error_graphs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.name)
    print(json.dumps(summary, indent=2), flush=True)
    print('Finished:', out, flush=True)


if __name__ == '__main__':
    main()
