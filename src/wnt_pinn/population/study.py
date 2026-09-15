"""Run the reproducible SciPy study and write tables, trajectories and figures.

Workers are ordinary CPU processes running SciPy, not learning algorithms.
Every perturbation is preconditioned separately and gets its own matched control.
The 160-sample ensemble uses a stated new seed because no numerical seed could
be found in the supplied PDF. Spearman coefficients are labeled as such, not PRCC.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import csv
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import uuid

import numpy as np
from scipy.stats import qmc, spearmanr

from .model import CONTROL, NOMINAL, Parameters, Protocol, REGIMES, STATE_NAMES, observables
from .simulation import Solver, audit, compare_at, maintenance_events, paired, precondition

UNCERTAIN = ('wCr', 'dAr', 'q13r', 'gAC', 'gLB', 'qLAC', 'v5r', 'dLr', 'vcr', 'krc')


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def source_comparison(results):
    """Audit printed numerical targets without tuning the model to those targets."""
    comparisons = []
    def add(label, tag, key, target, time='end'):
        row = results[tag]['end'] if time == 'end' else results[tag]['at'][str(float(time))]
        actual = row[key]
        comparisons.append({'comparison': label, 'scenario': tag, 'metric': key,
                            'pdf_value': target, 'computed_value': actual,
                            'absolute_difference': abs(actual-target)})
    for name, values in zip(REGIMES, [(.923,.617,.669,.757,2.156),
                           (.943,.663,.703,.777,1.661),(.954,.688,.722,.787,1.541),
                           (.957,.696,.728,.790,1.514)]):
        for key,value in zip(('NT_ratio','NS_ratio','FS_ratio','FA_ratio','IH_ratio'),values):
            add('Table 8',f'regime:{name}',key,value)
    for duration, values in [(24,(.987,.880,.130)),(48,(.957,.790,.147)),
                             (60,(.939,.753,.148)),(72,(.919,.721,.150))]:
        for key,value in zip(('NT_ratio','FA_ratio','IH_change'),values):
            add('Table 11',f'grid:{duration}:1.5',key,value)
    for label,tag,values in [('Table 12','protocol:Front-loaded',(.955,.854,.024)),
                             ('Table 12','protocol:Four pulses',(.959,.804,.144)),
                             ('Table 13','protocol:80% block',(.9570,.7605,.0217)),
                             ('Table 13','protocol:Complete block',(.9570,.7346,-.0102))]:
        for key,value in zip(('NT_ratio','FA_ratio','IH_change'),values):
            add(label,tag,key,value,time=88.)
    add('Section 10.9','variant:No differentiation','FA_ratio',1.133)
    add('Section 10.9','variant:No RA-to-HOX','IH_change',-.027)
    return comparisons


def _pair_task(tag, p, protocol, solver, keep=True):
    warm = precondition(p, solver)
    control, treated = paired(p, protocol, solver, warm.y[:, -1])
    rows = {str(t): compare_at(control, treated, p, t)
            for t in sorted(set([88., 112., 168., protocol.end, protocol.end+48.]))
            if t <= solver.horizon}
    event = maintenance_events(control, treated, p, protocol.end)
    result = {'tag': tag, 'parameters': p.to_dict(), 'protocol': asdict(protocol),
              'end': rows[str(protocol.end)], 'at': rows, 'events': event}
    if keep:
        result.update(t=treated.t, control=control.y, treated=treated.y,
                      audits={'control': audit(control, p, CONTROL, solver),
                              'treated': audit(treated, p, protocol, solver)},
                      preconditioning_daily_change=(warm.y[:, -1]-warm.y[:, 0]).tolist())
    return result


def run_study(config_path, *, root):
    """Create a fresh root/runs directory; preserve every earlier run."""
    root = Path(root)
    config = json.loads(Path(config_path).read_text())
    extra = set(config) - {'pipeline', 'workers', 'lhs_samples', 'seed', 'solver'}
    if extra or config.get('pipeline') != 'population_extension':
        raise ValueError(f'Invalid population configuration keys/pipeline: {extra}')
    solver = Solver(**config.get('solver', {}))
    if solver.horizon < 168 or solver.precondition_hours < 24 or solver.dt <= 0:
        raise ValueError('Study needs horizon >=168, preconditioning >=24, and dt >0.')
    workers, sample_count = int(config.get('workers', 4)), int(config.get('lhs_samples', 160))
    if workers < 1 or sample_count < 12:
        raise ValueError('workers >=1 and lhs_samples >=12 required.')
    start = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out = root / 'runs' / f'{stamp}_population_extension_{uuid.uuid4().hex[:4]}'
    out.mkdir(parents=True)
    for folder in ('figures', 'tables', 'data', 'source'):
        (out/folder).mkdir()
    print(f'RUN_DIRECTORY={out}', flush=True)
    write_json(out/'config.json', config)
    write_json(out/'parameters.json', Parameters().to_dict())
    source_pdf = root/'extension/wnt_hox_ra_revised_population_dynamics.pdf'
    source_files = list(Path(__file__).parent.glob('*.py')) + [root/'extension/run_population.py',
                    root/'extension/README.md', root/'tests/test_population_extension.py',
                    root/'src/wnt_pinn/cli.py', Path(config_path)]
    for src in source_files:
        dest = out/'source'/src.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    manifest = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'source_pdf': str(source_pdf.relative_to(root)),
                'source_pdf_sha256': sha256(source_pdf.read_bytes()).hexdigest(),
                'python': platform.python_version(),
                'versions': {x: version(x) for x in ('numpy', 'scipy', 'matplotlib')},
                'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                                        text=True).strip(),
                'state_order': STATE_NAMES, 'no_fitting': True,
                'source_hashes': {str(x.relative_to(root)): sha256(x.read_bytes()).hexdigest()
                                  for x in source_files}}
    write_json(out/'manifest.json', manifest)
    p = Parameters()
    tasks = []
    for name, (w, theta) in REGIMES.items():
        tasks.append((f'regime:{name}', p.changed(W=w, thetaP=theta), NOMINAL, solver, True))
    for duration in (24., 36., 48., 60., 72.):
        for amplitude in (.5, 1., 1.5, 2., 2.5):
            protocol = Protocol(f'{duration:g}h / {amplitude:g}', amplitude, ((40.,40.+duration,1.),))
            tasks.append((f'grid:{duration:g}:{amplitude:g}', p, protocol, solver, True))
    protocols = [Protocol('Front-loaded', 1.5, ((40.,64.,2.),)),
                 Protocol('Four pulses', 1.5, tuple((x,x+6.,2.) for x in (40.,54.,68.,82.))),
                 Protocol('80% block', block=.8), Protocol('Complete block', block=1.)]
    for protocol in protocols:
        tasks.append((f'protocol:{protocol.name}', p, protocol, solver, True))
    for label, change in [
        ('No differentiation', dict(dL5=0.,dA5=0.,dLr=0.,dAr=0.)),
        ('No RA-to-HOX', dict(v5r=0.,q13r=0.,wBr=0.,wCr=0.)),
        ('No CYP attenuation', dict(chiL=0.,chiA=0.)),
        ('Hill n=1', dict(n=1.)), ('Hill n=4', dict(n=4.)),
        ('Hill K=0.35', dict(K=.35)), ('Hill K=0.65', dict(K=.65))]:
        tasks.append((f'variant:{label}', p.changed(**change), NOMINAL, solver, True))
    tasks.append(('check:physical_beta_clock', p, NOMINAL, replace(solver,beta_clock='physical'), True))
    tasks.append(('check:Radau', p, NOMINAL, replace(solver,method='Radau',rtol=1e-10,atol=1e-12), True))
    design = qmc.LatinHypercube(d=len(UNCERTAIN), seed=int(config['seed'])).random(sample_count)
    multipliers = np.exp(np.log(.85) + design * (np.log(1.15)-np.log(.85)))
    write_csv(out/'tables/lhs_design.csv', [{'sample': i, **dict(zip(UNCERTAIN, row))}
                                            for i,row in enumerate(multipliers)])
    for i, factors in enumerate(multipliers):
        change = {k: getattr(p,k)*f for k,f in zip(UNCERTAIN,factors)}
        tasks.append((f'lhs:{i}', p.changed(**change), NOMINAL, solver, False))
    results = {}
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_pair_task, *task): task[0] for task in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                record = future.result()
                results[record['tag']] = record
                if i % 10 == 0 or i == len(tasks):
                    print(f'Completed {i}/{len(tasks)} matched comparisons ({time.monotonic()-start:.1f}s)', flush=True)
        nominal = results['regime:Severe APC loss']
        audit_results = {tag: row['audits'] for tag,row in results.items() if 'audits' in row}
        check = results['check:Radau']
        audit_results['BDF_vs_Radau'] = {
            'maximum_absolute_state_difference': float(np.max(np.abs(nominal['treated']-check['treated']))),
            'maximum_scaled_state_difference': float(np.max(np.abs(nominal['treated']-check['treated'])
                                                   / np.maximum(np.abs(check['treated']),1e-8)))}
        if audit_results['BDF_vs_Radau']['maximum_scaled_state_difference'] > 1e-5:
            raise RuntimeError('Independent solver agreement failed.')
        write_json(out/'validation.json', audit_results)
        np.savez_compressed(out/'data/trajectories.npz', **{
            f'{tag}__{key}': row[key] for tag,row in results.items() if 't' in row
            for key in ('t','control','treated')})
        serial = {tag: {k:v for k,v in row.items() if k not in ('t','control','treated')}
                  for tag,row in results.items()}
        write_json(out/'results.json', serial)
        write_csv(out/'tables/endpoints.csv', [{'scenario': tag, **row['end']}
                   for tag,row in sorted(results.items())])
        write_csv(out/'tables/regimes.csv', [{'regime': name, **results[f'regime:{name}']['end']}
                                            for name in REGIMES])
        write_csv(out/'tables/severe_timepoints.csv', list(nominal['at'].values()))
        write_csv(out/'tables/pdf_comparison.csv', source_comparison(results))
        oC, oA = observables(nominal['control'],p), observables(nominal['treated'],p)
        write_csv(out/'data/severe_APC_timeseries.csv', [{'time_h': float(t),
            **{f'control_{k}': float(v[i]) for k,v in oC.items()},
            **{f'ATRA_{k}': float(v[i]) for k,v in oA.items()}}
            for i,t in enumerate(nominal['t'])])
        lhs = [results[f'lhs:{i}'] for i in range(sample_count)]
        sensitivity = [{'parameter': k, 'spearman_rho': float(spearmanr(multipliers[:,j],
                        [x['end']['IH_change'] for x in lhs]).statistic)}
                       for j,k in enumerate(UNCERTAIN)]
        write_csv(out/'tables/spearman_sensitivity.csv', sensitivity)
        from .plotting import make_figures, make_brief
        make_figures(results, sensitivity, out)
        make_brief(results, audit_results, config, out)
        manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic()-start, matched_comparisons=len(tasks),
                        lhs_samples=sample_count)
        manifest['output_hashes'] = {str(x.relative_to(out)): sha256(x.read_bytes()).hexdigest()
                                    for x in out.rglob('*') if x.is_file() and x.name!='manifest.json'}
        write_json(out/'manifest.json', manifest)
        (root/'extension/LATEST_RUN.txt').write_text(str(out.relative_to(root))+'\n')
        print(f'COMPLETE: {out}', flush=True)
        return out
    except BaseException as error:
        manifest.update(status='failed', error=str(error))
        write_json(out/'manifest.json', manifest)
        raise
