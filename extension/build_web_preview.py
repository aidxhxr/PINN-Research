#!/usr/bin/env python3
"""Publish a static explanation alongside the existing localhost poster preview.

Reads completed results only; does not run or modify any simulation. The misspelled
``extenstion_report`` route deliberately matches the requested public URL. The
correctly spelled route is also provided. Python's static server redirects /html
to /html/ and serves index.html with the correct HTML content type.
"""
import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

CSS = '''
:root{color-scheme:light;--ink:#25364a;--blue:#215fa2;--muted:#5e6c7b}
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:var(--ink);font:18px/1.7 system-ui,sans-serif}
main{max-width:1080px;margin:auto;padding:30px 30px 80px;background:white}
h1{font-size:2.2rem;line-height:1.2}h2{font-size:1.5rem;margin-top:48px;border-top:1px solid #dce3eb;padding-top:25px}
h3{font-size:1.13rem;margin-bottom:8px}p{margin:14px 0}a{color:var(--blue)}nav{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}
nav a{padding:8px 14px;border:1px solid #c9d7e8;border-radius:6px;text-decoration:none;font-size:15px}
.lead{font-size:1.18rem}.muted{color:var(--muted);font-size:15px}.speech{background:#eef5fc;border-left:5px solid var(--blue);padding:20px 25px}
.note{background:#fff8e9;border-left:4px solid #c18a30;padding:15px 20px}.math{font-family:ui-monospace,monospace;background:#f3f5f8;padding:14px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:16px}td,th{padding:10px 12px;text-align:left;border-bottom:1px solid #dae2ec}
th{background:#f2f6fa}figure{margin:25px 0}img{width:100%;height:auto;border:1px solid #e5e9ef}figcaption{font-size:15px;color:var(--muted)}
.table{overflow-x:auto}li{margin-bottom:8px}details{padding:12px 0;border-bottom:1px solid #e5e9ef}summary{font-weight:650;cursor:pointer}
@media(max-width:650px){main{padding:20px 17px}body{font-size:16px}h1{font-size:1.9rem}td,th{padding:8px}}
@media print{body,main{background:white}nav{display:none}main{max-width:none;padding:0}h2{break-after:avoid}figure,table{break-inside:avoid}}
'''


def table(headers, rows):
    return '<div class="table"><table><thead><tr>'+''.join(f'<th>{escape(str(x))}</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(f'<td>{escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def figure(stem, caption):
    return f'<figure><a href="/extension_plots/figures/{stem}.pdf"><img loading="lazy" src="/extension_plots/figures/{stem}.png" alt="{escape(caption)}"></a><figcaption>{escape(caption)} Click the image for its vector PDF.</figcaption></figure>'


def publish(server):
    run = (ROOT/'extension/results').resolve()
    results = json.loads((run/'results.json').read_text())
    nominal = results['regime:Severe APC loss']; e = nominal['end']; event = nominal['events']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    destination = ROOT/'extension/web'/stamp
    report = destination/'extenstion_report/html'
    report.mkdir(parents=True)
    body = '''<h1>How to explain the population-extension results</h1>
<p class="lead">The main finding is a separation between <strong>how many stem-like cells remain</strong> and <strong>the transcriptional state of those cells</strong>. Under ATRA, the modeled population shrinks while the reduced HOX index rises.</p>
<p class="muted">Preliminary reproduction of the supplied 14-equation model • 15 September 2026 • SciPy integration, no neural network or fitted parameters</p>
<nav><a href="#say">60-second explanation</a><a href="#numbers">Main numbers</a><a href="#figures">Explain the plots</a><a href="#caveats">What needs clarification</a><a href="/extension_plots/index.html">All 14 plots</a><a href="/extension_plots/meeting_brief.pdf">Meeting PDF</a><a href="/extension_plots/all_figures.pdf">All plots PDF</a></nav>
<h2 id="say">Start with this explanation</h2>
<div class="speech"><p>“I implemented the complete 14-equation extension in Python using SciPy and the parameters in your report. I preconditioned each parameter set without treatment, then compared ATRA against a matched untreated trajectory.</p>
<p>Under severe APC loss, the 48-hour exposure gives about a 4.3% reduction in total cells, a 30.4% reduction in stem-like cells, and a 21.0% reduction in the ALDH-like fraction. At the same time, the reduced HOX index increases from 0.286 to 0.433. So abundance and the modeled transcriptional state move in different directions.</p>
<p>After washout, the stem-like population starts growing again about 6.9 hours later, while it is still about 31.5% below the untreated control. The main numbers match the report closely. Some mechanism-removal and blockade values differ, and I have kept those discrepancies visible rather than adjusting the parameters to force agreement.”</p></div>
<h2>What the 14 states represent</h2>
<p>The original seven molecular states are beta-catenin, APC, HOXA5, HOXA13, MYC, retinoic acid, and CYP26A1. Four additional states describe aggregate HOXB and HOXC programs in each of two stem-like phenotypes. The last three states track <strong>LGR5-like cells, ALDH-like cells, and differentiated cells</strong>.</p>
<p>The molecular signals set the growth, switching, and differentiation rates of the populations. LGR5-like and ALDH-like cells can switch between those two compartments, and both can differentiate. Growth slows as the total population approaches carrying capacity. In this model, the population states do not feed back into the molecular equations.</p>
<p>The program states are dimensionless scores, not measurements of four individual genes. The population states are normalized by an assumed carrying capacity of one million cells.</p>
<h2>What was actually done</h2>
<ol><li>Transcribed the equations and parameter tables into a documented 14-state right-hand-side function.</li>
<li>Ran 720 hours of untreated preconditioning for every parameter set. This prepares comparable starting states and avoids treating the arbitrary seed as the treatment baseline.</li>
<li>Integrated matched control and ATRA trajectories for 168 hours using SciPy BDF, with tight tolerances and a maximum internal step of 0.5 hours. Plots sample the solution every 0.125 hours.</li>
<li>Computed population fractions, the reduced HOX index, and renewal/exit fluxes from those solutions. No curves were drawn by hand or fitted to the PDF.</li>
<li>Ran four disease regimes, 25 dose-duration combinations, alternate schedules, mechanism interventions, Hill-shape variants, and 160 parameter samples: 202 matched comparisons in total.</li></ol>
<p>Five scientific tests passed. Positivity and population-balance checks passed for the audited trajectories. A tighter Radau solution agrees with BDF to a maximum scaled state difference of about 1.9 × 10<sup>−8</sup>.</p>
<h2 id="numbers">Main numbers: severe APC loss at exposure end</h2>
<p>ATRA exposure is nominally <strong>40–88 hours</strong>. The table compares treated and untreated trajectories at the same time, <strong>88 hours</strong>, which is 48 hours after treatment starts. Treatment amplitude is a dimensionless model input, not a patient dose.</p>'''
    rows = []
    for label,key in [('Total population','NT'),('Stem-like population','NS'),('ALDH-like fraction of all cells','FA'),('Absolute ALDH-like population','nA'),('Reduced HOX index','IH')]:
        change = f'{100*(e[key+"_ratio"]-1):+.2f}%' if key!='IH' else f'+{e["IH_change"]:.4f} index units'
        rows.append([label,f'{e[key+"_control"]:.4f}',f'{e[key+"_treated"]:.4f}',change])
    body += table(['Output','Control','ATRA','Change'],rows)
    body += '''<p>Population entries are divided by carrying capacity; the ALDH-like fraction and HOX index are unitless. The fraction 0.1271 means about 12.71% of all modeled cells are ALDH-like.</p>
<h3>Why total cells fall only 4.3% when stem-like cells fall 30.4%</h3>
<p>Differentiation removes cells from the stem-like pool but transfers them into the differentiated compartment. It does not immediately remove them from the total population. Here, the differentiated population rises from about 0.2312 to 0.3825 while the stem-like pool falls. Total-cell abundance and stem-like abundance therefore measure different responses.</p>
<h3>Why the two ALDH percentages can move in opposite directions</h3>
<div class="math">ALDH fraction of all cells = nA / (nL + nA + nD)<br>ALDH share of the stem-like pool = nA / (nL + nA)</div>
<p>The ALDH-like fraction of all cells falls from 12.71% to 10.03%. But ALDH cells make up a slightly larger share of the remaining stem-like pool: 17.46% to 18.94%. This happens because LGR5-like cells fall more strongly. It does <strong>not</strong> mean that ALDH cells increased in absolute number; their absolute population falls 24.43%.</p>
<h2 id="figures">Four plots to explain in the meeting</h2>
<h3>1. Molecular response — what ATRA changes upstream</h3>
<p>Read blue as control and orange as ATRA. During exposure, retinoic acid rises, HOXA5 rises about 4.12-fold, APC rises, and beta-catenin and HOXA13 fall. CYP26A1 rises about 1.39-fold; it clears retinoic acid and supplies negative feedback. The model also directly lets retinoic acid repress HOXA13 and activate the HOXC programs.</p>'''
    body += figure('fig02_molecular','Figure 2. The shaded region marks the nominal 40–88 h exposure. Smooth onset and washout come from the input function.')
    body += '''<h3>2. Population response — follow abundance separately from fractions</h3>
<p>The two stem-like populations decline and the differentiated population increases. The main interpretation is a shift out of the stem-like compartments under the assumed differentiation rates. This is a model response, not an independently measured cell-count result.</p>'''
    body += figure('fig03_population','Figure 3. NT is total abundance, NS is stem-like abundance, FS and FA are fractions; nL, nA and nD are individual compartments.')
    body += '''<h3>3. The central result — lower ALDH abundance can coexist with a higher HOX index</h3>
<p>The left panel falls and the right panel rises during treatment. The reduced HOX index combines an increase in population-weighted HOXC program activity with a decrease in the HOXA13 Hill response. It captures a proposed transcriptional direction. It is <strong>not</strong> a direct measure of remaining cell number, the complete eight-gene survival signature, or a probability of clinical recurrence.</p>'''
    body += figure('fig04_central_result','Figure 4. A decrease in ALDH-like fraction and an increase in the reduced HOX index occur simultaneously.')
    body += '''<h3>4. Maintenance and rebound — fewer cells can already be regrowing</h3>
<div class="math">Rate of change of stem-like population = renewal flux − exit flux<br>Maintenance index = renewal / (renewal + exit)</div>
<p>The threshold 0.5 follows exactly from the flux balance: below it, exit exceeds renewal and the stem-like population contracts; above it, renewal exceeds exit and it grows. It is not a fitted cutoff or a 50% survival probability. Switching between the two stem-like phenotypes cancels when their populations are added.</p>'''
    body += f'<p>At exposure end the maintenance index is {e["SSC"]:.3f}. It crosses upward through 0.5 at <strong>{event["crossing_h"]:.1f} h</strong> from simulation baseline, or <strong>{event["after_washout_h"]:.1f} h after washout</strong>. At that time the stem-like population remains {100*(1-event["NS_ratio"]):.1f}% below control. This means recovery has begun; it does not mean the population has recovered.</p>'
    body += figure('fig05_maintenance','Figure 5. The crossing of renewal and exit aligns with the minimum absolute stem-like population, as required by the balance equation.')
    body += '''<h2>What the additional analyses show</h2>
<h3>Disease regimes</h3><p>ATRA reduces the modeled populations in all four regimes. The relative stem-like reduction is stronger in Normal than in Severe APC loss under these parameter choices.</p>'''
    names=['Normal','Early adenoma','Advanced adenoma','Severe APC loss']
    body += table(['Regime','Total-cell change','Stem-like change','ALDH-fraction change'],[
        [name]+[f'{100*(results["regime:"+name]["end"][k+"_ratio"]-1):+.1f}%' for k in ['NT','NS','FA']] for name in names])
    body += '''<h3>Exposure duration — Figures 8–10</h3><p>At fixed amplitude, longer exposure deepens the population response while the HOX-index increase approaches a plateau. Each row is measured at its own exposure endpoint, so these are different assessment times.</p>'''
    body += table(['Duration','Assessment time','Total / control','ALDH fraction / control','HOX-index change'],[
        [f'{d} h',f'{40+d} h']+[f'{results[f"grid:{d}:1.5"]["end"][k]:.3f}' for k in ['NT_ratio','FA_ratio','IH_change']] for d in [24,48,60,72]])
    body += '''<h3>Equal input area — Figure 11</h3><p>Continuous, front-loaded, and four-pulse schedules all add 72 dimensionless-hours of input. Their trajectories still differ because timing, clearance, and time since washout matter. At the common 88-hour assessment, front-loading leaves a smaller HOX-index shift, but its ALDH-fraction reduction is weaker. This is a schedule comparison, not a claim of an optimal treatment.</p>
<h3>Selective blockade and mechanism removals — Figures 12–13</h3><p>The selective probe weakens only RA repression of HOXA13 and RA activation of HOXC during treatment. It greatly suppresses the HOX-index rise while leaving total-cell reduction nearly unchanged. In this model, those two transcriptional effects can therefore be separated from much of the differentiation response. The probe does not represent an identified drug.</p>
<p>Removing RA/HOXA5-dependent differentiation eliminates the ALDH-fraction reduction. Removing all direct RA-to-HOX links removes almost all of the HOX-index response. These tests expose what the equations assume; they do not establish the mechanisms experimentally. Some magnitudes differ from the PDF, as explained below.</p>
<h3>Parameter and Hill-shape checks — Figures 6, 14, and 17</h3><p>All 160 Latin hypercube samples preserve a lower ALDH-like fraction and a higher HOX index. The threshold-crossing 5–95% interval is approximately 94.45–95.50 h. These are results of a ±15% parameter perturbation exercise, not experimental confidence intervals or a Bayesian posterior. The figure ranks ordinary Spearman correlations, not partial correlations. The nominal and four alternative Hill shapes also preserve the main qualitative directions.</p>
<p><a href="/extension_plots/index.html">Browse all 14 figures</a> or <a href="/extension_plots/all_figures.pdf">open the complete figure PDF</a>.</p>
<h2 id="caveats">What matches, and what needs clarification</h2>
<p>The headline severe-APC population response and the post-washout crossing closely match the report. The following discrepancies are useful points for discussion:</p>'''
    body += table(['Check','PDF','Computed'],[
        ['No differentiation: ALDH-fraction ratio','1.133',f'{results["variant:No differentiation"]["end"]["FA_ratio"]:.4f}'],
        ['No RA-to-HOX: HOX-index change','−0.027',f'{results["variant:No RA-to-HOX"]["end"]["IH_change"]:.6f}'],
        ['80% block: ALDH-fraction ratio','0.7605',f'{results["protocol:80% block"]["end"]["FA_ratio"]:.4f}'],
        ['80% block: HOX-index change','0.0217',f'{results["protocol:80% block"]["end"]["IH_change"]:.4f}']])
    body += '''<p>Under the printed no-RA-to-HOX ablation, every direct RA path into the upstream molecular/HOX-program subsystem is removed. An ATRA-induced beta-catenin reduction cannot then explain a large negative HOX-index shift. The small remaining shift comes from population-composition weighting. The precise intended ablation or calculation is worth confirming.</p>
<p>Equation (6) also omits a beta-catenin time multiplier that adjacent prose seems to imply. Following the printed equation reproduces the reported beta-catenin ratio of 0.719; including the physical multiplier gives 0.718, with little effect on the population conclusions. Both versions were checked.</p>
<p>The four pulse endpoints and the numerical sampling seed were not explicitly provided. The reproduction states its choices: pulses at 40–46, 54–60, 68–74, 82–88 h, and seed 20260915.</p>
<div class="note"><strong>Keep the conclusion within what was computed.</strong> These are simulations with supplied calibrated parameters. No raw replicate data were fitted. The HOX index does not establish an effect on patient survival. The multistability searches (Figures 15–16) and identifiability analysis were not rerun, so do not present those as independently verified results from this implementation.</div>
<h2>Useful follow-up points for the meeting</h2>
<ul><li>Confirm the intended beta-catenin clock and the exact ablation definitions, especially the no-RA-to-HOX result.</li><li>Decide which experimental observations should be compared first: absolute viable cells, absolute and fractional ALDH/LGR5 measurements, and HOXA5/HOXA13/HOXC measurements at the same times.</li><li>Prioritize sampling after washout. The predicted regrowth signal appears before the population has returned to control abundance.</li></ul>
<h2>Files for reference</h2><nav><a href="/extension_plots/meeting_brief.pdf">Six-page brief</a><a href="/extension_plots/all_figures.pdf">All plots</a><a href="/extension_plots/tables/pdf_comparison.csv">PDF comparison audit</a><a href="/extension_plots/data/severe_APC_timeseries.csv">Raw severe-APC time series</a><a href="/extension_plots/RESULTS.md">Original results text</a></nav>
'''
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Population extension — meeting explanation</title><style>'+CSS+'</style></head><body><main>'+body+f'<p class="muted">Generated from completed run {escape(run.name)}. All displayed computed values come from its saved results.</p></main></body></html>'
    (report/'index.html').write_text(document)
    shutil.copy2(report/'index.html',report.parent/'index.html')
    # Existing poster files and process are left in place. Only requested aliases are added.
    for route,target in [('extension_plots',run),('extenstion_report',report.parent),('extension_report',report.parent)]:
        link=server/route
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise FileExistsError(f'Refusing to replace existing directory: {link}')
        link.symlink_to(target,target_is_directory=True)
    print('http://localhost:8003/extension_plots/index.html')
    print('http://localhost:8003/extension_plots/all_figures.pdf')
    print('http://localhost:8003/extenstion_report/html')
    print(f'HTML source: {report / "index.html"}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-root',type=Path,required=True)
    args=parser.parse_args()
    if not args.server_root.is_dir():
        parser.error('server-root must already exist')
    publish(args.server_root.resolve())
