"""Presentation figures and an automatically populated preliminary-results brief.

Matplotlib only draws values integrated by SciPy. PDF figure numbers retain the
source report's numbering; schematic Fig. 1 and attractor scans Figs. 15--16 are
outside this preliminary run and are explicitly listed as not reproduced.
"""
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .model import Parameters, observables

BLUE, ORANGE, TEAL = '#2463a7', '#dd7134', '#238776'
LABELS = {'b': r'$\beta$-catenin', 'p': 'APC', 'h5': 'HOXA5', 'h13': 'HOXA13',
          'm': 'MYC', 'r': 'Retinoic acid', 'c': 'CYP26A1', 'NT': 'Total population',
          'NS': 'Stem-like population', 'FS': 'Stem-like fraction', 'FA': 'ALDH-like fraction',
          'SA': 'ALDH share of stem pool', 'nL': 'LGR5-like cells', 'nA': 'ALDH-like cells',
          'nD': 'Differentiated cells', 'IH': 'Reduced HOX index', 'SSC': 'Maintenance index'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'semibold', 'axes.labelcolor': '#344054',
                     'axes.grid': True, 'grid.alpha': .18, 'figure.dpi': 110,
                     'savefig.dpi': 180, 'pdf.fonttype': 42})


def obs(row, kind='treated'):
    return observables(row[kind], Parameters(**row['parameters']))


def shade(ax):
    ax.axvspan(40,88,color=ORANGE,alpha=.09,lw=0)
    ax.set_xlabel('Time since simulation baseline (h)')


def make_figures(results, sensitivity, out):
    nominal = results['regime:Severe APC loss']
    t, control, treated = nominal['t'], obs(nominal,'control'), obs(nominal)
    figures = []
    with PdfPages(out/'all_figures.pdf') as book:
        def save(fig, stem, title):
            fig.suptitle(title, fontsize=15, fontweight='bold')
            fig.tight_layout(rect=(0,.035,1,.93))
            fig.text(.5,.008,'SciPy ODE simulation | severe APC loss unless stated | shaded interval: nominal ATRA exposure',
                     ha='center',fontsize=8,color='#667085')
            fig.savefig(out/'figures'/f'{stem}.png')
            fig.savefig(out/'figures'/f'{stem}.pdf')
            book.savefig(fig)
            figures.append((stem,title))
            plt.close(fig)
        def pair(ax,key):
            ax.plot(t,control[key],color=BLUE,label='Control',lw=1.8)
            ax.plot(t,treated[key],color=ORANGE,label='ATRA',lw=1.8)
            ax.set_title(LABELS.get(key,key))
            shade(ax)
        fig,axs=plt.subplots(2,4,figsize=(14,7))
        for ax,key in zip(axs.flat,('b','p','h5','h13','m','r','c')):
            pair(ax,key)
            ax.set_ylabel('Scaled concentration')
        for key,style in [('uCL','-'),('uCA','--')]:
            axs.flat[-1].plot(t,control[key],style,color=BLUE,label=f'{key} control')
            axs.flat[-1].plot(t,treated[key],style,color=ORANGE,label=f'{key} ATRA')
        axs.flat[-1].set_title('HOXC program scores')
        shade(axs.flat[-1])
        axs.flat[-1].legend(fontsize=8)
        axs.flat[0].legend()
        save(fig,'fig02_molecular','Figure 2 | Molecular response to the 48-hour ATRA exposure')
        fig,axs=plt.subplots(2,4,figsize=(14,7))
        for ax,key in zip(axs.flat,('NT','NS','FS','FA','nL','nA','nD','IH')):
            pair(ax,key)
            ax.set_ylabel('Population / carrying capacity' if key in ('NT','NS','nL','nA','nD') else 'Fraction / index')
        axs.flat[0].legend()
        save(fig,'fig03_population','Figure 3 | Cell depletion and transcriptional change coexist')
        fig,axs=plt.subplots(1,2,figsize=(11,4.8))
        pair(axs[0],'FA')
        pair(axs[1],'IH')
        axs[0].legend()
        save(fig,'fig04_central_result','Figure 4 | ALDH-like fraction falls while the reduced HOX index rises')
        fig,axs=plt.subplots(2,2,figsize=(11,8))
        pair(axs[0,0],'NS')
        axs[0,0].legend()
        for key,label,color in [('Jren','Renewal',TEAL),('Jexit','Exit',ORANGE)]:
            axs[0,1].plot(t,treated[key],label=label,color=color)
        shade(axs[0,1])
        axs[0,1].set_title('Stem-like cell fluxes')
        axs[0,1].legend()
        pair(axs[1,0],'SSC')
        axs[1,0].axhline(.5,color='gray',ls='--')
        crossing=nominal['events']['crossing_h']
        axs[1,0].annotate(f'Renewal resumes: {crossing:.1f} h',xy=(crossing,.5),xytext=(10,.60),
                          arrowprops={'arrowstyle':'->','color':'gray'},fontsize=9)
        scatter=axs[1,1].scatter(treated['NS']/control['NS'],treated['SSC'],c=t,s=5,cmap='viridis')
        axs[1,1].axhline(.5,color='gray',ls='--')
        axs[1,1].set_xlabel('Stem-like population: ATRA / control')
        axs[1,1].set_ylabel('Maintenance index')
        fig.colorbar(scatter,ax=axs[1,1],label='Time (h)')
        save(fig,'fig05_maintenance','Figure 5 | The stem-like pool begins regrowing before it recovers')
        lhs=[r for k,r in sorted(results.items()) if k.startswith('lhs:')]
        fig,axs=plt.subplots(1,2,figsize=(11,4.8))
        crossing_samples=[r['events']['crossing_h'] for r in lhs if r['events']['crossing_h'] is not None]
        axs[0].hist(crossing_samples,bins=15,color=BLUE,alpha=.85)
        axs[0].set_xlabel('Post-washout threshold crossing (h)')
        axs[0].set_ylabel('Samples')
        for i,time in enumerate((88.,112.,168.)):
            values=[r['at'][str(time)]['SSC'] for r in lhs]
            lo,med,hi=np.quantile(values,[.05,.5,.95])
            axs[1].errorbar(i,med,yerr=[[med-lo],[hi-med]],fmt='o',color=ORANGE,capsize=7)
        axs[1].set_xticks([0,1,2],['Exposure end\n88 h','24 h after washout\n112 h','80 h after washout\n168 h'])
        axs[1].axhline(.5,color='gray',ls='--')
        axs[1].set_ylabel('Maintenance index (median, 5–95%)')
        save(fig,'fig06_timing_ensemble',f'Figure 6 | Timing robustness in {len(lhs)} Latin hypercube samples')
        fig,ax=plt.subplots(figsize=(10,5))
        keys=['NT','FA','h5','h13','c']
        vals=[nominal['end'][f'{k}_ratio'] for k in keys]
        bars=ax.bar([LABELS[k] for k in keys],vals,color=[BLUE,BLUE,ORANGE,ORANGE,ORANGE])
        ax.axhline(1,color='gray',ls='--')
        ax.bar_label(bars,fmt='%.3f',padding=3)
        ax.set_ylim(0,max(vals)*1.18)
        ax.set_ylabel('ATRA / matched control at 88 h')
        save(fig,'fig07_endpoint','Figure 7 | End-of-exposure ratios reproduce the reported directions')
        durations=[24,36,48,60,72]
        doses=[.5,1.,1.5,2.,2.5]
        for number,stem,keys,timing,titles in [
            (8,'dose_duration',['NT_ratio','FA_ratio','IH_change'],'end',
             ['Total population ratio','ALDH-like fraction ratio','Reduced HOX index change']),
            (9,'grid_maintenance',['SSC','SSC','NS_ratio'],'mixed',
             ['Maintenance at exposure end','Maintenance 48 h after washout','Stem population ratio 48 h after washout'])]:
            fig,axs=plt.subplots(1,3,figsize=(14,4.6))
            for j,(ax,key,title) in enumerate(zip(axs,keys,titles)):
                arr=np.array([[results[f'grid:{d}:{amp:g}'][
                    'end' if timing=='end' or j==0 else 'at'][key]
                    if timing=='end' or j==0 else results[f'grid:{d}:{amp:g}']['at'][str(float(40+d+48))][key]
                    for amp in doses] for d in durations])
                im=ax.imshow(arr,origin='lower',aspect='auto',cmap='viridis')
                ax.set_xticks(range(5),doses)
                ax.set_yticks(range(5),durations)
                ax.set_xlabel('Dimensionless added RA input')
                ax.set_ylabel('Exposure duration (h)')
                ax.set_title(title)
                for yi in range(5):
                    for xi in range(5):
                        ax.text(xi,yi,f'{arr[yi,xi]:.2f}',ha='center',va='center',fontsize=8,
                                color='white' if arr[yi,xi] < (arr.min()+arr.max())/2 else 'black')
                fig.colorbar(im,ax=ax,shrink=.85)
            save(fig,f'fig{number:02d}_{stem}',f'Figure {number} | Dose and duration alter response depth and persistence')
        fig,axs=plt.subplots(2,3,figsize=(13,7))
        for d in (24,48,60,72):
            row=results[f'grid:{d}:1.5']
            a,c=obs(row),obs(row,'control')
            for ax,key in zip(axs.flat,('b','h5','h13','NT','FA','IH')):
                y=a[key] if key in ('b','h5','h13') else a[key]-c[key] if key=='IH' else a[key]/c[key]
                ax.plot(t,y,label=f'{d} h')
                ax.set_title(LABELS[key])
                ax.set_xlabel('Time (h)')
        axs.flat[0].legend()
        save(fig,'fig10_duration','Figure 10 | Longer exposure deepens depletion and delays recovery')
        for number,stem,rows,title in [
            (11,'schedules',[('Continuous',nominal),('Front-loaded',results['protocol:Front-loaded']),
                            ('Four pulses',results['protocol:Four pulses'])],'Equal nominal input area gives different trajectories'),
            (12,'selective_block',[('ATRA',nominal),('80% block',results['protocol:80% block']),
                                  ('Complete block',results['protocol:Complete block'])],'Selective HOX-arm blockade separates the responses')]:
            fig,axs=plt.subplots(1,3,figsize=(13,4.8))
            for label,row in rows:
                a,c=obs(row),obs(row,'control')
                for ax,key in zip(axs,('NT','FA','IH')):
                    ax.plot(t,a[key]-c[key] if key=='IH' else a[key]/c[key],label=label)
                    ax.set_title(LABELS[key])
                    ax.set_xlabel('Time (h)')
                    ax.set_ylabel('ATRA − control' if key=='IH' else 'ATRA / control')
            axs[0].legend(fontsize=9)
            save(fig,f'fig{number:02d}_{stem}',f'Figure {number} | {title}')
        fig,axs=plt.subplots(1,2,figsize=(12,5))
        labels=['Full model','No differentiation','No RA-to-HOX','No CYP attenuation']
        rows=[nominal]+[results[f'variant:{name}'] for name in labels[1:]]
        for ax,key in zip(axs,('FA_ratio','IH_change')):
            ax.bar(range(4),[r['end'][key] for r in rows],color=[BLUE,ORANGE,TEAL,'#805aa6'])
            ax.set_xticks(range(4),['Full model','No RA/HOXA5\ndifferentiation','No RA-to-HOX\nregulation','No CYP\nattenuation'],fontsize=8)
            ax.axhline(1 if key=='FA_ratio' else 0,color='gray',ls='--')
            ax.set_title('ALDH-like fraction ratio' if key=='FA_ratio' else 'Reduced HOX index change')
        save(fig,'fig13_ablations','Figure 13 | Removing mechanisms reveals which assumptions drive the result')
        fig,axs=plt.subplots(1,2,figsize=(12,5))
        scat=axs[0].scatter([r['end']['FA_ratio'] for r in lhs],[r['end']['IH_change'] for r in lhs],
                           c=[r['end']['NT_ratio'] for r in lhs],cmap='viridis',s=22)
        axs[0].set_xlabel('ALDH-like fraction ratio')
        axs[0].set_ylabel('Reduced HOX index change')
        fig.colorbar(scat,ax=axs[0],label='Total population ratio')
        ordered=sorted(sensitivity,key=lambda r:r['spearman_rho'])
        axs[1].barh([r['parameter'] for r in ordered],[r['spearman_rho'] for r in ordered],color=BLUE)
        axs[1].set_xlabel('Spearman rank correlation with HOX index change')
        axs[1].axvline(0,color='gray')
        save(fig,'fig14_sensitivity',f'Figure 14 | {len(lhs)} samples: ±15% log-uniform parameter perturbations')
        fig,axs=plt.subplots(1,2,figsize=(12,5))
        labels=['Nominal','n=1','n=4','K=0.35','K=0.65']
        rows=[nominal]+[results[f'variant:Hill {s}'] for s in labels[1:]]
        x=np.arange(5)
        for j,key in enumerate(('NT_ratio','NS_ratio','FA_ratio')):
            axs[0].bar(x+(j-1)*.24,[r['end'][key] for r in rows],width=.24,label=key.replace('_ratio',''))
        axs[0].set_xticks(x,labels)
        axs[0].axhline(1,color='gray',ls='--')
        axs[0].legend()
        axs[0].set_ylabel('ATRA / control at 88 h')
        axs[1].bar(x,[r['end']['IH_change'] for r in rows],color=ORANGE)
        axs[1].set_xticks(x,labels)
        axs[1].set_ylabel('Reduced HOX index change')
        save(fig,'fig17_hill_robustness','Figure 17 | Main directions survive the five tested Hill shapes')
    gallery='\n'.join(f'<section><h2>{title}</h2><a href="figures/{stem}.pdf">Vector PDF</a><br>'
                      f'<img src="figures/{stem}.png" alt="{title}"></section>' for stem,title in figures)
    (out/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Population extension results</title>'
        '<style>body{font:16px system-ui;margin:30px auto;max-width:1300px;color:#243044}img{max-width:100%}'
        'section{margin:45px 0}a{color:#2463a7}</style><h1>14-state population extension: preliminary reproduction</h1>'
        '<p><a href="meeting_brief.pdf">Meeting brief</a> · <a href="all_figures.pdf">All figures</a> · '
        '<a href="RESULTS.md">Results and limitations</a></p>'+gallery+'</html>')


def make_brief(results, audits, config, out):
    nominal=results['regime:Severe APC loss']
    e=nominal['end']
    event=nominal['events']
    lhs=[v for k,v in results.items() if k.startswith('lhs:')]
    qcross=np.quantile([x['events']['crossing_h'] for x in lhs if x['events']['crossing_h'] is not None],[.05,.5,.95])
    all_tension=sum(x['end']['FA_ratio']<1 and x['end']['IH_change']>0 for x in lhs)
    body=f'''# Preliminary SciPy reproduction: population extension

Model source: `extension/wnt_hox_ra_revised_population_dynamics.pdf` (September 2026).
All values below are recomputed from the 14 ODEs. No neural networks or parameter fitting.

## Main result to present

Under severe APC loss, a 48-hour ATRA input reduces cell abundance while increasing the
model's reduced HOX transcriptional index. These are different model outputs and can move
in opposite directions. After washout, stem-like cells start regrowing while still depleted.

| Output at 88 h | Control | ATRA | Change relative to control |
|---|---:|---:|---:|
| Total population / carrying capacity | {e['NT_control']:.4f} | {e['NT_treated']:.4f} | {100*(e['NT_ratio']-1):+.2f}% |
| Stem-like population / carrying capacity | {e['NS_control']:.4f} | {e['NS_treated']:.4f} | {100*(e['NS_ratio']-1):+.2f}% |
| ALDH-like fraction of all cells | {e['FA_control']:.4f} | {e['FA_treated']:.4f} | {100*(e['FA_ratio']-1):+.2f}% |
| Absolute ALDH-like population / carrying capacity | {e['nA_control']:.4f} | {e['nA_treated']:.4f} | {100*(e['nA_ratio']-1):+.2f}% |
| Reduced HOX index | {e['IH_control']:.4f} | {e['IH_treated']:.4f} | +{e['IH_change']:.4f} index units |

The ALDH share **within the residual stem pool** rises from {100*e['SA_control']:.2f}% to
{100*e['SA_treated']:.2f}%, even though the absolute ALDH population and its fraction of all cells
both fall. The denominator matters.

HOXA5 increases {e['h5_ratio']:.3f}-fold; HOXA13 falls to {e['h13_ratio']:.3f} times control;
CYP26A1 rises {e['c_ratio']:.3f}-fold. These are scaled model concentrations, not measured expression data.

The maintenance index is {e['SSC']:.4f} at exposure end (below the exact contraction threshold 0.5).
It re-crosses at {event['crossing_h']:.3f} h from simulation baseline, or
{event['after_washout_h']:.3f} h after washout. At that time the stem-like population remains
{100*(1-event['NS_ratio']):.2f}% below matched control. At 168 h it remains
{100*(1-nominal['at']['168.0']['NS_ratio']):.2f}% below control.

## What was completed

- Implemented all seven molecular, four HOX-program, and three population equations.
- Used the supplied parameters, separate untreated 720-h preconditioning for every parameter set,
  matched controls, BDF (rtol 1e-9, atol 1e-11, max step 0.5 h), and 0.125-h output spacing.
- Simulated all four disease regimes; a 5-by-5 dose-duration grid; equal-area schedules;
  selective HOX-arm blockades; three mechanism ablations; five Hill-shape settings.
- Ran {len(lhs)} Latin hypercube samples of the ten specified couplings over [0.85,1.15]
  log-uniform multipliers, seed {config['seed']}. {all_tension}/{len(lhs)} preserve lower ALDH-like
  fraction and higher HOX index. The crossing-time 5th/median/95th percentiles are
  {qcross[0]:.3f}, {qcross[1]:.3f}, {qcross[2]:.3f} h. These are parameter-perturbation intervals,
  not statistical confidence intervals or a Bayesian posterior.
- Reproduced numerical Figures 2–14 and 17 as 14 PNG/vector-PDF pairs. Figure 1 is a schematic;
  Figures 15–16 (multistability searches) and the identifiability analysis are not rerun here.
- Checked positivity, bounded programs/populations, exact population balance identities, and
  agreement with a tighter independent Radau integration. Maximum scaled state difference:
  {audits['BDF_vs_Radau']['maximum_scaled_state_difference']:.3g}.

## Reproduction choices and limits

**Not every printed number is reproduced.** Nominal, regime, duration and schedule
comparisons are audited in `tables/pdf_comparison.csv`. The no-differentiation
ALDH-fraction ratio is {results['variant:No differentiation']['end']['FA_ratio']:.4f}
(PDF: 1.133). Removing RA-to-HOX regulation gives HOX-index change
{results['variant:No RA-to-HOX']['end']['IH_change']:.6f} (PDF: -0.027).
The 80% selective-block condition gives ALDH-fraction ratio
{results['protocol:80% block']['end']['FA_ratio']:.4f} (PDF: 0.7605) and HOX-index change
{results['protocol:80% block']['end']['IH_change']:.4f} (PDF: 0.0217).
These discrepancies are retained, with no parameter tuning to conceal them.

In particular, setting v5r=q13r=wBr=wCr=0 removes every RA input to the upstream
molecular/HOX-program subsystem in the printed equations. An ATRA-induced reduction
in beta-catenin cannot then explain a large negative HOX-index change; only the
population-composition weighting still changes that index. This is a useful point
to clarify with the report author. The ablation figure reports the implemented
equations, not the unmatched numerical claim.

Equation (6) as printed has no dB multiplier, although adjacent prose implies one. The primary
run follows the printed equation; this gives beta-catenin ratio {e['b_ratio']:.6f}, consistent
with the PDF's 0.719. A separate physical-clock variant gives
{results['check:physical_beta_clock']['end']['b_ratio']:.6f}. Population conclusions barely change.
Other molecular clocks use the exact stated half-lives; population rates are already per hour.

The four 6-h pulse endpoints are not listed explicitly in the PDF. This reproduction assumes
40–46, 54–60, 68–74, 82–88 h at twice nominal amplitude. The LHS seed is also not numerically
specified in the provided report, so the seed above is our explicit choice. Schedule and ensemble
details may consequently differ. Smooth pulse tails are integrated, with no artificial cutoff.

The PDF calls an ordinary Spearman correlation a PRCC-style analysis. Here the same statistic is
correctly labeled Spearman; no partial-correlation claim is made. Time 94.9 h is measured from
simulation baseline, not treatment start (40 h); it is approximately 54.9 h after treatment start.

The 720-h warm-up is followed exactly; it is not proof that the slower populations are at a
perfect periodic equilibrium. Their one-day residual changes are recorded in results.json.
No raw experimental data were supplied or fitted. The HOX index is a reduced model score,
not the eight-gene clinical signature, and these simulations do not predict patient survival.
The selective block is a hypothetical intervention on two model couplings, not an identified drug.

## Suggested 60-second explanation

“I implemented the complete 14-equation extension directly in SciPy and reproduced the main
48-hour response using the supplied parameters and preconditioning. Under severe APC loss,
the model gives about a 4.3% reduction in total cells, a 30.4% reduction in stem-like cells,
and a 21.0% reduction in the ALDH-like fraction. At the same time, its reduced HOX index
rises from 0.286 to 0.433. The stem-like population starts growing again about 6.9 hours
after washout while still roughly 31.5% below control. The dose-duration and mechanism
comparisons help explain this separation. These are preliminary reproduction results,
not a fit to experimental data or a clinical conclusion.”

## Files

- `meeting_brief.pdf`: presentation brief and four main plots.
- `all_figures.pdf`: all 14 figure pages; `figures/`: individual PNG and vector PDF plots.
- `index.html`: local browser gallery.
- `tables/regimes.csv`, `endpoints.csv`, `severe_timepoints.csv`: numerical results.
- `data/severe_APC_timeseries.csv`: every severe-regime state and observable, control and ATRA.
- `data/trajectories.npz`: all non-ensemble trajectories, named by scenario.
- `results.json`, `validation.json`, `parameters.json`, `config.json`, `manifest.json`:
  full settings, audits, numerical results and file/source hashes; `source/`: code snapshot.
'''
    (out/'RESULTS.md').write_text(body)
    # A short standalone deck, with readable text pages and full-width key plots.
    with PdfPages(out/'meeting_brief.pdf') as pdf:
        def text_page(title, paragraphs):
            fig=plt.figure(figsize=(12.8,7.2))
            fig.patch.set_facecolor('white')
            fig.text(.06,.90,title,fontsize=24,fontweight='bold',color='#20324b')
            y=.77
            for text in paragraphs:
                wrapped=textwrap.fill(text,94)
                fig.text(.06,y,wrapped,fontsize=15,va='top',linespacing=1.5,color='#344054')
                y-=.06*(wrapped.count('\n')+1)+.035
            fig.text(.06,.045,'Preliminary model reproduction | 15 September 2026 | SciPy only',fontsize=10,color='#667085')
            pdf.savefig(fig)
            plt.close(fig)
        text_page('14-state population extension: preliminary results',[
            'Implemented 7 molecular states, 4 phenotype-conditioned HOX programs, and 3 cell populations using the supplied equations and parameters.',
            'Severe APC loss, 48-hour ATRA exposure: total cells −4.3%; stem-like cells −30.4%; ALDH-like fraction −21.0%.',
            f'Reduced HOX index: {e["IH_control"]:.3f} → {e["IH_treated"]:.3f}. Cell abundance and transcriptional state can move in different directions.',
            f'Stem-like regrowth resumes {event["after_washout_h"]:.1f} h after washout, while the population remains {100*(1-event["NS_ratio"]):.1f}% below control.'
        ])
        for stem in ('fig02_molecular','fig03_population','fig04_central_result','fig05_maintenance'):
            fig,ax=plt.subplots(figsize=(12.8,7.2))
            ax.imshow(plt.imread(out/'figures'/f'{stem}.png'))
            ax.axis('off')
            fig.tight_layout(pad=.2)
            pdf.savefig(fig)
            plt.close(fig)
        text_page('Checks, scope, and next discussion',[
            f'BDF and tighter Radau agree: maximum scaled state discrepancy {audits["BDF_vs_Radau"]["maximum_scaled_state_difference"]:.2g}. Positivity and population-flux identities pass.',
            f'{len(lhs)} parameter samples: {all_tension} preserve the main opposing response directions. Re-crossing 5–95% interval: {qcross[0]:.2f}–{qcross[2]:.2f} h.',
            'Generated 14 numerical figure types. Main endpoints match; some ablation and selective-block values differ from the PDF and are flagged in the results audit.',
            'Discuss: confirm the clock and ablation definitions. Multistability and identifiability scans were not rerun. No experimental fitting or clinical survival claim.'
        ])
