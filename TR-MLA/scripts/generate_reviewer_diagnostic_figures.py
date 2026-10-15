"""读取已完成审稿补充CSV，生成独立科学图；输入只读，不绘制未经估计的误差条。"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT/'research_artifacts/reviewer_supplement_20261009'
OUT = SUPPLEMENT/'figures'
COLORS = {'single_parent':'#64748b', 'multiple_parent':'#168c87', 'tr_mla':'#c86e2e',
          'typed_relational_myopic':'#316baf', 'marginal_lookahead_no_relations':'#875cab',
          'criticality_greedy':'#5b6570'}
NAMES = {'tr_mla':'TR-MLA', 'typed_relational_myopic':'Relational myopic',
         'marginal_lookahead_no_relations':'No-relation lookahead', 'criticality_greedy':'Criticality greedy'}


def save(fig,stem):
    for suffix in ('png','pdf','svg'):
        fig.savefig(OUT/f'{stem}.{suffix}',dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2,zorder=0)
    ax.set_axisbelow(True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'axes.titleweight':'semibold','svg.fonttype':'none','pdf.fonttype':42})
    inputs = {'parent_summary':SUPPLEMENT/'diagnostics/parent_ablation/variant_summary.csv',
              'noise_summary':SUPPLEMENT/'diagnostics/noise_replay/summary.csv',
              'efficiency_raw':SUPPLEMENT/'efficiency/raw_replicates.csv'}
    parent,noise,raw=(pd.read_csv(inputs[key]) for key in ('parent_summary','noise_summary','efficiency_raw'))
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5))
    for ax,target,title,ymax in zip(axes,('available_length_mm','available_width_mm'),
                                   ('(a) Available length','(b) Available width'),(1650,115),strict=True):
        group=parent[parent.target==target]
        seeds=sorted(group.seed.unique())
        for offset,variant,label in ((-.18,'single_parent','Original: one parent'),(.18,'multiple_parent','Local adaptation: two parents')):
            values=[group[(group.seed==seed)&(group.variant==variant)].iloc[0].mae for seed in seeds]
            bars=ax.bar(np.arange(3)+offset,values,width=.34,label=label,color=COLORS[variant],zorder=3)
            ax.bar_label(bars,labels=[f'{x:.1f}' for x in values],padding=3,fontsize=9)
        ax.axhline(10,ls=':',lw=1.5,color='#993131',label='Engineering tolerance: 10 mm')
        ax.set(xticks=np.arange(3),xticklabels=[str(s) for s in seeds],ylabel='Mean absolute error (mm)',
               xlabel='Independent data seed',title=title,ylim=(0,ymax))
        ax.text(.02,.95,'Reliable activation = 0% in all variants',transform=ax.transAxes,va='top',fontsize=9)
        style(ax)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.025),frameon=False,fontsize=9)
    fig.suptitle('Local parent adaptation: lower error does not ensure reliable inference',fontsize=13,y=.99)
    fig.subplots_adjust(bottom=.23,top=.83,wspace=.25)
    save(fig,'reviewer_continuous_parent_adaptation')

    fig,axes=plt.subplots(2,2,figsize=(12.3,9.4))
    groups=[(0.,0.),(.1,.05),(.1,.10),(.2,.05),(.2,.10)]
    ticks=['Clean','p=.10\na=.05','p=.10\na=.10','p=.20\na=.05','p=.20\na=.10']
    for ax,metric,title in ((axes[0,0],'final_sat_rate','(a) Fixed-query replay: final feasibility'),
                            (axes[0,1],'mean_rcrr_at_8','(b) Fixed-query replay: recovery at budget 8')):
        for policy in ('typed_relational_myopic','tr_mla'):
            for guard in (False,True):
                group=noise[(noise.policy_name==policy)&(noise.guard==guard)]
                values=[group[(group.error_probability==p)&(group.continuous_error_amplitude==a)].iloc[0][metric] for p,a in groups]
                ax.plot(np.arange(5),values,color=COLORS[policy],ls='-' if guard else '--',
                        marker='s' if guard else 'o',ms=5,label=f"{NAMES[policy]} | guard {'on' if guard else 'off'}")
        ax.set(xticks=np.arange(5),xticklabels=ticks,title=title,
               ylabel='Final SAT fraction' if metric=='final_sat_rate' else 'Mean RCRR@8',
               ylim=(0,1.05))
        style(ax)
    axes[0,0].legend(loc='lower left',fontsize=8,frameon=False)
    axes[0,0].text(.04,.69,'Both guard-on curves = 1.00',transform=axes[0,0].transAxes,fontsize=9)
    axes[0,1].text(.02,.03,'Same corrupted answers; fixed historical query order',transform=axes[0,1].transAxes,fontsize=8)
    endpoint=noise[(noise.error_probability==.2)&(noise.continuous_error_amplitude==.1)]
    endpoint_labels=['At p=.20, a=.10 (off / on):']
    for policy in ('typed_relational_myopic','tr_mla'):
        values=[endpoint[(endpoint.policy_name==policy)&(endpoint.guard==guard)].iloc[0].mean_rcrr_at_8 for guard in (False,True)]
        endpoint_labels.append(f'{NAMES[policy]}: {values[0]:.4f} / {values[1]:.4f}')
    axes[0,1].text(.04,.23,'\n'.join(endpoint_labels),transform=axes[0,1].transAxes,fontsize=9,linespacing=1.5)
    scale=raw[(raw.group=='scale')&(raw.status=='ok')].copy()
    aggregated=[]
    for policy in NAMES:
        selected=scale[scale.policy==policy]
        for candidates in (2,4,8,12):
            group=selected[selected.requested_candidates==candidates]
            if len(group)!=9:
                raise ValueError('Efficiency figure requires three specifications x three completed repetitions')
            aggregated.append({'policy':policy,'candidates':candidates,'n_repetitions':len(group),
                               'actual_eligible_median':group.actual_eligible.median(),
                               'decision_seconds_median':group.decision_seconds.median(),
                               'sat_calls_median':group.sat_calls.median()})
    medians=pd.DataFrame(aggregated)
    medians.to_csv(OUT/'reviewer_efficiency_plot_data.csv',index=False,encoding='utf-8-sig')
    for ax,metric,title in ((axes[1,0],'decision_seconds_median','(c) Pure decision time: measured median'),
                            (axes[1,1],'sat_calls_median','(d) Actual SAT calls: measured median')):
        for policy in NAMES:
            group=medians[medians.policy==policy]
            ax.plot(group.actual_eligible_median,group[metric],marker='o',lw=1.6,color=COLORS[policy],label=NAMES[policy])
        ax.set(xticks=[2,4,8,12],xlabel='Actual eligible candidates (fixed 24-field model)',title=title,
               ylabel='Median decision time (s, log scale)' if metric=='decision_seconds_median' else 'Median actual SAT calls')
        if metric=='decision_seconds_median':
            ax.set_yscale('log')
            ax.set_ylim(1e-5,1)
        style(ax)
    axes[1,1].legend(loc='upper left',fontsize=8,frameon=False)
    axes[1,1].text(.04,.57,'TR-MLA and no-relation lookahead\noverlap in actual SAT call counts',
                   transform=axes[1,1].transAxes,fontsize=9)
    fig.suptitle('Input protection and local candidate scaling: distinct diagnostic boundaries',fontsize=13,y=.99)
    fig.subplots_adjust(hspace=.47,wspace=.25,bottom=.11,top=.92)
    fig.text(.5,.025,'p: answer-error probability; a: relative continuous error. Scaling medians: 9 timing repeats of controlled public states.',ha='center',fontsize=9)
    save(fig,'reviewer_noise_and_efficiency')
    manifest={'input_hashes':{key:{'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for key,path in inputs.items()},
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'figure_names':['reviewer_continuous_parent_adaptation','reviewer_noise_and_efficiency'],
              'no_error_bars':'仅展示原CSV支持的均值/实际重复median，不绘制未估计CI',
              'efficiency_aggregation':'每policy/候选规模，3规格×3重复的9个原decision窗口耗时取median；SAT calls同法'}
    (OUT/'diagnostic_figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'figures':manifest['figure_names'],'output':str(OUT.resolve())},ensure_ascii=False))


if __name__=='__main__':
    main()
