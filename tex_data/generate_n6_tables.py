# Re-generate N=6 grouped/ordered LaTeX tables
# Usage: python3 generate_n6_tables.py
import pandas as pd
from pathlib import Path

mx=pd.read_csv('grid_summary_maximinmaximax_v5.csv')
mmr=pd.read_csv('grid_summary_minimax_regret_v3.csv')
df=pd.concat([mx, mmr], ignore_index=True)

n=6
out_dir=Path('generated_tables_n6')
out_dir.mkdir(parents=True, exist_ok=True)

method_groups=[
 ('クリスプ推定法', ['EV','GM']),
 ('基本区間推定法（単一モデル）', ['WMIN','WWMIN','DMIN']),
 ('中心固定型（単一モデル）', ['E-WMIN','E-WWMIN','E-DMIN','G-WMIN','G-WWMIN','G-DMIN']),
 ('アンサンブル型・区間包統合（MMR系）', ['MMRW','MMRWW','MMRD','MMRwc','E-MMRW','E-MMRWW','E-MMRD','G-MMRW','G-MMRWW','G-MMRD']),
 ('アンサンブル型・平均統合（AMR系）', ['AMRW','AMRWW','AMRD','AMRwc','E-AMRW','E-AMRWW','E-AMRD','G-AMRW','G-AMRWW','G-AMRD']),
 ('弱モデル内中心推定型・区間包統合', ['eMMRw','eMMRwc','eMMRd','eMMRdc','gMMRw','gMMRwc','gMMRd','gMMRdc']),
 ('弱モデル内中心推定型・平均統合', ['eAMRw','eAMRwc','eAMRd','eAMRdc','gAMRw','gAMRwc','gAMRd','gAMRdc']),
]
ordered_methods=[m for _, ms in method_groups for m in ms]

rules=['maximin','maximax','minimax_regret']
utilities=['u1','u2']
metrics=[('recall','sum_recall','再現数平均'),('precision','sum_precision','正解数平均'),('f1','sum_f1','F1平均')]

def fmt_int(x: float)->str:
    return f"{int(round(x)):,}"

def underline(val_str: str, rank: int)->str:
    if rank==1:
        return f"\uuline{{{val_str}}}"
    if rank==2:
        return f"\dashuline{{{val_str}}}"
    if rank==3:
        return f"\uline{{{val_str}}}"
    return val_str

include_map={
 'maximin': out_dir/'include_n6_maximin.tex',
 'maximax': out_dir/'include_n6_maximax.tex',
 'minimax_regret': out_dir/'include_n6_minimax_regret.tex',
}
for p in include_map.values():
    p.write_text('', encoding='utf-8')
(out_dir/'include_n6_tables.tex').write_text('', encoding='utf-8')

dfn=df[df['N']==n].copy()

for rule in rules:
  for utility in utilities:
    for metric_key, col, cap_prefix in metrics:
      sub=dfn[(dfn['rule']==rule)&(dfn['utility']==utility)].copy()
      pivot=sub.pivot_table(index='method', columns='tw', values=col, aggfunc='first')
      pivot=pivot.reindex(index=ordered_methods, columns=['A','B','C','D','E'])
      pivot['Total']=pivot.mean(axis=1)

      ranks={}
      for c in ['A','B','C','D','E','Total']:
        s=pivot[c]
        order=sorted(range(len(s)), key=lambda i:(-s.iloc[i], i))
        top_idx=order[:3]
        ranks[c]={pivot.index[i]:j+1 for j,i in enumerate(top_idx)}

      rule_j={'maximin':'マキシミン基準','maximax':'マキシマックス基準','minimax_regret':'ミニマックス・リグレット基準'}[rule]
      util_j={'u1':'効用値順序付け容易な場合','u2':'効用値順序付け困難な場合'}[utility]
      caption=f"{cap_prefix} : {util_j}（{rule_j}，$n={n}$）"

      fname=f"tab_n6_{rule}_{utility}_{metric_key}.tex"
      fpath=out_dir/fname

      lines=[]
      lines.append('\begin{table}[H]')
      lines.append('  \centering')
      lines.append('  \footnotesize')
      lines.append('  \setlength{\tabcolsep}{4.0pt}')
      lines.append('  \renewcommand{\arraystretch}{0.88}')
      lines.append(f'  \caption{{{caption}}}')
      lines.append('  \begin{adjustbox}{max width=\textwidth}')
      lines.append('  \begin{tabular}{l|r|r|r|r|r|r}\hline')
      lines.append('  手法名 & A & B & C & D & E & Total \\ \hline\hline')

      for gname, ms in method_groups:
        lines.append(f'  \multicolumn{{7}}{{l}}{{\textbf{{{gname}}}}} \\ \hline')
        for m in ms:
          row=[m]
          for c in ['A','B','C','D','E','Total']:
            v=pivot.loc[m,c]
            s=fmt_int(v)
            r=ranks[c].get(m, None)
            row.append(underline(s,r) if r else s)
          lines.append('  ' + ' & '.join(row) + ' \\')
        lines.append('  \hline')

      lines.append('  \end{tabular}')
      lines.append('  \end{adjustbox}')
      lines.append('\end{table}')
      fpath.write_text('
'.join(lines)+'
', encoding='utf-8')

      for inc in [include_map[rule], out_dir/'include_n6_tables.tex']:
        with inc.open('a', encoding='utf-8') as w:
          w.write(f"\input{{generated_tables_n6/{fname}}}
")

print('done')
