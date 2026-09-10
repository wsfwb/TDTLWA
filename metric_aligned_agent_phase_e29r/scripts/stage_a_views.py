from collections import Counter
def margin(row,src): return float(row[f'{src.lower()}_confidence'])-float(row[f'{src.lower()}_uncertainty'])
def stable_rank(rows,score_fn): return sorted(rows,key=lambda r:(-float(score_fn(r)),str(r['canonical_id'])))
def _lab(row,src): return int(row[f'{src.lower()}_reason_prediction'])
def _acc(row,src): return row[f'{src.lower()}_decision']=='ACCEPT_REASON'
def _item(row,label,supporters):
 ms=[margin(row,s) for s in supporters]
 return {'canonical_id':row['canonical_id'],'proposal_label':int(label),'supporters':list(supporters),'support':len(supporters),'score':max(ms),'mean_margin':sum(ms)/len(ms),'min_margin':min(ms),'max_margin':max(ms),'full_label':int(row['full_prediction']),'d3_label':int(row['residual_prediction'])}
def build_views(rows):
 names=('I1','I2','I3'); out={n:[] for n in ('V_ANY','V_I3I1','V_I3I2','V_UNANIMOUS','V_MAJORITY','V_BEST_SUPPORTED')}
 for r in rows:
  labels={s:_lab(r,s) for s in names}; l3=labels['I3']; supports=[s for s in names if labels[s]==l3]
  if _acc(r,'I3') and len(supports)>=2 and l3!=int(r['full_prediction']): out['V_ANY'].append(_item(r,l3,supports))
  if _acc(r,'I3') and _acc(r,'I1') and l3==labels['I1'] and l3!=int(r['full_prediction']): out['V_I3I1'].append(_item(r,l3,['I3','I1']))
  if _acc(r,'I3') and _acc(r,'I2') and l3==labels['I2'] and l3!=int(r['full_prediction']): out['V_I3I2'].append(_item(r,l3,['I3','I2']))
  cnt=Counter(labels.values()); top=cnt.most_common(1)[0]
  if top[1]==3 and top[0]!=int(r['full_prediction']): out['V_UNANIMOUS'].append(_item(r,top[0],names))
  if top[1]>=2 and top[0]!=int(r['full_prediction']): out['V_MAJORITY'].append(_item(r,top[0],[s for s in names if labels[s]==top[0]]))
  choices=[(lab,[s for s in names if labels[s]==lab]) for lab,n in cnt.items() if n>=2]
  if choices:
   lab,supp=max(choices,key=lambda z:(sum(margin(r,s) for s in z[1])/len(z[1]),z[0]))
   if lab!=int(r['full_prediction']): out['V_BEST_SUPPORTED'].append(_item(r,lab,supp))
 for k in out: out[k]=stable_rank(out[k],lambda x:x['score'])
 return out
def apply_prefix(view,k): return view[:min(int(k),len(view))]
