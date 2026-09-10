from __future__ import annotations
import json,re
from pathlib import Path
from openai import OpenAI
LABELS={'ang','sad','hap','neu','exc','fru'};DECISIONS={'KEEP_CHEAP','ACCEPT_REASON'}
def build_request_payload(prompt,schema):
    return {'model':'gpt-5.6-sol','reasoning':{'effort':'xhigh'},'temperature':0.0,'store':False,'input':[{'role':'user','content':[{'type':'input_text','text':prompt}]}],'text':{'format':{'type':'json_schema','name':'e24_reasoner_output','strict':True,'schema':schema}}}
def _obj(x):
    if isinstance(x,dict):return x
    if hasattr(x,'model_dump'):return x.model_dump()
    if hasattr(x,'to_dict'):return x.to_dict()
    raise ValueError('uninspectable response')
def extract_valid_output(response):
    r=_obj(response);items=r.get('output') or []
    for it in items:
        t=(it.get('type') if isinstance(it,dict) else getattr(it,'type',None))
        if t and t not in ('message','reasoning'):raise ValueError('tool/function call in response: '+str(t))
    text=r.get('output_text')
    if not text:
        for it in items:
            if (it.get('type') if isinstance(it,dict) else None)=='message':
                for c in it.get('content',[]):
                    if c.get('type')=='output_text':text=c.get('text');break
    if not text:raise ValueError('empty response body')
    try:o=json.loads(text)
    except Exception as e:raise ValueError('schema parse failure') from e
    if set(o)!={'final_label','decision','confidence','uncertainty','top2','rationale'}:raise ValueError('schema parse failure: fields')
    if o['final_label'] not in LABELS or o['decision'] not in DECISIONS:raise ValueError('schema parse failure: enum')
    if not (0<=float(o['confidence'])<=1 and 0<=float(o['uncertainty'])<=1):raise ValueError('schema parse failure: range')
    if not isinstance(o['top2'],list) or len(o['top2'])!=2 or not all(x in LABELS for x in o['top2']):raise ValueError('schema parse failure: top2')
    if not isinstance(o['rationale'],str) or len(o['rationale'])>160:raise ValueError('schema parse failure: rationale')
    return o,r
def load_auth():
    auth=json.loads(Path('/home/shangdongfang/.codex/auth.json').read_text());key=auth.get('OPENAI_API_KEY')
    if not key:raise RuntimeError('direct Responses API authentication unavailable')
    cfg=Path('/home/shangdongfang/.codex/config.toml').read_text();m=re.search(r'base_url\s*=\s*"([^"]+)"',cfg);base=m.group(1) if m else None
    if not base:raise RuntimeError('direct Responses API endpoint unavailable')
    return key,base
class DirectResponsesClient:
    def __init__(self,timeout=900):
        key,base=load_auth();self.base_url=base;self.key_summary={'present':True,'length':len(key),'prefix':key[:3]+'***'}
        self.client=OpenAI(api_key=key,base_url=base,timeout=timeout,max_retries=0)
    def create(self,prompt,schema):return self.client.responses.create(**build_request_payload(prompt,schema))
