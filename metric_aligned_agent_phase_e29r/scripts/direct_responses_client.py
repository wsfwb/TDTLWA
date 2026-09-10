import hashlib,json,re,time
from pathlib import Path
LABELS={'hap','sad','neu','ang','exc','fru'}
DECISIONS={'KEEP_BASE','ACCEPT_EXISTING_REASON','REVISE'}
SOURCES={'FULL','D3','E27','STAGE_A','I1','I2','I3','REVISED'}
CONSISTENCY={'LOW','MEDIUM','HIGH'}
class TransportFailure(RuntimeError):
    def __init__(self,status,detail,retry_after=None): super().__init__(detail); self.status=status; self.retry_after=retry_after
class ToolCallResponse(RuntimeError): pass
class SchemaFailure(RuntimeError):
    def __init__(self,message,envelope=None,raw_text=''): super().__init__(message); self.envelope=envelope or {}; self.raw_text=raw_text
def _redact(v): return re.sub(r'(?i)(authorization|api[_-]?key|cookie)\s*[:=]\s*[^,;\s]+',r'\1=[REDACTED]',str(v))[:3000]
def schema_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_auth():
    auth=json.loads(Path('/home/shangdongfang/.codex/auth.json').read_text()); key=auth.get('OPENAI_API_KEY')
    cfg=Path('/home/shangdongfang/.codex/config.toml').read_text(); m=re.search(r'base_url\s*=\s*"([^"]+)"',cfg)
    if not key or not m: raise RuntimeError('direct API auth/endpoint unavailable')
    return key,m.group(1).rstrip('/')
def payload(prompt,schema):
    return {'model':'gpt-5.6-sol','input':[{'role':'user','content':[{'type':'input_text','text':prompt}]}],'reasoning':{'effort':'xhigh'},'text':{'format':{'type':'json_schema','name':'e29_adjudicator_output','strict':True,'schema':schema}}}
def parse_sse_response(body):
    event_types=[]; completed=None; outputs={}; messages=[]
    for block in body.replace('\r\n','\n').split('\n\n'):
        event=None; data=[]
        for line in block.split('\n'):
            if line.startswith('event:'): event=line[6:].strip()
            elif line.startswith('data:'): data.append(line[5:].lstrip())
        if event: event_types.append(event)
        raw_data='\n'.join(data)
        if not raw_data or raw_data=='[DONE]': continue
        try: p=json.loads(raw_data)
        except Exception: continue
        candidate=p.get('response') if isinstance(p,dict) else None
        if isinstance(candidate,dict) and (event=='response.completed' or p.get('type')=='response.completed'): completed=candidate
        item=p.get('item') if isinstance(p,dict) else None; oi=p.get('output_index') if isinstance(p,dict) else None
        if isinstance(item,dict) and item.get('type')=='message':
            if isinstance(oi,int): outputs[oi]=item
            else: messages.append(item)
    if completed is not None:
        if not completed.get('output') and outputs: completed['output']=[outputs[i] for i in sorted(outputs)]
        return completed,{'sse_event_types':event_types,'sse_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'sse_bytes':len(body.encode())}
    if outputs: return {'output':[outputs[i] for i in sorted(outputs)]},{'sse_event_types':event_types,'sse_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'sse_bytes':len(body.encode())}
    if messages: return {'output':messages},{'sse_event_types':event_types,'sse_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'sse_bytes':len(body.encode())}
    raise ValueError('SSE did not contain completed response or message')
def extract(raw,status,ctype):
    items=raw.get('output') or []; types=[x.get('type') for x in items if isinstance(x,dict)]
    if any(t not in ('message','reasoning',None) for t in types): raise ToolCallResponse(','.join(str(t) for t in types))
    text=raw.get('output_text')
    if not text:
        for it in items:
            for c in it.get('content',[]) if isinstance(it,dict) else []:
                if c.get('type') in ('output_text','text'): text=c.get('text'); break
            if text: break
    env={'http_status':status,'content_type':ctype,'output_item_types':types,'content_item_types':[c.get('type') for i in items for c in (i.get('content') or []) if isinstance(c,dict)],'usage':raw.get('usage'),'response_status':raw.get('status'),'incomplete_details':raw.get('incomplete_details'),'response_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True,default=str).encode()).hexdigest()}
    if not text: raise SchemaFailure('empty_response',env,'')
    try: out=json.loads(text)
    except Exception as e: raise SchemaFailure('json_parse_failure',env,_redact(text)) from e
    required={'final_label','decision','preferred_source','confidence','uncertainty','evidence_consistency','short_reason'}
    if set(out)!=required or out['final_label'] not in LABELS or out['decision'] not in DECISIONS or out['preferred_source'] not in SOURCES or out['evidence_consistency'] not in CONSISTENCY or not (0<=float(out['confidence'])<=1) or not (0<=float(out['uncertainty'])<=1) or not isinstance(out['short_reason'],str) or not (1<=len(out['short_reason'])<=800):
        raise SchemaFailure('schema_validation_failure',env,_redact(text))
    return out,env
class DirectResponsesClient:
    def __init__(self,timeout=900):
        import httpx
        self.key,self.base_url=load_auth(); self.url=self.base_url+'/responses'; self.timeout=timeout; self.httpx=httpx
    def create(self,prompt,schema):
        payload_obj=payload(prompt,schema); start=time.time()
        try:
            with self.httpx.Client(timeout=self.timeout,trust_env=True) as c: r=c.post(self.url,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','Accept':'application/json'},json=payload_obj)
        except Exception as e: raise TransportFailure(None,_redact(e))
        ctype=r.headers.get('content-type')
        stream_meta={}
        if (ctype or '').lower().startswith('text/event-stream'):
            try: raw,stream_meta=parse_sse_response(r.text)
            except Exception: raw={}
        else:
            try: raw=r.json()
            except Exception: raw={}
        if r.status_code>=300: raise TransportFailure(r.status_code,_redact(raw or r.text),r.headers.get('retry-after'))
        out,env=extract(raw,r.status_code,ctype); env.update(stream_meta); env['latency_sec']=time.time()-start
        return out,env,payload_obj
