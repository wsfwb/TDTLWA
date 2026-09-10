from __future__ import annotations
import hashlib, json, re, time
from pathlib import Path

LABELS={'ang','sad','hap','neu','exc','fru'}
DECISIONS={'KEEP_CHEAP','ACCEPT_REASON'}

class TransportFailure(RuntimeError):
    def __init__(self, status, detail, retry_after=None):
        super().__init__(detail); self.status=status; self.detail=detail; self.retry_after=retry_after
class EmptyMessageResponse(RuntimeError):
    def __init__(self, envelope, raw_text=''):
        super().__init__('HTTP 200 response had no extractable message JSON'); self.envelope=envelope
        self.raw_text=_redact(raw_text)
class ToolCallResponse(RuntimeError): pass
class SchemaFailure(RuntimeError):
    def __init__(self, message, envelope=None, raw_text='', http_status=200):
        super().__init__(message)
        self.envelope=envelope or {}
        self.raw_text=_redact(raw_text)
        self.http_status=http_status

def build_v3_payload(prompt: str, schema: dict) -> dict:
    return {
        'model':'gpt-5.6-sol',
        'input':[{'role':'user','content':[{'type':'input_text','text':prompt}]}],
        'reasoning':{'effort':'xhigh'},
        'text':{'format':{'type':'json_schema','name':'e26_reasoner_output','strict':True,'schema':schema}},
    }

def _redact(value):
    text=str(value)
    return re.sub(r'(?i)(authorization|api[_-]?key|cookie)\s*[:=]\s*[^,;\s]+',r'\1=[REDACTED]',text)[:1600]

def _load_auth():
    auth=json.loads(Path('/home/shangdongfang/.codex/auth.json').read_text())
    key=auth.get('OPENAI_API_KEY')
    if not key: raise RuntimeError('direct Responses API authentication unavailable')
    config=Path('/home/shangdongfang/.codex/config.toml').read_text()
    match=re.search(r'base_url\s*=\s*"([^"]+)"',config)
    if not match: raise RuntimeError('direct Responses API endpoint unavailable')
    return key, match.group(1).rstrip('/')

def parse_sse_response(body: str):
    """Return the final Responses object from a complete SSE body without retaining text."""
    event_types=[]; messages=[]; completed=None; output_slots={}; content_slots={}
    for block in body.replace('\r\n','\n').split('\n\n'):
        lines=[line for line in block.split('\n') if line]
        event=None; data=[]
        for line in lines:
            if line.startswith('event:'): event=line[6:].strip()
            elif line.startswith('data:'): data.append(line[5:].lstrip())
        if event and event != 'message': event_types.append(event)
        raw_data='\n'.join(data)
        if not raw_data or raw_data=='[DONE]': continue
        try: payload=json.loads(raw_data)
        except Exception: continue
        candidate=payload.get('response') if isinstance(payload,dict) else None
        if isinstance(candidate,dict) and (event=='response.completed' or payload.get('type')=='response.completed'):
            completed=candidate
        item=payload.get('item') if isinstance(payload,dict) else None
        output_index=payload.get('output_index') if isinstance(payload,dict) else None
        event_type=payload.get('type') if isinstance(payload,dict) else event
        if isinstance(item,dict) and item.get('type')=='message':
            if isinstance(output_index,int): output_slots[output_index]=item
            else: messages.append(item)
        if event_type=='response.content_part.added' and isinstance(output_index,int):
            part=payload.get('part') or {}
            output_slots.setdefault(output_index,{'type':'message','content':[]})
            output_slots[output_index].setdefault('content',[]).append(part)
            content_slots[(output_index,payload.get('content_index',0))]=part
        if event_type=='response.output_text.delta' and isinstance(output_index,int):
            content_index=payload.get('content_index',0); key=(output_index,content_index)
            part=content_slots.get(key)
            if part is None:
                output_slots.setdefault(output_index,{'type':'message','content':[]})
                part={'type':'output_text','text':''}; output_slots[output_index]['content'].append(part);content_slots[key]=part
            part['text']=str(part.get('text',''))+str(payload.get('delta',''))
        if isinstance(payload,dict) and event_type=='response.output_item.done' and isinstance(item,dict): messages.append(item)
    assembled=[output_slots[index] for index in sorted(output_slots)]
    if completed is not None:
        response=completed
        if not response.get('output') and assembled: response['output']=assembled
    elif assembled: response={'output':assembled}
    elif messages: response={'output':messages}
    else: raise ValueError('SSE did not contain a completed response or message item')
    return response,{'sse_event_types':event_types,'sse_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'sse_bytes':len(body.encode())}

def _summary(raw: dict, http_status: int, content_type: str | None, stream_meta: dict | None = None) -> dict:
    items=raw.get('output') or []
    types=[]; content_types=[]
    for item in items:
        if isinstance(item,dict):
            types.append(item.get('type'))
            for content in item.get('content') or []:
                if isinstance(content,dict): content_types.append(content.get('type'))
    return {'http_status':http_status,'content_type':content_type,'response_id':raw.get('id'),'status':raw.get('status'),
            'incomplete_details':raw.get('incomplete_details'),'output_item_types':types,'content_item_types':content_types,
            'usage':raw.get('usage'),'raw_top_level_keys':sorted(raw.keys()),
            'response_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True,default=str).encode()).hexdigest(),
            **(stream_meta or {})}

def _schema_failure(message, envelope, raw_text, http_status):
    envelope=dict(envelope)
    envelope['diagnostic_output_text']=_redact(raw_text)
    envelope['diagnostic_output_text_sha256']=hashlib.sha256(str(raw_text).encode()).hexdigest()
    return SchemaFailure(message, envelope=envelope, raw_text=raw_text, http_status=http_status)

def extract_valid_response(raw: dict, http_status: int = 200, content_type: str | None = None, stream_meta: dict | None = None, raw_text: str | None = None):
    envelope=_summary(raw,http_status,content_type,stream_meta)
    item_types=envelope['output_item_types']
    disallowed=[t for t in item_types if t and t not in {'message','reasoning'}]
    if disallowed: raise ToolCallResponse('tool/function response items: '+','.join(disallowed))
    text=raw.get('output_text')
    if not text:
        for item in raw.get('output') or []:
            if item.get('type') == 'message':
                for content in item.get('content') or []:
                    if content.get('type') in {'output_text','text'}:
                        text=content.get('text'); break
            if text: break
    if not text: raise EmptyMessageResponse(envelope, raw_text or '')
    try: output=json.loads(text)
    except Exception as exc: raise _schema_failure('JSON parse failure', envelope, text, http_status) from exc
    expected={'final_label','decision','confidence','uncertainty','top2','rationale'}
    if set(output)!=expected: raise _schema_failure('schema fields mismatch', envelope, text, http_status)
    if output['final_label'] not in LABELS or output['decision'] not in DECISIONS: raise _schema_failure('schema enum mismatch', envelope, text, http_status)
    if not isinstance(output['confidence'],(int,float)) or not 0<=output['confidence']<=1: raise _schema_failure('confidence out of range', envelope, text, http_status)
    if not isinstance(output['uncertainty'],(int,float)) or not 0<=output['uncertainty']<=1: raise _schema_failure('uncertainty out of range', envelope, text, http_status)
    if not isinstance(output['top2'],list) or len(output['top2'])!=2 or not all(x in LABELS for x in output['top2']): raise _schema_failure('top2 mismatch', envelope, text, http_status)
    if not isinstance(output['rationale'],str) or len(output['rationale'])>80: raise _schema_failure('rationale mismatch', envelope, text, http_status)
    return output,envelope

class DirectResponsesClient:
    def __init__(self, timeout=900):
        import httpx
        self.key,self.base_url=_load_auth(); self.url=self.base_url+'/responses'; self.timeout=timeout; self.httpx=httpx
        self.key_summary={'present':True,'length':len(self.key),'redacted_prefix':self.key[:3]+'***'}
    def create(self, prompt: str, schema: dict):
        payload=build_v3_payload(prompt,schema); started=time.time()
        try:
            with self.httpx.Client(timeout=self.timeout,trust_env=True) as client:
                response=client.post(self.url,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','Accept':'application/json'},json=payload)
        except Exception as exc:
            raise TransportFailure(None,_redact(exc)) from exc
        latency=time.time()-started; content_type=response.headers.get('content-type'); retry_after=response.headers.get('retry-after')
        stream_meta=None
        if (content_type or '').lower().startswith('text/event-stream'):
            try: raw,stream_meta=parse_sse_response(response.text)
            except Exception:
                raw={}; stream_meta={'sse_body_sha256':hashlib.sha256(response.text.encode()).hexdigest(),'sse_bytes':len(response.text.encode()),'sse_event_types':[]}
        else:
            try: raw=response.json()
            except Exception: raw={}
        if response.status_code >= 300:
            raise TransportFailure(response.status_code,_redact(raw or response.text),retry_after=retry_after)
        output,envelope=extract_valid_response(raw,response.status_code,content_type,stream_meta,response.text)
        return output,envelope,latency,payload
