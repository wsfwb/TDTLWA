from __future__ import annotations
import random,re
BACKOFF=(15,30,60,120,300)
RETRY_PATTERNS=(r'upstream_error',r'HTTP\s+(?:429|500|502|503)',r'connection reset',r'connection aborted',r'remote end closed',r'timeout',r'incomplete response',r'empty response',r'schema parse')
def is_retryable(exc:BaseException)->bool:
    if isinstance(exc,TimeoutError): return True
    text=str(exc).lower()
    return any(re.search(p,text,re.I) for p in RETRY_PATTERNS)
def backoff_seconds(attempt:int,retry_after:float|None,jitter:float=0.0)->float:
    if retry_after is not None: return float(retry_after)
    base=BACKOFF[min(max(attempt,1)-1,len(BACKOFF)-1)]
    return base + (random.uniform(0,jitter) if jitter else 0)
def retry_after_from_exception(exc:BaseException):
    response=getattr(exc,'response',None); headers=getattr(response,'headers',None)
    if headers:
        value=headers.get('retry-after') or headers.get('Retry-After')
        if value:
            try:return float(value)
            except ValueError:return None
    return None
def classify_error(exc:BaseException)->str:
    t=str(exc).lower()
    for c in ('429','500','502','503'):
        if re.search(rf'\b{c}\b',t): return 'http_'+c
    if 'upstream' in t:return 'generic_upstream_error'
    if 'timeout' in t:return 'timeout'
    if 'connection' in t or 'remote end' in t:return 'connection_error'
    if 'schema' in t or 'parse' in t:return 'parse_failure'
    return 'nonretryable_error'
