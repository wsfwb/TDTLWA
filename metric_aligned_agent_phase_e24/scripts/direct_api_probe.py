#!/usr/bin/env python3
import json,sys,time,traceback
from pathlib import Path
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN))
from scripts.direct_responses_client import DirectResponsesClient,extract_valid_output,build_request_payload
from scripts.run_direct_reasoner import build_prompt,load_rows
from scripts.cache_store import atomic_write_json
from scripts.retry_policy import is_retryable,backoff_seconds,classify_error,retry_after_from_exception
def main():
 row=load_rows('canary_I1_train.jsonl')[0];schema=json.loads((RUN/'protocol/E24_OUTPUT_SCHEMA.json').read_text());prompt=build_prompt('I1',row)
 payload=build_request_payload(prompt,schema);audit={'keys':sorted(payload),'has_tools':'tools' in payload,'has_previous_response':'previous_response_id' in payload,'model':payload['model'],'reasoning':payload['reasoning'],'temperature':payload['temperature'],'store':payload['store']}
 atomic_write_json(RUN/'metadata/request_payload_audit.json',audit)
 started=time.time()
 events=[];client=DirectResponsesClient();output=raw=None;last=None
 for attempt in range(1,7):
  one=time.time()
  try:
   response=client.create(prompt,schema);output,raw=extract_valid_output(response);events.append({'attempt':attempt,'status':'success','latency_sec':time.time()-one});last=None;break
  except Exception as e:
   last=e;events.append({'attempt':attempt,'status':'failure','error_class':classify_error(e),'error':str(e),'latency_sec':time.time()-one})
   if attempt==6 or not is_retryable(e):break
   time.sleep(backoff_seconds(attempt,retry_after_from_exception(e),0))
 if last is None and output is not None:
  result={'status':'passed','canonical_id':row['canonical_id'],'model':'gpt-5.6-sol','reasoning_effort':'xhigh','latency_sec':time.time()-started,'valid_json':True,'tool_calls':0,'response_id':raw.get('id'),'usage':raw.get('usage'),'output':output,'endpoint_type':'direct Responses API','key_summary':client.key_summary,'events':events};rc=0
 else:
  result={'status':'transport_blocked' if last and is_retryable(last) else 'api_incompatible','canonical_id':row['canonical_id'],'model':'gpt-5.6-sol','reasoning_effort':'xhigh','latency_sec':time.time()-started,'valid_json':False,'tool_calls':None,'error_class':type(last).__name__ if last else None,'error':str(last) if last else 'empty response','traceback':traceback.format_exc()[-4000:],'events':events};rc=2
 atomic_write_json(RUN/'metadata/reasoner_api_status.json',result);print(json.dumps(result,ensure_ascii=False));return rc
if __name__=='__main__':raise SystemExit(main())
