from __future__ import annotations
def may_start_session5(state: dict) -> bool:
    return bool(state.get('probe',{}).get('passed') and state.get('canary',{}).get('passed'))
