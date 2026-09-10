from __future__ import annotations

ALLOW = {
    'I1': {'canonical_id','dialogue_id','speaker','turn_index','dialogue_length','current_utterance','preceding_turns','serialized_context','truncated_fixed_head_tail'},
    'I2': {'canonical_id','full_prediction','residual_prediction','full_probability','residual_probability','full_confidence','residual_confidence','full_entropy','residual_entropy','full_margin','residual_margin','agreement','confidence_delta','entropy_delta','margin_delta','label_names'},
    'I3': {'canonical_id','modality_summary_vector','modality_summary_dimensions','missing_modality_indicator','raw_audio_available','raw_video_available'},
}
FORBIDDEN = {'gold','label','session5_label','correctness','benefit','harm','neutral','test_wf1','test_metric','future_outcome','r1_prediction','r1_confidence','oracle','winner'}

def sanitize_input(interface: str, row: dict) -> dict:
    keys = {str(k).lower() for k in row}
    blocked = keys & FORBIDDEN
    if blocked: raise ValueError('forbidden input fields: ' + ','.join(sorted(blocked)))
    extra = set(row) - ALLOW[interface]
    if extra: raise ValueError('unapproved input fields: ' + ','.join(sorted(extra)))
    return {k: row[k] for k in row if k in ALLOW[interface]}

def prompt_view(interface: str, context: dict, evidence: dict | None = None, modality: dict | None = None) -> dict:
    # IDs remain local mapping keys and are deliberately excluded from remotely sent content.
    body = {'interface': interface, 'context': {k:v for k,v in sanitize_input('I1', context).items() if k != 'canonical_id'}}
    if interface in {'I2','I3'}:
        body['structured_evidence'] = {k:v for k,v in sanitize_input('I2', evidence or {}).items() if k != 'canonical_id'}
    if interface == 'I3':
        body['modality_summary'] = {k:v for k,v in sanitize_input('I3', modality or {}).items() if k != 'canonical_id'}
    return body
