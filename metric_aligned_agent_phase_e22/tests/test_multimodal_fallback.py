from pathlib import Path
import json
RUN=Path(__file__).resolve().parents[1]
def main():
 x=json.loads((RUN/'metadata/multimodal_status.json').read_text()); assert x['raw_audio_available'] is False and x['raw_video_available'] is False and x['summary_vectors_available'] is True; print('test_multimodal_fallback: PASS')
if __name__=='__main__':main()
