"""Run the real two-utterance demo against the running GPU service."""
import json, time
from pathlib import Path
import httpx
root=Path(__file__).resolve().parents[1]
report_dir=root/'artifacts/verification'
report_dir.mkdir(parents=True,exist_ok=True)
with httpx.Client(base_url='http://127.0.0.1:7860',timeout=900,trust_env=False) as client:
    while not client.get('/api/status').json()['ready']:
        time.sleep(10)
    sid=client.post('/api/sessions').json()['id']
    results=[]
    for name in ('intro','followup'):
        with open(root/f'static/samples/{name}.wav','rb') as audio:
            response=client.post(f'/api/sessions/{sid}/transcribe',files={'file':(name+'.wav',audio,'audio/wav')},data={'bias':8})
        response.raise_for_status()
        result=response.json()
        results.append(result)
        print(json.dumps({k:result[k] for k in ('sequence','baseline','biased','hotwords','updates','elapsed_s')},ensure_ascii=False),flush=True)
    (report_dir/'live-verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    assert '李煦' in [w['word'] for w in results[0]['hotwords']], 'LLM did not resolve explanation to 李煦'
    assert '李煦' in results[1]['applied_hotwords'], 'New hotword not applied next call'
    assert '李旭' in results[1]['baseline'] and '李煦' in results[1]['biased'], 'Sample correction failed'
    print('LIVE END-TO-END PASS',flush=True)
