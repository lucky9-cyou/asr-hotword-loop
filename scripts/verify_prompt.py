"""Real local-LLM regression checks for noisy explanations (not unit tests)."""
import json
import time
from pathlib import Path
import httpx

CASES = [
    ('你好，你好，我叫鱼明洞。鱼是年年有鱼的鱼。', ['余明洞']),
    ('我叫李旭，姓李。旭是春风和煦的旭。', ['李煦']),
    ('我叫王丽，是茉莉的莉。', ['王莉']),
    ('我姓鱼，鱼虾的鱼，不是多余的余，名字叫明洞。', ['鱼明洞']),
    ('我叫余明洞，余是年年有余的余。', ['余明洞']),
    ('你好你好，今天的天气不错。', []),
    ('我叫张明洞。', ['张明洞']),
]
if __name__ == '__main__':
    results=[]
    with httpx.Client(base_url='http://127.0.0.1:7860',timeout=900,trust_env=False) as client:
        deadline=time.monotonic()+900
        while not client.get('/api/status').json()['ready']:
            if time.monotonic()>deadline:
                raise TimeoutError('Model warmup did not complete')
            time.sleep(10)
        for text, expected in CASES:
            response=client.post('/api/sessions')
            response.raise_for_status()
            sid=response.json()['id']
            response=client.post(f'/api/sessions/{sid}/learn',json={'text':text})
            response.raise_for_status()
            data=response.json()
            words=[w['word'] for w in data['hotwords']]
            passed=words==expected
            result=dict(text=text,expected=expected,words=words,passed=passed,result=data)
            results.append(result)
            print(json.dumps({k:result[k] for k in ('text','expected','words','passed')},ensure_ascii=False),flush=True)
    report_dir=Path(__file__).resolve().parents[1]/'artifacts/verification'
    report_dir.mkdir(parents=True,exist_ok=True)
    (report_dir/'prompt-verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    assert all(x['passed'] for x in results), 'See artifacts/verification/prompt-verification.json for failures'
    print('PROMPT REGRESSIONS PASS',flush=True)
