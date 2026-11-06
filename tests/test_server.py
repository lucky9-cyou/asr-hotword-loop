"""API contract tests with injected engines; these do not assert model quality.

Real model smoke tests are separate. Here we verify ordering, validation and
failure isolation at the HTTP boundary without loading GPU weights.
"""
import asyncio
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server

INTRO = '我叫李旭，是春风和煦的煦。'
CANDIDATE = {'word': '李煦', 'evidence': '春风和煦的煦', 'reason': '说话人解释了名字用字'}


class FakeASR:
    def __init__(self):
        self.transcripts = []
        self.calls = []
        self.failure = None

    def status(self):
        return {'loaded': True}

    def transcribe(self, path, hotwords, bias):
        assert Path(path).is_file(), 'The upload must exist during inference.'
        self.calls.append({'words': list(hotwords), 'bias': bias, 'path': path})
        if self.failure:
            raise RuntimeError(self.failure)
        if self.transcripts:
            baseline = self.transcripts.pop(0)
        elif len(self.calls) == 1:
            baseline = INTRO
        else:
            baseline = '请把文件发给李旭。'
        return {
            'baseline': baseline,
            'biased': baseline.replace('李旭', '李煦') if '李煦' in hotwords else baseline,
            'probabilities': [],
            'model': 'test-double',
        }


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.candidates = [CANDIDATE.copy()]
        self.failure = None

    def status(self):
        return {'loaded': True}

    def close(self):
        pass

    def extract(self, text, existing):
        self.calls.append({'text': text, 'existing': list(existing)})
        if self.failure:
            raise RuntimeError(self.failure)
        return {'candidates': self.candidates, 'raw_output': 'test-double'}


@pytest.fixture
def api(monkeypatch, tmp_path):
    asr, llm = FakeASR(), FakeLLM()
    monkeypatch.setattr(server, 'engines', {'asr': asr, 'llm': llm})
    monkeypatch.setattr(server, 'sessions', {})
    monkeypatch.setattr(server, 'engine_errors', {})
    monkeypatch.setattr(server, 'inference_lock', asyncio.Lock())
    (tmp_path / 'tmp').mkdir()
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    with TestClient(server.app) as client:
        sid = client.post('/api/sessions').json()['id']
        yield client, sid, asr, llm
    assert not list((tmp_path / 'tmp').iterdir()), 'Uploads should be removed after every response.'


def wav(duration=.3, rate=16000):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(duration * rate), dtype=np.float32), rate, format='WAV')
    return buffer.getvalue()


def chunk(client, sid, duration=.3, bias=3):
    return client.post(f'/api/sessions/{sid}/chunk',
                       files={'file': ('chunk.wav', wav(duration), 'audio/wav')},
                       data={'bias': str(bias)})


def test_learning_changes_next_chunk_not_current_chunk(api):
    client, sid, asr, llm = api
    first = chunk(client, sid)
    assert first.status_code == 200, first.text
    first = first.json()
    assert first['sequence'] == 1
    assert first['applied_hotwords'] == []
    assert first['baseline'] == first['biased'] == INTRO
    assert [w['word'] for w in first['hotwords']] == ['李煦']
    assert first['updates'][0]['action'] == 'add'

    second = chunk(client, sid).json()
    assert second['sequence'] == 2
    assert second['applied_hotwords'] == ['李煦']
    assert second['baseline'] == '请把文件发给李旭。'
    assert second['biased'] == '请把文件发给李煦。'
    assert asr.calls[0]['words'] == []
    assert asr.calls[1]['words'] == ['李煦']
    assert llm.calls[1]['existing'] == ['李煦']
    assert INTRO in llm.calls[1]['text']
    assert '请把文件发给李煦。' in llm.calls[1]['text']
    assert second['updates'] == [], 'Repeated evidence must not appear as a new addition.'
    session = client.get(f'/api/sessions/{sid}').json()
    assert session['sequence'] == 2
    assert len(session['history']) == 2
    assert all(not Path(c['path']).exists() for c in asr.calls)


def test_text_learning_delete_and_session_isolation(api):
    client, sid, asr, llm = api
    learned = client.post(f'/api/sessions/{sid}/learn', json={'text': INTRO})
    assert learned.status_code == 200
    assert learned.json()['hotwords'][0]['word'] == '李煦'
    assert asr.calls == [], 'Text injection must skip acoustic inference.'
    other = client.post('/api/sessions').json()
    assert other['hotwords'] == []
    assert other['sequence'] == 0
    deleted = client.delete(f'/api/sessions/{sid}/hotwords/李煦')
    assert deleted.status_code == 200
    assert deleted.json()['hotwords'] == []
    llm.candidates = []
    next_chunk = chunk(client, sid).json()
    assert next_chunk['applied_hotwords'] == []
    assert next_chunk['hotwords'] == []
    assert client.get(f"/api/sessions/{other['id']}").json()['hotwords'] == []


@pytest.mark.parametrize('duration', [.05, 30.1])
def test_invalid_duration_rejected_before_model_invocation(api, duration):
    client, sid, asr, llm = api
    response = chunk(client, sid, duration=duration)
    assert response.status_code == 422
    assert asr.calls == []
    assert llm.calls == []
    assert client.get(f'/api/sessions/{sid}').json()['sequence'] == 0


@pytest.mark.parametrize('bias', [-.1, 8.1, 'nan', 'inf', '-inf'])
def test_invalid_bias_rejected_before_model_invocation(api, bias):
    client, sid, asr, llm = api
    response = chunk(client, sid, bias=bias)
    assert response.status_code == 422
    assert asr.calls == []
    assert llm.calls == []


def test_non_audio_upload_rejected(api):
    client, sid, asr, _ = api
    response = client.post(f'/api/sessions/{sid}/chunk',
                           files={'file': ('invalid.wav', b'not audio', 'audio/wav')})
    assert response.status_code == 422
    assert not asr.calls


def test_unsubstantiated_candidates_rejected(api):
    client, sid, _, llm = api
    llm.candidates = [
        {'word': '王旭'},
        {'word': '王煦', 'evidence': ''},
        {'word': '陈煦', 'evidence': '转写中不存在的说明'},
        {'word': '超' * 33, 'evidence': '春风和煦的煦'},
        CANDIDATE.copy(),
    ]
    response = client.post(f'/api/sessions/{sid}/learn', json={'text': INTRO})
    assert response.status_code == 200, response.text
    assert [w['word'] for w in response.json()['hotwords']] == ['李煦']


def test_llm_failure_preserves_transcript_and_prior_hotwords(api):
    client, sid, asr, llm = api
    assert client.post(f'/api/sessions/{sid}/learn', json={'text': INTRO}).status_code == 200
    llm.failure = 'controlled test failure'
    response = chunk(client, sid)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['baseline'] == INTRO
    assert data['biased'] == INTRO.replace('李旭', '李煦')
    assert data['applied_hotwords'] == ['李煦']
    assert [w['word'] for w in data['hotwords']] == ['李煦']
    assert data['updates'] == []
    assert data['llm']['error'] == 'controlled test failure'
    saved = client.get(f'/api/sessions/{sid}').json()
    assert saved['history'][0]['biased'] == data['biased']
    assert saved['hotwords'][0]['word'] == '李煦'


def test_asr_failure_returns_error_and_cleans_upload(api):
    client, sid, asr, llm = api
    asr.failure = 'controlled ASR failure'
    response = chunk(client, sid)
    assert response.status_code == 503
    assert 'controlled ASR failure' in response.json()['detail']
    assert llm.calls == []
    assert not Path(asr.calls[0]['path']).exists()


def test_unknown_session_returns_404(api):
    client, _, _, _ = api
    assert client.get('/api/sessions/unknown').status_code == 404
    assert client.post('/api/sessions/unknown/learn', json={'text': INTRO}).status_code == 404



def transcribe(client, sid, duration=.3, bias=8, route='transcribe'):
    return client.post(f'/api/sessions/{sid}/{route}',
                       files={'file': ('utterance.wav', wav(duration), 'audio/wav')},
                       data={'bias': str(bias)})


def test_manual_calls_keep_learning_and_apply_updates_only_next_call(api):
    client, sid, asr, llm = api
    second_text = '请把文件发给李旭。我叫王丽，是茉莉的莉。'
    asr.transcripts = [INTRO, second_text, '请联系李旭和王莉。']
    first_response = transcribe(client, sid)
    assert first_response.status_code == 200, first_response.text
    first = first_response.json()
    assert first['action'] == 'transcribe'
    assert first['sequence'] == 1
    assert first['applied_hotwords'] == []
    assert [w['word'] for w in first['hotwords']] == ['李煦']
    assert first['baseline'] == first['biased'] == INTRO
    assert first['probabilities'] == [], 'HTTP must not invent probability traces.'
    assert len(llm.calls) == 1

    llm.candidates = [{'word': '王莉', 'evidence': '我叫王丽，是茉莉的莉', 'reason': '茉莉确定莉'}]
    second_response = transcribe(client, sid)
    assert second_response.status_code == 200, second_response.text
    second = second_response.json()
    assert second['sequence'] == 2
    assert second['applied_hotwords'] == ['李煦']
    assert second['biased'] == second_text.replace('李旭', '李煦')
    assert [w['word'] for w in second['hotwords']] == ['李煦', '王莉']
    assert second['updates'][0]['word'] == '王莉'
    assert llm.calls[1]['existing'] == ['李煦']

    llm.candidates = []
    third_response = transcribe(client, sid)
    assert third_response.status_code == 200, third_response.text
    third = third_response.json()
    assert third['sequence'] == 3
    assert third['applied_hotwords'] == ['李煦', '王莉']
    assert third['updates'] == []
    assert len(asr.calls) == 3, 'Each complete recording causes one engine invocation.'
    assert len(llm.calls) == 3, 'Every nonempty audio result continues learning.'
    assert asr.calls[0]['words'] == []
    assert asr.calls[1]['words'] == ['李煦']
    assert asr.calls[2]['words'] == ['李煦', '王莉']
    assert INTRO in llm.calls[2]['text']
    assert '我叫王丽，是茉莉的莉' in llm.calls[2]['text']
    assert client.get(f'/api/sessions/{sid}').json()['sequence'] == 3


@pytest.mark.parametrize('route', ['transcribe', 'chunk'])
def test_complete_audio_limit_is_enforced_without_splitting(api, route):
    client, sid, asr, llm = api
    response = transcribe(client, sid, duration=30.1, route=route)
    assert response.status_code == 422
    assert '不会自动切块' in response.json()['detail']
    assert asr.calls == llm.calls == []


def test_status_describes_manual_calls_and_old_mode_routes_are_absent(api):
    client, sid, _, _ = api
    status = client.get('/api/status').json()
    assert status['workflow'] == 'manual_calls'
    assert status['streaming'] is False
    for removed_mode in ('extract', 'recognize'):
        response = client.post(f'/api/sessions/{sid}/{removed_mode}')
        assert response.status_code == 404
