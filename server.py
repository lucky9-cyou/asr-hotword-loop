"""Whole-recording ASR calls that use current hotwords and learn for the next call."""
import asyncio
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="ECHO · Explanations as Context for Hotword Optimization")
app.mount('/static', StaticFiles(directory=ROOT/'static'), name='static')
ASR_MODEL = os.getenv('ASR_MODEL', '/workspace/model/Qwen3-ASR-0.6B')
LLM_MODEL = os.getenv('LLM_MODEL', '/workspace/model/Qwen3.8-27B')
engines = {}
engine_errors = {}
inference_lock = asyncio.Lock()

@dataclass
class Session:
    id: str
    hotwords: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    sequence: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    listeners: list = field(default_factory=list)
    events: list = field(default_factory=list)
    def emit(self, stage, **data):
        event = dict(stage=stage, time=time.time(), **data)
        self.events.append(event)
        self.events = self.events[-100:]
        for queue in self.listeners:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)
    def state(self):
        return dict(id=self.id, hotwords=list(self.hotwords.values()), sequence=self.sequence, history=self.history[-30:])

sessions = {}

def get_session(sid):
    if sid not in sessions:
        raise HTTPException(404, '会话不存在，请新建会话')
    return sessions[sid]

def get_asr():
    if 'asr' not in engines:
        from asr_engine import ASREngine
        engines['asr'] = ASREngine()
    return engines['asr']

def get_llm():
    if 'llm' not in engines:
        from llm_engine import HotwordExtractor
        engines['llm'] = HotwordExtractor()
    return engines['llm']

@app.get('/')
async def index():
    return FileResponse(ROOT/'static/index.html')

@app.get('/api/status')
async def status():
    return dict(asr_model=ASR_MODEL, llm_model=LLM_MODEL,
                device=f"ASR {os.getenv('ASR_DEVICE','cuda:0')} · LLM {os.getenv('LLM_DEVICE','cuda:1')}",
                ready=all(key in engines and engines[key].status().get('loaded',False) for key in ('asr','llm')),
                errors=engine_errors, mode='local_models', workflow='manual_calls', streaming=False, max_audio_seconds=30)

@app.post('/api/warmup')
async def warmup():
    async with inference_lock:
        for key, factory in [('asr',get_asr),('llm',get_llm)]:
            try:
                engine = await asyncio.to_thread(factory)
                loader = getattr(engine, 'load', None)
                if loader:
                    await asyncio.to_thread(loader)
                engine_errors.pop(key, None)
            except Exception as exc:
                engine_errors[key] = str(exc)
                raise HTTPException(503, f'{key} 加载失败：{exc}')
    return await status()

@app.post('/api/sessions')
async def create_session():
    sid=uuid.uuid4().hex[:16]
    sessions[sid]=Session(sid)
    return sessions[sid].state()

@app.get('/api/sessions/{sid}')
async def session_state(sid: str):
    return get_session(sid).state()

@app.delete('/api/sessions/{sid}/hotwords/{word}')
async def delete_word(sid: str, word: str):
    session=get_session(sid)
    async with session.lock:
        session.hotwords.pop(word,None)
        session.emit('hotwords', hotwords=list(session.hotwords.values()))
    return session.state()

async def learn(session, text):
    session.emit('llm_start', text=text)
    result=await asyncio.to_thread(get_llm().extract, text, list(session.hotwords))
    updates=[]
    for item in result.get('candidates',[]):
        word=str(item.get('word','')).strip()
        evidence=str(item.get('evidence','')).strip()
        if not word or len(word)>32 or not evidence or evidence not in text:
            continue
        old=session.hotwords.get(word)
        record=dict(word=word, evidence=evidence, reason=item.get('reason',''),
                    updated_at=time.time(), sequence=session.sequence, count=(old or {}).get('count',0)+1)
        session.hotwords[word]=record
        if not old:
            updates.append(dict(action='add',**record))
    while len(session.hotwords)>32:
        expired=next(iter(session.hotwords))
        session.hotwords.pop(expired)
        updates.append(dict(action='evict',word=expired))
    session.emit('hotwords', hotwords=list(session.hotwords.values()), updates=updates, llm=result)
    return dict(hotwords=list(session.hotwords.values()), updates=updates, llm=result)

class LearnRequest(BaseModel):
    text: str = Field(min_length=1,max_length=6000)

@app.post('/api/sessions/{sid}/learn')
async def learn_text(sid: str, request: LearnRequest):
    session=get_session(sid)
    async with session.lock, inference_lock:
        try:
            return await learn(session, request.text)
        except Exception as exc:
            session.emit('error', error=str(exc))
            raise HTTPException(503, f'LLM 提取失败：{exc}')

async def process_audio(sid: str, file: UploadFile, bias: float):
    """Use a hotword snapshot for ASR, then update memory for the next call."""
    action = 'transcribe'
    if not 0 <= bias <= 8 or not np.isfinite(bias):
        raise HTTPException(422, '偏置强度应在 0–8 之间')
    session = get_session(sid)
    data = await file.read(12*1024*1024+1)
    if len(data) > 12*1024*1024:
        raise HTTPException(413, '单次音频最大 12 MB')
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=ROOT/'tmp') as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            info = sf.info(path)
            if not 0.1 <= info.duration <= 30:
                raise ValueError('单次完整音频应为 0.1–30 秒，不会自动切块')
            if info.channels > 2 or info.samplerate < 8000 or info.samplerate > 96000:
                raise ValueError('请上传 8–96 kHz 单声道或双声道 WAV')
        except Exception as exc:
            raise HTTPException(422, f'无法读取音频：{exc}')
        async with session.lock, inference_lock:
            started = time.monotonic()
            session.sequence += 1
            words = list(session.hotwords)
            session.emit('asr_start', sequence=session.sequence, hotwords=words, action=action)
            try:
                result = await asyncio.to_thread(get_asr().transcribe, path, words, bias)
            except Exception as exc:
                session.emit('error', error=str(exc))
                raise HTTPException(503, f'ASR 推理失败：{exc}')
            session.emit('asr_done', sequence=session.sequence, action=action, **result)
            transcript = result.get('biased', '')
            session.history.append(dict(sequence=session.sequence, action=action,
                                        baseline=result.get('baseline', ''), biased=result.get('biased', '')))
            session.history = session.history[-100:]
            learned = dict(hotwords=list(session.hotwords.values()), updates=[], llm={})
            if transcript.strip():
                context = '\n'.join(x['biased'] for x in session.history[-6:])[-4000:]
                try:
                    learned = await learn(session, context)
                except Exception as exc:
                    learned['llm'] = dict(error=str(exc))
                    session.emit('error', error=f'LLM 更新失败，保留已有热词：{exc}')
            result.update(sequence=session.sequence, action=action, mode=action,
                          applied_hotwords=words, duration_s=info.duration,
                          elapsed_s=round(time.monotonic()-started, 3), **learned)
            session.emit('request_done',
                         sequence=session.sequence, action=action, elapsed_s=result['elapsed_s'])
            return result
    finally:
        if path:
            Path(path).unlink(missing_ok=True)


@app.post('/api/sessions/{sid}/transcribe')
async def transcribe_audio(sid: str, file: UploadFile=File(...), bias: float=Form(8.0)):
    """Recognize one complete recording, then learn hotwords for future calls."""
    return await process_audio(sid, file, bias)


@app.post('/api/sessions/{sid}/chunk', deprecated=True)
async def chunk(sid: str, file: UploadFile=File(...), bias: float=Form(8.0)):
    """Compatibility URL for the same complete-recording operation; no slicing."""
    return await process_audio(sid, file, bias)

@app.get('/api/events/{sid}')
async def events(sid: str):
    session=get_session(sid)
    queue=asyncio.Queue(maxsize=100)
    session.listeners.append(queue)
    async def stream():
        try:
            yield ': connected\n\n'
            while True:
                try:
                    event=await asyncio.wait_for(queue.get(),15)
                    yield 'data: '+json.dumps(event,ensure_ascii=False)+'\n\n'
                except asyncio.TimeoutError:
                    yield ': keepalive\n\n'
        finally:
            session.listeners.remove(queue)
    return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

(ROOT/'tmp').mkdir(exist_ok=True)

@app.on_event('shutdown')
async def cleanup_workers():
    if 'llm' in engines:
        engines['llm'].close()
