const capturePanel=document.querySelector('#capture-panel'),transcript=document.querySelector('#transcript-text'),textForm=document.querySelector('#text-form'),textInput=document.querySelector('#text-input'),statusDot=document.querySelector('#status-dot'),healthLabel=document.querySelector('#health-label'),pipelineState=document.querySelector('#pipeline-state'),runId=document.querySelector('#run-id'),latencyTotal=document.querySelector('#latency-total'),latencyStatus=document.querySelector('#latency-status'),answerCard=document.querySelector('#answer-card'),answerInstruction=document.querySelector('#answer-instruction'),answerContent=document.querySelector('#answer-content'),answerText=document.querySelector('#answer-text'),answerStatus=document.querySelector('#answer-status'),answerMeta=document.querySelector('#answer-meta'),sources=document.querySelector('#sources-container'),sourcesList=document.querySelector('#sources-list'),historyBars=document.querySelector('#history-bars'),stages=['voice','stt','embed','retrieve','generate','guard'];
const corpusStat=document.querySelector('#corpus-stat');

// ============================================================================
// AUTHORITATIVE RECORDING LIFECYCLE STATE MACHINE
// ============================================================================
const STATE = Object.freeze({
  IDLE: 'IDLE',
  STARTING: 'STARTING',
  RECORDING: 'RECORDING',
  STOPPING: 'STOPPING',
  PROCESSING: 'PROCESSING'
});

let currentState = STATE.IDLE;
let recordingDuration = 0;
let durationInterval = null;
let mediaRecorderRef = null;
let streamRef = null;
let chunksRef = [];
let requestHistory = [];

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function stage(id, state = ''){ const el = document.querySelector(`#stage-${id}`); if(el) el.className = `stage ${state}`; }
function stageTime(id, value = '—'){ const el = document.querySelector(`#ms-${id}`); if(el) el.textContent = value; }
function resetTimeline(){ stages.forEach(id => { stage(id); stageTime(id); }); }
function setTranscript(text, partial = false){ if(transcript){ transcript.textContent = text; transcript.className = `transcript-text ${partial ? 'is-partial' : ''}`; } }
function makeTrace(){ return `TRACE / ${Math.random().toString(16).slice(2,8).toUpperCase()}`; }

async function checkHealth(){
  try {
    const response = await fetch('/health', { cache: 'no-store' });
    if (!response.ok) throw new Error('Health endpoint unavailable');
    if (statusDot) statusDot.className = 'status-dot is-healthy';
    if (healthLabel) healthLabel.textContent = 'SYSTEM HEALTHY';
  } catch {
    if (statusDot) statusDot.className = 'status-dot is-error';
    if (healthLabel) healthLabel.textContent = 'BACKEND OFFLINE';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

async function loadCorpusStats(){
  try {
    const response = await fetch('/api/stats', { cache: 'no-store' });
    const data = await response.json();
    const count = data?.stats?.total_vector_count;
    if (Number.isFinite(count) && corpusStat) corpusStat.textContent = `${count.toLocaleString()} CHUNKS / HYBRID RRF`;
  } catch { /* keep default label */ }
}
loadCorpusStats();

function resetView(){
  resetTimeline();
  if (latencyTotal) latencyTotal.textContent = '0';
  if (latencyStatus) latencyStatus.textContent = 'STANDBY';
  if (answerCard) answerCard.className = 'answer-card is-empty';
  if (answerInstruction) answerInstruction.hidden = false;
  if (answerContent) answerContent.hidden = true;
  if (sources) sources.hidden = true;
  if (sourcesList) sourcesList.replaceChildren();
  if (answerMeta) answerMeta.textContent = 'PIPELINE ACTIVE';
  if (answerStatus) answerStatus.textContent = 'ANSWER CHANNEL';
  if (pipelineState) pipelineState.textContent = 'STANDBY';
  if (runId) runId.textContent = makeTrace();
  if (textForm) textForm.style.borderColor = '';
}

function formatTime(secs) {
  const mins = Math.floor(secs / 60);
  const remainingSecs = secs % 60;
  return `${mins.toString().padStart(2, '0')}:${remainingSecs.toString().padStart(2, '0')}`;
}

function setLifecycleState(nextState, details = '') {
  console.log(`[MIC] State transition: ${currentState} -> ${nextState} ${details}`);
  currentState = nextState;

  const btn = document.querySelector('#pulse-record-btn');
  const iconIdle = document.querySelector('#icon-idle');
  const iconRecording = document.querySelector('#icon-recording');
  const durationEl = document.querySelector('#recording-duration');
  const statusTitle = document.querySelector('#voice-status-title');
  const panel = document.querySelector('#capture-panel');

  switch (nextState) {
    case STATE.IDLE:
      if (durationInterval) {
        clearInterval(durationInterval);
        durationInterval = null;
      }
      recordingDuration = 0;

      if (btn) {
        btn.className = 'pulse-record-btn is-idle';
        btn.disabled = false;
        btn.setAttribute('aria-label', 'Start voice recording');
      }
      if (iconIdle) iconIdle.hidden = false;
      if (iconRecording) iconRecording.hidden = true;
      if (durationEl) durationEl.hidden = true;
      if (statusTitle) statusTitle.textContent = 'TAP TO SPEAK';
      if (panel) panel.classList.remove('is-recording');

      // Stop stream tracks
      if (streamRef) {
        try {
          console.log('[MIC] stopping stream tracks');
          streamRef.getTracks().forEach(t => t.stop());
        } catch (e) {
          console.error('[MIC ERROR] stopping stream tracks', e);
        }
        streamRef = null;
      }
      mediaRecorderRef = null;
      chunksRef = [];
      console.log('[MIC] cleanup complete');
      break;

    case STATE.STARTING:
      if (btn) {
        btn.className = 'pulse-record-btn is-processing';
        btn.disabled = false;
        btn.setAttribute('aria-label', 'Requesting microphone access');
      }
      if (iconIdle) iconIdle.hidden = true;
      if (iconRecording) iconRecording.hidden = true;
      if (durationEl) durationEl.hidden = true;
      if (statusTitle) statusTitle.textContent = 'REQUESTING MIC...';
      if (panel) panel.classList.remove('is-recording');
      break;

    case STATE.RECORDING:
      if (btn) {
        btn.className = 'pulse-record-btn is-recording';
        btn.disabled = false;
        btn.setAttribute('aria-label', 'Stop voice recording');
      }
      if (iconIdle) iconIdle.hidden = true;
      if (iconRecording) iconRecording.hidden = false;
      if (durationEl) {
        durationEl.textContent = '00:00';
        durationEl.hidden = false;
      }
      if (statusTitle) statusTitle.textContent = 'TAP TO STOP';

      // REAL MEDIA STATE DRIVES VISUAL ORB
      if (panel) panel.classList.add('is-recording');

      if (durationInterval) clearInterval(durationInterval);
      recordingDuration = 0;
      durationInterval = setInterval(() => {
        recordingDuration += 1;
        if (durationEl) durationEl.textContent = formatTime(recordingDuration);
      }, 1000);
      break;

    case STATE.STOPPING:
      if (durationInterval) {
        clearInterval(durationInterval);
        durationInterval = null;
      }

      if (btn) {
        btn.className = 'pulse-record-btn is-processing';
        btn.disabled = false;
        btn.setAttribute('aria-label', 'Stopping recording');
      }
      if (iconIdle) iconIdle.hidden = true;
      if (iconRecording) iconRecording.hidden = true;
      if (durationEl) durationEl.hidden = true;
      if (statusTitle) statusTitle.textContent = 'STOPPING...';

      // ORB SMOOTHLY DISAPPEARS
      if (panel) panel.classList.remove('is-recording');
      break;

    case STATE.PROCESSING:
      if (durationInterval) {
        clearInterval(durationInterval);
        durationInterval = null;
      }

      if (btn) {
        btn.className = 'pulse-record-btn is-processing';
        btn.disabled = false;
        btn.setAttribute('aria-label', 'Transcribing audio');
      }
      if (iconIdle) iconIdle.hidden = true;
      if (iconRecording) iconRecording.hidden = true;
      if (durationEl) durationEl.hidden = true;
      if (statusTitle) statusTitle.textContent = 'TRANSCRIBING...';

      if (panel) panel.classList.remove('is-recording');
      break;
  }
}

async function handleMicInteraction(e) {
  if (e) {
    if (typeof e.preventDefault === 'function') e.preventDefault();
    if (typeof e.stopPropagation === 'function') e.stopPropagation();
  }

  console.log('[MIC] button clicked');
  console.log('[MIC] current recording state:', currentState);

  if (currentState === STATE.STOPPING || currentState === STATE.PROCESSING) {
    console.log('[MIC] Interaction ignored during transition state:', currentState);
    return;
  }

  if (currentState === STATE.STARTING) {
    console.log('[MIC] Clicked during STARTING — cancelling request');
    setLifecycleState(STATE.IDLE);
    return;
  }

  if (currentState === STATE.RECORDING) {
    // USER CLICKED STOP
    console.log('[MIC] stop requested');
    setLifecycleState(STATE.STOPPING);

    if (mediaRecorderRef && mediaRecorderRef.state !== 'inactive') {
      try {
        console.log('[MIC] recorder.stop()');
        mediaRecorderRef.stop();
      } catch (err) {
        console.error('[MIC ERROR] MediaRecorder stop:', err);
        setLifecycleState(STATE.IDLE);
      }
    } else {
      setLifecycleState(STATE.IDLE);
    }
  } else if (currentState === STATE.IDLE) {
    // USER CLICKED START
    console.log('[MIC] requesting microphone permission');
    resetView();
    setLifecycleState(STATE.STARTING);

    setTranscript('Listening for voice input…', true);
    stage('voice', 'is-active');
    if (pipelineState) pipelineState.textContent = 'LISTENING';
    if (latencyStatus) latencyStatus.textContent = 'RECORDING';

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia not supported in this browser/environment');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
      });
      console.log('[MIC] getUserMedia resolved');
      console.log('[MIC] stream tracks:', stream.getTracks().map(t => `${t.kind}:${t.readyState}`).join(', '));

      // Verify user didn't cancel during permission prompt
      if (currentState !== STATE.STARTING) {
        console.log('[MIC] Stream acquired but state changed, stopping tracks');
        stream.getTracks().forEach(t => t.stop());
        setLifecycleState(STATE.IDLE);
        return;
      }

      streamRef = stream;

      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/wav']
        .find(type => typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(type)) || '';

      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      mediaRecorderRef = recorder;
      chunksRef = [];
      console.log('[MIC] MediaRecorder created');

      recorder.onstart = () => {
        console.log('[MIC] recorder state: recording');
        setLifecycleState(STATE.RECORDING);
      };

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.push(event.data);
          console.log('[MIC] dataavailable — chunk size:', event.data.size);
        }
      };

      recorder.onerror = (event) => {
        console.error('[MIC ERROR] MediaRecorder', event.error || event);
        showError('Recording encountered an error. Please try again.');
        setLifecycleState(STATE.IDLE);
      };

      recorder.onstop = async () => {
        console.log('[MIC] recorder onstop — total chunks:', chunksRef.length);
        setLifecycleState(STATE.PROCESSING);

        // Stop microphone stream tracks immediately
        if (streamRef) {
          try {
            console.log('[MIC] stopping stream tracks');
            streamRef.getTracks().forEach(t => t.stop());
          } catch (e) {
            console.error('[MIC ERROR] stopping stream tracks on stop', e);
          }
          streamRef = null;
        }

        const blob = new Blob(chunksRef, { type: recorder.mimeType || 'audio/webm' });
        console.log('[MIC] blob created — size:', blob.size, 'bytes');

        if (!blob.size) {
          showError("Didn't catch that — try again or type your question.");
          setLifecycleState(STATE.IDLE);
          return;
        }

        stage('voice', 'is-success');
        stageTime('voice', 'CAPTURED');
        stage('stt', 'is-active');
        if (pipelineState) pipelineState.textContent = 'PROCESSING';
        if (latencyStatus) latencyStatus.textContent = 'RESOLVING';

        const formData = new FormData();
        formData.append('audio', blob, 'voice.webm');

        try {
          const res = await fetch('/api/query/voice', { method: 'POST', body: formData });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          console.log('[MIC] Voice query response:', data);

          if (data.query) setTranscript(`“${data.query}”`);
          else setTranscript("Didn't catch that — try again or type your question.");
          await renderResult(data, true);
        } catch (err) {
          console.error('[MIC ERROR] Voice query error:', err);
          showError('Voice transcription failed. Try typing your question below.');
        } finally {
          setLifecycleState(STATE.IDLE);
        }
      };

      console.log('[MIC] recorder.start()');
      recorder.start(100);

      // Immediate fallback if onstart event is synchronous in current engine
      if (recorder.state === 'recording' && currentState === STATE.STARTING) {
        console.log('[MIC] recorder state: recording (sync)');
        setLifecycleState(STATE.RECORDING);
      }

    } catch (error) {
      console.error('[MIC ERROR] getUserMedia', error);
      const denied = error?.name === 'NotAllowedError' || error?.name === 'SecurityError';
      showError(denied ? 'Mic access denied — please enable microphone permissions in your browser.' : 'Could not start microphone input.');
      setLifecycleState(STATE.IDLE);
    }
  }
}
window.handlePulseToggle = handleMicInteraction;

function initMicHandlers() {
  console.log('[MIC] initializing mic handlers');
  setLifecycleState(STATE.IDLE);
  if (pipelineState) pipelineState.textContent = 'STANDBY';
  if (latencyStatus) latencyStatus.textContent = 'STANDBY';
  if (transcript) transcript.textContent = 'Waiting for a question.';

  const btn = document.querySelector('#pulse-record-btn');
  const statusBlock = document.querySelector('.voice-status-block');
  const waveContainer = document.querySelector('#wave-container');

  if (btn) {
    btn.removeEventListener('click', handleMicInteraction);
    btn.addEventListener('click', handleMicInteraction);
  }
  if (statusBlock) {
    statusBlock.removeEventListener('click', handleMicInteraction);
    statusBlock.addEventListener('click', handleMicInteraction);
  }
  if (waveContainer) {
    waveContainer.removeEventListener('click', handleMicInteraction);
    waveContainer.addEventListener('click', handleMicInteraction);
  }
}

// Ensure handlers are bound immediately and on DOM readiness
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initMicHandlers);
} else {
  initMicHandlers();
}

window.addEventListener('load', () => {
  setLifecycleState(STATE.IDLE);
});

window.addEventListener('beforeunload', () => {
  if (streamRef) {
    try { streamRef.getTracks().forEach(t => t.stop()); } catch (e) {}
  }
});

function startTextFlow(query){
  if (!query || currentState !== STATE.IDLE) return;
  resetView();
  setTranscript(`“${query}”`);
  stage('voice', 'is-success'); stageTime('voice', 'BYPASS');
  stage('stt', 'is-success'); stageTime('stt', 'BYPASS');
  sendTextQuery(query);
}

if (textForm) {
  textForm.addEventListener('submit', event => {
    event.preventDefault();
    const query = textInput.value.trim();
    startTextFlow(query);
    textInput.value = '';
  });
}

document.querySelectorAll('.guardrail-chip').forEach(button => {
  button.addEventListener('click', () => startTextFlow(button.dataset.query));
});

async function sendTextQuery(query){
  stage('embed','is-active');
  try {
    const response = await fetch('/api/query/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    if (!response.ok) throw new Error(`Query failed (${response.status})`);
    await renderResult(await response.json(), false);
  } catch(e) {
    console.error('[API] Text Query Error:', e);
    showError('Backend unavailable — check API connection.');
  } finally {
    setLifecycleState(STATE.IDLE);
  }
}

async function renderResult(data, isVoice){
  const latency = data.latency_breakdown || {};
  const steps = [
    ['stt', latency.stt_ms, isVoice],
    ['embed', latency.embedding_ms, true],
    ['retrieve', latency.retrieval_ms, true],
    ['generate', latency.generation_ms, true],
    ['guard', (latency.guardrail_pre_ms || 0) + (latency.guardrail_post_ms || 0), true]
  ];
  const reason = String(data.refusal_reason || '').toLowerCase();
  const refusalAt = data.refused ? (reason === 'unsafe' || reason === 'off_topic' || reason === 'ungrounded' ? 'guard' : 'retrieve') : null;
  
  for (const [id, time, visible] of steps) {
    if (!visible) continue;
    stage(id, 'is-active');
    await sleep(80);
    stageTime(id, `${Math.round(time || 0)}ms`);
    if (id === refusalAt) {
      stage(id, 'is-refusal');
      break;
    }
    stage(id, 'is-success');
  }
  
  if (data.refused && refusalAt) {
    let after = false;
    for (const [id] of steps) {
      if (after) stageTime(id, '—');
      if (id === refusalAt) after = true;
    }
  }
  
  renderAnswer(data);
  const total = Math.round(data.latency_ms || 0);
  if (latencyTotal) latencyTotal.textContent = total;
  if (latencyStatus) latencyStatus.textContent = data.refused ? 'REFUSED' : 'GROUNDED';
  if (pipelineState) pipelineState.textContent = data.refused ? 'REFUSED' : 'COMPLETE';
  if (answerMeta) answerMeta.textContent = `${data.refused ? 'GUARDRAIL' : 'GROUNDING'} / ${data.trace_id || 'VERIFIED'}`;
  addHistory(total, data.refused);
}

function renderAnswer(data){
  if (answerInstruction) answerInstruction.hidden = true;
  if (answerContent) answerContent.hidden = false;
  if (data.refused) {
    if (answerCard) answerCard.className = 'answer-card is-refused';
    if (answerStatus) answerStatus.textContent = 'RESPONSE WITHHELD';
    if (answerText) answerText.textContent = data.refusal_message || 'No grounded passage found for this question.';
    if (sources) sources.hidden = true;
    return;
  }
  if (answerCard) answerCard.className = 'answer-card is-grounded';
  if (answerStatus) answerStatus.textContent = 'GROUNDED ANSWER';
  if (answerText) answerText.textContent = data.answer || 'No answer was returned.';
  const chunks = (data.retrieved_chunks || []).slice(0, 3);
  if (sources) sources.hidden = !chunks.length;
  if (sourcesList) {
    sourcesList.replaceChildren();
    chunks.forEach(chunk => {
      const item = document.createElement('div');
      item.className = 'source-snippet';
      const value = String(chunk.text || '').replace(/\s+/g, ' ').trim();
      item.textContent = value.length > 180 ? `${value.slice(0, 180)}…` : value;
      sourcesList.append(item);
    });
  }
}

function showError(message){
  setLifecycleState(STATE.IDLE);
  if (answerCard) answerCard.className = 'answer-card is-refused';
  if (answerInstruction) answerInstruction.hidden = true;
  if (answerContent) answerContent.hidden = false;
  if (answerStatus) answerStatus.textContent = 'SYSTEM NOTICE';
  if (answerMeta) answerMeta.textContent = 'ACTION REQUIRED';
  if (answerText) answerText.textContent = message;
  if (sources) sources.hidden = true;
  if (pipelineState) pipelineState.textContent = 'INTERRUPTED';
  if (latencyStatus) latencyStatus.textContent = 'ERROR';
}

function addHistory(ms, estimatedRefusal){
  requestHistory.push({ ms, refused: estimatedRefusal });
  if (requestHistory.length > 5) requestHistory.shift();
  if (historyBars) {
    historyBars.replaceChildren();
    historyBars.classList.add('has-data');
    const max = Math.max(...requestHistory.map(entry => entry.ms), 150);
    requestHistory.forEach((entry, index) => {
      const bar = document.createElement('div');
      bar.className = `history-bar${entry.refused ? ' is-refusal' : ''}`;
      bar.style.height = `${Math.max(9, Math.round(entry.ms / max * 48))}px`;
      bar.style.opacity = 0.35 + (index + 1) / requestHistory.length * 0.65;
      bar.dataset.ms = `${entry.ms}ms`;
      historyBars.append(bar);
    });
  }
}
