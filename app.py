# -*- coding: utf-8 -*-
"""
[서용엔지니어링] 상수관망 누수음 지능형 진단 클라우드 API & 모바일 PWA 앱
- 클라우드 24시간 상시 호스팅 및 스마트폰 단독 앱(PWA) 지원
- 듀얼 AI 판정 엔진: 순수 음향(76.0%) vs 배관 물리 결합(84.9%)
- 현장 배관 파라미터(관종, 관경, 심도, 수압) 직접 입력 패널
- 2D Mel 스펙트로그램 & PSD 시각화 차트 base64 자동 렌더링
- 현장 굴착 실증 피드백(일치/불일치) 데이터베이스 누적 기능 탑재
"""
import os
import sys
import io
import csv
import base64
import pickle
import datetime
import tempfile
import subprocess
import shutil
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft, welch
from scipy.fftpack import dct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string, Response

# 한글 폰트 설정 (서버 환경 NanumGothic 또는 맑은 고딕 fallback)
plt.rcParams['font.family'] = ['Malgun Gothic', 'NanumGothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_EXTS = ('.wav', '.mp4', '.m4a', '.mp3', '.mov', '.aac', '.flac', '.ogg', '.wma')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# ----------------------------------------------------------------------
# 1. 신호 처리 및 특징 추출 엔진
# ----------------------------------------------------------------------
def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
def mel_to_hz(mel): return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def get_mel_filterbank(sr=8000, n_fft=1024, n_mels=128, fmin=0.0, fmax=4000.0):
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus: filters[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m: filters[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return filters

MEL_FB = get_mel_filterbank()

def extract_features_from_signal(signal, sr=8000):
    _, _, zxx = stft(signal, fs=sr, nperseg=1024, noverlap=768)
    power_spec = np.abs(zxx)**2
    mel_spec = np.dot(MEL_FB, power_spec)
    log_mel = 10.0 * np.log10(mel_spec + 1e-9)
    norm_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    
    mel_mean = np.mean(norm_mel, axis=1)
    mel_std = np.std(norm_mel, axis=1)
    delta_mean = np.mean(np.abs(np.diff(norm_mel, axis=1)), axis=1)
    diff2 = np.diff(np.diff(norm_mel, axis=1), axis=1)
    delta2_mean = np.mean(np.abs(diff2), axis=1)
    mfcc = dct(mel_mean, type=2, norm='ortho')[:20]
    return np.hstack([mel_mean, mel_std, delta_mean, delta2_mean, mfcc])

def get_ffmpeg_path():
    candidates = [
        shutil.which("ffmpeg"),
        os.path.join(BASE_DIR, "ffmpeg"),
        os.path.join(BASE_DIR, "ffmpeg.exe"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

def load_audio_signal(file_path, target_sr=8000):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.wav':
        try:
            sr, raw_y = wavfile.read(file_path)
            if raw_y.ndim > 1: raw_y = np.mean(raw_y, axis=1)
            y = raw_y.astype(np.float32)
            max_val = np.max(np.abs(y))
            if max_val > 1.0: y = y / (max_val + 1e-6)
            if sr != target_sr:
                new_len = int(round(len(y) * float(target_sr) / float(sr)))
                indices = np.linspace(0, len(y) - 1, new_len)
                y = np.interp(indices, np.arange(len(y)), y).astype(np.float32)
            return y, target_sr
        except Exception:
            pass

    ffmpeg_bin = get_ffmpeg_path()
    if ffmpeg_bin:
        try:
            cmd = [
                ffmpeg_bin, '-y', '-v', 'error',
                '-i', file_path,
                '-f', 's16le', '-acodec', 'pcm_s16le',
                '-ac', '1', '-ar', str(target_sr), '-'
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, _ = p.communicate()
            if p.returncode == 0 and len(out) > 0:
                raw_y = np.frombuffer(out, dtype=np.int16).astype(np.float32)
                max_val = np.max(np.abs(raw_y))
                if max_val > 0: raw_y = raw_y / 32768.0
                return raw_y, target_sr
        except Exception:
            pass

    return None, None

def extract_sliding_features(file_path, win_sec=6.0, hop_sec=3.0, target_sr=8000):
    y, sr = load_audio_signal(file_path, target_sr=target_sr)
    if y is None or len(y) == 0:
        return None, 0.0, None, "READ_ERROR"
    
    dur = len(y) / float(sr)
    if dur < 3.5:
        return None, dur, y, "TOO_SHORT"

    status = "NORMAL"
    if 3.5 <= dur < 5.5:
        repeat_cnt = int(np.ceil((win_sec * sr) / len(y))) + 1
        y = np.tile(y, repeat_cnt)
        status = "LOOP_PADDED"

    win_samples = int(win_sec * sr)
    hop_samples = int(hop_sec * sr)
    
    segs = []
    if len(y) < win_samples:
        pad_len = win_samples - len(y)
        y_padded = np.pad(y, (0, pad_len), mode='wrap')
        segs.append(extract_features_from_signal(y_padded, sr=sr))
    else:
        for start in range(0, len(y) - win_samples + 1, hop_samples):
            w = y[start : start + win_samples]
            segs.append(extract_features_from_signal(w, sr=sr))
        if len(segs) == 0:
            segs.append(extract_features_from_signal(y[:win_samples], sr=sr))
            
    return np.array(segs, dtype=np.float32), dur, y, status

# ----------------------------------------------------------------------
# 2. AI 모델 로드
# ----------------------------------------------------------------------
def load_models():
    pipe_model_path = os.path.join(BASE_DIR, "pipe_profiler_model.pkl")
    pure_model_path = os.path.join(BASE_DIR, "models", "xgb_pure_acoustic.pkl")
    pipe_leak_path = os.path.join(BASE_DIR, "models", "xgb_pipe_physical.pkl")

    pipe_pkg = None
    if os.path.exists(pipe_model_path):
        with open(pipe_model_path, 'rb') as f:
            pipe_pkg = pickle.load(f)

    pure_clf = None
    if os.path.exists(pure_model_path):
        with open(pure_model_path, 'rb') as f:
            p_payload = pickle.load(f)
        pure_clf = p_payload['model'] if isinstance(p_payload, dict) else p_payload

    pipe_clf = None
    if os.path.exists(pipe_leak_path):
        with open(pipe_leak_path, 'rb') as f:
            p_payload = pickle.load(f)
        pipe_clf = p_payload['model'] if isinstance(p_payload, dict) else p_payload

    return pure_clf, pipe_clf, pipe_pkg

pure_clf, pipe_clf, pipe_pkg = load_models()
print(f"[클라우드 초기화] 순수모델: {pure_clf is not None}, 결합모델: {pipe_clf is not None}, 프로파일러: {pipe_pkg is not None}")

# ----------------------------------------------------------------------
# 3. 진단 엔진 및 시각화 생성
# ----------------------------------------------------------------------
def analyze_audio(fp, eff_depth=0.7, mop_code=-1.0, pipe_di=-1.0, before_pre=-1.0):
    fname = os.path.basename(fp)
    segs, dur, raw_audio, status = extract_sliding_features(fp)
    
    if status == "READ_ERROR" or segs is None or len(segs) == 0:
        if status == "TOO_SHORT":
            return {
                '파일명': fname,
                '음원길이': round(dur, 1),
                '누수_판정': "판정보류",
                '누수_확률': 0.0,
                '순수음향_확률': 0.0,
                '배관결합_확률': None,
                '상태분류': f"음원부족({dur:.1f}초 - 3.5초 이상 필요)",
                '설명': "최소 분석 기준(3.5초) 미달로 신호 왜곡 방지를 위해 판정을 보류합니다. (5초 이상 녹음 권장)"
            }
        return {
            '파일명': fname,
            '음원길이': 0.0,
            '누수_판정': "분석불가",
            '누수_확률': 0.0,
            '순수음향_확률': 0.0,
            '배관결합_확률': None,
            '상태분류': "판독실패",
            '설명': "지원되지 않는 오디오 형식이거나 파일이 손상되었습니다."
        }

    # 1. 듀얼 AI 판정
    pure_leak_p = 0.0
    if pure_clf is not None:
        try:
            pure_probs = pure_clf.predict_proba(segs)[:, 1]
            pure_leak_p = float(np.mean(pure_probs)) * 100.0
        except Exception:
            pure_leak_p = 50.0

    has_pipe_input = (mop_code > 0 or pipe_di > 0)
    pipe_leak_p = None
    if has_pipe_input and pipe_clf is not None:
        meta_vec = np.array([pipe_di, eff_depth, before_pre, mop_code, -1.0], dtype=np.float32)
        try:
            X_pipe_segs = np.array([np.hstack([s, meta_vec]) for s in segs], dtype=np.float32)
            pipe_probs = pipe_clf.predict_proba(X_pipe_segs)[:, 1]
            pipe_leak_p = float(np.mean(pipe_probs)) * 100.0
        except Exception:
            pipe_leak_p = None

    if has_pipe_input and pipe_leak_p is not None:
        leak_p = pipe_leak_p
        primary_model = "배관 물리 결합 모델 (84.9% 검증)"
    else:
        leak_p = pure_leak_p
        primary_model = "순수 음향 진단 모델 (76.0% 검증)"

    is_leak = (leak_p >= 50.0)
    res_str = "누수" if is_leak else "비누수"

    # 2. 물리 음향 스펙트럼 해석
    sr = 8000
    start_idx = int(1.5 * sr)
    steady = raw_audio[start_idx : int(min(len(raw_audio), 5.5 * sr))] if len(raw_audio) > start_idx + int(0.5 * sr) else raw_audio
    steady_norm = steady - np.mean(steady)
    if np.std(steady_norm) > 1e-6:
        steady_norm = steady_norm / np.std(steady_norm)

    f, psd = welch(steady_norm, fs=sr, nperseg=512, noverlap=256)
    alpha_soil = 0.0004 * f * (eff_depth - 0.7)
    psd_calib = psd * np.exp(alpha_soil)
    total_p = np.sum(psd_calib) + 1e-12

    p_b1 = np.sum(psd_calib[(f >= 0) & (f < 300)]) / total_p
    p_b2 = np.sum(psd_calib[(f >= 300) & (f < 700)]) / total_p
    p_b3 = np.sum(psd_calib[(f >= 700) & (f < 1500)]) / total_p
    p_b4 = np.sum(psd_calib[(f >= 1500) & (f < 3000)]) / total_p
    p_b5 = np.sum(psd_calib[(f >= 3000) & (f <= 4000)]) / total_p

    spectral_centroid = float(np.sum(f * psd_calib) / total_p)
    peak_freq = float(f[np.argmax(psd_calib)])
    hf_ratio = float((p_b4 + p_b5) / (p_b2 + 1e-6))
    p_high = (p_b4 + p_b5) * 100.0

    # 2D Mel 스펙트로그램
    _, _, zxx_full = stft(steady_norm, fs=sr, nperseg=512, noverlap=384)
    mel_fb_512 = get_mel_filterbank(sr=sr, n_fft=512, n_mels=64)
    mel_2d = np.dot(mel_fb_512, np.abs(zxx_full)**2)
    log_mel_2d = 10.0 * np.log10(mel_2d + 1e-9)

    # 3. 배관 속성 및 분출 형태 역추정
    mat_disp, di_disp = "미확정", "미확정"
    leak_type_title = "정상 관로"
    if pipe_pkg is not None:
        feat_arr = np.array([[eff_depth, p_b1, p_b2, p_b3, p_b4, p_b5, spectral_centroid, 4000.0, peak_freq, hf_ratio]])
        try:
            mat_pred = pipe_pkg['mat_model'].predict(feat_arr)[0]
            mat_disp = "금속관" if "금속" in str(mat_pred) else "플라스틱관"
            di_pred = pipe_pkg['di_model'].predict(feat_arr)[0]
            di_disp = str(di_pred).replace("배관", "").strip()
        except Exception:
            pass

        z_jet = 0.008 * (spectral_centroid - 600.0) + 14.0 * (hf_ratio - 0.07) + 0.18 * (p_high - 3.5)
        jet_prob = float(1.0 / (1.0 + np.exp(-np.clip(z_jet, -6.0, 6.0))) * 100.0)
        if abs(jet_prob - 50.0) <= 6.0: leak_type_title = "복합/경계 분출형"
        elif jet_prob > 50.0: leak_type_title = f"고속 제트 분출형 ({jet_prob:.0f}%)"
        else: leak_type_title = f"대량 유출 파열형 ({100.0 - jet_prob:.0f}%)"

    # 차트 이미지 생성
    chart_b64 = None
    try:
        fig = plt.Figure(figsize=(6.4, 5.6), dpi=100, facecolor='#0B132B')
        gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.40, wspace=0.35, left=0.10, right=0.92, top=0.90, bottom=0.12)
        ax_mel = fig.add_subplot(gs[0, :])
        ax_psd = fig.add_subplot(gs[1, 0])
        ax_bar = fig.add_subplot(gs[1, 1])

        for ax in [ax_mel, ax_psd, ax_bar]:
            ax.set_facecolor('#13244D')
            ax.tick_params(colors='#8FA8D6', labelsize=8)
            for s in ax.spines.values(): s.set_color('#27417D')

        # 멜 스펙트로그램
        ax_mel.imshow(log_mel_2d, aspect='auto', origin='lower', cmap='plasma', extent=[0, dur, 0, 4000])
        ax_mel.set_title(f"2D Mel-Spectrogram ({leak_type_title})", color='#FFFFFF', fontsize=9, fontweight='bold')
        ax_mel.set_xlabel("시간 (초)", color='#8FA8D6', fontsize=8)
        ax_mel.set_ylabel("주파수 (Hz)", color='#8FA8D6', fontsize=8)

        # PSD
        ax_psd.plot(f, psd_calib, color='#38BDF8', lw=1.5, label='PSD')
        ax_psd.axvline(1500, color='#F59E0B', ls='--', lw=1.0, label='1.5kHz')
        ax_psd.plot(peak_freq, np.interp(peak_freq, f, psd_calib), 'ro', markersize=4, label=f'{peak_freq:.0f}Hz')
        ax_psd.set_title("파워 스펙트럼 밀도 (PSD)", color='#FFFFFF', fontsize=9, fontweight='bold')
        ax_psd.set_xlabel("주파수 (Hz)", color='#8FA8D6', fontsize=8)
        ax_psd.legend(facecolor='#0E1C3E', edgecolor='#27417D', labelcolor='#E2E8F0', fontsize=7, loc='upper right')

        # 듀얼 확률 게이지
        labels = ['순수(76%)', '결합(85%)']
        if pipe_leak_p is not None:
            values = [pure_leak_p, pipe_leak_p]
            colors = ['#F87171' if v >= 50.0 else '#38BDF8' for v in values]
            bars = ax_bar.barh(labels, values, color=colors, height=0.45, edgecolor='#FFFFFF', lw=0.5)
            for bar, val in zip(bars, values):
                ax_bar.text(min(val + 2, 85), bar.get_y() + bar.get_height()/2.0, f"{val:.1f}%", va='center', color='#FFFFFF', fontsize=8, fontweight='bold')
        else:
            values = [pure_leak_p, 0.0]
            colors = ['#F87171' if pure_leak_p >= 50.0 else '#38BDF8', '#334155']
            bars = ax_bar.barh(labels, values, color=colors, height=0.45, edgecolor='#FFFFFF', lw=0.5)
            ax_bar.text(min(pure_leak_p + 2, 85), bars[0].get_y() + bars[0].get_height()/2.0, f"{pure_leak_p:.1f}%", va='center', color='#FFFFFF', fontsize=8, fontweight='bold')
            ax_bar.text(8, bars[1].get_y() + bars[1].get_height()/2.0, "미지정 (입력 시 활성)", va='center', color='#FBBF24', fontsize=7, fontweight='bold')

        ax_bar.axvline(50.0, color='#EF4444', ls=':', lw=1.2)
        ax_bar.set_xlim(0, 100)
        ax_bar.set_title("AI 듀얼 판정 확률 대조", color='#FFFFFF', fontsize=9, fontweight='bold')
        ax_bar.set_xlabel("누수 확률 (%)", color='#8FA8D6', fontsize=8)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    except Exception as e:
        print("[차트 렌더링 실패]", e)

    return {
        '파일명': fname,
        '음원길이': round(dur, 1),
        '누수_판정': res_str,
        '누수_확률': round(leak_p, 1),
        '순수음향_확률': round(pure_leak_p, 1),
        '배관결합_확률': round(pipe_leak_p, 1) if pipe_leak_p is not None else None,
        '추정_관로재질': mat_disp,
        '추정_구경범주': di_disp,
        '고주파잔존비': round(hf_ratio, 2),
        '중심주파수': round(spectral_centroid, 1),
        '피크주파수': round(peak_freq, 1),
        '분출형태': leak_type_title,
        '적용모델': primary_model,
        '차트_base64': chart_b64
    }

# ----------------------------------------------------------------------
# 4. 모바일 PWA 웹 인터페이스
# ----------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>서용 누수진단 (모바일 현장용)</title>
  <link rel="manifest" href="/manifest.json">
  <meta name="theme-color" content="#1B3065">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #0B132B; color: #E2E8F0; font-family: -apple-system, BlinkMacSystemFont, "Malgun Gothic", sans-serif; -webkit-tap-highlight-color: transparent; }
    .card { background-color: #13244D; border: 1px solid #27417D; }
    .input-field { background-color: #0E1C3E; border: 1px solid #2D4A8A; color: #FFFFFF; }
    .btn-primary { background-color: #356DFA; }
    .btn-primary:active { background-color: #2554C7; transform: scale(0.98); }
  </style>
</head>
<body class="p-3 max-w-lg mx-auto pb-14">

  <!-- 헤더 -->
  <header class="card rounded-2xl p-4 mb-3 shadow-lg flex items-center justify-between">
    <div>
      <div class="flex items-center gap-1.5">
        <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
        <h1 class="text-base font-bold text-white tracking-tight">서용 누수진단 AI</h1>
      </div>
      <p class="text-[11px] text-blue-300">현장 청음기 연동 의사결정지원 시스템</p>
    </div>
    <span class="bg-blue-900/80 border border-blue-500/50 text-blue-300 text-[10px] font-semibold px-2.5 py-1 rounded-full">Cloud v2.5</span>
  </header>

  <!-- 입력 폼 카드 -->
  <div class="card rounded-2xl p-4 mb-3 shadow-lg">
    <h2 class="text-xs font-bold text-emerald-400 mb-2.5 flex items-center gap-1">
      <span>⚙️ 현장 음원 및 배관 파라미터 직접 입력</span>
    </h2>

    <div class="space-y-3 text-xs">
      <div>
        <label class="block font-medium text-slate-300 mb-1">🎵 청음기 저장 음원 선택 (WAV, MP4, M4A)</label>
        <input type="file" id="audioFile" accept="audio/*,video/*,.wav,.mp4,.m4a,.mp3"
          class="block w-full text-xs text-slate-300 file:mr-2 file:py-1.5 file:px-3 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-500 cursor-pointer">
      </div>

      <div>
        <label class="block font-medium text-slate-300 mb-1">관종(재질)</label>
        <select id="mopCode" class="input-field w-full rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-blue-400">
          <option value="-1.0">미지정 (순수 음향 76.0% 모델로 진단)</option>
          <option value="1.0">금속관 (주철 / 강관 / 동관 / 스테인리스)</option>
          <option value="2.0">플라스틱관 (PE / PVC / HI-VP / PB)</option>
        </select>
      </div>

      <div class="grid grid-cols-3 gap-2">
        <div>
          <label class="block font-medium text-slate-300 mb-1">관경 (mm)</label>
          <input type="number" id="pipeDi" placeholder="예: 50" class="input-field w-full rounded-xl px-2.5 py-2 text-xs focus:outline-none focus:border-blue-400">
        </div>
        <div>
          <label class="block font-medium text-slate-300 mb-1">심도 (m)</label>
          <input type="number" step="0.1" id="pipeDp" value="0.7" class="input-field w-full rounded-xl px-2.5 py-2 text-xs focus:outline-none focus:border-blue-400">
        </div>
        <div>
          <label class="block font-medium text-slate-300 mb-1">수압 (kgf)</label>
          <input type="number" step="0.1" id="beforePre" placeholder="선택" class="input-field w-full rounded-xl px-2.5 py-2 text-xs focus:outline-none focus:border-blue-400">
        </div>
      </div>

      <button id="btnDiagnose" onclick="runDiagnosis()" class="btn-primary w-full py-3 rounded-xl text-white font-bold text-sm shadow-md mt-1 flex items-center justify-center gap-1.5 cursor-pointer">
        <span>🚀 현장 즉시 AI 정밀 진단</span>
      </button>
    </div>
  </div>

  <!-- 로딩 스피너 -->
  <div id="loadingBox" class="hidden card rounded-2xl p-6 mb-3 text-center">
    <div class="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent mb-2"></div>
    <p class="text-xs text-blue-300 font-medium">클라우드 532차원 분석 및 듀얼 모델 추론 중...</p>
  </div>

  <!-- 결과 패널 -->
  <div id="resultBox" class="hidden space-y-3">
    <!-- 주 판정 배너 -->
    <div id="resBadgeBox" class="rounded-2xl p-4 text-center shadow-lg border">
      <p class="text-[11px] text-slate-300 mb-1" id="resModelName">적용 모델</p>
      <h3 class="text-2xl font-black mb-1" id="resDecision">누수 의심</h3>
      <p class="text-sm font-bold" id="resProb">판정 확률: 85.4%</p>
    </div>

    <!-- 듀얼 확률 게이지 -->
    <div class="card rounded-2xl p-4 shadow-lg">
      <h3 class="text-xs font-bold text-slate-300 mb-2.5 flex items-center justify-between">
        <span>📊 듀얼 AI 판정 대조</span>
        <span class="text-[10px] text-slate-400">기준선 50%</span>
      </h3>
      
      <div class="mb-2">
        <div class="flex justify-between text-[11px] mb-1">
          <span class="text-slate-300">순수 음향 모델 (76.0% 실측)</span>
          <span class="font-bold text-white" id="valPure">0.0%</span>
        </div>
        <div class="w-full bg-slate-800 rounded-full h-2.5">
          <div id="barPure" class="bg-sky-400 h-2.5 rounded-full" style="width: 0%"></div>
        </div>
      </div>

      <div>
        <div class="flex justify-between text-[11px] mb-1">
          <span class="text-slate-300">배관 물리 결합 (84.9% 실측)</span>
          <span class="font-bold text-white" id="valPipe">-</span>
        </div>
        <div class="w-full bg-slate-800 rounded-full h-2.5">
          <div id="barPipe" class="bg-red-400 h-2.5 rounded-full" style="width: 0%"></div>
        </div>
      </div>
    </div>

    <!-- 스펙트로그램 차트 -->
    <div class="card rounded-2xl p-3 shadow-lg">
      <h3 class="text-xs font-bold text-slate-300 mb-2">📈 음향 스펙트로그램 & PSD 시각화</h3>
      <div class="rounded-xl overflow-hidden border border-slate-700 bg-slate-900">
        <img id="chartImg" src="" alt="분석 차트" class="w-full h-auto">
      </div>
    </div>

    <!-- 4단계 요약 카드 -->
    <div class="card rounded-2xl p-4 shadow-lg text-xs space-y-2">
      <h3 class="font-bold text-slate-300 mb-1">📋 물리 음향 분석 세부 지표</h3>
      <div class="grid grid-cols-2 gap-2 text-[11px] text-slate-300">
        <div class="bg-slate-900/60 p-2 rounded-lg">분출 형태: <span id="resJet" class="font-bold text-white">-</span></div>
        <div class="bg-slate-900/60 p-2 rounded-lg">추정 재질: <span id="resMat" class="font-bold text-white">-</span></div>
        <div class="bg-slate-900/60 p-2 rounded-lg">추정 구경: <span id="resDia" class="font-bold text-white">-</span></div>
        <div class="bg-slate-900/60 p-2 rounded-lg">고주파잔존비: <span id="resHf" class="font-bold text-white">-</span></div>
      </div>
    </div>

    <!-- 현장 굴착 실증 피드백 기록 영역 (TRL 6 도약용) -->
    <div class="card rounded-2xl p-4 shadow-lg border-emerald-600/40">
      <h3 class="text-xs font-bold text-emerald-400 mb-1.5 flex items-center gap-1">
        <span>📝 현장 굴착 실증 결과 피드백 (필드 검증용)</span>
      </h3>
      <p class="text-[10px] text-slate-400 mb-2.5">실제 땅을 파보았을 때 누수가 맞았는지 기록하여 실증 데이터를 누적합니다.</p>
      
      <div class="flex gap-2 mb-2">
        <button onclick="sendFeedback('누수확인')" class="flex-1 py-2 rounded-xl bg-red-600/80 hover:bg-red-600 text-white font-bold text-xs">
          🎯 실제 누수 맞음
        </button>
        <button onclick="sendFeedback('오탐_정상')" class="flex-1 py-2 rounded-xl bg-slate-700 hover:bg-slate-600 text-slate-200 font-bold text-xs">
          ❌ 꽝 (정상이었음)
        </button>
      </div>
      <input type="text" id="fbMemo" placeholder="현장 특이사항 메모 (예: 25mm PE 핀홀, 차량 소음 심함)" class="input-field w-full rounded-xl px-2.5 py-1.5 text-[11px] focus:outline-none">
      <p id="fbStatus" class="text-[10px] text-emerald-400 mt-1.5 hidden"></p>
    </div>
  </div>

  <!-- 푸터 -->
  <footer class="mt-4 p-3 bg-slate-900/60 rounded-2xl border border-slate-800 text-[10px] text-slate-400 leading-normal">
    <p class="font-bold text-slate-300 mb-1">⚠️ 현장 실증 원칙 (TRL 4~5 기준)</p>
    <p>• 온실 속 90% 과적합을 배제하고, 순수 음향 76.0%, 배관 결합 84.9% 실측치를 투명하게 기준으로 삼습니다.</p>
    <p>• 지하 감쇠 및 도로 소음에 따른 오탐이 있을 수 있으므로, 최종 굴착 전 상관식 탐사기와 밸브 조작을 병행하십시오.</p>
  </footer>

  <script>
    let currentResult = null;

    async function runDiagnosis() {
      const fileInput = document.getElementById('audioFile');
      if (!fileInput.files || fileInput.files.length === 0) {
        alert("스마트폰에 저장된 음원 파일(WAV, MP4 등)을 선택하세요.");
        return;
      }

      const file = fileInput.files[0];
      const mop = document.getElementById('mopCode').value;
      const dia = document.getElementById('pipeDi').value || "-1.0";
      const dp = document.getElementById('pipeDp').value || "0.7";
      const pre = document.getElementById('beforePre').value || "-1.0";

      const formData = new FormData();
      formData.append('audio', file);
      formData.append('mop_code', mop);
      formData.append('pipe_di', dia);
      formData.append('pipe_dp', dp);
      formData.append('before_pre', pre);

      document.getElementById('loadingBox').classList.remove('hidden');
      document.getElementById('resultBox').classList.add('hidden');
      document.getElementById('fbStatus').classList.add('hidden');

      try {
        const res = await fetch('/api/diagnose', { method: 'POST', body: formData });
        const data = await res.json();
        document.getElementById('loadingBox').classList.add('hidden');

        if (!data.success) {
          alert("진단 실패: " + (data.error || "오류 발생"));
          return;
        }

        currentResult = data;
        renderResult(data);
      } catch (err) {
        document.getElementById('loadingBox').classList.add('hidden');
        alert("통신 오류: " + err.message);
      }
    }

    function renderResult(data) {
      document.getElementById('resultBox').classList.remove('hidden');

      const badgeBox = document.getElementById('resBadgeBox');
      const isLeak = (data.leak_decision === '누수');
      const isHold = (data.leak_decision === '판정보류');

      if (isHold) {
        badgeBox.className = "rounded-2xl p-4 text-center shadow-lg border bg-amber-950/40 border-amber-500/50 text-amber-300";
        document.getElementById('resDecision').innerText = "판정보류 (음원 부족)";
        document.getElementById('resProb').innerText = "음원 길이 3.5초 미달";
      } else if (isLeak) {
        badgeBox.className = "rounded-2xl p-4 text-center shadow-lg border bg-red-950/50 border-red-500/60 text-red-400";
        document.getElementById('resDecision').innerText = "🚨 누수 의심 (이상 진동 포착)";
        document.getElementById('resProb').innerText = "누수 확률: " + data.primary_prob.toFixed(1) + "%";
      } else {
        badgeBox.className = "rounded-2xl p-4 text-center shadow-lg border bg-sky-950/50 border-sky-500/60 text-sky-400";
        document.getElementById('resDecision').innerText = "✅ 정상 통수 (비누수)";
        document.getElementById('resProb').innerText = "정상 확률: " + (100.0 - data.primary_prob).toFixed(1) + "%";
      }
      document.getElementById('resModelName').innerText = data.primary_model;

      const pureP = data.pure_prob || 0.0;
      document.getElementById('valPure').innerText = pureP.toFixed(1) + "%";
      document.getElementById('barPure').style.width = Math.min(100, pureP) + "%";
      document.getElementById('barPure').className = (pureP >= 50.0 ? "bg-red-400" : "bg-sky-400") + " h-2.5 rounded-full";

      if (data.pipe_prob !== null && data.pipe_prob !== undefined) {
        const pipeP = data.pipe_prob;
        document.getElementById('valPipe').innerText = pipeP.toFixed(1) + "%";
        document.getElementById('barPipe').style.width = Math.min(100, pipeP) + "%";
        document.getElementById('barPipe').className = (pipeP >= 50.0 ? "bg-red-400" : "bg-sky-400") + " h-2.5 rounded-full";
      } else {
        document.getElementById('valPipe').innerText = "미지정 (입력 시 산출)";
        document.getElementById('barPipe').style.width = "0%";
      }

      if (data.chart_base64) {
        document.getElementById('chartImg').src = "data:image/png;base64," + data.chart_base64;
      }

      document.getElementById('resJet').innerText = data.leak_type || "-";
      document.getElementById('resMat').innerText = data.pipe_material || "-";
      document.getElementById('resDia').innerText = data.pipe_diameter || "-";
      document.getElementById('resHf').innerText = data.hf_ratio ? data.hf_ratio.toFixed(2) : "-";
    }

    async function sendFeedback(outcome) {
      if (!currentResult) return;
      const memo = document.getElementById('fbMemo').value;
      try {
        const res = await fetch('/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: currentResult.filename,
            ai_decision: currentResult.leak_decision,
            ai_prob: currentResult.primary_prob,
            actual_outcome: outcome,
            memo: memo
          })
        });
        const d = await res.json();
        if (d.success) {
          const st = document.getElementById('fbStatus');
          st.innerText = "✓ 현장 실증 피드백이 클라우드 DB에 안전하게 기록되었습니다.";
          st.classList.remove('hidden');
        }
      } catch (err) {
        alert("피드백 전송 실패: " + err.message);
      }
    }
  </script>
</body>
</html>
"""

# ----------------------------------------------------------------------
# 5. REST API 및 PWA 엔드포인트
# ----------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_PAGE)

@app.route('/manifest.json', methods=['GET'])
def manifest():
    manifest_data = {
        "name": "서용 누수진단 AI",
        "short_name": "누수진단",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0B132B",
        "theme_color": "#1B3065",
        "description": "서용엔지니어링 상수관망 누수음 의사결정지원 시스템"
    }
    return jsonify(manifest_data)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'time': datetime.datetime.now().isoformat(),
        'models': {
            'pure': pure_clf is not None,
            'pipe': pipe_clf is not None,
            'profiler': pipe_pkg is not None
        }
    })

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    f = request.files.get('audio') or request.files.get('file')
    if not f or f.filename == '':
        return jsonify({'success': False, 'error': '음원 파일이 누락되었습니다.'}), 400

    try: mop_code = float(request.form.get('mop_code', -1.0))
    except Exception: mop_code = -1.0
    try: pipe_di = float(request.form.get('pipe_di', -1.0))
    except Exception: pipe_di = -1.0
    try: pipe_dp = float(request.form.get('pipe_dp', 0.7))
    except Exception: pipe_dp = 0.7
    try: before_pre = float(request.form.get('before_pre', -1.0))
    except Exception: before_pre = -1.0

    ext = os.path.splitext(f.filename)[1].lower() or '.wav'
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(tmp_fd)
    try:
        f.save(tmp_path)
        res = analyze_audio(tmp_path, eff_depth=pipe_dp, mop_code=mop_code, pipe_di=pipe_di, before_pre=before_pre)
        return jsonify({
            'success': True,
            'filename': f.filename,
            'duration_sec': res.get('음원길이', 0.0),
            'leak_decision': res.get('누수_판정', '판정보류'),
            'primary_prob': res.get('누수_확률', 0.0),
            'pure_prob': res.get('순수음향_확률', 0.0),
            'pipe_prob': res.get('배관결합_확률'),
            'pipe_material': res.get('추정_관로재질', '-'),
            'pipe_diameter': res.get('추정_구경범주', '-'),
            'hf_ratio': res.get('고주파잔존비', 0.0),
            'leak_type': res.get('분출형태', '-'),
            'primary_model': res.get('적용모델', '-'),
            'chart_base64': res.get('차트_base64')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """현장 굴착 실증 결과 피드백 누적 저장"""
    try:
        data = request.get_json() or {}
        log_path = os.path.join(BASE_DIR, "field_validation_log.csv")
        file_exists = os.path.exists(log_path)
        
        with open(log_path, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['기록일시', '파일명', 'AI_판정', 'AI_확률(%)', '실제_굴착결과', '현장메모'])
            writer.writerow([
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                data.get('filename', ''),
                data.get('ai_decision', ''),
                data.get('ai_prob', ''),
                data.get('actual_outcome', ''),
                data.get('memo', '')
            ])
        return jsonify({'success': True, 'msg': '기록 완료'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
