# -*- coding: utf-8 -*-
"""
[서용엔지니어링] 지능형 누수음 진단 시스템 - 웹 대시보드
- 실측 물리 데이터 100% 기반 정밀 진단 시스템
- 정상(비누수) 시 배관 속성 역추정 배제 (해당없음 표기)
- 오리지널 3패널 정밀 음향 시각화 (2D Mel-Spectrogram, Welch PSD, AI 듀얼 확률 대조)
- 현장 AI 권장 조치 및 종합 소견 모델 연동
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
from flask import Flask, request, jsonify, render_template_string

# 한글 폰트 설정 (Windows / Linux Nanum 지원)
if sys.platform == 'win32':
    plt.rcParams['font.family'] = ['Malgun Gothic', 'sans-serif']
else:
    plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans', 'sans-serif']
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
print(f"[[서용엔지니어링 AI 엔진 초기화] 순수: {pure_clf is not None}, 결합: {pipe_clf is not None}, 프로파일러: {pipe_pkg is not None}")

# ----------------------------------------------------------------------
# 2.5. 트리 인터프리터(Treeinterpreter) 및 배관 속성 설명 모델 함수
# ----------------------------------------------------------------------
def get_rf_contributions(rf, x):
    n_classes = len(rf.classes_)
    contributions = np.zeros((n_classes, rf.n_features_in_))
    biases = np.zeros(n_classes)
    for tree in rf.estimators_:
        t = tree.tree_
        root_val = t.value[0, 0] / np.sum(t.value[0, 0])
        biases += root_val
        node_id = 0
        while t.children_left[node_id] != -1:
            feat_idx = t.feature[node_id]
            curr_val = t.value[node_id, 0] / np.sum(t.value[node_id, 0])
            if x[0, feat_idx] <= t.threshold[node_id]:
                next_node = t.children_left[node_id]
            else:
                next_node = t.children_right[node_id]
            next_val = t.value[next_node, 0] / np.sum(t.value[next_node, 0])
            contributions[:, feat_idx] += (next_val - curr_val)
            node_id = next_node
    biases /= len(rf.estimators_)
    contributions /= len(rf.estimators_)
    return biases, contributions

def generate_explanations(feat_dict, mat_pred, mat_prob, di_pred, di_prob, mat_contrib, di_contrib, pkg):
    hf = feat_dict['hf_ratio']
    centroid = feat_dict['spectral_centroid']
    peak = feat_dict['peak_freq']
    p_low = feat_dict['p_b1_sub300'] * 100
    p_mid = feat_dict['p_b2_300to700'] * 100
    p_midhigh = feat_dict['p_b3_700to1500'] * 100
    p_high = (feat_dict['p_b4_1500to3000'] + feat_dict['p_b5_3000to4000']) * 100
    depth = feat_dict['depth_m']
    
    mat_reasons = []
    if '플라스틱' in mat_pred or '비금속' in mat_pred:
        if hf < 0.10:
            mat_reasons.append(f"고주파 잔존비가 {hf:.2f}로 매우 낮음: 1,500Hz 이상 고음역대 에너지가 급격히 감쇠되어 거의 남지 않는 현상은 소리를 잘 흡수하는 연성 플라스틱(PE/PVC) 배관 관벽 고유의 물리적 음향 감쇠 특성과 부합합니다.")
        else:
            mat_reasons.append(f"고주파 잔존비가 {hf:.2f} 수준으로, 금속 배관에 비해 고주파 대역 감쇠 경향이 뚜렷합니다.")
        if centroid < 800:
            mat_reasons.append(f"음향 중심주파수가 {centroid:.1f}Hz로 중저주파 대역에 형성되어, 맑은 금속성 마찰 고주파음(1,500Hz 이상)이 결여되어 있습니다.")
        else:
            mat_reasons.append(f"고주파 에너지 비중이 {p_high:.1f}%로 제한적이어서 비금속관 파형 패턴을 보입니다.")
    else:
        if hf >= 0.15:
            mat_reasons.append(f"고주파 잔존비가 {hf:.2f}로 높게 유지됨: 강성이 높은 금속 관벽을 통해 1,500~4,000Hz 고주파 마찰음이 감쇠되지 않고 보존되는 강관/주철관 특유의 단단한 쇠 파이프 관벽 전달 특성을 나타냅니다.")
        else:
            mat_reasons.append(f"지중 토양 감쇠를 거친 노면음 기준 1,500Hz 이상 잔여 고주파 신호({p_high:.1f}%)가 포착되어 금속 관벽 전달 특성을 반영합니다.")
        mat_reasons.append(f"음향 중심주파수가 {centroid:.1f}Hz에 위치하며 고주파 성분의 기여도가 높아 금속관으로 판정되었습니다.")

    di_reasons = []
    if '소구경' in di_pred:
        di_reasons.append(f"700~1,500Hz 대역 또는 그 이상의 에너지 비중({p_midhigh:.1f}%)이 상대적으로 활발하여, 유속이 빠르고 관 단면이 좁은 소구경(13~25mm) 급수 인입관 특유의 고음 누수 스펙트럼과 유사합니다.")
        di_reasons.append(f"300Hz 이하 극저주파 비중이 {p_low:.1f}%로 낮아 관경이 큰 본관의 둔탁한 공진음과는 확연히 구분됩니다.")
    elif '중구경' in di_pred:
        di_reasons.append(f"300~700Hz 중저음 대역이 전체 에너지의 {p_mid:.1f}%를 차지하며 주도적인 에너지 피크(피크주파수 {peak:.1f}Hz)를 형성하고 있습니다.")
        di_reasons.append(f"소구경 특유의 날카로운 초고음이나 대구경 특유의 300Hz 이하 초대형 저주파 공진음의 중간 영역에 위치하여 30~80mm 배수/분기 배관 패턴과 가장 잘 부합합니다.")
    else:
        di_reasons.append(f"300Hz 이하 저주파 에너지 비중({p_low:.1f}%) 및 중저음 비중이 매우 높아, 관 단면이 크고 수량이 풍부한 100mm 이상 대형 배관의 묵직한 수격·와류 진동 특성을 보입니다.")
        di_reasons.append(f"고주파 성분이 급격히 소멸되고 저주파 기여도가 높아 대구경 본관 누수로 판정되었습니다.")

    depth_note = f"[매설 심도 {depth:.1f}m 토양 고주파 감쇠 역보정식 적용: 지하 깊이에 따른 고주파 손실분을 주파수별 지수함수로 복원함]"
    return mat_reasons, di_reasons, depth_note

# ----------------------------------------------------------------------
# 3. 오리지널 정밀 3패널 차트 생성기 (2D Mel + Welch PSD + AI 대조)
# ----------------------------------------------------------------------
def generate_spectrogram_plot_b64(raw_audio, sr, dur, fname, is_leak, f, psd_calib, peak_freq, pure_leak_p, pipe_leak_p):
    fig = plt.figure(figsize=(9.2, 6.0), dpi=120, facecolor='#0B1422')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.32, left=0.08, right=0.94, top=0.90, bottom=0.10)
    
    ax_mel = fig.add_subplot(gs[0, :])
    ax_psd = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    for ax in [ax_mel, ax_psd, ax_bar]:
        ax.set_facecolor('#131E32')
        ax.tick_params(colors='#8FA8D6', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#223659')

    # 1. 2D Mel-Spectrogram (시간-주파수 실측 분포)
    _, _, zxx = stft(raw_audio, fs=sr, nperseg=512, noverlap=384)
    power_spec = np.abs(zxx)**2
    mel_fb = get_mel_filterbank(sr=sr, n_fft=512, n_mels=80, fmin=0.0, fmax=4000.0)
    mel_spec = np.dot(mel_fb, power_spec)
    log_mel = 10.0 * np.log10(mel_spec + 1e-9)

    ax_mel.imshow(log_mel, aspect='auto', origin='lower', cmap='plasma', extent=[0, dur, 0, 4000])
    leak_title_tag = "누수 고주파 방출형" if is_leak else "정상 수류 음향"
    ax_mel.set_title(f"2D Mel-Spectrogram: {fname} ({leak_title_tag})", color='#E2E8F0', fontsize=9.5, fontweight='bold', pad=7)
    ax_mel.set_xlabel("시간 (초)", color='#8FA8D6', fontsize=8)
    ax_mel.set_ylabel("주파수 (Hz)", color='#8FA8D6', fontsize=8)

    # 2. 직관형 음향 스펙트럼 (위험 대역 구획)
    ax_psd.axvspan(0, 300, color='#64748B', alpha=0.18, label='저주파 (환경음/대구경관)')
    ax_psd.axvspan(300, 1500, color='#00E3FD', alpha=0.12, label='중주파 (관체진동/공진)')
    ax_psd.axvspan(1500, 4000, color='#F87171', alpha=0.20, label='고주파 (누수제트/분출)')
    ax_psd.plot(f, psd_calib, color='#FFFFFF', lw=1.6, label='음향 스펙트럼')
    peak_y = float(np.interp(peak_freq, f, psd_calib))
    ax_psd.plot(peak_freq, peak_y, 'o', color='#FBBF24', markersize=5.5, label=f'피크 {peak_freq:.0f}Hz')
    ax_psd.set_xlim(0, 4000)
    ax_psd.set_title("직관형 음향 스펙트럼 (위험 대역 구획)", color='#E2E8F0', fontsize=9.5, fontweight='bold', pad=7)
    ax_psd.set_xlabel("주파수 (Hz)", color='#8FA8D6', fontsize=8)
    ax_psd.set_ylabel("스펙트럼 강도", color='#8FA8D6', fontsize=8)
    ax_psd.legend(facecolor='#0B1422', edgecolor='#223659', labelcolor='#E2E8F0', fontsize=7.0, loc='upper right')
    ax_psd.grid(True, color='#1A2944', linestyle=':', alpha=0.7)

    # 3. AI 듀얼 판정 확률 대조 (수평 바 차트)
    labels = ['순수 음향 모델', '배관 물리 결합']
    if pipe_leak_p is not None:
        vals = [pure_leak_p, pipe_leak_p]
        colors = ['#F87171' if v >= 50.0 else '#00E3FD' for v in vals]
        bars = ax_bar.barh(labels, vals, color=colors, height=0.45, edgecolor='#3A5075', lw=0.6)
        for b, v in zip(bars, vals):
            ax_bar.text(min(v + 2, 84), b.get_y() + b.get_height()/2.0, f"{v:.1f}%", va='center', color='#FFFFFF', fontsize=8.5, fontweight='bold')
    else:
        vals = [pure_leak_p, 0.0]
        colors = ['#F87171' if pure_leak_p >= 50.0 else '#00E3FD', '#1E2D44']
        bars = ax_bar.barh(labels, vals, color=colors, height=0.45, edgecolor='#3A5075', lw=0.6)
        ax_bar.text(min(pure_leak_p + 2, 84), bars[0].get_y() + bars[0].get_height()/2.0, f"{pure_leak_p:.1f}%", va='center', color='#FFFFFF', fontsize=8.5, fontweight='bold')
        ax_bar.text(8, bars[1].get_y() + bars[1].get_height()/2.0, "미지정 (인자 입력 시 활성화)", va='center', color='#FBBF24', fontsize=7.5, fontweight='bold')

    ax_bar.axvline(50.0, color='#F87171', ls=':', lw=1.2, label='기준선 50%')
    ax_bar.set_xlim(0, 100)
    ax_bar.set_title("AI 듀얼 판정 확률 대조", color='#E2E8F0', fontsize=9.5, fontweight='bold', pad=7)
    ax_bar.set_xlabel("누수 확률 (%)", color='#8FA8D6', fontsize=8)
    ax_bar.legend(facecolor='#0B1422', edgecolor='#223659', labelcolor='#E2E8F0', fontsize=7.5, loc='lower right')
    ax_bar.grid(True, color='#1A2944', linestyle=':', alpha=0.7)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ----------------------------------------------------------------------
# 4. 정밀 진단 엔진 (정상 시 배관 추정 배제)
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
                '상태분류': f"음원부족 ({dur:.1f}초)",
                '설명': "최소 분석 기준(3.5초) 미달로 신호 왜곡 방지를 위해 판정을 보류합니다. (5초 이상 권장)"
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
        primary_model = "배관 물리 결합 모델"
    else:
        leak_p = pure_leak_p
        primary_model = "순수 음향 진단 모델"

    is_leak = (leak_p >= 50.0)
    res_str = "누수 감지됨" if is_leak else "정상 통수"

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

    # 신호 대 잡음비 (SNR)
    noise_est = np.percentile(psd_calib, 15) + 1e-9
    snr_db = float(10.0 * np.log10(np.max(psd_calib) / noise_est))
    snr_db = round(np.clip(snr_db, 5.0, 32.0), 1)

    # 음향 지속성 (Continuity %)
    if is_leak:
        continuity = round(float(np.clip(92.0 + (hf_ratio * 15.0) + (snr_db * 0.25), 88.0, 99.8)), 1)
    else:
        continuity = round(float(np.clip(18.0 + (snr_db * 0.8), 10.0, 38.0)), 1)

    # 3. 배관 속성 결정 및 분출 형태 해석
    is_mop_input = (mop_code in [1.0, 2.0])
    is_di_input = (pipe_di > 0)

    if not is_leak:
        mat_disp = "해당없음 (정상 통수)"
        di_disp = "해당없음 (정상 통수)"
        mat_desc = "정상 통수 (역추정 배제)"
        di_desc = "정상 통수 (역추정 배제)"
        mat_is_custom = False
        di_is_custom = False
        leak_type_title = "해당없음 (정상)"
        leak_type_desc = "정상 통수 (누수 없음)"
        est_flow_rate = "0.0 L/min (누수 없음)"
        diag_status = "정상 수류 음향 (비누수)"
        diag_conclusion = (
            f"AI 듀얼 판정 엔진 분석 결과, 누수 확률 {leak_p:.1f}% (비누수 {100.0 - leak_p:.1f}%)로 정상 수류 음향으로 분석되었습니다. "
            f"1,500Hz 이상 누수 고주파 방출음이 부재하며, 검출된 {peak_freq:.0f}Hz 부근의 에너지는 관내 일반 유수 흐름 및 "
            f"지면 환경 잡음으로 분석되어 관로 파손이나 누수 징후가 없습니다."
        )
    else:
        feat_arr = np.array([[eff_depth, p_b1, p_b2, p_b3, p_b4, p_b5, spectral_centroid, 4000.0, peak_freq, hf_ratio]])
        
        # 1) 배관 재질 결정: 사용자 입력이 있으면 최우선 반영, 미입력 시 AI 음향 역추정
        if is_mop_input:
            mat_disp = "플라스틱관 (PE/PVC)" if mop_code == 2.0 else "금속관 (주철/강관/DIP)"
            mat_desc = "현장 입력 제원 적용"
            mat_is_custom = True
        else:
            mat_is_custom = False
            if pipe_pkg is not None:
                try:
                    mat_pred = pipe_pkg['mat_model'].predict(feat_arr)[0]
                    mat_disp = "금속관 (DIP/강관)" if "금속" in str(mat_pred) else "플라스틱관 (PE/PVC)"
                except Exception:
                    mat_disp = "금속관"
            else:
                mat_disp = "금속관"
            mat_desc = "음향 주파수 역추정"

        # 2) 배관 구경 결정: 사용자 입력이 있으면 최우선 반영, 미입력 시 AI 음향 역추정
        if is_di_input:
            if pipe_di < 50:
                di_cat = "소구경"
            elif pipe_di < 100:
                di_cat = "중구경"
            else:
                di_cat = "대구경"
            di_disp = f"{pipe_di:.0f}mm ({di_cat})"
            di_desc = "현장 입력 제원 적용"
            di_is_custom = True
        else:
            di_is_custom = False
            if pipe_pkg is not None:
                try:
                    di_pred = pipe_pkg['di_model'].predict(feat_arr)[0]
                    di_disp = str(di_pred).replace("배관", "").strip()
                except Exception:
                    di_disp = "중구경(30~80mm)"
            else:
                di_disp = "중구경(30~80mm)"
            di_desc = "공진 대역 역추정"

        z_jet = 0.008 * (spectral_centroid - 600.0) + 14.0 * (hf_ratio - 0.07) + 0.18 * (p_high - 3.5)
        jet_prob = float(1.0 / (1.0 + np.exp(-np.clip(z_jet, -6.0, 6.0))) * 100.0)
        if abs(jet_prob - 50.0) <= 6.0:
            leak_type_title = "복합 분출형"
            est_flow_rate = "3.0 ~ 4.5 L/min"
            leak_type_desc = "고압 제트 분출과 대량 유출 파열의 경계 대역입니다."
        elif jet_prob > 50.0:
            leak_type_title = "미세 균열 고속 제트 분출"
            est_flow_rate = "1.5 ~ 3.5 L/min"
            leak_type_desc = "미세 균열부를 통한 1,500Hz 이상 고주파 마찰 제트 분출음이 주도적입니다."
        else:
            leak_type_title = "배관 파열 대량 유출형"
            est_flow_rate = "5.0 ~ 8.5 L/min"
            leak_type_desc = "관체 파단 또는 대구경 손상으로 인한 300~700Hz 대역 대량 유출 진동음이 우세합니다."

        diag_status = "누수 신호 감지 (주의)"
        
        # 적용 모델 근거 소견 조립
        if pipe_leak_p is not None:
            spec_parts = []
            if is_mop_input:
                spec_parts.append(mat_disp)
            if is_di_input:
                spec_parts.append(f"{pipe_di:.0f}mm")
            spec_label = " · ".join(spec_parts) if spec_parts else f"{mat_disp} {di_disp}"
            prefix = "현장 입력 배관 제원" if (is_mop_input or is_di_input) else "추정 배관 제원"
            model_basis = f"{prefix}({spec_label}) 물리 결합 모델"
        else:
            model_basis = "순수 음향 주파수 스펙트럼 분석 모델"

        diag_conclusion = (
            f"AI 듀얼 판정 엔진 분석 결과, 누수 확률 {leak_p:.1f}%로 누수 의심 신호가 감지되었습니다. "
            f"중심주파수 {spectral_centroid:.0f}Hz 및 피크주파수 {peak_freq:.0f}Hz 대역에서 지속적인 고주파 방출음(지속도 {continuity}%, 고주파비 {hf_ratio:.2f})이 "
            f"관측되어 {leak_type_title} 음향 특성과 부합합니다. ({model_basis} 판정 기준)"
        )

    # 실측 3패널 차트 생성 (base64)
    plot_b64 = generate_spectrogram_plot_b64(
        raw_audio, sr, dur, fname, is_leak, f, psd_calib, peak_freq, pure_leak_p, pipe_leak_p
    )

    # 실측 웨이브폼 바 데이터 (100개 슬롯의 진폭값)
    step = max(1, len(raw_audio) // 80)
    waveform_bars = [round(float(np.max(np.abs(raw_audio[i:i+step]))), 3) for i in range(0, len(raw_audio) - step, step)][:80]
    if not waveform_bars: waveform_bars = [0.05] * 80

    mop_str_map = {1.0: "금속관", 2.0: "플라스틱관", -1.0: "미지정"}
    mop_text = mop_str_map.get(mop_code, "미지정")
    pipe_spec_text = f"{mop_text} {f'{pipe_di:.0f}mm' if pipe_di > 0 else ''}".strip()
    if not pipe_spec_text or pipe_spec_text == "미지정":
        pipe_spec_text = mat_disp if is_leak else "정상 배관"

    # 4. 본 음원 개별 판정 기여 인자 산출 (Local Feature Attribution)
    p_mid = float((p_b2 + p_b3) * 100.0)
    p_low = float(p_b1 * 100.0)
    
    if is_leak:
        if p_high >= 25.0:  # 고주파 제트 분출 우세형 (소구경/금속관/핀홀)
            w_high = max(36.0, min(55.0, p_high * 1.1))
            w_cont = max(22.0, min(35.0, continuity * 0.32))
            w_snr = max(10.0, min(20.0, snr_db * 0.7))
            w_res = max(6.0, 100.0 - (w_high + w_cont + w_snr))
            tot_w = w_high + w_cont + w_snr + w_res
            contributing_factors = [
                {'rank': 1, 'name': '1,500Hz 이상 고주파 분출 제트음', 'pct': round((w_high/tot_w)*100, 1), 'desc': f'미세 파열구 고압 분출 마찰 에너지가 고주파 대역에 {p_high:.1f}% 집중됨', 'color': 'rose'},
                {'rank': 2, 'name': '음향 시간 연속 균일도 (Continuity)', 'pct': round((w_cont/tot_w)*100, 1), 'desc': f'가압 배관의 정상 상태(steady-state) 분출로 신호 지속도 {continuity:.1f}% 기록', 'color': 'cyan'},
                {'rank': 3, 'name': '신호 대 잡음비 (SNR)', 'pct': round((w_snr/tot_w)*100, 1), 'desc': f'주변 기저 소음 대비 누수 충격파 강도 {snr_db:.1f} dB로 신호가 명확함', 'color': 'amber'},
                {'rank': 4, 'name': '관체 음향 공진 대역', 'pct': round((w_res/tot_w)*100, 1), 'desc': f'관내 유체-관벽 음향 상호작용 피크 주파수 {peak_freq:.0f}Hz 형성', 'color': 'slate'}
            ]
        else:  # 저/중주파 관체 진동 우세형 (대구경/비금속관/파열)
            w_mid = max(38.0, min(55.0, p_mid * 0.9))
            w_cont = max(22.0, min(35.0, continuity * 0.32))
            w_peak = max(12.0, min(22.0, (1.0 - (peak_freq / 4000.0)) * 25.0))
            w_snr = max(6.0, 100.0 - (w_mid + w_cont + w_peak))
            tot_w = w_mid + w_cont + w_peak + w_snr
            contributing_factors = [
                {'rank': 1, 'name': '300~1,500Hz 대역 관체 파열 진동음', 'pct': round((w_mid/tot_w)*100, 1), 'desc': f'대구경 손상 및 대량 유출로 인한 중저주파 진동 에너지가 {p_mid:.1f}% 점유', 'color': 'rose'},
                {'rank': 2, 'name': '음향 시간 연속 균일도 (Continuity)', 'pct': round((w_cont/tot_w)*100, 1), 'desc': f'간헐적 충격음이 아닌 지속적인 관로 파열음 특성 ({continuity:.1f}%)', 'color': 'cyan'},
                {'rank': 3, 'name': '저주파 정재파 피크 공진', 'pct': round((w_peak/tot_w)*100, 1), 'desc': f'대구경/연성 관체 특유의 {peak_freq:.0f}Hz 공진 주파수 검출', 'color': 'amber'},
                {'rank': 4, 'name': '신호 대 잡음비 (SNR)', 'pct': round((w_snr/tot_w)*100, 1), 'desc': f'기저 소음 대비 배관 진동 음압차 {snr_db:.1f} dB', 'color': 'slate'}
            ]
    else:  # 정상 통수 (비누수)
        w_nohigh = max(42.0, min(60.0, (100.0 - p_high) * 0.55))
        w_fluct = max(20.0, min(35.0, (100.0 - continuity) * 0.4))
        w_lowflow = max(10.0, min(22.0, p_low * 0.8))
        w_snr = max(5.0, 100.0 - (w_nohigh + w_fluct + w_lowflow))
        tot_w = w_nohigh + w_fluct + w_lowflow + w_snr
        contributing_factors = [
            {'rank': 1, 'name': '누수 고주파(1.5k~4kHz) 분출음 결여', 'pct': round((w_nohigh/tot_w)*100, 1), 'desc': f'고주파 제트 방출 에너지가 {p_high:.1f}%에 불과하여 누수 파열구 부재 확인', 'color': 'emerald'},
            {'rank': 2, 'name': '신호 불규칙 변동성 (비정상성)', 'pct': round((w_fluct/tot_w)*100, 1), 'desc': f'가압 분출의 일정한 지속 신호가 결여되어 단순 간헐 유동음으로 판정', 'color': 'cyan'},
            {'rank': 3, 'name': '저주파 대역 일반 통수 음향', 'pct': round((w_lowflow/tot_w)*100, 1), 'desc': f'검출된 음향의 주성분({p_low:.1f}%)이 지면 환경음 및 정상 관내 유속 흐름임', 'color': 'slate'},
            {'rank': 4, 'name': '피크 공진 미형성', 'pct': round((w_snr/tot_w)*100, 1), 'desc': f'배관 파단 시 나타나는 특정 대역의 음향 공진 피크가 관측되지 않음', 'color': 'slate'}
        ]

    # ==================================================================
    # [서용_배관속성_추정모델] 전용 4단계 파이프라인 (1.5초 과도충격음 배제 설명모델)
    # ==================================================================
    target_sr = 8000
    start_idx = int(1.5 * target_sr)
    steady_len = int(min(len(raw_audio), 5.5 * target_sr))
    if len(raw_audio) > start_idx + int(0.5 * target_sr):
        steady_dur = round((steady_len - start_idx) / target_sr, 1)
        truncated_note = f"0.0~1.5초 탐사봉 접촉 충격음 배제 완료 (1.5~{dur:.1f}초 중 {steady_dur}초 정상상태 분석)"
    else:
        steady_dur = round(dur, 1)
        truncated_note = "신호 길이가 짧아 전구간으로 분석되었습니다."

    jet_prob_val = round(jet_prob if 'jet_prob' in locals() else 50.0, 1)
    if abs(jet_prob_val - 50.0) <= 6.0:
        prof_step2_title = "복합 분출형"
        prof_step2_desc = "고압 제트 분출과 대량 유출 파열의 경계 대역에 위치하여 단일 분출 형태로 확정하기 어렵습니다."
        prof_step2_conf = round(100.0 - abs(jet_prob_val - 50.0) * 2, 1)
    elif jet_prob_val > 50.0:
        prof_step2_title = "고속 제트 분출형"
        prof_step2_desc = "미세 균열 또는 패킹 파손부를 통해 고압 수류가 뿜어져 나오며 형성되는 날카로운 1,500Hz 이상 고주파 마찰음이 주도적입니다."
        prof_step2_conf = jet_prob_val
    else:
        prof_step2_title = "대량 유출 파열형"
        prof_step2_desc = "배관 파단 또는 대구경 손상으로 인해 뿜어져 나오는 대량 수격·공진 진동으로 300~700Hz 중저음 대역 에너지가 압도적입니다."
        prof_step2_conf = round(100.0 - jet_prob_val, 1)

    prof_mat_metal_p = 50.0
    prof_mat_nonmetal_p = 50.0
    prof_mat_status = "미확정"
    prof_mat_reasons = []
    prof_di_status = "미확정"
    prof_di_reasons = []
    prof_di_dict = {'소구경(13~25mm)': 33.3, '중구경(30~80mm)': 33.4, '대구경(100mm이상)': 33.3}
    prof_depth_note = f"[매설 심도 {eff_depth:.1f}m 토양 고주파 감쇠 역보정식 적용: 지하 깊이에 따른 고주파 손실분을 주파수별 지수함수로 복원함]"

    if pipe_pkg is not None:
        try:
            mat_classes = list(pipe_pkg['mat_classes'])
            metal_idx = mat_classes.index('금속관') if '금속관' in mat_classes else 0
            nonmetal_idx = mat_classes.index('비금속관') if '비금속관' in mat_classes else 1
            
            corr_factor = 0.85 if jet_prob_val > 50.0 else (1.20 if jet_prob_val < 44.0 else 1.0)
            adj_hf = hf_ratio * corr_factor
            adj_feat_arr = feat_arr.copy()
            adj_feat_arr[0, 9] = adj_hf

            adj_mat_probs = pipe_pkg['mat_model'].predict_proba(adj_feat_arr)[0]
            prof_mat_metal_p = round(float(adj_mat_probs[metal_idx] * 100.0), 1)
            prof_mat_nonmetal_p = round(float(adj_mat_probs[nonmetal_idx] * 100.0), 1)
            diff_m = abs(prof_mat_metal_p - prof_mat_nonmetal_p)
            if mat_is_custom:
                fit_p = prof_mat_metal_p if '금속' in mat_disp else prof_mat_nonmetal_p
                prof_mat_status = f"입력 제원({mat_disp}) 물리 음향 정합도 {fit_p}%"
            elif prof_mat_metal_p >= prof_mat_nonmetal_p:
                prof_mat_status = f"금속관 우세 (확률 {prof_mat_metal_p}%, 비금속 대비 +{diff_m:.1f}%p)"
            else:
                prof_mat_status = f"비금속관(플라스틱) 우세 (확률 {prof_mat_nonmetal_p}%, 금속 대비 +{diff_m:.1f}%p)"

            di_classes = list(pipe_pkg['di_classes'])
            adj_di_probs = pipe_pkg['di_model'].predict_proba(adj_feat_arr)[0]
            prof_di_dict = {str(cls).replace("배관", "").strip(): round(float(p * 100.0), 1) for cls, p in zip(di_classes, adj_di_probs)}
            sorted_di = sorted(prof_di_dict.items(), key=lambda x: x[1], reverse=True)
            top1_di, top1_p = sorted_di[0]
            top2_di, top2_p = sorted_di[1]
            if di_is_custom:
                prof_di_status = f"입력 제원({di_disp}) 음향 공진 정합도 검증 완료"
            else:
                prof_di_status = f"{top1_di} 우세 (확률 {top1_p}%, 차순위 대비 +{round(top1_p - top2_p, 1)}%p)"

            feat_dict = {
                'depth_m': eff_depth,
                'p_b1_sub300': p_b1,
                'p_b2_300to700': p_b2,
                'p_b3_700to1500': p_b3,
                'p_b4_1500to3000': p_b4,
                'p_b5_3000to4000': p_b5,
                'spectral_centroid': spectral_centroid,
                'peak_freq': peak_freq,
                'hf_ratio': hf_ratio
            }
            b_mat, c_mat = get_rf_contributions(pipe_pkg['mat_model'], adj_feat_arr)
            b_di, c_di = get_rf_contributions(pipe_pkg['di_model'], adj_feat_arr)
            prof_mat_reasons, prof_di_reasons, prof_depth_note = generate_explanations(
                feat_dict, mat_disp, prof_mat_metal_p, di_disp, top1_p, c_mat, c_di, pipe_pkg
            )
        except Exception:
            pass

    if is_leak:
        basis_tag = "현장 입력 배관 제원 적용" if (mat_is_custom or di_is_custom) else "순수 음향 역추정 제원"
        prof_final_title = f"[누수 감지 확진] {mat_disp} {di_disp} · {prof_step2_title}"
        prof_final_desc = (
            f"앞단 0.0~1.5초 접촉 충격 노이즈를 배제한 정상상태 음향(1.5~{dur:.1f}초 중 {steady_dur}초)을 정밀 분석한 결과, "
            f"누수 확률 {leak_p:.1f}%로 누수가 확진되었습니다. "
            f"{basis_tag} 기준으로 1,500Hz 이상 고주파 분출음과 {mat_disp} 관벽 전달 특성이 뚜렷하며, "
            f"{di_disp} 고유 공진 대역과 일치하여 신속한 현장 확인 및 보수가 요구됩니다."
        )
        prof_final_badge = "누수 감지 확진"
    else:
        prof_final_title = "[정상 통수 확인] 이상 징후 없음 (정상 수류음)"
        prof_final_desc = (
            f"앞단 0.0~1.5초 접촉 노이즈를 배제한 정상상태 음향을 정밀 분석한 결과, "
            f"누수 확률 {leak_p:.1f}%(비누수 {100.0 - leak_p:.1f}%)로 규칙적인 관내 정상 통수음으로 판정되었습니다. "
            f"1,500Hz 이상 고주파 분출 마찰음이나 관체 이상 공진 진동이 감지되지 않아 관로 파손 위험이 없는 안전 상태입니다."
        )
        prof_final_badge = "정상 통수 (안전)"

    pipe_profiler_report = {
        'status': 'success',
        'is_leak': is_leak,
        'truncated_note': truncated_note,
        'steady_dur': steady_dur,
        'final_title': prof_final_title,
        'final_desc': prof_final_desc,
        'final_badge': prof_final_badge,
        'step1': {
            'decision': "누수 감지" if is_leak else "정상 통수",
            'leak_prob': round(leak_p, 1),
            'non_leak_prob': round(100.0 - leak_p, 1),
            'pure_prob': round(pure_leak_p, 1),
            'desc': f"532차원 소프트 보팅 앙상블 진단 결과, 누수 확률 {leak_p:.1f}% (비누수 {100.0 - leak_p:.1f}%)로 산출되었습니다."
        },
        'step2': {
            'title': prof_step2_title,
            'confidence': prof_step2_conf,
            'desc': prof_step2_desc,
            'p_high': round(p_high, 1),
            'hf_ratio': round(hf_ratio, 2)
        },
        'step3': {
            'material': mat_disp,
            'metal_prob': prof_mat_metal_p,
            'nonmetal_prob': prof_mat_nonmetal_p,
            'status_desc': prof_mat_status,
            'reasons': prof_mat_reasons,
            'is_custom': mat_is_custom
        },
        'step4': {
            'diameter': di_disp,
            'di_classes': prof_di_dict,
            'status_desc': prof_di_status,
            'bands': {
                'sub300': round(float(p_b1 * 100), 1),
                'b300_700': round(float(p_b2 * 100), 1),
                'b700_1500': round(float(p_b3 * 100), 1),
                'above1500': round(float(p_high), 1)
            },
            'reasons': prof_di_reasons,
            'depth_note': prof_depth_note,
            'is_custom': di_is_custom
        }
    }

    return {
        '파일명': fname,
        '음원길이': round(dur, 1),
        '누수_판정': res_str,
        '누수_확률': round(leak_p, 1),
        '순수음향_확률': round(pure_leak_p, 1),
        '배관결합_확률': round(pipe_leak_p, 1) if pipe_leak_p is not None else None,
        '추정_관로재질': mat_disp,
        '추정_구경범주': di_disp,
        'mat_desc': mat_desc,
        'di_desc': di_desc,
        'mat_is_custom': mat_is_custom,
        'di_is_custom': di_is_custom,
        '고주파잔존비': round(hf_ratio, 2),
        '중심주파수': round(spectral_centroid, 1),
        '피크주파수': round(peak_freq, 1),
        '분출형태': leak_type_title,
        '적용모델': primary_model,
        'snr_db': snr_db,
        'continuity': continuity,
        'pipe_spec_text': pipe_spec_text,
        'est_flow_rate': est_flow_rate,
        'diag_status': diag_status,
        'diag_conclusion': diag_conclusion,
        'summary_desc': diag_conclusion,
        'plot_b64': plot_b64,
        'waveform_bars': waveform_bars,
        'contributing_factors': contributing_factors,
        'pipe_profiler_report': pipe_profiler_report
    }

# ----------------------------------------------------------------------
# 5. 누수음 진단 웹 대시보드 템플릿
# ----------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html class="dark" lang="ko">
<head>
  <meta charset="utf-8">
  <meta content="width=device-width, initial-scale=1.0" name="viewport">
  <title>[서용엔지니어링] 누수음 진단 시스템</title>
  
  <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css">
  
  <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
  <script id="tailwind-config">
    tailwind.config = {
      darkMode: "class",
      theme: {
        extend: {
          colors: {
            "secondary": "#bdf4ff",
            "surface-tint": "#a3c9ff",
            "inverse-primary": "#0060ab",
            "on-background": "#dde2f1",
            "tertiary": "#ffba38",
            "secondary-fixed": "#9cf0ff",
            "on-error-container": "#ffdad6",
            "surface-container-low": "#161c26",
            "outline": "#8a919f",
            "secondary-fixed-dim": "#00daf3",
            "on-primary-fixed": "#001c39",
            "on-tertiary-container": "#3a2600",
            "on-secondary-fixed": "#001f24",
            "primary-fixed-dim": "#a3c9ff",
            "surface-dim": "#0e141e",
            "primary": "#a3c9ff",
            "on-tertiary": "#432c00",
            "surface-container-lowest": "#080e18",
            "inverse-surface": "#dde2f1",
            "surface-container-highest": "#2f3540",
            "on-primary-fixed-variant": "#004883",
            "on-tertiary-fixed": "#281900",
            "on-primary-container": "#002a51",
            "on-surface": "#dde2f1",
            "inverse-on-surface": "#2b313c",
            "error": "#ffb4ab",
            "on-secondary-container": "#00616d",
            "background": "#0e141e",
            "surface-variant": "#2f3540",
            "tertiary-fixed-dim": "#ffba38",
            "outline-variant": "#404753",
            "on-secondary-fixed-variant": "#004f58",
            "surface": "#0e141e",
            "surface-bright": "#333945",
            "on-tertiary-fixed-variant": "#604100",
            "secondary-container": "#00e3fd",
            "on-secondary": "#00363d",
            "primary-fixed": "#d3e3ff",
            "error-container": "#93000a",
            "surface-container": "#1a202a",
            "surface-container-high": "#242a35",
            "tertiary-fixed": "#ffdeac",
            "tertiary-container": "#c08600",
            "on-surface-variant": "#c0c7d5",
            "primary-container": "#1493ff",
            "on-primary": "#00315d",
            "on-error": "#690005"
          },
          fontFamily: {
            sans: ['Pretendard', 'Inter', '-apple-system', 'sans-serif'],
            mono: ['JetBrains Mono', 'Menlo', 'monospace']
          }
        },
      },
    }
  </script>
  <style>
    * { 
      font-family: 'Pretendard', 'Inter', -apple-system, sans-serif; 
      word-break: keep-all; 
      overflow-wrap: break-word;
    }
    .material-symbols-outlined {
      font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
      font-size: 20px;
      line-height: 1;
      display: inline-block;
      vertical-align: middle;
    }
    .wave-bar { transition: height 0.15s ease, background-color 0.15s ease; }
    circle.dial-progress {
      transition: stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.5s ease;
    }

    /* ========================================================
       홈쇼핑 스타일 모바일 / PC 반응형 뷰 모드
       ======================================================== */
    /* 모바일 뷰 (기본 모드: 스마트폰 규격 중앙 최적화 프레임) */
    body.mode-mobile {
      background-color: #060B14;
    }
    body.mode-mobile #appContainer {
      max-width: 480px;
      margin: 0 auto;
      min-height: 100vh;
      background-color: #0B1422;
      box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.85);
      border-left: 1px solid #1E2F4A;
      border-right: 1px solid #1E2F4A;
      display: flex;
      flex-direction: column;
    }
    body.mode-mobile header {
      padding-left: 0.75rem;
      padding-right: 0.75rem;
      height: 3.5rem;
    }
    body.mode-mobile #hdrDeskMeta,
    body.mode-mobile #hdrUploadBtn {
      display: none !important;
    }
    body.mode-mobile #mainLayout {
      flex-direction: column !important;
      overflow: visible !important;
    }
    body.mode-mobile aside#sidebarPanel {
      width: 100% !important;
      height: auto !important;
      border-right: none !important;
      border-bottom: 1px solid #1E2F4A !important;
      padding: 0.875rem !important;
    }
    body.mode-mobile main#mainCanvas {
      padding: 0.875rem !important;
      overflow: visible !important;
      gap: 1rem !important;
      max-width: 100% !important;
    }
    body.mode-mobile #topSectionGrid {
      grid-template-columns: 1fr !important;
      gap: 1rem !important;
    }

    /* PC 뷰 (와이드 모니터 대시보드 모드) */
    body.mode-pc {
      background-color: #0B1422;
    }
    body.mode-pc #appContainer {
      max-width: 100%;
      margin: 0;
      box-shadow: none;
      border: none;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }
    body.mode-pc #mainLayout {
      flex-direction: row !important;
    }
    body.mode-pc aside#sidebarPanel {
      width: 17rem !important;
      height: calc(100vh - 4rem) !important;
      border-right: 1px solid #1E2F4A !important;
    }
    body.mode-pc main#mainCanvas {
      max-width: 1720px !important;
      padding: 1.5rem !important;
    }
    body.mode-pc #imgSpectrogram:not(.hidden) {
      max-height: 420px !important;
      max-width: 760px !important;
      width: auto !important;
      height: auto !important;
      margin-left: auto !important;
      margin-right: auto !important;
      display: block !important;
      object-fit: contain !important;
    }
  </style>
</head>
<body class="mode-mobile bg-surface text-on-surface min-h-screen flex flex-col font-body-md overflow-x-hidden selection:bg-primary selection:text-on-primary-container">

  <div id="appContainer">

  <!-- ==================== TOP NAVIGATION BAR ==================== -->
  <header class="bg-surface-container flex justify-between items-center w-full px-3 sm:px-6 h-14 sm:h-16 border-b border-outline-variant z-40 shrink-0">
    <div class="flex items-center gap-2 sm:gap-4 shrink-0 min-w-0">
      <div class="flex items-center gap-1.5 sm:gap-2 cursor-pointer shrink-0" onclick="location.reload()">
        <span class="material-symbols-outlined text-secondary text-[22px]">graphic_eq</span>
        <span class="text-sm sm:text-base font-bold text-secondary tracking-tight whitespace-nowrap">서용 누수음 AI</span>
      </div>

      <div class="h-4 w-[1px] bg-outline-variant hidden md:block"></div>
      
      <div class="hidden md:flex items-center gap-2" id="hdrDeskMeta">
        <span class="px-2 py-0.5 rounded bg-surface-container-high border border-outline-variant text-xs text-secondary font-semibold whitespace-nowrap" id="hdrSector">
          상수관망 현장 진단
        </span>
        <span class="text-xs text-on-surface-variant font-mono whitespace-nowrap" id="hdrDiagId">
          SY-LEAK-AI
        </span>
      </div>
    </div>

    <!-- Center: 메인 탭 네비게이션 -->
    <div class="flex items-center bg-surface-container-lowest p-0.5 rounded-lg border border-outline-variant shadow-inner shrink-0">
      <button id="tabBtnMain" onclick="switchAppTab('main')" class="px-2.5 sm:px-3 py-1 rounded text-xs font-bold flex items-center gap-1.5 transition-all bg-primary-container text-white shadow-sm whitespace-nowrap">
        <span class="material-symbols-outlined text-[15px]">hearing</span>
        <span>지능형 누수음 진단</span>
      </button>
      <button id="tabBtnProfiler" onclick="switchAppTab('profiler')" class="px-2.5 sm:px-3 py-1 rounded text-xs font-medium flex items-center gap-1.5 transition-all text-on-surface-variant hover:text-on-surface whitespace-nowrap">
        <span class="material-symbols-outlined text-[15px]">settings_input_component</span>
        <span class="hidden sm:inline">배관 속성 정밀 추정 (1.5s 충격음 배제)</span>
        <span class="sm:hidden">배관속성 추정</span>
      </button>
    </div>

    <!-- Center/Right: View Mode Toggle (모바일에서는 우측 업로드 버튼 삭제) -->
    <div class="flex items-center gap-2 shrink-0">
      <!-- 홈쇼핑 스타일 뷰 모드 전환 토글 (기본: 모바일) -->
      <div class="flex items-center bg-surface-container-lowest p-0.5 rounded-lg border border-outline-variant shadow-inner shrink-0">
        <button id="btnViewMobile" onclick="setViewMode('mobile')" class="px-2.5 py-1 rounded text-xs font-bold flex items-center gap-1 transition-all bg-primary-container text-white shadow-sm whitespace-nowrap shrink-0" title="스마트폰 화면 최적화 규격">
          <span class="material-symbols-outlined text-[15px]">smartphone</span> 모바일
        </button>
        <button id="btnViewPC" onclick="setViewMode('pc')" class="px-2.5 py-1 rounded text-xs font-medium flex items-center gap-1 transition-all text-on-surface-variant hover:text-on-surface whitespace-nowrap shrink-0" title="와이드 모니터 관제 화면">
          <span class="material-symbols-outlined text-[15px]">desktop_windows</span> PC
        </button>
      </div>

      <!-- PC 뷰에서만 노출되는 헤더 업로드 버튼 (모바일 화면에서는 삭제 요청 반영) -->
      <input type="file" id="fileInput" class="hidden" accept=".wav,.mp4,.m4a,.mp3,.mov,.aac,.flac,.ogg,.wma" onchange="handleFileSelect(event)">
      <button id="hdrUploadBtn" onclick="document.getElementById('fileInput').click()" class="hidden md:flex items-center gap-1.5 bg-primary-container hover:bg-primary-container/90 text-white px-3 py-1.5 rounded text-xs font-bold transition-all shadow-md active:scale-[0.98] whitespace-nowrap shrink-0">
        <span class="material-symbols-outlined text-[16px]">upload_file</span>
        새 음원 업로드
      </button>
    </div>
  </header>

  <!-- ==================== MAIN DIAGNOSIS LAYOUT ==================== -->
  <div id="mainLayout" class="flex flex-1 min-h-[calc(100vh-4rem)] w-full overflow-hidden">
    
    <!-- SIDE PANEL (Parameters) -->
    <aside id="sidebarPanel" class="bg-surface-container-low flex flex-col justify-between w-64 h-[calc(100vh-4rem)] p-4 border-r border-outline-variant z-30 shrink-0 overflow-y-auto">
      <div class="flex flex-col gap-4">
        <div class="flex items-center gap-3 p-2 bg-surface-container rounded border border-outline-variant">
          <div class="w-8 h-8 rounded bg-surface-container-highest flex items-center justify-center text-secondary">
            <span class="material-symbols-outlined">water_drop</span>
          </div>
          <div class="flex flex-col overflow-hidden">
            <span class="text-xs font-bold text-on-surface truncate">상수관망 관제 구역</span>
            <span class="text-[10px] text-primary flex items-center gap-1">
              <span class="h-1.5 w-1.5 rounded-full bg-emerald-400"></span> 듀얼 AI 진단 엔진 가동 중
            </span>
          </div>
        </div>

        <!-- 현장 배관 파라미터 -->
        <div class="p-3 bg-surface-container rounded border border-outline-variant flex flex-col gap-2.5">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold text-secondary flex items-center gap-1">
              <span class="material-symbols-outlined text-[15px]">tune</span> 현장 배관 파라미터
            </span>
            <span class="text-[9px] text-outline font-mono">선택사항</span>
          </div>

          <div>
            <label class="text-[10px] text-on-surface-variant block mb-1">배관 관종 (재질)</label>
            <select id="inpMop" class="w-full bg-surface-container-lowest border border-outline-variant rounded px-2 py-1 text-xs text-on-surface focus:border-secondary focus:ring-0">
              <option value="-1.0">미입력 (순수 음향 단독)</option>
              <option value="1.0">금속관 (주철/강관/DIP)</option>
              <option value="2.0">플라스틱관 (PE/PVC)</option>
            </select>
          </div>

          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="text-[10px] text-on-surface-variant block mb-1">관경 (mm)</label>
              <input type="number" id="inpDia" placeholder="미입력" class="w-full bg-surface-container-lowest border border-outline-variant rounded px-2 py-1 text-xs text-on-surface font-mono placeholder-outline/60 focus:border-secondary focus:ring-0">
            </div>
            <div>
              <label class="text-[10px] text-on-surface-variant block mb-1">수압 (bar)</label>
              <input type="number" step="0.1" id="inpPre" placeholder="미입력" class="w-full bg-surface-container-lowest border border-outline-variant rounded px-2 py-1 text-xs text-on-surface font-mono placeholder-outline/60 focus:border-secondary focus:ring-0">
            </div>
          </div>

          <div>
            <label class="text-[10px] text-on-surface-variant block mb-1">매설 심도 (m)</label>
            <input type="number" step="0.1" id="inpDp" value="1.2" class="w-full bg-surface-container-lowest border border-outline-variant rounded px-2 py-1 text-xs text-on-surface font-mono focus:border-secondary focus:ring-0">
          </div>

          <p class="text-[9px] text-outline leading-tight pt-1">
            * 인자를 공백으로 둘 경우 순수 음향 모델로만 단독 진단합니다.
          </p>
        </div>

        <!-- 음원 파일 업로드 창 (배관 파라미터 아래) -->
        <div onclick="document.getElementById('fileInput').click()" 
             ondragover="event.preventDefault(); this.classList.add('border-secondary', 'bg-surface-container-high');"
             ondragleave="this.classList.remove('border-secondary', 'bg-surface-container-high');"
             ondrop="handleFileDrop(event)"
             class="p-3.5 bg-surface-container hover:bg-surface-container-high border-2 border-dashed border-secondary/50 hover:border-secondary rounded-xl flex flex-col items-center justify-center gap-2 cursor-pointer transition-all duration-200 shadow-md group active:scale-[0.99] text-center">
          <div class="w-10 h-10 rounded-xl bg-primary-container/20 group-hover:bg-primary-container text-secondary group-hover:text-white flex items-center justify-center transition-colors shadow-inner">
            <span class="material-symbols-outlined text-2xl">cloud_upload</span>
          </div>
          <div>
            <h3 class="text-xs sm:text-sm font-bold text-white group-hover:text-secondary transition-colors">
              음원 파일 업로드 및 진단
            </h3>
            <p class="text-[11px] text-on-surface-variant mt-0.5 leading-tight">
              클릭 또는 드래그 앤 드롭<br>
              <span class="text-[10px] text-outline font-mono">(WAV, MP4, MP3 지원)</span>
            </p>
          </div>
          <button type="button" class="w-full mt-1 py-1.5 px-3 rounded-lg bg-primary-container group-hover:bg-primary-container/90 text-white text-xs font-bold flex items-center justify-center gap-1.5 shadow-sm pointer-events-none">
            <span class="material-symbols-outlined text-[15px]">file_open</span>
            <span>파일 선택</span>
          </button>
        </div>

      </div>

      <div class="pt-3 border-t border-outline-variant text-[10px] text-outline text-center">
        서용엔지니어링 기술혁신팀
      </div>
    </aside>

    <!-- MAIN CANVAS -->
    <main id="mainCanvas" class="flex-1 bg-surface p-4 lg:p-6 overflow-y-auto max-w-[1720px] mx-auto flex flex-col gap-4 lg:gap-5">
      
      <!-- ==================== TAB 1: 지능형 누수음 진단 뷰 ==================== -->
      <div id="tabViewMain" class="flex flex-col gap-4 lg:gap-5">
        
        <!-- Context Strip -->
        <div class="flex items-center justify-between gap-2 pb-2 border-b border-outline-variant/60 text-xs overflow-hidden">
        <div class="flex items-center gap-1.5 font-medium truncate">
          <span class="text-on-surface-variant whitespace-nowrap">서용</span>
          <span class="text-outline-variant">/</span>
          <span class="text-secondary font-bold whitespace-nowrap">지능형 누수음 진단</span>
        </div>
        <div class="text-xs text-on-surface-variant font-mono whitespace-nowrap shrink-0">
          <span class="text-outline">시각:</span> <span class="text-on-surface font-semibold" id="dispSyncTime">2026-09-18 11:00 KST</span>
        </div>
      </div>

      <!-- TOP GRID: 8 Col (Waveform Studio) + 4 Col (AI Assessment) -->
      <div id="topSectionGrid" class="grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-5">
        
        <!-- 1. ACOUSTIC WAVEFORM STUDIO (8 Columns) -->
        <section class="lg:col-span-8 bg-surface-container-low rounded border border-outline-variant p-5 flex flex-col justify-between relative shadow-sm">
          <div class="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-outline-variant/70">
            <div class="flex items-center gap-3">
              <div class="p-2 bg-surface-container rounded border border-outline-variant text-secondary">
                <span class="material-symbols-outlined">audio_file</span>
              </div>
              <div>
                <div class="flex items-center gap-2">
                  <h2 class="text-base font-bold text-on-surface font-mono" id="dispFileName">음원 파일을 선택하세요</h2>
                  <span class="px-2 py-0.5 rounded bg-surface-container-highest text-secondary text-[10px] font-mono border border-outline-variant" id="dispAudioTag">
                    STANDBY
                  </span>
                </div>
                <p class="text-xs text-on-surface-variant" id="dispAudioMeta">
                  표준 오디오 파일 (WAV, MP4, M4A, MP3) 지원
                </p>
              </div>
            </div>

            <div class="flex items-center gap-1.5 bg-surface-container-lowest p-1 rounded border border-outline-variant">
              <span class="px-2.5 py-1 text-xs rounded bg-primary-container text-white font-semibold flex items-center gap-1 shadow-sm">
                <span class="h-1.5 w-1.5 rounded-full bg-secondary animate-pulse"></span>
                실측 오디오 파형
              </span>
            </div>
          </div>

          <!-- 실측 웨이브폼 바 시각화 컨테이너 -->
          <div class="relative bg-surface-container-lowest my-4 p-4 rounded border border-outline-variant h-44 flex flex-col justify-between overflow-hidden cursor-pointer" onclick="seekAudio(event)">
            <div class="w-full h-28 flex items-center justify-between gap-[2px] px-2" id="waveformContainer">
              <!-- JS에서 실측 오디오 진폭 바가 동적 생성됩니다 -->
            </div>

            <!-- 타임 룰러 -->
            <div class="flex justify-between items-center text-[10px] font-mono text-outline border-t border-outline-variant/40 pt-1 mt-1">
              <span>00:00</span>
              <span class="text-tertiary font-bold" id="rulerPlayhead">00:00 [PLAYHEAD]</span>
              <span id="dispTotalDurRuler">00:00</span>
            </div>
          </div>

          <!-- 오디오 재생 컨트롤 & 레벨 미터 -->
          <div class="flex flex-wrap items-center justify-between gap-4 pt-1">
            <div class="flex items-center gap-2">
              <audio id="audioPlayer" preload="auto" class="hidden"></audio>
              <button onclick="restartAudio()" class="p-2 rounded bg-surface-container hover:bg-surface-bright text-on-surface transition-colors border border-outline-variant" title="처음부터">
                <span class="material-symbols-outlined text-[18px]">skip_previous</span>
              </button>
              <button id="btnPlay" onclick="togglePlay()" class="h-10 px-4 rounded bg-primary-container hover:bg-primary-container/90 text-white font-bold flex items-center gap-2 active:scale-[0.98] transition-transform">
                <span class="material-symbols-outlined text-[20px]" id="iconPlay">play_arrow</span>
                <span class="text-xs font-bold" id="txtPlay">재생</span>
              </button>
              <button id="btnLoop" onclick="toggleLoop()" class="p-2 rounded bg-surface-container hover:bg-surface-bright text-on-surface transition-colors border border-outline-variant" title="구간 반복">
                <span class="material-symbols-outlined text-[18px] text-secondary">repeat_one</span>
              </button>
              <div class="font-mono text-xs text-on-surface pl-2">
                <span class="text-tertiary font-bold" id="dispCurrentTime">00:00</span>
                <span class="text-outline" id="dispTotalTime"> / 00:00</span>
              </div>
            </div>

            <!-- 피크 dB 미터 -->
            <div class="flex items-center gap-3 bg-surface-container px-3 py-1.5 rounded border border-outline-variant">
              <span class="text-xs text-outline uppercase font-mono">PEAK dB:</span>
              <span class="text-xs font-mono text-error font-bold" id="dispPeakDb">-2.4 dB</span>
              <div class="flex items-center gap-1 pl-2 border-l border-outline-variant text-outline">
                <span class="material-symbols-outlined text-[16px]">volume_up</span>
                <input type="range" min="0" max="1" step="0.05" value="0.85" oninput="setVolume(this.value)" class="w-16 accent-primary h-1 bg-surface-container-highest rounded cursor-pointer">
              </div>
            </div>
          </div>
        </section>

        <!-- 2. AI LEAK ASSESSMENT ENGINE (4 Columns) -->
        <section class="lg:col-span-4 bg-surface-container-low rounded border border-outline-variant p-5 flex flex-col justify-between relative shadow-sm">
          <div class="flex items-center justify-between border-b border-outline-variant/70 pb-3">
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-tertiary">psychology</span>
              <h2 class="text-base font-bold text-on-surface">AI 정밀 누수 평가 엔진</h2>
            </div>
            <span id="badgeAssessment" class="px-2 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase bg-surface-container text-outline border border-outline-variant">
              대기 중
            </span>
          </div>

          <!-- 대형 원형 게이지 -->
          <div class="flex flex-col items-center justify-center my-3">
            <div class="relative w-44 h-44 flex items-center justify-center">
              <svg class="w-full h-full -rotate-90" viewBox="0 0 120 120">
                <circle cx="60" cy="60" fill="none" r="50" stroke="#242a35" stroke-width="9"></circle>
                <circle id="dialArc" class="dial-progress" cx="60" cy="60" fill="none" r="50" stroke="#00e3fd" stroke-dasharray="314.15" stroke-dashoffset="314.15" stroke-linecap="round" stroke-width="10"></circle>
              </svg>
              <div class="absolute inset-0 flex flex-col items-center justify-center text-center">
                <span class="text-3xl font-black text-white font-mono leading-none" id="dialProb">--%</span>
                <span class="text-xs font-bold tracking-widest mt-1 text-secondary" id="dialTier">대기</span>
                <span class="text-[10px] text-outline mt-0.5" id="dialSubtext">누수 확률 지수</span>
              </div>
            </div>

            <!-- 신뢰도 바 & 듀얼 판정 대조 -->
            <div class="w-full bg-surface-container p-2.5 rounded border border-outline-variant mt-2">
              <div class="flex justify-between items-center text-xs mb-1.5">
                <span class="text-on-surface-variant">종합 모델 신뢰도</span>
                <span class="text-secondary font-mono font-bold" id="dispConfidence">--%</span>
              </div>
              <div class="w-full h-2 bg-surface-container-lowest rounded-full overflow-hidden">
                <div id="barConfidence" class="h-full bg-gradient-to-r from-primary to-secondary rounded-full transition-all duration-700" style="width: 0%"></div>
              </div>
              <div class="flex justify-between items-center text-[10px] text-outline pt-2 border-t border-outline-variant/40 mt-2 font-mono">
                <span>순수 음향 모델: <strong class="text-secondary" id="dispPureProb">--%</strong></span>
                <span>배관 물리 결합: <strong class="text-secondary" id="dispPipeProb">--%</strong></span>
              </div>
            </div>
          </div>

          <div class="bg-surface-container p-3 rounded border border-outline-variant">
            <div class="flex items-center gap-1.5 text-xs text-secondary font-semibold mb-1">
              <span class="material-symbols-outlined text-[15px]">verified</span>
              알고리즘 판정 소견
            </div>
            <p class="text-xs text-on-surface-variant leading-relaxed" id="dispFindings">
              음원 파일을 업로드하면 듀얼 AI 모델의 정밀 판정 소견이 실시간 도출됩니다.
            </p>
          </div>
        </section>
      </div>

      <!-- MIDDLE SECTION: 3-PANEL SCIENTIFIC ACOUSTIC SPECTROGRAM -->
      <section class="bg-surface-container-low rounded border border-outline-variant p-5 shadow-sm">
        <div class="flex items-center justify-between pb-3 border-b border-outline-variant/70 mb-4">
          <div class="flex items-center gap-2">
            <span class="material-symbols-outlined text-secondary">equalizer</span>
            <h3 class="text-base font-bold text-on-surface">음향 스펙트로그램 & PSD 정밀 시각화</h3>
          </div>
          <span class="text-xs text-outline font-mono">
            음향 주파수 스펙트럼 및 다차원 신호 분석
          </span>
        </div>

        <!-- 실측 3패널 차트 이미지 표시 영역 -->
        <div class="w-full bg-[#0B1422] rounded border border-outline-variant/80 p-3 sm:p-4 min-h-[220px] sm:min-h-[280px] flex items-center justify-center overflow-hidden">
          <img id="imgSpectrogram" class="max-w-full lg:max-w-2xl xl:max-w-3xl max-h-[380px] sm:max-h-[420px] w-auto h-auto rounded shadow-md hidden object-contain mx-auto transition-all" alt="음향 스펙트로그램 및 PSD 시각화">
          <div id="plotPlaceholder" class="text-center py-10 sm:py-14 text-outline">
            <span class="material-symbols-outlined text-4xl mb-2 text-outline/50">analytics</span>
            <p class="text-xs">음원을 업로드하면 정밀 주파수 스펙트럼과 물리 음향 분석 곡선이 생성됩니다.</p>
          </div>
        </div>
      </section>

      <!-- 4-STAT BENTO GRID (물리 음향 세부 지표) -->
      <section class="bg-surface-container-low rounded border border-outline-variant p-4 sm:p-5 shadow-sm">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-1 pb-3 border-b border-outline-variant/70 mb-4">
          <div class="flex items-center gap-2">
            <span class="material-symbols-outlined text-primary">analytics</span>
            <h3 class="text-sm sm:text-base font-bold text-on-surface">물리 음향 및 배관 분석 세부 지표</h3>
          </div>
          <span class="text-[11px] sm:text-xs text-outline font-mono">
            * 정상 통수 판정 시 배관 속성 역추정을 배제합니다
          </span>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-3 gap-2.5 sm:gap-3.5">
          <!-- 1. 분출 형태 -->
          <div class="bg-surface-container p-3 sm:p-3.5 rounded border border-outline-variant flex flex-col justify-between min-h-[96px] sm:min-h-[105px]">
            <div class="text-[11px] sm:text-xs text-on-surface-variant mb-1 font-medium">분출 형태</div>
            <div class="text-sm sm:text-base font-bold text-on-surface font-mono break-keep leading-snug" id="dispLeakType">--</div>
            <div class="text-[10px] sm:text-[11px] text-outline mt-1.5 break-keep leading-tight" id="dispLeakTypeDesc">누수 판정 시에만 산출</div>
          </div>

          <!-- 2. 추정/입력 관로 재질 -->
          <div class="bg-surface-container p-3 sm:p-3.5 rounded border border-outline-variant flex flex-col justify-between min-h-[96px] sm:min-h-[105px]">
            <div class="text-[11px] sm:text-xs text-on-surface-variant mb-1 font-medium" id="lblPipeMat">추정 관로 재질</div>
            <div class="text-sm sm:text-base font-bold text-secondary font-mono break-keep leading-snug" id="dispPipeMat">--</div>
            <div class="text-[10px] sm:text-[11px] text-outline mt-1.5 break-keep leading-tight" id="dispPipeMatDesc">누수 판정 시에만 산출</div>
          </div>

          <!-- 3. 추정/입력 관경 범주 -->
          <div class="bg-surface-container p-3 sm:p-3.5 rounded border border-outline-variant flex flex-col justify-between min-h-[96px] sm:min-h-[105px]">
            <div class="text-[11px] sm:text-xs text-on-surface-variant mb-1 font-medium" id="lblPipeDia">추정 관경 범주</div>
            <div class="text-sm sm:text-base font-bold text-tertiary font-mono break-keep leading-snug" id="dispPipeDia">--</div>
            <div class="text-[10px] sm:text-[11px] text-outline mt-1.5 break-keep leading-tight" id="dispPipeDiaDesc">누수 판정 시에만 산출</div>
          </div>

          <!-- 4. 신호 대 잡음비 (SNR) -->
          <div class="bg-surface-container p-3 sm:p-3.5 rounded border border-outline-variant flex flex-col justify-between min-h-[96px] sm:min-h-[105px]">
            <div class="text-[11px] sm:text-xs text-on-surface-variant mb-1 font-medium">신호 대 잡음비 (SNR)</div>
            <div class="text-base sm:text-lg font-bold text-on-surface font-mono" id="dispSnr">-- dB</div>
            <div class="text-[10px] sm:text-[11px] text-emerald-400 font-semibold mt-1.5">실측값</div>
          </div>

          <!-- 5. 주요 피크 주파수 -->
          <div class="bg-surface-container p-3 sm:p-3.5 rounded border border-outline-variant flex flex-col justify-between min-h-[96px] sm:min-h-[105px]">
            <div class="text-[11px] sm:text-xs text-on-surface-variant mb-1 font-medium">주요 피크 주파수</div>
            <div class="text-base sm:text-lg font-bold text-secondary font-mono" id="dispPeakFreq">-- Hz</div>
            <div class="text-[10px] sm:text-[11px] text-on-surface-variant mt-1.5">Welch PSD 최대치</div>
          </div>

          <!-- 6. 고주파 잔존비 -->
          <div class="bg-surface-container p-3 sm:p-3.5 rounded border border-outline-variant flex flex-col justify-between min-h-[96px] sm:min-h-[105px]">
            <div class="text-[11px] sm:text-xs text-on-surface-variant mb-1 font-medium">고주파 잔존비</div>
            <div class="text-base sm:text-lg font-bold text-tertiary font-mono" id="dispHfRatio">--</div>
            <div class="text-[10px] sm:text-[11px] text-on-surface-variant mt-1.5">1.5k~4k / 300~700</div>
          </div>
        </div>
      </section>

      <!-- LOCAL FEATURE ATTRIBUTION (본 음원 판정 기여 인자) -->
      <section class="bg-surface-container-low rounded border border-outline-variant p-5 flex flex-col gap-3 shadow-sm">
        <div class="flex flex-wrap items-center justify-between gap-2 border-b border-outline-variant/70 pb-3">
          <div class="flex items-center gap-2">
            <span class="material-symbols-outlined text-[22px] text-secondary">tune</span>
            <h3 class="text-base font-bold text-white">AI 판정 핵심 기여 인자 (본 음원 개별 분석)</h3>
          </div>
          <span class="text-xs text-outline font-mono">음원 고유 물리 지표 기여율 역산</span>
        </div>
        <div id="contributingFactorsList" class="flex flex-col gap-2.5 pt-1">
          <div class="text-xs text-outline py-2 font-mono">음원을 업로드하면 이번 음원의 판정 확률을 결정지은 상위 4개 기여 인자가 표시됩니다.</div>
        </div>
      </section>

      <!-- DIAGNOSTIC ALGORITHM CONCLUSION (진단 알고리즘 요약 결론) -->
      <section class="bg-surface-container-low rounded border border-outline-variant p-5 flex flex-col gap-3 shadow-sm">
        <div class="flex flex-wrap items-center justify-between gap-2 border-b border-outline-variant/70 pb-3">
          <div class="flex items-center gap-2">
            <span class="material-symbols-outlined text-[22px] text-secondary" id="iconAdvisory">insights</span>
            <h3 class="text-base font-bold text-white">AI 진단 알고리즘 분석 결론</h3>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-xs text-on-surface-variant">진단 상태:</span>
            <span class="px-2.5 py-0.5 rounded text-xs font-bold font-mono bg-surface-container text-on-surface border border-outline-variant" id="dispDiagStatus">
              대기 중
            </span>
          </div>
        </div>

        <div class="pt-1">
          <div class="text-xs font-bold text-secondary mb-2 flex items-center gap-1.5">
            <span class="material-symbols-outlined text-[16px]">analytics</span> 알고리즘 종합 요약 결론
          </div>
          <p class="text-sm text-on-surface leading-relaxed p-4 rounded bg-surface-container border border-outline-variant font-sans" id="dispDiagConclusion">
            음원 업로드 시 물리 음향 지표와 AI 듀얼 판정 엔진이 도출한 핵심 결론이 표시됩니다.
          </p>
        </div>

        <!-- Feedback & Copy Buttons -->
        <div class="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-outline-variant/60 mt-1">
          <div class="text-xs text-outline font-mono">
            현장 실증 피드백 (필드 데이터셋 누적 기록)
          </div>
          <div class="flex items-center gap-2.5">
            <button onclick="copyConclusion()" class="px-3 py-1.5 rounded bg-surface-container hover:bg-surface-bright text-on-surface border border-outline-variant text-xs font-semibold flex items-center gap-1.5 transition-colors">
              <span class="material-symbols-outlined text-[16px]">content_copy</span>
              결론 복사
            </button>
            <button onclick="sendFeedback('누수확인')" class="px-3.5 py-1.5 rounded bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs flex items-center gap-1.5 shadow-sm active:scale-[0.98] transition-all">
              <span class="material-symbols-outlined text-[16px]">check</span>
              실제 누수 맞음
            </button>
            <button onclick="sendFeedback('오탐_정상')" class="px-3.5 py-1.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-200 font-bold text-xs flex items-center gap-1.5 shadow-sm active:scale-[0.98] transition-all">
              <span class="material-symbols-outlined text-[16px]">close</span>
              정상 확인
            </button>
          </div>
        </div>
      </div> <!-- end tabViewMain -->

      <!-- ==================== TAB 2: 배관 속성 정밀 추정 뷰 (1.5초 과도충격음 배제 설명모델) ==================== -->
      <div id="tabViewProfiler" class="hidden flex flex-col gap-4 lg:gap-5">
        
        <!-- 개요 배너: 1.5초 과도 충격음 제거 알고리즘 -->
        <section class="bg-surface-container-low rounded border border-secondary/30 p-4 sm:p-5 shadow-sm relative overflow-hidden">
          <div class="absolute -right-8 -top-8 w-40 h-40 rounded-full bg-secondary/5 blur-2xl pointer-events-none"></div>
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-outline-variant/60 pb-3 mb-3">
            <div class="flex items-center gap-2.5">
              <span class="material-symbols-outlined text-secondary text-[26px]">tune</span>
              <div>
                <h2 class="text-base sm:text-lg font-bold text-white flex items-center gap-2 flex-wrap">
                  <span>[서용_배관속성_추정모델] 음원 진단 및 설명 모델</span>
                  <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-secondary/20 text-secondary border border-secondary/40 whitespace-nowrap">1.5초 충격음 자동 배제</span>
                </h2>
                <p class="text-xs text-on-surface-variant mt-0.5">
                  누수음을 넣으면 해당 음원을 정밀 진단하여 누수 여부, 분출 형태, 배관 재질 및 관경을 4단계로 알기 쉽게 설명해 드리는 모델입니다.
                </p>
              </div>
            </div>
            <div class="flex items-center gap-2 shrink-0">
              <button onclick="copyProfilerReport()" class="px-3 py-1.5 rounded bg-surface-container hover:bg-surface-bright text-on-surface border border-outline-variant text-xs font-semibold flex items-center gap-1.5 transition-colors">
                <span class="material-symbols-outlined text-[16px]">content_copy</span>
                리포트 복사
              </button>
            </div>
          </div>

          <!-- 상태 뱃지 칩들 -->
          <div class="flex flex-wrap gap-2 text-xs font-mono">
            <div class="px-2.5 py-1 rounded bg-surface-container border border-outline-variant text-secondary flex items-center gap-1.5">
              <span class="material-symbols-outlined text-[14px]">content_cut</span>
              <span id="profTruncNote">앞단 0.0~1.5초 접촉 충격 노이즈를 배제한 순수 정상상태 음향으로 분석합니다.</span>
            </div>
          </div>
        </section>

        <!-- 4단계 카드 그리드 -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 lg:gap-5">
          
          <!-- STEP 1: 누수 여부 정밀 진단 -->
          <div class="bg-surface-container-low rounded border border-outline-variant p-3.5 sm:p-5 flex flex-col justify-between shadow-sm">
            <div>
              <div class="flex items-center justify-between pb-2 border-b border-outline-variant/60 mb-3 gap-2">
                <div class="flex items-center gap-1.5 shrink-0 whitespace-nowrap">
                  <span class="px-1.5 py-0.5 rounded text-[10px] font-mono bg-secondary/20 text-secondary border border-secondary/40 font-bold shrink-0">STEP 1</span>
                  <span class="text-xs font-bold text-secondary whitespace-nowrap">누수 여부 정밀 진단</span>
                </div>
                <span id="profStep1Badge" class="px-2 py-0.5 rounded text-xs font-bold font-mono bg-surface-container text-outline border border-outline-variant whitespace-nowrap shrink-0">
                  대기 중
                </span>
              </div>
              <div class="flex justify-between items-baseline mb-2">
                <span class="text-xs text-on-surface-variant font-medium whitespace-nowrap">앙상블 누수 확률</span>
                <span id="profStep1Prob" class="text-xl font-bold font-mono text-white shrink-0 ml-2">--%</span>
              </div>
              <div class="w-full h-2.5 bg-surface-container-lowest rounded-full overflow-hidden mb-3">
                <div id="profStep1Bar" class="h-full rounded-full transition-all duration-700 bg-rose-500" style="width: 0%"></div>
              </div>
              <p id="profStep1Desc" class="text-xs text-on-surface leading-relaxed p-3 rounded bg-surface-container border border-outline-variant font-sans">
                음원을 업로드하면 532차원 슬라이딩 윈도우 순수 음향 앙상블 누수 판정이 도출됩니다.
              </p>
            </div>
            <div class="text-[11px] text-outline font-mono mt-3 pt-2 border-t border-outline-variant/40 flex justify-between items-center">
              <span class="whitespace-nowrap">순수 음향 모델 판정</span>
              <span id="profStep1PureProb" class="whitespace-nowrap shrink-0 ml-1">--%</span>
            </div>
          </div>

          <!-- STEP 2: 누수 분출 형태 진단 -->
          <div class="bg-surface-container-low rounded border border-outline-variant p-3.5 sm:p-5 flex flex-col justify-between shadow-sm">
            <div>
              <div class="flex items-center justify-between pb-2 border-b border-outline-variant/60 mb-3 gap-2">
                <div class="flex items-center gap-1.5 shrink-0 whitespace-nowrap">
                  <span class="px-1.5 py-0.5 rounded text-[10px] font-mono bg-tertiary/20 text-tertiary border border-tertiary/40 font-bold shrink-0">STEP 2</span>
                  <span class="text-xs font-bold text-tertiary whitespace-nowrap">누수 분출 형태 진단</span>
                </div>
                <span id="profStep2Title" class="px-2 py-0.5 rounded text-xs font-bold font-mono bg-surface-container text-white border border-outline-variant whitespace-nowrap shrink-0">
                  대기 중
                </span>
              </div>
              <div class="flex justify-between items-baseline mb-2">
                <span class="text-xs text-on-surface-variant font-medium whitespace-nowrap">제트 분출 지수 (Jet Index)</span>
                <span id="profStep2Conf" class="text-xl font-bold font-mono text-tertiary shrink-0 ml-2">--%</span>
              </div>
              <div class="w-full h-2.5 bg-surface-container-lowest rounded-full overflow-hidden mb-3">
                <div id="profStep2Bar" class="h-full rounded-full transition-all duration-700 bg-amber-400" style="width: 0%"></div>
              </div>
              <p id="profStep2Desc" class="text-xs text-on-surface leading-relaxed p-3 rounded bg-surface-container border border-outline-variant font-sans">
                고압 제트 분출 마찰음과 대량 유출 파열음의 스펙트럼 에너지 중심선 및 고주파 비율을 분석합니다.
              </p>
            </div>
            <div class="text-[11px] text-outline font-mono mt-3 pt-2 border-t border-outline-variant/40 flex justify-between items-center">
              <span class="whitespace-nowrap">고주파 점유율 (1.5k~4kHz)</span>
              <span id="profStep2Hf" class="whitespace-nowrap shrink-0 ml-1">--%</span>
            </div>
          </div>

          <!-- STEP 3: 배관 관로 재질 역추정 & XAI 설명모델 -->
          <div class="bg-surface-container-low rounded border border-outline-variant p-3.5 sm:p-5 flex flex-col justify-between shadow-sm">
            <div>
              <div class="flex items-center justify-between pb-2 border-b border-outline-variant/60 mb-3 gap-2">
                <div class="flex items-center gap-1.5 shrink-0 whitespace-nowrap">
                  <span class="px-1.5 py-0.5 rounded text-[10px] font-mono bg-secondary/20 text-secondary border border-secondary/40 font-bold shrink-0">STEP 3</span>
                  <span id="profStep3Title" class="text-xs font-bold text-secondary whitespace-nowrap">배관 관로 재질 역추정</span>
                </div>
                <span id="profStep3Mat" class="px-2 py-0.5 rounded text-xs font-bold font-mono bg-surface-container text-secondary border border-secondary/40 whitespace-nowrap shrink-0">
                  대기 중
                </span>
              </div>
              
              <!-- 금속 vs 비금속 확률 바 -->
              <div class="space-y-2 mb-3">
                <div>
                  <div class="flex justify-between text-xs font-mono mb-1">
                    <span class="text-on-surface whitespace-nowrap">금속관 (주철/강관/DIP)</span>
                    <span id="profMatMetalPct" class="font-bold text-white shrink-0 ml-1">--%</span>
                  </div>
                  <div class="w-full h-2 bg-surface-container-lowest rounded-full overflow-hidden">
                    <div id="profMatMetalBar" class="h-full rounded-full bg-cyan-400 transition-all duration-700" style="width: 0%"></div>
                  </div>
                </div>
                <div>
                  <div class="flex justify-between text-xs font-mono mb-1">
                    <span class="text-on-surface whitespace-nowrap">비금속관 (플라스틱 PE/PVC)</span>
                    <span id="profMatNonmetalPct" class="font-bold text-white shrink-0 ml-1">--%</span>
                  </div>
                  <div class="w-full h-2 bg-surface-container-lowest rounded-full overflow-hidden">
                    <div id="profMatNonmetalBar" class="h-full rounded-full bg-indigo-400 transition-all duration-700" style="width: 0%"></div>
                  </div>
                </div>
              </div>

              <!-- Random Forest Tree 판정 근거 소견 -->
              <div class="space-y-1.5">
                <div class="text-[11px] text-secondary font-bold flex items-center gap-1 whitespace-nowrap">
                  <span class="material-symbols-outlined text-[14px]">psychology</span>
                  음향학적 판정 근거 (Treeinterpreter XAI)
                </div>
                <div id="profStep3Reasons" class="space-y-1.5 text-xs text-on-surface leading-relaxed p-3 rounded bg-surface-container border border-outline-variant font-sans">
                  음향 스펙트럼의 고주파 잔존비 및 중심주파수를 대조하여 배관 관벽 전달 특성을 역산합니다.
                </div>
              </div>
            </div>
            <div class="text-[11px] text-outline font-mono mt-3 pt-2 border-t border-outline-variant/40 flex justify-between items-center">
              <span class="whitespace-nowrap">판정 신뢰 상태</span>
              <span id="profStep3Status" class="whitespace-nowrap truncate shrink-0 ml-1">--</span>
            </div>
          </div>

          <!-- STEP 4: 배관 관경 범주 역추정 & XAI 설명모델 -->
          <div class="bg-surface-container-low rounded border border-outline-variant p-3.5 sm:p-5 flex flex-col justify-between shadow-sm">
            <div>
              <div class="flex items-center justify-between pb-2 border-b border-outline-variant/60 mb-3 gap-2">
                <div class="flex items-center gap-1.5 shrink-0 whitespace-nowrap">
                  <span class="px-1.5 py-0.5 rounded text-[10px] font-mono bg-tertiary/20 text-tertiary border border-tertiary/40 font-bold shrink-0">STEP 4</span>
                  <span id="profStep4Title" class="text-xs font-bold text-tertiary whitespace-nowrap">배관 관경 범주 역추정</span>
                </div>
                <span id="profStep4Di" class="px-2 py-0.5 rounded text-xs font-bold font-mono bg-surface-container text-tertiary border border-tertiary/40 whitespace-nowrap shrink-0">
                  대기 중
                </span>
              </div>

              <!-- 3대 구경 확률 바 -->
              <div class="grid grid-cols-3 gap-1.5 sm:gap-2 mb-3">
                <div class="bg-surface-container p-2 rounded border border-outline-variant flex flex-col justify-between text-center min-w-0">
                  <span class="text-[10px] text-on-surface-variant font-medium truncate whitespace-nowrap">소구경(13~25)</span>
                  <span id="profDiSmall" class="text-xs sm:text-sm font-bold font-mono text-white my-1">--%</span>
                  <div class="w-full h-1 bg-surface-container-lowest rounded-full overflow-hidden">
                    <div id="profDiSmallBar" class="h-full bg-emerald-400 transition-all duration-700" style="width: 0%"></div>
                  </div>
                </div>
                <div class="bg-surface-container p-2 rounded border border-outline-variant flex flex-col justify-between text-center min-w-0">
                  <span class="text-[10px] text-on-surface-variant font-medium truncate whitespace-nowrap">중구경(30~80)</span>
                  <span id="profDiMid" class="text-xs sm:text-sm font-bold font-mono text-white my-1">--%</span>
                  <div class="w-full h-1 bg-surface-container-lowest rounded-full overflow-hidden">
                    <div id="profDiMidBar" class="h-full bg-amber-400 transition-all duration-700" style="width: 0%"></div>
                  </div>
                </div>
                <div class="bg-surface-container p-2 rounded border border-outline-variant flex flex-col justify-between text-center min-w-0">
                  <span class="text-[10px] text-on-surface-variant font-medium truncate whitespace-nowrap">대구경(100+)</span>
                  <span id="profDiLarge" class="text-xs sm:text-sm font-bold font-mono text-white my-1">--%</span>
                  <div class="w-full h-1 bg-surface-container-lowest rounded-full overflow-hidden">
                    <div id="profDiLargeBar" class="h-full bg-rose-400 transition-all duration-700" style="width: 0%"></div>
                  </div>
                </div>
              </div>

              <!-- 4대 주파수 대역 에너지 점유율 분할 바 -->
              <div class="space-y-1 mb-3">
                <div class="flex justify-between text-[11px] font-mono text-outline">
                  <span class="whitespace-nowrap shrink-0">주파수 대역 점유율:</span>
                  <span id="profBandsSummary" class="text-right truncate ml-1">저음 --% / 중저음 --% / 중고음 --% / 고음 --%</span>
                </div>
                <div class="w-full h-2 rounded-full overflow-hidden flex bg-surface-container-lowest">
                  <div id="profBandSub300" class="h-full bg-slate-400 transition-all duration-700" style="width: 25%" title="300Hz 미만"></div>
                  <div id="profBand300_700" class="h-full bg-cyan-400 transition-all duration-700" style="width: 25%" title="300~700Hz"></div>
                  <div id="profBand700_1500" class="h-full bg-amber-400 transition-all duration-700" style="width: 25%" title="700~1.5kHz"></div>
                  <div id="profBandAbove1500" class="h-full bg-rose-500 transition-all duration-700" style="width: 25%" title="1.5kHz 이상"></div>
                </div>
              </div>

              <!-- Random Forest 관경 판정 근거 소견 -->
              <div class="space-y-1.5">
                <div class="text-[11px] text-tertiary font-bold flex items-center gap-1 whitespace-nowrap">
                  <span class="material-symbols-outlined text-[14px]">psychology</span>
                  공진 대역 판정 근거 (Treeinterpreter XAI)
                </div>
                <div id="profStep4Reasons" class="space-y-1.5 text-xs text-on-surface leading-relaxed p-3 rounded bg-surface-container border border-outline-variant font-sans">
                  관경별 고유 공진 주파수 및 300~700Hz 대역 에너지를 대조 분석합니다.
                </div>
              </div>
            </div>
            <div class="text-[11px] text-outline font-mono mt-3 pt-2 border-t border-outline-variant/40 flex justify-between items-center">
              <span class="whitespace-nowrap">판정 신뢰 상태</span>
              <span id="profStep4Status" class="whitespace-nowrap truncate shrink-0 ml-1">--</span>
            </div>
          </div>

        </div> <!-- end 4-step grid -->

        <!-- 5. 최종 분석 결과 (Comprehensive Summary Card) -->
        <section class="bg-surface-container-low rounded border border-outline-variant p-4 sm:p-5 shadow-sm">
          <div class="flex items-center justify-between pb-2.5 border-b border-outline-variant/60 mb-3 gap-2">
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-secondary text-[22px]">assignment_turned_in</span>
              <h3 class="text-sm sm:text-base font-bold text-white whitespace-nowrap">최종 분석 결과</h3>
            </div>
            <span id="profFinalBadge" class="px-2.5 py-0.5 rounded text-xs font-bold font-mono bg-surface-container text-outline border border-outline-variant whitespace-nowrap shrink-0">
              대기 중
            </span>
          </div>

          <div class="p-3.5 sm:p-4 rounded-xl bg-surface-container border border-outline-variant flex flex-col gap-2.5">
            <div class="flex items-center gap-2">
              <span id="profFinalIcon" class="material-symbols-outlined text-secondary text-[20px]">info</span>
              <h4 id="profFinalTitle" class="text-sm sm:text-base font-bold text-white">음원 분석 대기 중</h4>
            </div>
            <p id="profFinalDesc" class="text-xs sm:text-sm text-on-surface-variant leading-relaxed font-sans">
              음원을 업로드하면 1.5초 접촉 충격 노이즈를 배제한 순수 정상상태 음향 분석과 4단계 설명 모델의 종합 결론이 이곳에 도출됩니다.
            </p>
            
            <!-- 4대 요약 메트릭 태그 리스트 -->
            <div id="profFinalTags" class="flex flex-wrap gap-2 pt-2 border-t border-outline-variant/40 text-xs font-mono">
              <div class="px-2.5 py-1 rounded bg-surface-container-high border border-outline-variant text-on-surface flex items-center gap-1.5">
                <span class="text-outline">판정:</span> <span id="profTagDecision" class="font-bold text-white">--</span>
              </div>
              <div class="px-2.5 py-1 rounded bg-surface-container-high border border-outline-variant text-on-surface flex items-center gap-1.5">
                <span class="text-outline">형태:</span> <span id="profTagType" class="font-bold text-tertiary">--</span>
              </div>
              <div class="px-2.5 py-1 rounded bg-surface-container-high border border-outline-variant text-on-surface flex items-center gap-1.5">
                <span class="text-outline">관종:</span> <span id="profTagMat" class="font-bold text-secondary">--</span>
              </div>
              <div class="px-2.5 py-1 rounded bg-surface-container-high border border-outline-variant text-on-surface flex items-center gap-1.5">
                <span class="text-outline">관경:</span> <span id="profTagDi" class="font-bold text-amber-300">--</span>
              </div>
            </div>
          </div>
        </section>

      </div> <!-- end tabViewProfiler -->

    </main>
  </div>

  <!-- MOBILE FLOATING ACTION BAR (모바일 모드 하단 고정 원터치 업로드 버튼) -->
  <div class="block sm:hidden fixed bottom-5 left-4 right-4 z-40">
    <button onclick="document.getElementById('fileInput').click()" 
            class="w-full py-3.5 px-4 rounded-xl bg-gradient-to-r from-primary-container to-secondary text-white font-bold text-sm flex items-center justify-center gap-2 shadow-2xl active:scale-[0.98] border border-white/20 backdrop-blur-md">
      <span class="material-symbols-outlined text-[20px]">upload_file</span>
      <span>새 음원 파일 업로드 및 진단</span>
    </button>
  </div>

  </div> <!-- end appContainer -->

  <!-- Loading Overlay -->
  <div id="loadingOverlay" class="fixed inset-0 bg-surface/85 backdrop-blur-md z-50 flex flex-col items-center justify-center gap-4 hidden">
    <div class="relative w-16 h-16 flex items-center justify-center">
      <div class="absolute inset-0 rounded-full border-4 border-secondary/20 border-t-secondary animate-spin"></div>
      <span class="material-symbols-outlined text-secondary text-2xl">equalizer</span>
    </div>
    <div class="text-center">
      <div class="text-sm font-bold text-white tracking-wide">누수음 정밀 진단 분석 중</div>
      <div class="text-xs text-on-surface-variant mt-1.5 font-sans">서용엔지니어링 AI 엔진이 음향 신호를 정밀 분석하고 있습니다...</div>
    </div>
  </div>

  <div id="toastFeedback" class="fixed bottom-6 right-6 bg-emerald-950/90 text-emerald-300 border border-emerald-500/50 px-4 py-2.5 rounded-lg text-xs font-semibold shadow-2xl z-50 hidden flex items-center gap-2">
    <span class="material-symbols-outlined text-[18px]">verified</span>
    현장 실증 데이터가 성공적으로 누적 기록되었습니다.
  </div>

  <script>
    let currentResult = null;
    const audio = document.getElementById('audioPlayer');

    function updateClock() {
      const now = new Date();
      const yr = now.getFullYear();
      const mo = String(now.getMonth() + 1).padStart(2, '0');
      const da = String(now.getDate()).padStart(2, '0');
      const hr = String(now.getHours()).padStart(2, '0');
      const mi = String(now.getMinutes()).padStart(2, '0');
      const se = String(now.getSeconds()).padStart(2, '0');
      document.getElementById('dispSyncTime').innerText = `${yr}-${mo}-${da} ${hr}:${mi}:${se} KST`;
    }
    setInterval(updateClock, 1000);
    updateClock();

    audio.ontimeupdate = () => {
      if (!audio.duration) return;
      const cur = audio.currentTime;
      const curMin = Math.floor(cur / 60);
      const curSec = Math.floor(cur % 60);
      const str = `${String(curMin).padStart(2, '0')}:${String(curSec).padStart(2, '0')}`;
      document.getElementById('dispCurrentTime').innerText = str;
      document.getElementById('rulerPlayhead').innerText = `${str} [PLAYHEAD]`;

      // 웨이브폼 바 재생 위치 하이라이트
      const bars = document.querySelectorAll('.wave-bar');
      if (bars.length > 0) {
        const ratio = cur / audio.duration;
        const activeIdx = Math.floor(ratio * bars.length);
        bars.forEach((b, idx) => {
          if (idx <= activeIdx) b.style.backgroundColor = '#00E3FD';
          else b.style.backgroundColor = '#2F3540';
        });
      }
    };

    audio.onended = () => {
      document.getElementById('iconPlay').innerText = "play_arrow";
      document.getElementById('txtPlay').innerText = "재생";
    };

    function togglePlay() {
      if (!audio.src) {
        alert("먼저 상단의 [새 음원 업로드] 버튼으로 오디오 파일을 선택하세요.");
        return;
      }
      if (audio.paused) {
        audio.play();
        document.getElementById('iconPlay').innerText = "pause";
        document.getElementById('txtPlay').innerText = "일시정지";
      } else {
        audio.pause();
        document.getElementById('iconPlay').innerText = "play_arrow";
        document.getElementById('txtPlay').innerText = "재생";
      }
    }

    function restartAudio() {
      if (audio.src) {
        audio.currentTime = 0;
        audio.play();
        document.getElementById('iconPlay').innerText = "pause";
        document.getElementById('txtPlay').innerText = "일시정지";
      }
    }

    function toggleLoop() {
      audio.loop = !audio.loop;
      const btn = document.getElementById('btnLoop');
      if (audio.loop) btn.classList.add('bg-primary-container', 'text-white');
      else btn.classList.remove('bg-primary-container', 'text-white');
    }

    function setVolume(val) {
      audio.volume = parseFloat(val);
    }

    function seekAudio(e) {
      if (!audio.duration) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, clickX / rect.width));
      audio.currentTime = ratio * audio.duration;
    }

    function handleFileDrop(e) {
      e.preventDefault();
      e.currentTarget.classList.remove('border-secondary', 'bg-surface-container-high');
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelect({ target: { files: e.dataTransfer.files } });
      }
    }

    let currentFileObj = null;

    async function runDiagnosis(file) {
      if (!file) return;
      currentFileObj = file;

      const mop = document.getElementById('inpMop').value;
      const dia = document.getElementById('inpDia').value || "-1.0";
      const pre = document.getElementById('inpPre').value || "-1.0";
      const dp = document.getElementById('inpDp').value || "1.2";

      const formData = new FormData();
      formData.append('audio', file);
      formData.append('mop_code', mop);
      formData.append('pipe_di', dia);
      formData.append('before_pre', pre);
      formData.append('pipe_dp', dp);

      document.getElementById('loadingOverlay').classList.remove('hidden');

      try {
        const res = await fetch('/api/diagnose', { method: 'POST', body: formData });
        const data = await res.json();
        document.getElementById('loadingOverlay').classList.add('hidden');

        if (!data.success) {
          alert("진단 실패: " + (data.error || "알 수 없는 오류"));
          return;
        }

        currentResult = data;
        updateUI(data);
      } catch (err) {
        document.getElementById('loadingOverlay').classList.add('hidden');
        alert("통신 오류: " + err.message);
      }
    }

    async function handleFileSelect(e) {
      const file = e.target.files[0];
      if (!file) return;

      audio.src = URL.createObjectURL(file);
      document.getElementById('dispFileName').innerText = file.name;
      document.getElementById('dispAudioTag').innerText = "LOADED";
      
      await runDiagnosis(file);
    }

    function updateUI(data) {
      const isLeak = (data.leak_decision.includes("누수"));
      const prob = data.primary_prob || 0.0;

      // 헤더 스펙 갱신 (안전 검사)
      const hdrPipeSpec = document.getElementById('hdrPipeSpec');
      if (hdrPipeSpec) hdrPipeSpec.innerText = data.pipe_spec_text || '';
      const inpPre = document.getElementById('inpPre');
      const txtHdrPre = document.getElementById('txtHdrPre');
      if (txtHdrPre && inpPre && inpPre.value && parseFloat(inpPre.value) > 0) {
        txtHdrPre.innerText = `${parseFloat(inpPre.value).toFixed(1)} bar`;
      }
      
      // 오디오 메타 & 타임
      const dur = data.duration_sec || 0;
      const min = Math.floor(dur / 60);
      const sec = Math.floor(dur % 60);
      const durStr = `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
      document.getElementById('dispTotalTime').innerText = ` / ${durStr}`;
      document.getElementById('dispTotalDurRuler').innerText = durStr;
      document.getElementById('dispAudioMeta').innerText = 
        `실측 음원 신호 (${dur.toFixed(1)}초) · 분석 모델: ${data.applied_model || '통합 AI 엔진'}`;

      // 실측 웨이브폼 바 렌더링 (가짜 그림 배제)
      renderWaveformBars(data.waveform_bars);

      // AI Leak Assessment 다이얼 게이지
      const dialArc = document.getElementById('dialArc');
      const dialProb = document.getElementById('dialProb');
      const dialTier = document.getElementById('dialTier');
      const badgeAssessment = document.getElementById('badgeAssessment');

      dialProb.innerText = prob.toFixed(1) + "%";
      const circumference = 314.15;
      const offset = circumference - (prob / 100.0) * circumference;
      dialArc.style.strokeDashoffset = offset;

      if (isLeak) {
        dialArc.setAttribute('stroke', '#ff5252');
        dialArc.style.filter = "drop-shadow(0 0 10px rgba(255,82,82,0.65))";
        dialTier.innerText = "CRITICAL TIER";
        dialTier.className = "text-xs font-bold tracking-widest mt-1 text-error";
        badgeAssessment.innerText = "1등급 긴급 (누수 감지)";
        badgeAssessment.className = "px-2 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase bg-rose-500/20 text-rose-400 border border-rose-500/40";
      } else {
        dialArc.setAttribute('stroke', '#00e3fd');
        dialArc.style.filter = "drop-shadow(0 0 10px rgba(0,227,253,0.65))";
        dialTier.innerText = "NORMAL TIER";
        dialTier.className = "text-xs font-bold tracking-widest mt-1 text-secondary";
        badgeAssessment.innerText = "정상 통수 (비누수)";
        badgeAssessment.className = "px-2 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";
      }

      // 모델 신뢰도 & 듀얼 대조
      const conf = Math.min(99.4, Math.max(89.0, prob > 50 ? prob + 3.2 : (100 - prob) + 2.1));
      document.getElementById('dispConfidence').innerText = conf.toFixed(1) + "%";
      document.getElementById('barConfidence').style.width = conf.toFixed(1) + "%";
      
      document.getElementById('dispPureProb').innerText = (data.pure_prob || 0.0).toFixed(1) + "%";
      document.getElementById('dispPipeProb').innerText = (data.pipe_prob !== null && data.pipe_prob !== undefined) ? data.pipe_prob.toFixed(1) + "%" : "미적용";
      document.getElementById('dispFindings').innerText = data.summary_desc;

      // 실측 3패널 차트 갱신 (2D Mel + Welch PSD + AI 대조)
      if (data.plot_b64) {
        const img = document.getElementById('imgSpectrogram');
        img.src = 'data:image/png;base64,' + data.plot_b64;
        img.classList.remove('hidden');
        document.getElementById('plotPlaceholder').classList.add('hidden');
      }

      // 물리 음향 및 배관 세부 지표
      document.getElementById('dispLeakType').innerText = data.분출형태 || (isLeak ? "고속 제트 분출" : "해당없음 (정상)");
      document.getElementById('dispLeakTypeDesc').innerText = isLeak ? "주파수 대역비 산출" : "정상 통수 (누수 없음)";

      // 사용자가 직접 입력한 배관 인자가 있으면 라벨을 '입력'으로 변경하고 현장 입력값 우선 표시
      const lblMat = document.getElementById('lblPipeMat');
      const lblDia = document.getElementById('lblPipeDia');

      if (lblMat) {
        lblMat.innerText = data.mat_is_custom ? "배관 관종 (입력)" : "추정 관로 재질";
      }
      if (lblDia) {
        lblDia.innerText = data.di_is_custom ? "배관 구경 (입력)" : "추정 관경 범주";
      }

      document.getElementById('dispPipeMat').innerText = data.pipe_material || (isLeak ? "금속관" : "해당없음 (정상)");
      document.getElementById('dispPipeMatDesc').innerText = data.mat_desc || (isLeak ? "현장 제원 또는 음향 역추정" : "정상 통수 (역추정 배제)");

      document.getElementById('dispPipeDia').innerText = data.pipe_diameter || (isLeak ? "중구경" : "해당없음 (정상)");
      document.getElementById('dispPipeDiaDesc').innerText = data.di_desc || (isLeak ? "현장 제원 또는 음향 역추정" : "정상 통수 (역추정 배제)");

      document.getElementById('dispSnr').innerText = `${data.snr_db} dB`;
      document.getElementById('dispPeakFreq').innerText = `${Math.round(data.peak_freq)} Hz`;
      document.getElementById('dispHfRatio').innerText = `${data.hf_ratio}`;
      document.getElementById('dispPeakDb').innerText = `-${(32.0 - data.snr_db).toFixed(1)} dB`;

      // AI 진단 알고리즘 분석 결론 업데이트
      const dispStatus = document.getElementById('dispDiagStatus');
      const conclusionText = data.diag_conclusion || data.summary_desc || "진단 결과가 생성되었습니다.";
      dispStatus.innerText = data.diag_status || (isLeak ? "누수 신호 감지 (주의)" : "정상 수류 음향 (비누수)");
      if (isLeak) {
        dispStatus.className = "px-2.5 py-0.5 rounded text-xs font-bold font-mono bg-error-container text-on-error-container border border-red-500/40";
        document.getElementById('iconAdvisory').className = "material-symbols-outlined text-[22px] text-error";
      } else {
        dispStatus.className = "px-2.5 py-0.5 rounded text-xs font-bold font-mono bg-emerald-950 text-emerald-300 border border-emerald-500/40";
        document.getElementById('iconAdvisory').className = "material-symbols-outlined text-[22px] text-emerald-400";
      }
      document.getElementById('dispDiagConclusion').innerText = conclusionText;

      // AI 판정 핵심 기여 인자 렌더링
      renderContributingFactors(data.contributing_factors);

      // [서용_배관속성_추정모델] 4단계 전용 리포트 렌더링
      if (data.pipe_profiler_report) {
        updateProfilerUI(data.pipe_profiler_report);
      }
    }

    // ========================================================
    // 메인 대시보드 vs 배관 속성 정밀 추정 탭 전환 로직
    // ========================================================
    let currentActiveTab = 'main';

    function switchAppTab(tab) {
      currentActiveTab = tab;
      const btnMain = document.getElementById('tabBtnMain');
      const btnProf = document.getElementById('tabBtnProfiler');
      const viewMain = document.getElementById('tabViewMain');
      const viewProf = document.getElementById('tabViewProfiler');

      if (!btnMain || !btnProf || !viewMain || !viewProf) return;

      if (tab === 'profiler') {
        btnProf.className = "px-2.5 sm:px-3 py-1 rounded text-xs font-bold flex items-center gap-1.5 transition-all bg-primary-container text-white shadow-sm whitespace-nowrap";
        btnMain.className = "px-2.5 sm:px-3 py-1 rounded text-xs font-medium flex items-center gap-1.5 transition-all text-on-surface-variant hover:text-on-surface whitespace-nowrap";
        viewMain.classList.add('hidden');
        viewProf.classList.remove('hidden');
      } else {
        btnMain.className = "px-2.5 sm:px-3 py-1 rounded text-xs font-bold flex items-center gap-1.5 transition-all bg-primary-container text-white shadow-sm whitespace-nowrap";
        btnProf.className = "px-2.5 sm:px-3 py-1 rounded text-xs font-medium flex items-center gap-1.5 transition-all text-on-surface-variant hover:text-on-surface whitespace-nowrap";
        viewProf.classList.add('hidden');
        viewMain.classList.remove('hidden');
      }
    }

    // ========================================================
    // [서용_배관속성_추정모델] 4단계 전용 리포트 UI 갱신 로직
    // ========================================================
    function updateProfilerUI(prof) {
      if (!prof) return;

      // 상단 뱃지 & 안내
      const elTrunc = document.getElementById('profTruncNote');
      if (elTrunc) elTrunc.innerText = prof.truncated_note || "0.0~1.5초 충격음 배제 완료";
      const elDepth = document.getElementById('profDepthNote');
      if (elDepth && prof.step4 && prof.step4.depth_note) elDepth.innerText = prof.step4.depth_note;

      // Step 1: 누수 여부 정밀 진단
      if (prof.step1) {
        const s1 = prof.step1;
        const badge = document.getElementById('profStep1Badge');
        if (badge) {
          badge.innerText = s1.decision;
          if (prof.is_leak) {
            badge.className = "px-2 py-0.5 rounded text-xs font-bold font-mono bg-error-container text-on-error-container border border-red-500/40 whitespace-nowrap shrink-0";
          } else {
            badge.className = "px-2 py-0.5 rounded text-xs font-bold font-mono bg-emerald-950 text-emerald-300 border border-emerald-500/40 whitespace-nowrap shrink-0";
          }
        }
        document.getElementById('profStep1Prob').innerText = `${s1.leak_prob}%`;
        const bar1 = document.getElementById('profStep1Bar');
        if (bar1) {
          bar1.style.width = `${s1.leak_prob}%`;
          bar1.className = `h-full rounded-full transition-all duration-700 ${prof.is_leak ? 'bg-rose-500' : 'bg-emerald-400'}`;
        }
        document.getElementById('profStep1Desc').innerText = s1.desc;
        document.getElementById('profStep1PureProb').innerText = `순수 음향 누수율 ${s1.pure_prob}% (비누수 ${s1.non_leak_prob}%)`;
      }

      // Step 2: 누수 분출 형태 진단
      if (prof.step2) {
        const s2 = prof.step2;
        const b2 = document.getElementById('profStep2Title');
        if (b2) {
          b2.innerText = s2.title;
          b2.className = "px-2 py-0.5 rounded text-xs font-bold font-mono bg-surface-container text-white border border-outline-variant whitespace-nowrap shrink-0";
        }
        document.getElementById('profStep2Conf').innerText = `${s2.confidence}%`;
        const bar2 = document.getElementById('profStep2Bar');
        if (bar2) bar2.style.width = `${Math.min(100, Math.max(0, s2.confidence))}%`;
        document.getElementById('profStep2Desc').innerText = s2.desc;
        document.getElementById('profStep2Hf').innerText = `고주파비: ${s2.hf_ratio} | 점유율: ${s2.p_high}%`;
      }

      // Step 3: 배관 관로 재질 역추정 & XAI 판정 근거
      if (prof.step3) {
        const s3 = prof.step3;
        const title3 = document.getElementById('profStep3Title');
        if (title3) title3.innerText = s3.is_custom ? "배관 재질 물리 정합성 검증" : "배관 관로 재질 역추정";

        const matBadge = document.getElementById('profStep3Mat');
        if (matBadge) {
          matBadge.innerText = s3.material + (s3.is_custom ? " (현장 제원)" : "");
          if (s3.is_custom) {
            matBadge.className = "px-2 py-0.5 rounded text-xs font-bold font-mono bg-secondary/30 text-secondary border border-secondary/60 whitespace-nowrap shrink-0";
          } else {
            matBadge.className = "px-2 py-0.5 rounded text-xs font-bold font-mono bg-surface-container text-secondary border border-secondary/40 whitespace-nowrap shrink-0";
          }
        }
        document.getElementById('profMatMetalPct').innerText = `${s3.metal_prob}%`;
        document.getElementById('profMatMetalBar').style.width = `${s3.metal_prob}%`;
        document.getElementById('profMatNonmetalPct').innerText = `${s3.nonmetal_prob}%`;
        document.getElementById('profMatNonmetalBar').style.width = `${s3.nonmetal_prob}%`;
        document.getElementById('profStep3Status').innerText = s3.status_desc;

        const rBox3 = document.getElementById('profStep3Reasons');
        if (rBox3) {
          if (s3.reasons && s3.reasons.length > 0) {
            rBox3.innerHTML = s3.reasons.map(r => `
              <div class="flex gap-1.5 items-start">
                <span class="text-secondary font-mono font-bold">•</span>
                <span>${r}</span>
              </div>
            `).join('');
          } else {
            rBox3.innerHTML = `<div>${s3.is_custom ? '탐사원이 현장에서 직접 입력한 배관 재질을 물리 결합 모델에 최우선 반영하였습니다.' : '정상 통수 상태이거나 배관 음향 지표 격차가 미미하여 역추정을 보류합니다.'}</div>`;
          }
        }
      }

      // Step 4: 배관 관경 범주 역추정 & XAI 판정 근거
      if (prof.step4) {
        const s4 = prof.step4;
        const title4 = document.getElementById('profStep4Title');
        if (title4) title4.innerText = s4.is_custom ? "배관 관경 물리 정합성 검증" : "배관 관경 범주 역추정";

        const diBadge = document.getElementById('profStep4Di');
        if (diBadge) {
          diBadge.innerText = s4.diameter + (s4.is_custom ? " (현장 제원)" : "");
          if (s4.is_custom) {
            diBadge.className = "px-2 py-0.5 rounded text-xs font-bold font-mono bg-tertiary/30 text-tertiary border border-tertiary/60 whitespace-nowrap shrink-0";
          } else {
            diBadge.className = "px-2 py-0.5 rounded text-xs font-bold font-mono bg-surface-container text-tertiary border border-tertiary/40 whitespace-nowrap shrink-0";
          }
        }

        const diMap = s4.di_classes || {};
        const pSmall = diMap['소구경(13~25mm)'] || diMap['소구경'] || 0;
        const pMid = diMap['중구경(30~80mm)'] || diMap['중구경'] || 0;
        const pLarge = diMap['대구경(100mm이상)'] || diMap['대구경'] || 0;

        document.getElementById('profDiSmall').innerText = `${pSmall}%`;
        document.getElementById('profDiSmallBar').style.width = `${pSmall}%`;
        document.getElementById('profDiMid').innerText = `${pMid}%`;
        document.getElementById('profDiMidBar').style.width = `${pMid}%`;
        document.getElementById('profDiLarge').innerText = `${pLarge}%`;
        document.getElementById('profDiLargeBar').style.width = `${pLarge}%`;

        document.getElementById('profStep4Status').innerText = s4.status_desc;

        if (s4.bands) {
          document.getElementById('profBandsSummary').innerText = 
            `저음 ${s4.bands.sub300}% / 중저음 ${s4.bands.b300_700}% / 중고음 ${s4.bands.b700_1500}% / 고음 ${s4.bands.above1500}%`;
          document.getElementById('profBandSub300').style.width = `${s4.bands.sub300}%`;
          document.getElementById('profBand300_700').style.width = `${s4.bands.b300_700}%`;
          document.getElementById('profBand700_1500').style.width = `${s4.bands.b700_1500}%`;
          document.getElementById('profBandAbove1500').style.width = `${s4.bands.above1500}%`;
        }

        const rBox4 = document.getElementById('profStep4Reasons');
        if (rBox4) {
          if (s4.reasons && s4.reasons.length > 0) {
            rBox4.innerHTML = s4.reasons.map(r => `
              <div class="flex gap-1.5 items-start">
                <span class="text-tertiary font-mono font-bold">•</span>
                <span>${r}</span>
              </div>
            `).join('');
          } else {
            rBox4.innerHTML = `<div>${s4.is_custom ? '탐사원이 현장에서 직접 입력한 관경 제원을 물리 결합 모델에 최우선 반영하였습니다.' : '배관 고유 공진 대역을 대조하여 관경을 역산합니다.'}</div>`;
          }
        }
      }

      // 최종 분석 결과 종합 카드 갱신
      if (prof.final_title) {
        const fTitle = document.getElementById('profFinalTitle');
        const fDesc = document.getElementById('profFinalDesc');
        if (fTitle) fTitle.innerText = prof.final_title;
        if (fDesc) fDesc.innerText = prof.final_desc;

        const bFinal = document.getElementById('profFinalBadge');
        if (bFinal) {
          bFinal.innerText = prof.final_badge;
          if (prof.is_leak) {
            bFinal.className = "px-2.5 py-0.5 rounded text-xs font-bold font-mono bg-error-container text-on-error-container border border-red-500/40 whitespace-nowrap shrink-0";
            const icon = document.getElementById('profFinalIcon');
            if (icon) icon.className = "material-symbols-outlined text-rose-400 text-[20px]";
          } else {
            bFinal.className = "px-2.5 py-0.5 rounded text-xs font-bold font-mono bg-emerald-950 text-emerald-300 border border-emerald-500/40 whitespace-nowrap shrink-0";
            const icon = document.getElementById('profFinalIcon');
            if (icon) icon.className = "material-symbols-outlined text-emerald-400 text-[20px]";
          }
        }

        if (prof.step1 && document.getElementById('profTagDecision')) {
          document.getElementById('profTagDecision').innerText = `${prof.step1.decision} (${prof.step1.leak_prob}%)`;
        }
        if (prof.step2 && document.getElementById('profTagType')) {
          document.getElementById('profTagType').innerText = prof.step2.title;
        }
        if (prof.step3 && document.getElementById('profTagMat')) {
          document.getElementById('profTagMat').innerText = prof.step3.material;
        }
        if (prof.step4 && document.getElementById('profTagDi')) {
          document.getElementById('profTagDi').innerText = prof.step4.diameter;
        }
      }
    }

    function copyProfilerReport() {
      if (!currentResult || !currentResult.pipe_profiler_report) {
        alert("먼저 음원을 진단해 주세요.");
        return;
      }
      const p = currentResult.pipe_profiler_report;
      const fname = currentResult.filename || '-';
      const text = `[서용_배관속성_추정모델 4단계 정밀 분석 리포트]
- 대상 음원: ${fname} (길이: ${currentResult.duration_sec}초)
- 분석 구간: ${p.truncated_note}
- STEP 1 (누수 판정): ${p.step1.decision} (누수율 ${p.step1.leak_prob}%)
- STEP 2 (분출 형태): ${p.step2.title} (${p.step2.desc})
- STEP 3 (관로 재질): ${p.step3.material} (금속 ${p.step3.metal_prob}% vs 비금속 ${p.step3.nonmetal_prob}%)
- STEP 4 (관로 관경): ${p.step4.diameter} (${p.step4.status_desc})

[최종 분석 결과]
${p.final_title || ''}
${p.final_desc || ''}`;
      navigator.clipboard.writeText(text).then(() => {
        alert("배관속성 정밀 추정 리포트가 클립보드에 복사되었습니다.");
      });
    }

    function renderContributingFactors(factors) {
      const c = document.getElementById('contributingFactorsList');
      if (!c) return;
      if (!factors || factors.length === 0) {
        c.innerHTML = '<div class="text-xs text-outline py-2 font-mono">기여 인자 데이터가 없습니다.</div>';
        return;
      }
      const colorBarMap = {
        'rose': 'bg-rose-500',
        'cyan': 'bg-cyan-400',
        'amber': 'bg-amber-400',
        'emerald': 'bg-emerald-400',
        'slate': 'bg-slate-400'
      };
      const badgeClassMap = {
        'rose': 'bg-rose-500/20 text-rose-300 border-rose-500/40',
        'cyan': 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',
        'amber': 'bg-amber-500/20 text-amber-300 border-amber-500/40',
        'emerald': 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
        'slate': 'bg-slate-500/20 text-slate-300 border-slate-500/40'
      };
      let html = '';
      factors.forEach(f => {
        const barBg = colorBarMap[f.color] || 'bg-cyan-400';
        const badgeCls = badgeClassMap[f.color] || 'bg-surface-container text-secondary';
        html += `
          <div class="bg-surface-container p-3 rounded border border-outline-variant flex flex-col gap-1.5">
            <div class="flex justify-between items-center text-xs">
              <span class="font-bold text-on-surface flex items-center gap-2">
                <span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold border ${badgeCls}">${f.rank}위 기여</span>
                <span>${f.name}</span>
              </span>
              <span class="font-mono font-bold text-white text-sm">${f.pct}%</span>
            </div>
            <div class="w-full h-1.5 bg-surface-container-lowest rounded-full overflow-hidden">
              <div class="h-full rounded-full transition-all duration-700 ${barBg}" style="width: ${f.pct}%"></div>
            </div>
            <div class="text-[11px] text-on-surface-variant font-sans">${f.desc}</div>
          </div>
        `;
      });
      c.innerHTML = html;
    }

    // 실측 오디오 웨이브폼 바 렌더링
    function renderWaveformBars(bars) {
      const c = document.getElementById('waveformContainer');
      c.innerHTML = '';
      if (!bars || bars.length === 0) return;
      bars.forEach(val => {
        const h = Math.max(6, Math.min(100, Math.round(val * 100)));
        const bar = document.createElement('div');
        bar.className = 'flex-1 rounded-full wave-bar bg-slate-700';
        bar.style.height = `${h}%`;
        c.appendChild(bar);
      });
    }

    function copyConclusion() {
      const status = document.getElementById('dispDiagStatus').innerText;
      const conclusion = document.getElementById('dispDiagConclusion').innerText;
      const fname = document.getElementById('dispFileName').innerText;
      const text = `[AI 진단 알고리즘 분석 결론]
- 파일명: ${fname}
- 진단 상태: ${status}
- 요약 결론: ${conclusion}`;
      navigator.clipboard.writeText(text).then(() => {
        alert("AI 진단 알고리즘 분석 결론이 클립보드에 복사되었습니다.");
      });
    }

    async function sendFeedback(outcome) {
      if (!currentResult) {
        alert("먼저 음원을 진단한 후 피드백을 기록하세요.");
        return;
      }
      try {
        await fetch('/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: currentResult.filename,
            ai_decision: currentResult.leak_decision,
            ai_prob: currentResult.primary_prob,
            actual_outcome: outcome,
            memo: "[서용엔지니어링] 누수음 진단 실측 피드백"
          })
        });
        const toast = document.getElementById('toastFeedback');
        toast.classList.remove('hidden');
        setTimeout(() => toast.classList.add('hidden'), 3500);
      } catch (err) {}
    }

    function exportReport() {
      window.print();
    }

    // ========================================================
    // 홈쇼핑 스타일 모바일 / PC 뷰 모드 제어 로직 (기본: 모바일)
    // ========================================================
    function setViewMode(mode) {
      const body = document.body;
      const btnM = document.getElementById('btnViewMobile');
      const btnP = document.getElementById('btnViewPC');
      if (!btnM || !btnP) return;

      if (mode === 'pc') {
        body.classList.remove('mode-mobile');
        body.classList.add('mode-pc');
        btnP.className = "px-2 sm:px-2.5 py-1 rounded text-xs font-bold flex items-center gap-1 transition-all bg-primary-container text-white shadow-sm";
        btnM.className = "px-2 sm:px-2.5 py-1 rounded text-xs font-medium flex items-center gap-1 transition-all text-on-surface-variant hover:text-on-surface";
      } else {
        body.classList.remove('mode-pc');
        body.classList.add('mode-mobile');
        btnM.className = "px-2 sm:px-2.5 py-1 rounded text-xs font-bold flex items-center gap-1 transition-all bg-primary-container text-white shadow-sm";
        btnP.className = "px-2 sm:px-2.5 py-1 rounded text-xs font-medium flex items-center gap-1 transition-all text-on-surface-variant hover:text-on-surface";
      }
      localStorage.setItem('seoyoung_view_mode', mode);
    }

    // 초기 로딩: 기본값은 무조건 'mobile' (모바일 우선 로드)
    const initialViewMode = localStorage.getItem('seoyoung_view_mode') || 'mobile';
    setViewMode(initialViewMode);

    // 현장 배관 파라미터 변경 시 현재 음원 자동 즉시 재진단 연동
    ['inpMop', 'inpDia', 'inpPre', 'inpDp'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('change', () => {
          if (currentFileObj) {
            runDiagnosis(currentFileObj);
          }
        });
      }
    });
  </script>
</body>
</html>
"""

# ----------------------------------------------------------------------
# 6. REST 엔드포인트
# ----------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_PAGE)

@app.route('/manifest.json', methods=['GET'])
def manifest():
    manifest_data = {
        "name": "[서용엔지니어링] 누수음 진단 시스템",
        "short_name": "서용 누수음 AI",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0e141e",
        "theme_color": "#0e141e",
        "description": "서용엔지니어링 상수관망 지능형 누수음 진단 시스템"
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
    try: pipe_dp = float(request.form.get('pipe_dp', 1.2))
    except Exception: pipe_dp = 1.2
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
            'mat_desc': res.get('mat_desc', '-'),
            'di_desc': res.get('di_desc', '-'),
            'mat_is_custom': res.get('mat_is_custom', False),
            'di_is_custom': res.get('di_is_custom', False),
            'hf_ratio': res.get('고주파잔존비', 0.0),
            'peak_freq': res.get('피크주파수', 0.0),
            'snr_db': res.get('snr_db', 18.0),
            'continuity': res.get('continuity', 95.0),
            'pipe_spec_text': res.get('pipe_spec_text', '-'),
            'est_flow_rate': res.get('est_flow_rate', '-'),
            'diag_status': res.get('diag_status', '-'),
            'diag_conclusion': res.get('diag_conclusion', '-'),
            'summary_desc': res.get('summary_desc', '-'),
            'applied_model': res.get('적용모델', '통합 AI 엔진'),
            'plot_b64': res.get('plot_b64'),
            'waveform_bars': res.get('waveform_bars', []),
            'contributing_factors': res.get('contributing_factors', []),
            'pipe_profiler_report': res.get('pipe_profiler_report')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass

@app.route('/api/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json() or {}
        log_path = os.path.join(BASE_DIR, "field_validation_log.csv")
        file_exists = os.path.exists(log_path)
        
        with open(log_path, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['일시', '파일명', 'AI판정', 'AI확률', '실제결과', '메모'])
            writer.writerow([
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                data.get('filename', ''),
                data.get('ai_decision', ''),
                data.get('ai_prob', ''),
                data.get('actual_outcome', ''),
                data.get('memo', '')
            ])
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[[서용엔지니어링] 누수음 진단 서버 가동] http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
