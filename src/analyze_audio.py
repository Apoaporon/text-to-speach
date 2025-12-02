"""
音声ファイルの情報を取得・分析するスクリプト
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import soundfile as sf


def analyze_audio_file(file_path: str) -> Optional[Dict[str, Union[str, int, float]]]:
    """
    音声ファイルの詳細情報を取得する

    Args:
        file_path: 音声ファイルのパス

    Returns:
        音声情報の辞書（失敗時はNone）
        {
            'filename': ファイル名,
            'file_size_mb': ファイルサイズ(MB),
            'duration': 長さ(秒),
            'sample_rate': サンプリングレート(Hz),
            'channels': チャンネル数,
            'channel_type': 'モノラル' or 'ステレオ' or 'マルチチャンネル',
            'bit_depth': ビット深度(bit),
            'format': ファイル形式,
            'total_samples': 総サンプル数,
            'peak_amplitude': ピーク振幅,
            'rms_level': RMS レベル(dB)
        }
    """
    try:
        # ファイルの存在確認
        if not os.path.exists(file_path):
            print(f"エラー: ファイルが見つかりません - {file_path}")
            return None

        file_path_obj = Path(file_path)

        # soundfileで詳細情報を取得
        info = sf.info(file_path)

        # librosで音声データを読み込み（分析用）
        y, _ = librosa.load(file_path, sr=None, mono=False)

        # チャンネル数の判定
        if y.ndim == 1:
            channels = 1
            channel_type = "モノラル"
        else:
            channels = y.shape[0]
            if channels == 2:
                channel_type = "ステレオ"
            else:
                channel_type = f"マルチチャンネル({channels}ch)"

        # ピーク振幅とRMSレベルを計算
        if y.ndim == 1:
            peak_amplitude = float(np.max(np.abs(y)))
            rms_level = float(np.sqrt(np.mean(y**2)))
        else:
            peak_amplitude = float(np.max(np.abs(y)))
            rms_level = float(np.sqrt(np.mean(y**2)))

        # RMSレベルをdBに変換（0を避ける）
        rms_db = 20 * np.log10(rms_level) if rms_level > 0 else -np.inf

        # ファイルサイズ
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        # ビット深度の取得
        bit_depth = info.subtype_info.split('_')[-1] if info.subtype_info else 'N/A'
        if bit_depth.startswith('PCM'):
            bit_depth = bit_depth.replace('PCM', '')

        return {
            'filename': file_path_obj.name,
            'file_size_mb': round(file_size_mb, 2),
            'duration': round(info.duration, 2),
            'sample_rate': info.samplerate,
            'channels': channels,
            'channel_type': channel_type,
            'bit_depth': bit_depth,
            'format': info.format,
            'total_samples': info.frames,
            'peak_amplitude': round(peak_amplitude, 4),
            'rms_level': round(float(rms_db), 2)
        }

    except (OSError, ValueError, RuntimeError) as e:
        print(f"エラーが発生しました: {e}")
        return None


def analyze_audio_levels(
    file_path: str
) -> Optional[Dict[str, float]]:
    """
    音声ファイルのレベル情報を詳細に分析する

    Args:
        file_path: 音声ファイルのパス

    Returns:
        レベル情報の辞書
        {
            'peak_db': ピークレベル(dBFS),
            'rms_db': RMSレベル(dBFS),
            'lufs': ラウドネス(LUFS推定値),
            'crest_factor': クレストファクター(dB),
            'clipping_samples': クリッピングサンプル数,
            'clipping_percentage': クリッピング率(%),
            'dynamic_range': ダイナミックレンジ(dB),
            'headroom': ヘッドルーム(dB)
        }
    """
    try:
        # 音声データを読み込み
        y, sr = librosa.load(file_path, sr=None, mono=False)

        # ステレオの場合は平均化
        if y.ndim > 1:
            y = np.mean(y, axis=0)

        # ピークレベル（dBFS）
        peak_amplitude = np.max(np.abs(y))
        peak_db = 20 * np.log10(peak_amplitude) if peak_amplitude > 0 else -np.inf

        # RMSレベル（dBFS）
        rms = np.sqrt(np.mean(y**2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf

        # ラウドネス（簡易LUFS推定）
        lufs_estimate = rms_db - 23.0

        # クレストファクター（ピークとRMSの比）
        crest_factor = peak_db - rms_db

        # クリッピング検出（振幅が0.99以上のサンプル）
        clipping_threshold = 0.99
        clipping_samples = np.sum(np.abs(y) >= clipping_threshold)
        clipping_percentage = (clipping_samples / len(y)) * 100

        # ダイナミックレンジ
        frame_length = sr
        hop_length = sr // 2
        frame_rms = librosa.feature.rms(y=y, frame_length=int(frame_length), hop_length=int(hop_length))[0]
        noise_floor = np.percentile(frame_rms, 5)
        noise_floor_db = 20 * np.log10(noise_floor) if noise_floor > 0 else -np.inf
        dynamic_range = peak_db - noise_floor_db

        # ヘッドルーム
        headroom = 0.0 - peak_db

        return {
            'peak_db': round(peak_db, 2),
            'rms_db': round(rms_db, 2),
            'lufs': round(lufs_estimate, 2),
            'crest_factor': round(crest_factor, 2),
            'clipping_samples': int(clipping_samples),
            'clipping_percentage': round(clipping_percentage, 4),
            'dynamic_range': round(dynamic_range, 2),
            'headroom': round(headroom, 2)
        }

    except (OSError, ValueError, RuntimeError) as e:
        print(f"エラーが発生しました: {e}")
        return None


def detect_noise_types(file_path: str) -> Optional[Dict[str, Union[bool, float]]]:
    """
    音声ファイルに含まれるノイズの種類を検出する

    Args:
        file_path: 音声ファイルのパス

    Returns:
        ノイズ検出結果の辞書
        {
            'white_noise_level': ホワイトノイズレベル(dB),
            'has_white_noise': ホワイトノイズ有無,
            'hum_60hz_level': 60Hzハムノイズレベル(dB),
            'hum_50hz_level': 50Hzハムノイズレベル(dB),
            'has_hum_noise': ハムノイズ有無,
            'click_count': クリック・ポップノイズ数,
            'has_click_noise': クリック・ポップノイズ有無,
            'background_noise_level': 背景ノイズレベル(dB),
            'has_background_noise': 背景ノイズ有無
        }
    """
    try:
        # 音声データを読み込み
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # 1. ホワイトノイズ検出
        # 最初の0.5秒をノイズ区間として分析
        noise_samples = int(0.5 * sr)
        noise_segment = y[:noise_samples]
        white_noise_level = 20 * np.log10(np.std(noise_segment)) if np.std(noise_segment) > 0 else -np.inf
        has_white_noise = white_noise_level > -60.0  # -60dB以上をノイズありと判定

        # 2. ハムノイズ検出（50Hz/60Hz）
        # FFTでスペクトル分析
        fft = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), 1/sr)
        magnitude_db = 20 * np.log10(np.abs(fft) + 1e-10)

        # 60Hz付近のピークを検出
        idx_60hz = np.argmin(np.abs(freqs - 60))
        hum_60hz_level = float(magnitude_db[idx_60hz])

        # 50Hz付近のピークを検出
        idx_50hz = np.argmin(np.abs(freqs - 50))
        hum_50hz_level = float(magnitude_db[idx_50hz])

        # 周囲の平均と比較
        window = 10
        avg_around_60 = np.mean(magnitude_db[max(0, int(idx_60hz)-window):int(idx_60hz)+window])
        avg_around_50 = np.mean(magnitude_db[max(0, int(idx_50hz)-window):int(idx_50hz)+window])

        has_hum_noise = bool((hum_60hz_level - avg_around_60 > 20) or (hum_50hz_level - avg_around_50 > 20))

        # 3. クリック・ポップノイズ検出
        # 微分の急激な変化を検出
        diff = np.diff(y, prepend=y[0])
        std_diff = np.std(diff)
        clicks = np.abs(diff) > (3.0 * std_diff)
        click_count = int(np.sum(clicks))
        has_click_noise = click_count > 10  # 10個以上をノイズありと判定

        # 4. 背景ノイズ検出
        # 音声のない区間（低振幅区間）のノイズレベルを測定
        rms_per_frame = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        threshold = np.percentile(rms_per_frame, 10)  # 下位10%を無音区間とする
        silent_frames = rms_per_frame < threshold

        if np.sum(silent_frames) > 0:
            background_noise_level = 20 * np.log10(np.mean(rms_per_frame[silent_frames]))
        else:
            background_noise_level = -np.inf

        has_background_noise = background_noise_level > -50.0  # -50dB以上を背景ノイズありと判定

        return {
            'white_noise_level': round(white_noise_level, 2),
            'has_white_noise': has_white_noise,
            'hum_60hz_level': round(hum_60hz_level, 2),
            'hum_50hz_level': round(hum_50hz_level, 2),
            'has_hum_noise': has_hum_noise,
            'click_count': click_count,
            'has_click_noise': has_click_noise,
            'background_noise_level': round(background_noise_level, 2),
            'has_background_noise': has_background_noise
        }

    except (OSError, ValueError, RuntimeError) as e:
        print(f"エラーが発生しました: {e}")
        return None


def print_noise_detection(info: Dict[str, Union[bool, float]], filename: str) -> None:
    """
    ノイズ検出結果を整形して出力する

    Args:
        info: ノイズ検出情報の辞書
        filename: ファイル名
    """
    print("\n" + "=" * 70)
    print(f"【ノイズ検出分析】: {filename}")
    print("=" * 70)

    # ホワイトノイズ
    status = "🔴 検出" if info['has_white_noise'] else "✅ なし"
    print(f"ホワイトノイズ    : {status}")
    print(f"  レベル: {info['white_noise_level']:+.2f} dB")
    if info['has_white_noise']:
        print("  → スペクトル減算法による除去を推奨")

    print("-" * 70)

    # ハムノイズ
    status = "🔴 検出" if info['has_hum_noise'] else "✅ なし"
    print(f"ハムノイズ        : {status}")
    print(f"  60Hz: {info['hum_60hz_level']:+.2f} dB")
    print(f"  50Hz: {info['hum_50hz_level']:+.2f} dB")
    if info['has_hum_noise']:
        print("  → ノッチフィルタによる除去を推奨")

    print("-" * 70)

    # クリック・ポップノイズ
    status = "🔴 検出" if info['has_click_noise'] else "✅ なし"
    print(f"クリック/ポップ  : {status}")
    print(f"  検出数: {info['click_count']}個")
    if info['has_click_noise']:
        print("  → デクリック処理を推奨")

    print("-" * 70)

    # 背景ノイズ
    status = "🔴 検出" if info['has_background_noise'] else "✅ なし"
    print(f"背景ノイズ        : {status}")
    print(f"  レベル: {info['background_noise_level']:+.2f} dB")
    if info['has_background_noise']:
        print("  → 軽めのノイズ抑制を推奨（強すぎるとロボ声になるので注意）")

    print("=" * 70 + "\n")


def analyze_frequency_characteristics(file_path: str) -> Optional[Dict[str, Union[float, bool]]]:
    """
    音声ファイルの周波数特性を分析する

    Args:
        file_path: 音声ファイルのパス

    Returns:
        周波数特性分析結果の辞書
        {
            'low_freq_energy': 低域エネルギー(80Hz以下) dB,
            'needs_highpass': ハイパスフィルタ必要性,
            'voice_band_energy': 音声帯域エネルギー(300-4000Hz) dB,
            'high_freq_energy': 高域エネルギー(8kHz以上) dB,
            'needs_lowpass': ローパスフィルタ必要性,
            'sibilance_level': シビランス(6-10kHz)レベル dB,
            'needs_deesser': ディエッサー必要性,
            'spectral_centroid': スペクトル重心 Hz,
            'voice_clarity': 音声明瞭度 (0-1),
            'needs_eq_boost': 音声帯域ブースト必要性
        }
    """
    try:
        # 音声データを読み込み
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # STFTでスペクトル分析
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # 周波数ビンのインデックスを取得
        def get_freq_range(f_min, f_max):
            idx_min = np.argmin(np.abs(freqs - f_min))
            idx_max = np.argmin(np.abs(freqs - f_max))
            return idx_min, idx_max

        # 1. 低域エネルギー（80Hz以下）
        idx_low_start, idx_low_end = get_freq_range(0, 80)
        low_freq_magnitude = magnitude[idx_low_start:idx_low_end, :]
        low_freq_energy = 20 * np.log10(np.mean(low_freq_magnitude) + 1e-10)

        # 2. 音声帯域エネルギー（300-4000Hz）
        idx_voice_start, idx_voice_end = get_freq_range(300, 4000)
        voice_band_magnitude = magnitude[idx_voice_start:idx_voice_end, :]
        voice_band_energy = 20 * np.log10(np.mean(voice_band_magnitude) + 1e-10)

        # 3. 高域エネルギー（8kHz以上）
        idx_high_start, idx_high_end = get_freq_range(8000, sr/2)
        high_freq_magnitude = magnitude[idx_high_start:idx_high_end, :]
        high_freq_energy = 20 * np.log10(np.mean(high_freq_magnitude) + 1e-10)

        # 4. シビランス帯域（6-10kHz）- サ行の音
        if sr >= 20000:  # サンプリングレートが十分高い場合
            idx_sib_start, idx_sib_end = get_freq_range(6000, 10000)
            sibilance_magnitude = magnitude[idx_sib_start:idx_sib_end, :]
            sibilance_level = 20 * np.log10(np.mean(sibilance_magnitude) + 1e-10)
        else:
            sibilance_level = -np.inf

        # 5. スペクトル重心（音色の明るさ指標）
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        # 6. 音声明瞭度（音声帯域 vs 全体のエネルギー比）
        total_magnitude = magnitude
        total_energy = np.mean(total_magnitude)
        voice_energy = np.mean(voice_band_magnitude)
        voice_clarity = float(voice_energy / (total_energy + 1e-10))

        # 判定基準
        needs_highpass = low_freq_energy > -40.0  # 低域が-40dB以上なら除去推奨
        needs_lowpass = high_freq_energy > -35.0 and high_freq_energy > (voice_band_energy - 10)  # 高域ノイズ判定
        needs_deesser = sibilance_level > -30.0 and sibilance_level > (voice_band_energy - 5)  # シビランス強い
        needs_eq_boost = voice_clarity < 0.3  # 音声帯域が弱い

        return {
            'low_freq_energy': round(low_freq_energy, 2),
            'needs_highpass': needs_highpass,
            'voice_band_energy': round(voice_band_energy, 2),
            'high_freq_energy': round(high_freq_energy, 2),
            'needs_lowpass': needs_lowpass,
            'sibilance_level': round(sibilance_level, 2),
            'needs_deesser': needs_deesser,
            'spectral_centroid': round(spectral_centroid, 2),
            'voice_clarity': round(voice_clarity, 3),
            'needs_eq_boost': needs_eq_boost
        }

    except (OSError, ValueError, RuntimeError) as e:
        print(f"エラーが発生しました: {e}")
        return None


def print_frequency_analysis(info: Dict[str, Union[float, bool]], filename: str) -> None:
    """
    周波数特性分析結果を整形して出力する

    Args:
        info: 周波数特性情報の辞書
        filename: ファイル名
    """
    print("\n" + "=" * 70)
    print(f"【周波数特性分析】: {filename}")
    print("=" * 70)

    # 低域（ハイパスフィルタ判定）
    status = "🔴 要処理" if info['needs_highpass'] else "✅ 適切"
    print(f"低域エネルギー (〜80Hz)  : {status}")
    print(f"  レベル: {info['low_freq_energy']:+.2f} dB")
    if info['needs_highpass']:
        print("  → ハイパスフィルタ推奨（80〜120Hz以下カット）")
        print("     低い唸り・振動・風音を除去")

    print("-" * 70)

    # 音声帯域
    status = "🔴 弱い" if info['needs_eq_boost'] else "✅ 良好"
    print(f"音声帯域 (300-4000Hz)    : {status}")
    print(f"  レベル: {info['voice_band_energy']:+.2f} dB")
    print(f"  明瞭度: {info['voice_clarity']:.3f}")
    if info['needs_eq_boost']:
        print("  → 音声帯域のブースト推奨")
        print("     300Hz〜4kHzを+2〜3dB程度持ち上げ")

    print("-" * 70)

    # 高域（ローパスフィルタ判定）
    status = "🔴 要処理" if info['needs_lowpass'] else "✅ 適切"
    print(f"高域エネルギー (8kHz〜)  : {status}")
    print(f"  レベル: {info['high_freq_energy']:+.2f} dB")
    if info['needs_lowpass']:
        print("  → ローパスフィルタ推奨（16kHz付近でカット）")
        print("     高域ノイズを除去")

    print("-" * 70)

    # シビランス（ディエッサー判定）
    status = "🔴 強い" if info['needs_deesser'] else "✅ 適切"
    print(f"シビランス (6-10kHz)     : {status}")
    print(f"  レベル: {info['sibilance_level']:+.2f} dB")
    if info['needs_deesser']:
        print("  → ディエッサー推奨")
        print("     サ行の刺さる音（シビランス）を抑制")

    print("-" * 70)

    # スペクトル重心
    print(f"スペクトル重心           : {info['spectral_centroid']:.2f} Hz")
    if info['spectral_centroid'] < 1000:
        print("  音色が暗め・こもり気味")
    elif info['spectral_centroid'] > 3000:
        print("  音色が明るめ・シャープ")
    else:
        print("  音色バランス良好")

    print("=" * 70 + "\n")


def analyze_silence_and_voice(file_path: str, silence_thresh_db: float = -40.0) -> Optional[Dict[str, Union[float, int, List[tuple]]]]:
    """
    音声ファイルの無音区間と音声区間を分析する

    Args:
        file_path: 音声ファイルのパス
        silence_thresh_db: 無音判定の閾値（dBFS）

    Returns:
        無音・音声区間分析結果の辞書
        {
            'total_duration': 総再生時間(秒),
            'silence_duration': 無音区間の合計時間(秒),
            'voice_duration': 音声区間の合計時間(秒),
            'silence_ratio': 無音区間の割合(0-1),
            'silence_segments': 無音区間のリスト[(開始時刻, 終了時刻), ...],
            'voice_segments': 音声区間のリスト[(開始時刻, 終了時刻), ...],
            'leading_silence': 先頭の無音時間(秒),
            'trailing_silence': 末尾の無音時間(秒),
            'longest_silence': 最長無音区間(秒),
            'voice_segment_count': 音声区間の数,
            'needs_trim': 先頭・末尾のトリミング必要性,
            'needs_compression': 長い無音圧縮必要性,
            'needs_vad_split': VAD分割必要性
        }
    """
    try:
        # 音声を読み込み
        y, sr = librosa.load(file_path, sr=None, mono=True)
        total_duration = len(y) / sr

        # RMSエネルギーを計算（フレーム単位）
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # dBFSに変換
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # フレームの時刻を計算
        frames = range(len(rms_db))
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

        # 無音判定
        is_silence = rms_db < silence_thresh_db

        # 区間を検出（連続する無音/音声をグループ化）
        silence_segments = []
        voice_segments = []

        i = 0
        while i < len(is_silence):
            if is_silence[i]:
                # 無音区間の開始
                start = times[i]
                while i < len(is_silence) and is_silence[i]:
                    i += 1
                end = times[i-1] if i < len(times) else total_duration
                silence_segments.append((float(start), float(end)))
            else:
                # 音声区間の開始
                start = times[i]
                while i < len(is_silence) and not is_silence[i]:
                    i += 1
                end = times[i-1] if i < len(times) else total_duration
                voice_segments.append((float(start), float(end)))

        # 統計情報を計算
        silence_duration = sum(end - start for start, end in silence_segments)
        voice_duration = sum(end - start for start, end in voice_segments)
        silence_ratio = silence_duration / total_duration if total_duration > 0 else 0

        # 先頭と末尾の無音
        leading_silence = silence_segments[0][1] - silence_segments[0][0] if silence_segments and silence_segments[0][0] < 0.1 else 0.0
        trailing_silence = silence_segments[-1][1] - silence_segments[-1][0] if silence_segments and silence_segments[-1][1] > (total_duration - 0.1) else 0.0

        # 最長無音区間
        longest_silence = max((end - start for start, end in silence_segments), default=0.0)

        # 音声区間の数
        voice_segment_count = len(voice_segments)

        # 処理の必要性判定
        needs_trim = leading_silence > 0.5 or trailing_silence > 0.5  # 0.5秒以上の無音
        needs_compression = longest_silence > 1.0  # 1秒以上の無音がある
        needs_vad_split = voice_segment_count > 5 and total_duration > 30  # 長いファイルで複数の音声区間

        return {
            'total_duration': round(total_duration, 2),
            'silence_duration': round(silence_duration, 2),
            'voice_duration': round(voice_duration, 2),
            'silence_ratio': round(silence_ratio, 3),
            'silence_segments': silence_segments,
            'voice_segments': voice_segments,
            'leading_silence': round(leading_silence, 2),
            'trailing_silence': round(trailing_silence, 2),
            'longest_silence': round(longest_silence, 2),
            'voice_segment_count': voice_segment_count,
            'needs_trim': needs_trim,
            'needs_compression': needs_compression,
            'needs_vad_split': needs_vad_split
        }

    except (OSError, ValueError, RuntimeError) as e:
        print(f"エラーが発生しました: {e}")
        return None


def print_silence_analysis(info: Dict[str, Union[float, int, List[tuple]]], filename: str) -> None:
    """
    無音・音声区間分析結果を整形して出力する

    Args:
        info: 無音・音声区間情報の辞書
        filename: ファイル名
    """
    print("\n" + "=" * 70)
    print(f"【無音・音声区間分析】: {filename}")
    print("=" * 70)

    # 基本統計
    print(f"総再生時間          : {info['total_duration']:.2f} 秒")
    silence_ratio = float(info['silence_ratio']) if isinstance(info['silence_ratio'], (int, float)) else 0.0
    print(f"音声区間            : {info['voice_duration']:.2f} 秒 ({(1-silence_ratio)*100:.1f}%)")
    print(f"無音区間            : {info['silence_duration']:.2f} 秒 ({silence_ratio*100:.1f}%)")

    print("-" * 70)

    # 先頭・末尾の無音
    status = "🔴 要処理" if info['needs_trim'] else "✅ 適切"
    print(f"先頭・末尾の無音    : {status}")
    print(f"  先頭: {info['leading_silence']:.2f} 秒")
    print(f"  末尾: {info['trailing_silence']:.2f} 秒")
    if info['needs_trim']:
        print("  → 先頭・末尾のトリミング推奨")

    print("-" * 70)

    # 長い無音区間
    status = "🔴 要処理" if info['needs_compression'] else "✅ 適切"
    print(f"最長無音区間        : {status}")
    print(f"  長さ: {info['longest_silence']:.2f} 秒")
    if info['needs_compression']:
        print("  → 長い無音の圧縮推奨（例: 1.0秒以上→0.5秒に短縮）")

    print("-" * 70)

    # VAD分割
    status = "🔴 推奨" if info['needs_vad_split'] else "✅ 不要"
    print(f"VAD分割             : {status}")
    print(f"  音声区間数: {info['voice_segment_count']}個")
    if info['needs_vad_split']:
        print("  → VADによる音声区間ごとの分割を推奨")
        print("     モデル学習用データとして効率的")

    print("-" * 70)

    # 音声区間の詳細（最初の5個まで）
    print("音声区間の詳細 (最初の5個):")
    voice_segments = info.get('voice_segments', [])
    if isinstance(voice_segments, list):
        for i, (start, end) in enumerate(voice_segments[:5], 1):
            duration = end - start
            print(f"  {i}. {start:.2f}s - {end:.2f}s (長さ: {duration:.2f}s)")

        if len(voice_segments) > 5:
            print(f"  ... 他 {len(voice_segments) - 5}個の音声区間")

    print("=" * 70 + "\n")


def print_level_analysis(info: Dict[str, float], filename: str) -> None:
    """
    レベル分析結果を整形して出力する

    Args:
        info: レベル情報の辞書
        filename: ファイル名
    """
    print("\n" + "=" * 70)
    print(f"【音量・クリッピング分析】: {filename}")
    print("=" * 70)
    print(f"ピークレベル      : {info['peak_db']:+.2f} dBFS")
    print(f"RMSレベル         : {info['rms_db']:+.2f} dBFS")
    print(f"ラウドネス(推定)  : {info['lufs']:+.2f} LUFS")
    print(f"クレストファクター: {info['crest_factor']:+.2f} dB")
    print(f"ダイナミックレンジ: {info['dynamic_range']:+.2f} dB")
    print(f"ヘッドルーム      : {info['headroom']:+.2f} dB")
    print("-" * 70)
    print(f"クリッピング      : {info['clipping_samples']}サンプル ({info['clipping_percentage']:.4f}%)")

    # 推奨事項を表示
    print("-" * 70)
    print("【推奨事項】")

    # ピークレベルのチェック
    if info['peak_db'] > -1.0:
        print("⚠️  ピークが高すぎます（-1dBFS以上）→ ピークノーマライズ推奨（-1〜-3dBFS）")
    elif info['peak_db'] < -10.0:
        print("⚠️  ピークが低すぎます（-10dBFS以下）→ ゲインアップ推奨")
    else:
        print("✅ ピークレベルは適切です")

    # ラウドネスのチェック
    if info['lufs'] > -16.0:
        print("⚠️  ラウドネスが高すぎます → ラウドネスノーマライズ推奨（-20〜-16 LUFS）")
    elif info['lufs'] < -30.0:
        print("⚠️  ラウドネスが低すぎます → ラウドネスノーマライズ推奨")
    else:
        print("✅ ラウドネスは適切です")

    # クリッピングのチェック
    if info['clipping_percentage'] > 0.01:
        print(f"🔴 クリッピング検出！ ({info['clipping_percentage']:.4f}%) → デクリップ処理推奨")
    elif info['clipping_percentage'] > 0:
        print(f"⚠️  わずかなクリッピングあり ({info['clipping_percentage']:.4f}%)")
    else:
        print("✅ クリッピングなし")

    # ヘッドルームのチェック
    if info['headroom'] < 1.0:
        print("⚠️  ヘッドルームが不足 → ピークを下げることを推奨")

    print("=" * 70 + "\n")


def print_audio_info(info: Dict[str, Union[str, int, float]]) -> None:
    """
    音声情報を整形して出力する

    Args:
        info: 音声情報の辞書
    """
    print("\n" + "=" * 70)
    print("【音声ファイル情報】")
    print("=" * 70)
    print(f"ファイル名        : {info['filename']}")
    print(f"ファイルサイズ    : {info['file_size_mb']} MB")
    print(f"形式              : {info['format']}")
    print("-" * 60)
    print(f"長さ              : {info['duration']} 秒")
    print(f"サンプリングレート: {info['sample_rate']} Hz")
    print(f"チャンネル        : {info['channel_type']} ({info['channels']}ch)")
    print(f"ビット深度        : {info['bit_depth']} bit")
    print(f"総サンプル数      : {info['total_samples']:,}")
    print("-" * 60)
    print(f"ピーク振幅        : {info['peak_amplitude']}")
    print(f"RMSレベル         : {info['rms_level']} dB")
    print("=" * 60 + "\n")


def analyze_directory(directory: str, extensions: Optional[List[str]] = None) -> List[Dict]:
    """
    ディレクトリ内のすべての音声ファイルを分析する

    Args:
        directory: 対象ディレクトリ
        extensions: 対象とする拡張子リスト（デフォルト: ['.wav', '.mp3', '.m4a', '.flac', '.ogg']）

    Returns:
        分析結果のリスト
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']

    results: List[Dict[str, Union[str, int, float]]] = []
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"エラー: ディレクトリが見つかりません - {directory}")
        return results

    # 音声ファイルを検索
    audio_files: List[Path] = []
    for ext in extensions:
        audio_files.extend(dir_path.glob(f"*{ext}"))

    print(f"\n{len(audio_files)}個の音声ファイルが見つかりました\n")

    for audio_file in sorted(audio_files):
        print(f"分析中: {audio_file.name}...")
        info = analyze_audio_file(str(audio_file))
        if info:
            results.append(info)
            print_audio_info(info)

    return results


def generate_summary_report(results: List[Dict], output_file: str = "audio_analysis_report.txt") -> None:
    """
    分析結果のサマリーレポートを生成する

    Args:
        results: 分析結果のリスト
        output_file: 出力ファイル名
    """
    if not results:
        print("分析結果がありません")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("音声ファイル分析レポート\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"分析ファイル数: {len(results)}\n\n")

        # 各ファイルの詳細
        for i, info in enumerate(results, 1):
            f.write(f"\n--- ファイル {i} ---\n")
            f.write(f"ファイル名        : {info['filename']}\n")
            f.write(f"ファイルサイズ    : {info['file_size_mb']} MB\n")
            f.write(f"形式              : {info['format']}\n")
            f.write(f"長さ              : {info['duration']} 秒\n")
            f.write(f"サンプリングレート: {info['sample_rate']} Hz\n")
            f.write(f"チャンネル        : {info['channel_type']}\n")
            f.write(f"ビット深度        : {info['bit_depth']} bit\n")
            f.write(f"総サンプル数      : {info['total_samples']:,}\n")
            f.write(f"ピーク振幅        : {info['peak_amplitude']}\n")
            f.write(f"RMSレベル         : {info['rms_level']} dB\n")

        # サマリー統計
        f.write("\n" + "=" * 80 + "\n")
        f.write("サマリー統計\n")
        f.write("=" * 80 + "\n")

        total_duration = sum(r['duration'] for r in results)
        total_size = sum(r['file_size_mb'] for r in results)
        sample_rates = [r['sample_rate'] for r in results]
        channels = [r['channels'] for r in results]

        f.write(f"総再生時間        : {total_duration:.2f} 秒 ({total_duration/60:.2f} 分)\n")
        f.write(f"総ファイルサイズ  : {total_size:.2f} MB\n")
        f.write(f"サンプリングレート: {set(sample_rates)}\n")
        f.write(f"チャンネル数      : {set(channels)}\n")

    print(f"\nレポートを保存しました: {output_file}")


def main():
    """メイン処理"""
    # 単一ファイルの分析例
    print("=" * 60)
    print("【単一ファイルの分析】")
    print("=" * 60)

    audio_file = "downloads/audio/【粗品】最近のSNSニュース斬った【1人賛否】.wav"  # 分析対象ファイル

    if os.path.exists(audio_file):
        # 基本情報の分析
        info = analyze_audio_file(audio_file)
        if info:
            print_audio_info(info)

        # レベル分析
        level_info = analyze_audio_levels(audio_file)
        if level_info:
            print_level_analysis(level_info, Path(audio_file).name)

        # ノイズ検出
        noise_info = detect_noise_types(audio_file)
        if noise_info:
            print_noise_detection(noise_info, Path(audio_file).name)

        # 周波数特性分析
        freq_info = analyze_frequency_characteristics(audio_file)
        if freq_info:
            print_frequency_analysis(freq_info, Path(audio_file).name)

        # 無音・音声区間分析
        silence_info = analyze_silence_and_voice(audio_file, silence_thresh_db=-40.0)
        if silence_info:
            print_silence_analysis(silence_info, Path(audio_file).name)
    else:
        print(f"ファイルが見つかりません: {audio_file}")

    # ディレクトリ内の全ファイルを分析する例
    # print("\n" + "=" * 60)
    # print("【ディレクトリ分析】")
    # print("=" * 60)

    # directory = "wav_file"  # 分析対象ディレクトリ
    # results = analyze_directory(directory)

    # if results:
    #     generate_summary_report(results, "audio_analysis_report.txt")


if __name__ == "__main__":
    main()
