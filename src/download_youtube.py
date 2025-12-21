"""
yt-dlpを使用してYouTube動画をダウンロードするスクリプト
"""
from pathlib import Path
from typing import Optional

try:
    import yt_dlp  # type: ignore
except ImportError:
    print("yt-dlpがインストールされていません。pip install yt-dlp を実行してください。")
    raise


def download_youtube_audio(
    url: str,
    output_dir: str = "downloads",
    audio_format: str = "wav",
    quality: str = "best"
) -> Optional[str]:
    """
    YouTube動画から音声をダウンロードする

    Args:
        url: YouTube動画のURL
        output_dir: 出力ディレクトリ (デフォルト: downloads)
        audio_format: 出力形式 ('wav', 'mp3', 'm4a' など, デフォルト: wav)
        quality: 音質 ('best', 'worst', デフォルト: best)

    Returns:
        ダウンロードしたファイルのパス（失敗時はNone）
    """
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # yt-dlpの設定
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
    }

    # 形式に応じた後処理設定
    if audio_format == 'wav':
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
    elif audio_format == 'mp3':
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192' if quality == 'best' else '128',
        }]
    elif audio_format == 'm4a':
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }]

    try:
        print(f"ダウンロード開始: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 動画情報を取得
            info = ydl.extract_info(url, download=True)

            # ダウンロードされたファイル名を取得
            if info:
                filename = ydl.prepare_filename(info)
                # 拡張子を変換後のものに変更
                output_file = str(Path(filename).with_suffix(f'.{audio_format}'))
                print(f"ダウンロード完了: {output_file}")
                return output_file
            return None

    except yt_dlp.utils.DownloadError as e:
        print(f"ダウンロードエラー: {e}")
        return None
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        return None


def download_youtube_video(
    url: str,
    output_dir: str = "downloads",
    resolution: str = "best"
) -> Optional[str]:
    """
    YouTube動画をダウンロードする（映像+音声）

    Args:
        url: YouTube動画のURL
        output_dir: 出力ディレクトリ (デフォルト: downloads)
        resolution: 解像度 ('best', '1080p', '720p', '480p' など)

    Returns:
        ダウンロードしたファイルのパス（失敗時はNone）
    """
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 解像度に応じたフォーマット指定
    if resolution == "best":
        format_spec = "bestvideo+bestaudio/best"
    else:
        format_spec = f"bestvideo[height<={resolution.replace('p', '')}]+bestaudio/best"

    # yt-dlpの設定
    ydl_opts = {
        'format': format_spec,
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': False,
    }

    try:
        print(f"動画ダウンロード開始: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            if info:
                filename = ydl.prepare_filename(info)
                # mp4形式に統一
                output_file = str(Path(filename).with_suffix('.mp4'))
                print(f"ダウンロード完了: {output_file}")
                return output_file
            return None

    except yt_dlp.utils.DownloadError as e:
        print(f"ダウンロードエラー: {e}")
        return None
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        return None


def get_video_info(url: str) -> Optional[dict]:
    """
    YouTube動画の情報を取得する

    Args:
        url: YouTube動画のURL

    Returns:
        動画情報の辞書（失敗時はNone）
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if info:
                return {
                    'title': info.get('title', 'N/A'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'N/A'),
                    'view_count': info.get('view_count', 0),
                    'description': info.get('description', 'N/A'),
                }
            return None

    except yt_dlp.utils.DownloadError as e:
        print(f"情報取得エラー: {e}")
        return None
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        return None


def main():
    """メイン処理"""
    # 使用例
    
    # YouTube動画のURL
    url = "https://youtu.be/2GSESXvHKko"  # サンプルURL
    
    # 動画情報を取得
    print("=" * 50)
    print("動画情報を取得中...")
    info = get_video_info(url)
    if info:
        print(f"タイトル: {info['title']}")
        print(f"長さ: {info['duration']}秒")
        print(f"アップローダー: {info['uploader']}")
        print(f"再生回数: {info['view_count']:,}")
    print("=" * 50)
    
    # 音声のみをダウンロード（WAV形式）
    print("\n音声ダウンロード開始...")
    audio_file = download_youtube_audio(
        url=url,
        output_dir="downloads/audio",
        audio_format="wav",
        quality="best"
    )
    
    if audio_file:
        print(f"\n音声ファイルが保存されました: {audio_file}")
    else:
        print("\n音声ダウンロードに失敗しました")
    
    # 動画をダウンロード（1080p以下）
    # print("\n動画ダウンロード開始...")
    # video_file = download_youtube_video(
    #     url=url,
    #     output_dir="downloads/video",
    #     resolution="1080p"
    # )


if __name__ == "__main__":
    main()
