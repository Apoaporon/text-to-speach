"""Convert M4A audio files to WAV format."""
import argparse
from pathlib import Path

from pydub import AudioSegment

# ===== 設定: ここでパスを指定 =====
INPUT_PATH = "m4a_file/takaichi2.m4a"  # 変換元のM4Aファイルまたはディレクトリ
OUTPUT_PATH = "wav_file"  # 出力先（Noneの場合は入力と同じ場所に保存）
IS_DIRECTORY = False  # Trueの場合はディレクトリ内の全M4Aファイルを変換
# ================================


def convert_m4a_to_wav(input_path: Path, output_path: Path | None = None) -> None:
    """
    Convert M4A file to WAV format.

    Args:
        input_path: Path to input M4A file
        output_path: Path to output WAV file (optional, defaults to same name with .wav extension)
    """
    if output_path is None:
        output_path = input_path.with_suffix(".wav")
    elif output_path.is_dir():
        # 出力先がディレクトリの場合、入力ファイル名を使ってパスを生成
        output_path = output_path / input_path.with_suffix(".wav").name
    
    # 出力先ディレクトリが存在しない場合は作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting: {input_path} -> {output_path}")

    # M4Aファイルを読み込み
    audio = AudioSegment.from_file(str(input_path), format="m4a")

    # WAV形式で保存
    audio.export(str(output_path), format="wav")

    print(f"✅ 変換完了: {output_path}")


def convert_directory(input_dir: Path, output_dir: Path | None = None) -> None:
    """
    Convert all M4A files in a directory to WAV format.

    Args:
        input_dir: Directory containing M4A files
        output_dir: Output directory (optional, defaults to input directory)
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # すべてのM4Aファイルを取得
    m4a_files = list(input_dir.glob("*.m4a"))

    if not m4a_files:
        print(f"❌ M4Aファイルが見つかりません: {input_dir}")
        return

    print(f"📁 {len(m4a_files)} 個のM4Aファイルを変換します...")

    for m4a_file in m4a_files:
        output_path = output_dir / m4a_file.with_suffix(".wav").name
        try:
            convert_m4a_to_wav(m4a_file, output_path)
        except Exception as e:
            print(f"❌ エラー ({m4a_file.name}): {e}")

    print(f"\n✅ すべての変換が完了しました！")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert M4A files to WAV format")
    parser.add_argument(
        "input", type=str, nargs="?", default=INPUT_PATH, help="Input M4A file or directory"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output WAV file or directory (optional)"
    )
    parser.add_argument(
        "-d",
        "--directory",
        action="store_true",
        help="Convert all M4A files in directory",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else (Path(OUTPUT_PATH) if OUTPUT_PATH else None)

    if not input_path.exists():
        print(f"❌ ファイルまたはディレクトリが見つかりません: {input_path}")
        return

    if args.directory or IS_DIRECTORY or input_path.is_dir():
        # ディレクトリモード
        convert_directory(input_path, output_path)
    else:
        # 単一ファイルモード
        if not input_path.suffix.lower() == ".m4a":
            print(f"❌ M4Aファイルではありません: {input_path}")
            return
        convert_m4a_to_wav(input_path, output_path)


if __name__ == "__main__":
    main()
