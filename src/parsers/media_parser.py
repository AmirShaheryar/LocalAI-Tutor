import os
import json
from faster_whisper import WhisperModel


def transcribe_lecture(media_path, output_dir="outputs", model_size="base"):
    """
    Transcribes audio/video files and generates timestamped transcripts.
    
    Args:
        media_path (str): Path to input .mp4, .mp3, or .wav file.
        output_dir (str): Folder where transcript JSON will be saved.
        model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium').
    """
    if not os.path.exists(media_path):
        print(f" Error: File not found at '{media_path}'")
        return None

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎙️ Loading Whisper model ('{model_size}')...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"⏳ Transcribing '{os.path.basename(media_path)}'...")
    segments, info = model.transcribe(media_path, beam_size=5)

    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    transcript_data = {
        "file_name": os.path.basename(media_path),
        "language": info.language,
        "duration_seconds": round(info.duration, 2),
        "transcript_segments": []
    }

    for segment in segments:
        segment_entry = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "timestamp_formatted": f"[{format_timestamp(segment.start)} -> {format_timestamp(segment.end)}]",
            "text": segment.text.strip()
        }
        transcript_data["transcript_segments"].append(segment_entry)

    output_filename = os.path.splitext(os.path.basename(media_path))[0] + "_transcript.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)

    print(f" Transcription complete! Saved to '{output_path}'")
    return transcript_data

def format_timestamp(seconds):
    """Converts seconds into MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

if __name__ == "__main__":
    
    current_script_path = os.path.abspath(__file__)
    parsers_dir = os.path.dirname(current_script_path)
    src_dir = os.path.dirname(parsers_dir)
    project_root = os.path.dirname(src_dir)

    sample_media = os.path.join(project_root, "data", "raw_media", "Lecture.mkv")

    if os.path.exists(sample_media):
        transcribe_lecture(sample_media)
    else:
        print(f" Please place a sample audio or video file at: '{sample_media}' to test.")