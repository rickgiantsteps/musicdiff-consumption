from src.clap_score import clap_score
from pathlib import Path
import pandas as pd

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
audio_folder = parent_dir / "audios" / "genaudios"
out_folder = current_dir / "output"
out_folder.mkdir(parents=True, exist_ok=True)

baselines = ["musiccaps", "songdescriber"]
models_dict = {
    "MusicGen": {"type": "size", "values": ["Small", "Medium", "Large"]},
    "Magnet":   {"type": "size", "values": ["Small", "Medium"]},
    "musicAudioLDM": {"type": "steps", "values": [10, 25, 50, 100, 150, 200]},
    "musicAudioLDM2": {"type": "steps", "values": [10, 25, 50, 100, 150, 200]},
    "musicSAO":      {"type": "steps", "values": [10, 25, 50, 100, 150, 200]},
    "MusicLDM":      {"type": "steps", "values": [10, 25, 50, 100, 150, 200]},
    "musicTango":    {"type": "steps", "values": [10, 25, 50, 100, 150, 200]},
    "ACEStep":       {"type": "steps", "values": [10, 25, 50, 100, 150, 200]},
    "AudioX":        {"type": "steps", "values": [10, 25, 50, 100, 150, 200]},
}

def get_renamed_id(file_name, model, val, config_type):
    numeric_part = Path(file_name).stem
    if config_type == "size":
        return f"{model}{val}-{numeric_part}"
    else:
        return f"{model}_{val}-steps-{numeric_part}"


for model, config in models_dict.items():
    config_type = config["type"]

    if config_type == "steps":
        # collect all steps for this model into a single CSV: {model}_clap_scores.csv
        model_scores = []

        for val in config["values"]:
            for base in baselines:
                # ex: audio_folder / model / base / 10
                generated_path = audio_folder / model / base / str(val)
                csv_file_path = current_dir / f"{base}_captions.csv"

                if not generated_path.exists() or not generated_path.is_dir():
                    print(f"Skipping (no audio dir): {generated_path}")
                    continue

                if not csv_file_path.exists():
                    print(f"Skipping (no captions CSV): {csv_file_path}")
                    continue

                # check .wav files exist
                wavs = list(generated_path.glob("*.wav"))
                if not wavs:
                    print(f"Skipping (no .wav files) in {generated_path}")
                    continue

                print(f"Computing CLAP for {model} (steps={val}) on {base}...")

                df = pd.read_csv(csv_file_path)
                if 'file_name' not in df.columns or 'caption' not in df.columns:
                    print(f"CSV missing required columns: {csv_file_path}")
                    continue

                df['file_name'] = df['file_name'].apply(lambda x: get_renamed_id(x, model, val, config_type))
                id2text = df.set_index('file_name')['caption'].to_dict()

                try:
                    clp = clap_score(
                        id2text,
                        str(generated_path),
                        audio_files_extension='.wav',
                        clap_model='music_audioset_epoch_15_esc_90.14.pt'
                    )
                    score_item = clp.item()
                    model_scores.append({
                        'model': model,
                        'steps': val,
                        'baseline': base,
                        'clap_score': score_item
                    })
                except Exception as e:
                    print(f"Error processing {model} steps={val} ({base}): {e}")

        # save aggregated per-model CSV for diffusion models
        if model_scores:
            model_df = pd.DataFrame(model_scores, columns=['model', 'steps', 'baseline', 'clap_score'])
            output_file_path = out_folder / f"{model}_clap_scores.csv"
            model_df.to_csv(output_file_path, index=False)
            print(f"Saved results to {output_file_path}")

    else:  # config_type == "size"
        # for nondiff models, create and save a CSV per size {model}{size}_clap_scores.csv
        for val in config["values"]:
            model_scores = []
            for base in baselines:
                # ex: audio_folder / model / Small / musiccaps
                generated_path = audio_folder / model / val / base
                csv_file_path = current_dir / f"{base}_captions.csv"

                if not generated_path.exists() or not generated_path.is_dir():
                    print(f"Skipping (no audio dir): {generated_path}")
                    continue

                if not csv_file_path.exists():
                    print(f"Skipping (no captions CSV): {csv_file_path}")
                    continue

                wavs = list(generated_path.glob("*.wav"))
                if not wavs:
                    print(f"Skipping (no .wav files) in {generated_path}")
                    continue

                print(f"Computing CLAP for {model} (size={val}) on {base}...")

                df = pd.read_csv(csv_file_path)
                if 'file_name' not in df.columns or 'caption' not in df.columns:
                    print(f"CSV missing required columns: {csv_file_path}")
                    continue

                df['file_name'] = df['file_name'].apply(lambda x: get_renamed_id(x, model, val, config_type))
                id2text = df.set_index('file_name')['caption'].to_dict()

                try:
                    clp = clap_score(
                        id2text,
                        str(generated_path),
                        audio_files_extension='.wav',
                        clap_model='music_audioset_epoch_15_esc_90.14.pt'
                    )
                    score_item = clp.item()
                    model_scores.append({
                        'model': model,
                        'size': val,
                        'baseline': base,
                        'clap_score': score_item
                    })
                except Exception as e:
                    print(f"Error processing {model} size={val} ({base}): {e}")

            # save per-model+size CSV
            if model_scores:
                model_df = pd.DataFrame(model_scores, columns=['model', 'size', 'baseline', 'clap_score'])
                output_file_path = out_folder / f"{model}{val}_clap_scores.csv"
                model_df.to_csv(output_file_path, index=False)
                print(f"Saved results to {output_file_path}")