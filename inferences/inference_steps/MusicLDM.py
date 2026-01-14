import scipy
import torch
import os
import glob
from diffusers import MusicLDMPipeline
from codecarbon import EmissionsTracker

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

repo_id = "ucsd-reach/musicldm"
pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to(device="cuda", dtype=torch.float16)
prompt = "An organ is playing chords and melodies while a male voice is singing soulfully."
negative_prompt = "Low quality."
generator = torch.Generator("cuda").manual_seed(0)

n_step = [10, 25, 50, 100, 150, 200]
runs = 5

for k in range(runs):
    for x in n_step:

        tracker = EmissionsTracker(project_name=f"MusicLDM-inference_{x}-steps", tracking_mode="process",
                                   gpu_ids="0",
                                   output_dir="../../results/inference_steps/MusicLDM",
                                   output_file=f"MusicLDM-emissions-run{k+1}.csv", allow_multiple_runs=False)
        tracker.start_task(f"Inference emissions with {x} steps")

        try:
            # run the generation
            audio = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=x,
                audio_length_in_s=10.0,
                num_waveforms_per_prompt=1,
                generator=generator,
            ).audios

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        scipy.io.wavfile.write(
            f"genaudios/MusicLDM_{x}-steps.wav",
            rate=16000, data=audio[0])
        print(model_emissions)

emissions_base = glob.glob('../../results/inference_steps/MusicLDM/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
