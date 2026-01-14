from transformers import AutoProcessor, MusicgenForConditionalGeneration
from codecarbon import EmissionsTracker
import scipy
import glob
import os


processor = AutoProcessor.from_pretrained("facebook/musicgen-large", device='cuda')
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")
model = model.to('cuda:0')

inputs = processor(
    text=["An organ is playing chords and melodies while a male voice is singing soulfully."],
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to('cuda:0')

runs = 5

for k in range(runs):

    tracker = EmissionsTracker(project_name=f"MusicGenLarge-inference", tracking_mode="process", gpu_ids="0",
                                output_dir="../../../results/inference_steps/MusicGen",
                                output_file=f"MusicGenLarge-emissions-run{k+1}.csv", allow_multiple_runs=True)
    tracker.start_task(f"Inference emissions")

    try:

        audio_values = model.generate(**inputs, max_new_tokens=500)

        model_emissions = tracker.stop_task()

    finally:
        _ = tracker.stop()

    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(f"genaudios/MusicGenLarge.wav",
                           rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

emissions_base = glob.glob('../../../results/inference_steps/MusicGen/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
