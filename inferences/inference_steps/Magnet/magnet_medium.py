from audiocraft.models import MAGNeT
from codecarbon import EmissionsTracker
from audiocraft.data.audio import audio_write
import glob
import os

model = MAGNeT.get_pretrained("facebook/magnet-medium-10secs")

descriptions=["An organ is playing chords and melodies while a male voice is singing soulfully."]

runs = 5

for k in range(runs):

    tracker = EmissionsTracker(project_name=f"MagnetMedium-inference", tracking_mode="process",  gpu_ids="0",
                                output_dir="../../../results/inference_steps/Magnet",
                                output_file=f"MagnetMedium-emissions-run{k+1}.csv", allow_multiple_runs=True)
    tracker.start_task(f"Inference emissions")

    try:

        wav = model.generate(descriptions)

        model_emissions = tracker.stop_task()

    finally:
        _ = tracker.stop()

    for idx, one_wav in enumerate(wav):
       audio_write("genaudios/MagnetMedium", one_wav.cpu(), model.sample_rate, strategy="loudness")

emissions_base = glob.glob('../../../results/inference_steps/Magnet/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
