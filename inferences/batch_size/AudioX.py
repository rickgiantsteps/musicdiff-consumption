import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.utils import read_video, merge_video_audio
from stable_audio_tools.data.utils import load_and_process_audio
from codecarbon import EmissionsTracker
from itertools import chain
import glob
import os

emissions = glob.glob('../../results/batch_size/AudioX/*')
emissions1 = glob.glob('../../results/batch_size/AudioX/batch1/*')
emissions2 = glob.glob('../../results/batch_size/AudioX/batch2/*')
emissions5 = glob.glob('../../results/batch_size/AudioX/batch5/*')
genaudios = glob.glob('genaudios/AudioX/*')
for f in chain(emissions, emissions1, emissions2, emissions5, genaudios):
    if os.path.isfile(f):
    	os.remove(f)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda:0")

# Download model
model, model_config = get_pretrained_model("HKUSTAudio/AudioX")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
target_fps = model_config["video_fps"]
seconds_start = 0
seconds_total = 10

model = model.to(device)

# only text-to-music generation
video_path = None
text_prompt = "An organ is playing chords and melodies while a male voice is singing soulfully."
audio_path = None

video_tensor = read_video(video_path, seek_time=0, duration=seconds_total, target_fps=target_fps)
audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total)

single_conditioning = {
    "video_prompt": [video_tensor.unsqueeze(0)],     
    "text_prompt": text_prompt,
    "audio_prompt": audio_tensor.unsqueeze(0),
    "seconds_start": seconds_start,
    "seconds_total": seconds_total
}

batch_size = 10

conditioning = [single_conditioning] * batch_size

n_step = 100
runs = 5

for k in range(runs):

    tracker = EmissionsTracker(project_name=f"AudioX-inference_batch10", tracking_mode="process",
                            gpu_ids="0",
                            output_dir="../../results/batch_size/AudioX",
                            output_file=f"AudioX-emissions-batch10-run{k+1}.csv")
    tracker.start_task(f"Inference emissions with 10 waveforms per prompt")

    # Generate stereo audio
    output = generate_diffusion_cond(
        model,
        batch_size = 10,
        steps=n_step,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    model_emissions = tracker.stop_task()
    _ = tracker.stop()
    print(model_emissions)

    for i in range(output.shape[0]):
        single_audio_clip = output[i]
        normalized_clip = single_audio_clip.to(torch.float32).div(torch.max(torch.abs(single_audio_clip))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()        
        filename = f"genaudios/AudioX/AudioX-batch10-n{i}.wav"
        torchaudio.save(filename, normalized_clip, sample_rate)

batch_size = 5
conditioning = [single_conditioning] * batch_size

for k in range(runs):
    file_counter = 0
    for i in range(2):

        tracker = EmissionsTracker(project_name=f"AudioX-inference_batch5", tracking_mode="process",
                            gpu_ids="0",
                            output_dir="../../results/batch_size/AudioX/batch5",
                            output_file=f"AudioX-emissions-batch5-n{i}-run{k+1}.csv")
        tracker.start_task(f"Inference emissions with 5 waveforms per prompt")

        # Generate stereo audio
        output = generate_diffusion_cond(
            model,
            batch_size = 5,
            steps=n_step,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )

        model_emissions = tracker.stop_task()
        _ = tracker.stop()
        print(model_emissions)

        for i in range(output.shape[0]):
            single_audio_clip = output[i]
            normalized_clip = single_audio_clip.to(torch.float32).div(torch.max(torch.abs(single_audio_clip))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()        
            filename = f"genaudios/AudioX/AudioX-batch5-n{file_counter}.wav"
            torchaudio.save(filename, normalized_clip, sample_rate)
            file_counter += 1

batch_size = 2
conditioning = [single_conditioning] * batch_size

for k in range(runs):
    file_counter = 0
    for i in range(5):

        tracker = EmissionsTracker(project_name=f"AudioX-inference_batch2", tracking_mode="process",
                            gpu_ids="0",
                            output_dir="../../results/batch_size/AudioX/batch2",
                            output_file=f"AudioX-emissions-batch2-n{i}-run{k+1}.csv")
        tracker.start_task(f"Inference emissions with 2 waveforms per prompt")

        # Generate stereo audio
        output = generate_diffusion_cond(
            model,
            batch_size = 2,
            steps=n_step,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )

        model_emissions = tracker.stop_task()
        _ = tracker.stop()
        print(model_emissions)

        for i in range(output.shape[0]):
            single_audio_clip = output[i]
            normalized_clip = single_audio_clip.to(torch.float32).div(torch.max(torch.abs(single_audio_clip))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()        
            filename = f"genaudios/AudioX/AudioX-batch2-n{file_counter}.wav"
            torchaudio.save(filename, normalized_clip, sample_rate)
            file_counter += 1

conditioning = [{
    "video_prompt": [video_tensor.unsqueeze(0)],        
    "text_prompt": text_prompt,
    "audio_prompt": audio_tensor.unsqueeze(0),
    "seconds_start": seconds_start,
    "seconds_total": seconds_total
}]

for k in range(runs):
    for i in range(10):

        tracker = EmissionsTracker(project_name=f"AudioX-inference_batch1", tracking_mode="process",
                            gpu_ids="0",
                            output_dir="../../results/batch_size/AudioX/batch1",
                            output_file=f"AudioX-emissions-batch1-n{i}-run{k+1}.csv")
        tracker.start_task(f"Inference emissions with 1 waveforms per prompt")

        # Generate stereo audio
        output = generate_diffusion_cond(
            model,
            steps=n_step,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )

        model_emissions = tracker.stop_task()
        _ = tracker.stop()
        print(model_emissions)

        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()     
        filename = f"genaudios/AudioX/AudioX-batch1-n{i}.wav"
        torchaudio.save(filename, output, sample_rate)

emissions_base = glob.glob('../../results/batch_size/AudioX/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../results/batch_size/AudioX/batch5/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../results/batch_size/AudioX/batch2/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../results/batch_size/AudioX/batch1/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
