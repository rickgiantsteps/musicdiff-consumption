<div align="left">

# Inference scripts

</div>

This folder contains all the scripts used to collect energy consumption data of the 9 models for each of the three performed experiments.<br>

For the quality metrics experiment, the scripts necessary to compute the CLAP, FAD, and Audiobox scores have also been included.<br>
The `getCLAPscores.py` file, contained in the CLAP folder, uses the 300 randomly chosen captions for each baseline to obtain the prompt coherence of the generations (using the Stable Audio Metrics library: https://github.com/Stability-AI/stable-audio-metrics). The FAD folder contains all the scripts necessary to get the objective quality measurement for all configurations (using the FADTK library: https://github.com/microsoft/fadtk).
The Audiobox folder contains the scripts used to compute Audiobox scores (Production Quality, Production Complexity, Content Enjoyment, and Content Usefulness), which are obtained using the audiobox-aesthetics model (https://github.com/facebookresearch/audiobox-aesthetics).

## Install & Usage

The scripts related to AudioLDM, AudioLDM2, MusicLDM, MusicGen, and Stable Audio Open can be run in the environment provided with this repository, as they are hosted on HuggingFace (https://huggingface.co/cvssp/audioldm-s-full-v2, https://huggingface.co/cvssp/audioldm2-music, https://huggingface.co/docs/diffusers/api/pipelines/musicldm, https://huggingface.co/facebook/musicgen-small, https://huggingface.co/stabilityai/stable-audio-open-1.0). 
Stable Audio Open requires a HuggingFace user token for access. Ensure you are logged in via huggingface-cli login or have your token configured as an environment variable (HF_TOKEN) before running the model. Acceptance of the terms on the model's HF page is necessary to perform inferences.

For the remaining models, cloning their respective GitHub repositories, along with creating custom environments, is required: 

- **ACEStep**: https://github.com/ace-step/ACE-Step
- **AudioX**: https://github.com/ZeyueT/AudioX
- **MAGNeT**: https://huggingface.co/facebook/magnet-small-10secs
- **Tango**: https://huggingface.co/declare-lab/tango-music-af-ft-mc