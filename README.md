<div align="center">

# Analyzing the Energy Consumption of Generative Text-to-Music Models 
 
[Francesca Ronchini](https://www.linkedin.com/in/francesca-ronchini/)*<sup>1</sup>, [Riccardo Passoni](https://www.linkedin.com/in/riccardo-passoni/?locale=en_US)\*<sup>2</sup>, [Luca Comanducci](https://lucacoma.github.io/)<sup>1</sup>, [Romain Serizel](https://members.loria.fr/RSerizel/)<sup>3</sup>, [Fabio Antonacci](https://www.deib.polimi.it/ita/personale/dettagli/573870)<sup>1</sup>

<sup>1</sup> Dipartimento di Elettronica, Informazione e Bioingegneria - Politecnico di Milano, Milan, Italy <br>
<sup>2</sup> Institute of Sound Recording (IoSR), University of Surrey, Guildford, UK <br>
<sup>3</sup> Université de Lorraine, CNRS, Inria, Loria, Nancy, France <br>
*These authors contributed equally

</div>

- [Abstract](#abstract)
- [Install & Usage](#install--usage)
- [Additional information](#additional-information)
    
## Abstract

Music generation via multimodal machine learning generative models has recently gained significant traction in both the research community and the general public. Among such models, an important category is represented by Text-To-Music, which take only a textual description or caption as input to specify the type of composition the user wishes to generate. While generative artificial intelligence models achieve remarkable performance, they also require substantial computational power during both training and inference. The latter, in particular, becomes increasingly critical as these models are deployed more widely in real-world music-making applications. In this paper, we address this issue by analyzing the energy consumption of seven diffusion-based and two autoregressive text-to-music models. Through a series of experiments, we evaluate how model-dependent generation parameters influence energy usage during inference. Since audio generation quality, prompt adherence, and energy efficiency are all important, we employ Pareto-optimal analysis to explore the trade-offs among these competing objectives. Our results highlight the balance between model performance and energy usage, offering guidance for designing more environmentally conscious generative audio systems. 

## Install & Usage

In order to run the Jupyter notebooks, you need to clone the repo, create a virtual environment, and install the needed packages.

You can create the virtual environment and install the needed packages using conda with the following command: 

```
conda env create -f requirements.yml
```

Once everything is installed, you can run the Jupyter Notebook following the instruction reported on it and reproduce the results. <br>

The scripts contained in the 'inferences' folder can be run by creating environments specific to the desired model; further information is provided in the folder's README.


## Additional information

For more details:
"[Analyzing the Energy Consumption of Generative Text-to-Music Models]()" (Francesca Ronchini, Riccardo Passoni, Luca Comanducci, Romain Serizel, Fabio Antonacci)

If you use code or comments from this work, please cite:

```BibTex
@ARTICLE{,
  author={Ronchini, Francesca and Passoni, Riccardo and Comanducci, Luca and Serizel, Romain and Antonacci, Fabio},
  journal={IEEE Transactions on Technology & Society}, 
  title={Analyzing the Energy Consumption of Generative Text-to-Music Models}, 
  year={2026},
  volume={},
  number={},
  pages={},
  keywords={},
  doi={}}
```

