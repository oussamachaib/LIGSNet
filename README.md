# LIGSNet

> Laser-induced grating spectroscopy (LIGS) is an advanced laser diagnostic adept at scalar (usually temperature and species) measurements in high-pressure gas turbine engines. The measurement is in the form a short-lived oscillating-decaying time series, whose frequency and amplitude are proportional to temperature and species concentration, respectively. In non-homogeneous environments, the technique suffers from a number of shortcomings which greatly limit its scope for use in realistic jet engines. One of these shortcomings is a result of the curse of dimensionality, which obstructs the visualization and interpretation of high-dimensional LIGS data sets. Another shortcoming concerns the prediction of harmonic constants $n_h$, which are necessary for temperature computation, but in general unknown and challenging to model.

The following two solutions address the computational shortcomings of LIGS:
* ```LIGSConvAENet```: A deep convolutional autoencoder trained to minimize the reconstruction error of LIGS time series. The convolutional backbone of the encoder-decoder blocks was designed via receptive fields reflecting LIGS-specific domain knowledge. A single-unit bottleneck is used to reconstruct the latent, physically-meaningful flamelet coordinate, which is inherently one-dimensional.
* ```LIGSConvNet```: A deep, translation-invariant, convolutional neural network for binary classification. It solves the long-standing problem of predicting correct harmonic constants ($n_h \in {1,2}$) which plagues the analysis of non-premixed flames. It leverages the pretrained encoder of ```LIGSConvAENet```.

Both networks were trained in the PyTorch framework on a hydrogen-air LIGS data set acquired in a high-pressure gas turbine @ the Gas Turbine Research Centre (Cardiff, Wales, UK) in 2023. Details on the setup and diagnostics can be found in [(Chaib et al. 2024)](https://doi.org/10.1115/1.4065996). The data set is in the form of $25{,}000$ observations (signals) of $6{,}000$ features (voltage measurements) each.

**Note**: Certain parts of the repository are still work-in-progress, but the trained models are readily-available for use. The notebooks illustrate how to get started. The trainer classes used to train both networks are provided in ```trainers.py```.

## Architectures
### LIGSConvAENet
To design the convolutional backbone of the autoencoder, we assumed a "good" LIGS manifold learner would make use of receptive fields that span the range of frequencies we expect to find in such periodic signal data sets, and in a relatively smooth fashion. This information was used to compute both kernel size and stride at each hidden layer, after an initial aggressive downsampling in the first layer. By the time the signal reaches the final convolutional layer, the successive downsamplings (which can be thought of a successive low-pass filters) result in signatures in the form of a single-peak signal, indicative of the decay rate of the signature.
<p align = "center">
<img src="https://github.com/user-attachments/assets/db33ecc1-042e-405a-9823-f69bffe7717f" width="600"/>
</p>

### LIGSConvNet
The classifier is initialized with the encoder block of the previous autoencoder, and trained to minimize the binary cross entropy loss. CNNs are not architecturally translation-invariant (though it is sometimes falsely claimed they are: see the excellent ICML paper of [Biscione and Bowers](https://arxiv.org/abs/2110.05861) on the topic), but may learn to become so via suitable augmentation. Thus, the data was augmented with random translations across the 1D canvas through which the CNN was able to acquire translation invariance.
<p align = "center">
<img src="https://github.com/user-attachments/assets/3801e0ee-7f04-44ff-81e7-2683d3ee871b" width="600"/>
</p>

## Model evaluation
### Embeddings
The performance of ```LIGSConvAENet``` was validated against the embeddings obtained via ```tulip```, via Spearman and Pearson correlation coefficients. It is important to state the Spearman coefficient is the more reliable metric here, given the objective is to order the different observations along the latent trajectory in a consistent way. In other words, the relationship between embeddings learned via either ```LIGSConvAENet``` or ```tulip``` need not be linear, so long the ordinal ranking of individual observations is comparable.

<p align = "center">
<img src = "https://github.com/user-attachments/assets/4739023b-286e-49d2-84bd-32bbf7aef9b2" width = "600"/>
</p>

### Classification
The performance of ```LIGSConvNet``` was validated using the standard validation set approach (80-20 train-test split). To assess invariance to translation, we performed cross-dataset testing, wherein the CNN trained on the translated (untranslated) data set was evaluated on the untranslated (translated) data.

<div align = "center">

| **Classifier**               | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **AUC** | **TN**  | **FP**  | **FN**  | **TP**  |
|-----------------------------|------------|------------|--------|----------|------|------|------|------|------|
| *Exp. 1: Like-for-like*     |            |            |        |          |      |      |      |      |      |
| CNN                         | 0.97       | 0.97       | 0.97   | 0.97     | 0.97 | 2251 | 68   | 75   | 2606 |
| CNN (augmented)             | 0.90       | 0.90       | 0.91   | 0.91     | 0.90 | 2066 | 260  | 236  | 2438 |
| *Exp. 2: Cross-dataset*     |            |            |        |          |      |      |      |      |      |
| CNN                         | 0.48       | 0.56       | 0.18   | 0.28     | 0.51 | 9579 | 1933 | 11022 | 2466 |
| CNN (augmented)             | 0.90       | 0.85       | 0.98   | 0.91     | 0.89 | 9249 | 2263 | 304   | 13184 |

</div>

## Preview
### Unsupervised visualization of latent dynamics and trajectories
By learning the intrinsic, physically-interpretable, one-dimensional manifold, one is able to reduce a large data set of $25,000$+ noisy time series to a finite set of *ordered* time series spanning the dynamics across the domain, from initial (left) to terminal (right) states.
<p align = "center">
<img src = "https://github.com/user-attachments/assets/29433e44-647a-46c0-854d-57f56529e0d7" width = "600"/>
</p>

### Translation-invariant prediction of harmonic constants
By leveraging the encoder pretrained unsupervised, and augmentation of the original training set with random translations, the network is able to acquire translation-invariance, with prediction accuracies of the order of 90%. 
<p align = "center">
<img src = "https://github.com/user-attachments/assets/dc0f65b7-742c-4f26-8921-485ea5764812" width = "600"/>
</p>

## Navigating the repository
Some of the data folders are saved locally, for the moment (for confidentiality). They will be uploaded to the repo in due course.
```python3
.
├── data
│   ├── raw
│   │   └── data.pkl (raw data set -- upcoming)
│   ├── preprocessed
│   │   └── data.pkl (preprocessed data set -- upcoming)
│   ├── reconstruction
│   │   └── data.pkl (reconstructed signals)
│   ├── pseudomixture
│   │   └── data.pkl (pseudomixture data)
│   └── splits
│       └── *.pkl (wip)
├── models
│   ├── checkpoints (pretrained models)
│   │   └── ConvClassifier.pth
│   │   └── ConvAutoencoder.pth
│   ├── ConvClassifier.py (classifier class)
│   └── ConvAutoencoder.py (autoencoder class)
├── notebooks
│   ├── Demo1.ipynb (demonstration 1: classification)
│   └── Demo2.ipynb (demonstration 2: manifold learning)
├── source
├── loaders.py (data loading)
└── trainers.py (model trainer classes)
```

## Citations
Machine Learning for Combustion Meeting 2024 (London, UK)
```
@inproceedings{ChaibCombML2024,
    title   = {Feature learning from laser-induced grating spectroscopy of reacting flows},
    author  = {Chaib, Oussama and Weller, Lee, and Hochgreb Simone},
    year    = {2024},
              }
```

ASME TurboExpo 24 (London, UK)
```
@article{ChaibASME2024,
    author = {Chaib, Oussama and Weller, Lee and Giles, Anthony and Morris, Steve and Williams, Benjamin A. O. and Hochgreb, Simone},
    title = "{Spatial Temperature Measurements in a Swirl-Stabilized Hydrogen–Air Diffusion Flame at Elevated Pressure Using Laser-Induced Grating Spectroscopy}",
    journal = {Journal of Engineering for Gas Turbines and Power},
    volume = {146},
    number = {11},
    pages = {111020},
    year = {2024},
    month = {08},
    doi = {10.1115/1.4065996},
    url = {https://doi.org/10.1115/1.4065996},
    eprint = {https://asmedigitalcollection.asme.org/gasturbinespower/article-pdf/146/11/111020/7365958/gtp\_146\_11\_111020.pdf},
        }
```
