# LIGSNet

> Laser-induced grating spectroscopy (LIGS) is an advanced laser diagnostic adept at scalar (usually temperature and species) measurements in high-pressure gas turbine engines. The measurement is in the form a short-lived oscillating-decaying time series, whose frequency and amplitude are proportional to temperature and species concentration, respectively. In non-homogeneous environments, the technique suffers from a number of shortcomings which greatly limit its scope for use in realistic jet engines. One of these shortcomings is a result of the curse of dimensionality, which obstructs the visualization and interpretation of high-dimensional LIGS data sets. Another shortcoming concerns the prediction of harmonic constants $n_h$ which are necessary for temperature computation, but are in general unknown and challenging to model.

Two solutions were thus developed to address both shortcomings:
* ```LIGSNetCAE```: A deep convolutional autoencoder trained to minimize the reconstruction error of LIGS time series. The convolutional backbone of the encoder-decoder blocks was designed via receptive fields reflecting LIGS-specific domain knowledge. A single-unit bottleneck is used to reconstruct the latent, physically-meaningful flamelet coordinate, which is inherently one-dimensional.
* ```LIGSConvNet```: A deep, translation-invariant, convolutional neural network for binary classification. It solves the long-standing problem of predicting correct harmonic constants ($n_h \in {1,2}$) which plagues the analysis of non-premixed flames. It leverages the pretrained encoder of ```LIGSNetCAE```.

Both networks were trained in the PyTorch framework on a hydrogen-air LIGS data set acquired in a high-pressure gas turbine in the Gas Turbine Research Centre (Cardiff, Wales, UK) in 2023. Details on the setup and diagnostics can be found in [(Chaib et al. 2024)](https://doi.org/10.1115/1.4065996).

**Note**: Certain parts of the repository are still work-in-progress, but the trained models are readily-available for use. The notebooks illustrate how to get started and use both networks. The trainer classes used to train both networks are provided in ```trainers.py```.

## Architectures
### LIGSNetCAE
To design the convolutional backbone of the autoencoder, we assumed a "good" LIGS manifold learner would make use of receptive fields that span the range of frequencies we except to find in such periodic signal data sets, and in a relatively smooth fashion. This information was used to compute both kernel size and stride at each hidden layer, after an initial aggressive downsampling in the first layer. By the time the signal reaches the final convolutional layer, the successive downsamplings (which can be thought of a successive low-pass filters) result in signatures in the form of a single-peak signal, indicative of the decay rate of the signature.
<p align = "center">
<img src="https://github.com/user-attachments/assets/db33ecc1-042e-405a-9823-f69bffe7717f" width="600"/>
</p>

### LIGSConvNet
The classifier is initialized with the encoder block of the previous autoencoder, and trained to minimize the binary cross entropy loss. CNNs are not architecturally translation-invariant (though it is sometimes falsely claimed they are -- see the excellent ICML paper of [Biscione and Bowers](https://arxiv.org/abs/2110.05861)), but may learn to become so via suitable augmentation. Thus, the data was augmented with random translations across the 1D canvas through which the CNN was able to acquire translation invariance.
<p align = "center">
<img src="https://github.com/user-attachments/assets/3801e0ee-7f04-44ff-81e7-2683d3ee871b" width="600"/>
</p>

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
