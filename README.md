# LIGSNet
A collection of neural networks for representation learning and classification of LIGS time series.
> Laser-induced grating spectroscopy (LIGS) is an advanced laser diagnostic adept at scalar (usually temperature and species) measurements in high-pressure gas turbine engines. The measurement is in the form a short-lived oscillating-decaying time series, whose frequency and amplitude are proportional to temperature and species concentration, respectively. In non-homogeneous environments, the technique suffers from a number of shortcomings which greatly limit its scope for use in realistic jet engines. One of these shortcomings is a result of the curse of dimensionality, which obstructs the visualization and interpreation of LIGS data sets. Another shortcoming concerns the prediction of harmonic constants $n_h$ which are necessary for temperature computation, but are in general unknown and challenging to model.

Two solutions were thus developed to address both shortcomings:
* ```LIGSNetCAE```: A deep convolutional autoencoder trained to minimize the reconstruction error of LIGS time series. The convolutional backbone of the encoder-decoder blocks was designed via receptive fields reflecting LIGS-specific domain knowledge. A single-unit bottleneck is used to reconstruct the latent, physically-meaningful flamelet coordinate, which is inherently one-dimensional.
* ```LIGSConvNet```: A deep, translation-invariant, convolutional neural network for binary classification. It solves the long-standing problem of predicting correct harmonic constants ($n_h \in {1,2}$) which plagues the analysis of non-premixed flames. It leverages the pretrained encoder of ```LIGSNetCAE``.

Both networks were trained on a hydrogen-air LIGS data set acquired in a high-pressure gas turbine in the Gas Turbine Research Centre (Cardiff, Wales, UK) in 2023. Details on the setup and diagnostics can be found in [(Chaib et al. 2024)](https://doi.org/10.1115/1.4065996).

## Architectures
### LIGSNetCAE
<p align = "center">
<img src="https://github.com/user-attachments/assets/db33ecc1-042e-405a-9823-f69bffe7717f" width="900"/>
</p>

### LIGSConvNet
<p align = "center">
<img src="https://github.com/user-attachments/assets/3801e0ee-7f04-44ff-81e7-2683d3ee871b" width="900"/>
</p>


## Preview
### Unsupervised visualization of latent dynamics and trajectories

<p align = "center">
<img src = "https://github.com/user-attachments/assets/2a853214-af27-49e2-83ea-cb796831e147" width = "900"/>
</p>

