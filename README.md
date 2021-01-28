> # Blind Deconvolution of Turbulent Flows using Super-Resolution Reconstruction

#Abstract

Turbulence is a complex flow phenomenon characterized by chaotic multiscale interactions which can be accurately modeled using direct numerical simulations (DNS). However, DNS is computationally prohibitive in many cases due to the fine grid requirement to resolve the high frequency content of the flow. Large eddy simulation (LES)
attempts to relax this computational burden by resolving only large scales and modeling small scale interactions which is termed as the subgrid scale (SGS) modeling. One of the approaches to SGS modeling is approximate deconvolution (AD) procedure. In this work, we propose to use a machine learning algorithm called generative
adversarial networks (GAN) to determine the deconvolved flow field instead of using traditional inverse filtering operation of AD. GANs have been demonstrated for image super-resolution by up sampling low resolution images. We use the enhanced super-resolution GAN (ESRGAN) framework for upscaling physical turbulence fields i.e,
low-resolution LES field to super-resolution (close to DNS) field. The resulting super-resolved field can then be utilized to compute the SGS closure model instead of solving AD procedure numerically. In this work, ESRGAN framework is tested to super resolve two-dimensional decaying homogeneous turbulence flow field with physics bases constraints. The reconstruction of turbulent small scales by the proposed framework is evaluated by comparing to energy spectra on both super resolved and DNS field.

<img src="comp.png" width="50%">


**Enhanced Super Resolution Generative Adversarial Networks (ESRGAN):**  

Generator (G) |  Discriminator (D)
----- | -----
<img src="gen1.png" width="90%">| <img src="disc1.png" width="90%" >


<div align="center">
<img src="gan.png" width="70%">
<p>ESRGAN framework</p>
</div>

>**2D decaying tubulence Problem:**
<div align="center">
<img src="2dturb.png" width="50%">
</div>

>**Reuslts:**
<div align="center">
<img src="si-1.png" width="90%">
<p>Stream funtion super resolution</p>
</div>

<div align="center">
<img src="omega-1.png" width="90%">
<p>Vorticity feild super resolution</p>
</div>


<div align="center">
<img src="en-1.png" width="40%">
<p>Energy Spectrum</p>
</div>

References:
##Please find the report in the repository for more detailed analysis##

> Future Work:
# Stabilize the trained GAN model in place of Approximate Deconvolution (AD) structural turbulence model in the LES solver
