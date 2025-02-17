# FPM_benchmarks

The repository provides MATLAB codes for 6 Fourier ptychographic microscopy (FPM) reconstruction algorithms for testing their performance. 
Run "gen_img_batch.m" to generate the simulated FPM data. 
One can add noise signals and LED shifting during the generation of FPM data, to simulate the comment challenges in FPM experiments. 

## FPM algorithms
<center>
  
| Name of Algos | Notes                                                                                                                                                                                                                                                                                         |
|:--------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| m-FPM        | conventional FPM reconstruction using extended ptychographic iterative engine (ePIE) equipped with momentum. The codes are adapted from [here](https://github.com/SmartImagingLabUConn/Fourier-Ptychography). [[Literatures]](https://smartimaging.uconn.edu/fourier-ptychtography/) |
| AS-FPM       | conventional FPM reconstruction with adaptive step-size. The codes are adapted from [here](https://www.scilaboratory.com/code.html). [[Paper]](https://opg.optica.org/oe/fulltext.cfm?uri=oe-24-18-20724&id=349656)                                                                           |
| ADMM-FPM     | FPM reconstruction using ADMM. [[Paper]](https://www.mdpi.com/2073-4409/11/9/1512)                                                                                                                                                                                                            |
| APIC         | Closed-formed FPM reconstruction. The codes are adapted from [here](https://github.com/rzcao/APIC-analytical-complex-field-reconstruction). [[Paper]](https://www.nature.com/articles/s41467-024-49126-y)                                                                                     |
| FD-FPM       | FPM reconstruction using feature-domain loss function. [[Paper]](https://opg.optica.org/abstract.cfm?uri=optica-11-5-634)                                                                                                                                                                     |
| VEM-FPM      | Variational EM algorithm for FPM reconstruction.  

</center>

## How to use
-**Step 1** run "gen_img_batch.m" to generate the simulated FPM data. The data is saved with .mat file in the provided folders. <br>
-**Step 2** run "recover/run_benchmarks.m" to perform reconstruction with different algorithms. The data loading path should be set so that the code will automatically load the .mat files generated from the first step. <br>

Users may adjust the noise signal and LED shifting strength to simulate the noise signals and LED uncertainties. <br> 
The parameters for data generation can be also adjusted to test the code in different experimental conditions. 


## Parameters in simulations

<div align="center">
<img src="https://github.com/ShuheZhang-MUMC/FPM_benchmarks/blob/main/sources/LED_layouts.jpg" width = "260" alt="" align=center />
</div><br>
The code simulated the data generation for a ring-shaped LED panel, shown for example in the image above. Other parameters for simulation are listed below.
<div align="center">
  
| Params name               | Values                                 |
|---------------------------|----------------------------------------|
| Pixel element size        | 4.8        um                 |
| Wavelength                | 0.62863    um                  |
| Numerical aperture        | 0.25                                 |
| Magnification             | 10                                   |
| LED height                | 50       mm                          |
| LED distance              | 6        mm                          |
| LED shape                 | Ring      [1,8,12,16,24,36,48,54] |
| Pixels for Low-res image  | 128                                   |
| Pixels for High-res image | 512                                    |
| Downsample rate           | 4                                      |

</div>
