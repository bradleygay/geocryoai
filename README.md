## GeoCryoAI
###### Bradley A. Gay, PhD | NASA Postdoctoral Program Fellow | JPL, California Institute of Technology
##### GeoCryoAI is a hybridized ensemble learning framework composed of stacked convolutional layers and memory-encoded recurrent neural networks. This multimodal deep learning architecture simultaneously ingests and analyzes _in situ_ measurements, airborne remote sensing observations, and process-based modeling outputs exhibiting disparate spatiotemporal sampling and data densities. If these resources prove helpful and are incorporated, repurposed, and/or modules are extracted and reused, please cite this repository, the companion dataset and source code in the [ORNL DAAC](https://doi.org/10.3334/ORNLDAAC/2371) repository, and the JGR-MLC manuscript.
> ###### Gay, B. A., Pastick, N. J., Watts, J. D., et al., 2025. Decoding the Spatiotemporal Complexities of the Permafrost Carbon Feedback with Multimodal Ensemble Learning. Journal of Geophysical Research, Machine Learning and Computation. Under Review. </br>
> ###### Gay, B.A., et al. 2025. GeoCryoAI | Decoding the Spatiotemporal Complexities of the Permafrost Carbon Feedback with Multimodal Ensemble Learning. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2371 </br>
> ###### Gay, B., Pastick, N., Watts, J., Armstrong, A., Miner, K., & Miller, C. (2024). geocryoai (Version 1.0.0) [Computer software]. https://www.github.com/bradleygay/geocryoai </br>

### Relevant Manuscripts and Datasets
> ###### Gay, B. A., Pastick, N. J., Watts, J. D., et al., 2025. Decoding the Spatiotemporal Complexities of the Permafrost Carbon Feedback with Multimodal Ensemble Learning. Journal of Geophysical Research, Machine Learning and Computation. In Production. </br>
> ###### Gay, B.A., N.J. Pastick, J.D. Watts, A.H. Armstrong, K. Miner, and C.E. Miller. 2025. GeoCryoAI Permafrost, Thaw Depth and Carbon Flux in Alaska, 1969-2022. Preprint. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2371
> ###### Gay, B. A., Pastick, N. J., Watts, J. D., et al., 2024. Forecasting Permafrost Carbon Dynamics in Alaska with Earth Observation Data and Artificial Intelligence, ESS Open Archive. https://essopenarchive.org/users/524229/articles/1225858-forecasting-permafrost-carbon-dynamics-in-alaska-with-earth-observation-data-and-artificial-intelligence </br>
> ###### Gay, B. A., Züfle, A. E., Armstrong, A. H., et al. Investigating Permafrost Carbon Dynamics in Alaska with Artificial Intelligence, December 26, 2023. ESS Open Archive. https://doi.org/10.22541/essoar.170355056.64772303/v1 </br>
> ###### Gay, B. A., Züfle, A. E., Armstrong, A. H., et al. Investigating High-Latitude Permafrost Carbon Dynamics with Artificial Intelligence and Earth System Data Assimilation, December 26, 2023. ESS Open Archive. https://doi.org/10.22541/essoar.170355053.35677457/v1 </br>
> ###### Gay, B.A., Pastick, N.J., Züfle, A.E., Armstrong, A.H., Miner, K.R., Qu, J.J., 2023. Investigating permafrost carbon dynamics in Alaska with artificial intelligence. Environmental Research Letters 18. https://doi.org/10.1088/1748-9326/ad0607 </br>
> ###### Gay, B. A., (2023). Investigating High-Latitude Permafrost Carbon Dynamics with Artificial Intelligence and Earth System Data Assimilation. (Order No. 30488695, George Mason University). ProQuest Dissertations and Theses, 281. Retrieved from https://www.proquest.com/dissertations-theses/investigating-high-latitude-permafrost-carbon/docview/2826111475/se-2 </br>


#####
##### Large Data Files
###### This repository contains large data files that have been chunked for storage. To reconstruct the original files:
###### 1. Clone the repository
###### 2. Install required packages:
    pip install h5py pandas numpy
###### 3. Run the reconstruction script:
    python chunk_reassembly.py
###### The script will reconstruct:
- ensemble_tensor.h5 (from h5_chunks)
- final_fcfch4alt_monthly_1km_ds.parquet (from parquet_chunks)
