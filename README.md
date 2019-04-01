# connectivity
Probabilistic connectivity kernel model of social interactions

## Important Files

1. `egosbm.py` contains Python classes `EgocentricSBM`, `StochasticBlockModel` and `NetworkData`
2. `geography.txt` contains code dictionary of local authorities in the UK
3. `usoc_wave3.ego` contains the inferred SBM model for UK from USoc Wave 3
4. `test_pi_lon.tsv` contains distribution of people in Blau space (defined by `usoc_wave3.ego`) in London (according to 2011 UK Census)
5. `test_pi_uk.tsv` contains distribution of people in Blau space (defined by `usoc_wave3.ego`) in all of UK (according to 2011 UK Census)

## Important Notebooks

1. `sbm_connectivity_kernel_v3.ipynb`: definition of the SBM model for egocentric data
2. `cont2cat.ipynb` and `cont2cat_mcmc.ipynb`: converting continuous dimensions to categorical ones for SBM model
3. `census_usoc.ipynb`: exploring Understanding Society data to learn an SBM kernel for the UK, alongside UK Census 2011 to find social access statistic for different LAs of UK, or boroughs of London
4. `egosbm.ipynb`: tutorial exploring the Python classes defined in `egosbm.py`
