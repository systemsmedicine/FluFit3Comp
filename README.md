# fluFit3: Multicompartment Modeling of Influenza A Replication Along the Murine Respiratory Tract

This repository contains code and data for fitting mathematical models of influenza A (H3N2) infection dynamics in mice across three anatomical compartments: **nose**, **trachea**, and **lungs**. Parameter estimation and model fitting are performed using a Python code via Jupyter Notebook.

## Repository Structure

fluFit3/
│
├── fluFit3.ipynb # Main notebook: model definition, ODE solving, plots, and analysis
├── data/
│ ├── viralDataset.csv # Experimental viral load data (3 compartments)
│ ├── bestPars_M1.csv # Estimated parameters for Model 1
│ ├── bestPars_M2.csv # Estimated parameters for Model 2
│ ├── bestPars_M3.csv # Estimated parameters for Model 3
│ ├── bestPars_M4.csv # Estimated parameters for Model 4
│ └── bootParam.csv # Bootstrap parameter samples for Model 1

# Cite

Blanco-Rodriguez et al. (2026). *Multicompartment Modeling of Influenza A Replication Along the Murine Respiratory Tract*. Frontiers in Virology.

