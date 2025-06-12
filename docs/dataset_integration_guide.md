# Dataset Integration Guide
Integrate a dataset to reformat into Zarr.

## Overview
Integrating a dataset in dynamical.org `reformatters` is done by subclassing a trio of base classes, customizing their behavior based on the unique characteristics of your dataset.

There are three core base classes to subclass.
1. `TemplateConfig` defines the dataset **structure**.
1. `RegionConfig` defines the **process** by which a region of that dataset is reformatted: **downloading, reading, rewriting.**
1. `DynamicalDataset` brings together a `TemplateConfig` and `RegionJob` and defines the compute resources to operationally update and validate a dataset.*

### Words
* **Provider** - the agency or organization that publishes the source data. e.g. ECMWF
* **Model** - the model or system that produced the data. e.g. GFS
* **Variant** - the specific subset and structure of data from the model. e.g. forecast, analysis, climatology. Variant may include any other information needed to distinquish datasets from the same model.
* **Dataset** - a specific provider-model-variant. e.g. `noaa/gfs/forecast`



## Integration steps

### Set up template

Copy it
```
mkdir -p src/reformatters/<provider>/<model>/<variant>
cp -r src/reformatters/example/ src/reformatters/<provider>/<model>/<variant>
```

Renames
Find and replace:
1. 