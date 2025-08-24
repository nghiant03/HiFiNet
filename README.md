# HiFiNet: Hierarchical Fault Identification in Wireless Sensor Networks via Edge‑Based Classification and Graph Aggregation

## Installation

- Python
- Pip
- [PDM](https://github.com/pdm-project/pdm)

## Usage
HiFiNet experiments can be run using the intergrated CLI.
```bash
hifinet --help
```
### Inject
There are default adaptor for the 2 datasets used in the paper. Therefore the dataset can be downloaded and placed inside the
directories inside `data/`. The injecting command for each will then be simply:
```bash
hifinet inject intel
hifinet inject merra2
```
The injection functionality can be used with any dataset `.csv` file. The data need to be in the following schema:
- `node_id` column: Unique ID for each node of the WSN
- `target` column: The target feature needed to be injected
- `feature`/`feature_*` columns: The additional features presented in the data
See the available option to configure the injection functionality:
```bash
hifinet inject --help
```
