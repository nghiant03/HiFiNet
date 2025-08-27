# HiFiNet: Hierarchical Fault Identification in Wireless Sensor Networks via Edge‑Based Classification and Graph Aggregation

## Dependencies

- Python
- Pip
- [PDM](https://github.com/pdm-project/pdm)

## Installation

Clone the repo:
```bash
git clone https://github.com/nghiant03/HiFiNet
```
Navigate to the repo directory. Then install the repo as a package with:
```bash
cd HiFiNet
pdm install
```
*Optional*: Install linting and lsp depedencies for using type check. HiFiNet repo is strongly linted and type hinted:
```bash
pdm install -G:all
```

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
```bash
hifinet inject path/to/dataset
```
To customize fault parameters, supply a JSON file via `--fault-config`:
```bash
hifinet inject path/to/dataset --fault-config path/to/config.json
```
`config.json` should map fault names (for example, `drift` or `spike`) to their parameter overrides:
```json
{
  "drift": {"sigma": 0.02},
  "spike": {"bias_range": [2.0, 4.0]}
}
```
Inline JSON strings are supported for quick tweaks:
```bash
hifinet inject path/to/dataset --fault-config '{"drift": {"sigma": 0.02}}'
```
See the available option to configure the injection functionality:
```bash
hifinet inject --help
```
