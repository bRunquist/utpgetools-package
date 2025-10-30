# utpgetools

A collection of utility tools for your UT PGE projects.

## Installation

Install directly from GitHub with real time updates:

```bash
git clone https://github.com/bRunquist/utpgetools-package
cd \utpgetools-package
pip install -e .
```
Install from GitHub without realtime updates
```bash
git clone https://github.com/bRunquist/utpgetools-package
cd \utpgetools
pip install .
```

Install from PyPi:

```bash
pip install utpgetools
```

## Usage

```python
from utpgetools import hello_world

print(hello_world())
```
```python
import utpgetools.artificial_lift as al
al.pcpdesign()
```

## Development

- Clone the repo
- Install dependencies (`pip install -e .`)
- Modify/add your tools in `utpgetools/`

## License

Custom Academic Use License
