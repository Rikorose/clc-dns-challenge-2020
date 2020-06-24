# CLCNet results for the DNS Challenge 2020

Implementation for the Paper [CLC: Complex Linear Coding for the DNS 2020 Challenge](https://arxiv.org/abs/2006.13077).

To run this model on some noisy audio files, use the python script
`clcnet-dns2020/enhance_jit.py` and a model file in `models`:
```py
python enhance_jit.py models/clc.pt <input_noisy_dir> <output_enhanced_dir>
```
Citation:
```bibtex
@misc{schrter2020clc,
    title={CLC: Complex Linear Coding for the DNS 2020 Challenge},
    author={Hendrik Schröter and Tobias Rosenkranz and Alberto N. Escalante-B. and Andreas Maier},
    year={2020},
    eprint={2006.13077},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
