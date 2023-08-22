# hf-albert-early-exit
ELBERT model based on huggingface's albert implementation with early exit mechanism. 
Based on transformers version 4.31.0

based on the paper: https://arxiv.org/pdf/2107.00175.pdf \
and the corresponding github repo: https://github.com/shakeley/ELBERT/tree/master 

## Installation
```bash
pip install -r requirements.txt
pip install transformers[torch]
```
and then to run the experiments simply run the main.py file or:
```python
from main import run
run()
```



