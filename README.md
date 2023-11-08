## Installation ⚙️ 

Clone this repository

```
git clone https://github.com/Yifei-ZHAO96/pytorch-llama.git
cd pytorch-llama
```

Install the requirements

```
python3 -m venv env
. ./env/bin/activate
pip install -r requirements.txt
```

## Use Example ##

```python
from inference import LLaMAInference

llama = LLaMAInference(llama_path, "70B")
print(llama.generate(["Chat:\nHuman: Tell me a joke about artificial intelligence.\nAI:"])[0])
```

## Convert LLaMA weights

To convert LLaMA weights to a plain pytorch state-dict run

```
python convert.py --llama-path <ORIGINAL-LLAMA-WEIGHTS> --model <MODEL> --output-path <CONVERTED-WEIGHTS-PATH>
```

## References

- [vanilla-llama](https://github.com/galatolofederico/vanilla-llama)
- [pytorch-llama](https://github.com/hkproj/pytorch-llama)
- [llama-int8](https://github.com/tloen/llama-int8/tree/main)