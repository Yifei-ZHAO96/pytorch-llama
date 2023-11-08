import time
import json
import torch
from typing import Tuple, List, Dict
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm import tqdm
from pathlib import Path
import os
from llama import ModelArgs, Tokenizer, Transformer, LLaMA, default_quantize

class LLaMAInference:
    def __init__(self, llama_path: str, model_size: str, device_map='auto', quantize: bool = True, **kwargs) -> None:
        """
        Initialize LLaMA instance by loading the pre-trained model.

        Args:
            llama_path (str): Path to the model weights.
            model_size (str): Model size.
            device_map (str, optional): Device map. Defaults to 'auto'.
            quantize (bool, optional): Where quantize the model to 8 bits. Defaults to True.
        
        """
        # load model abd tokenizer
        state_dict = os.path.join(llama_path, model_size, "state_dict.pth")
        params_file = os.path.join(llama_path, model_size, "params.json")
        tokenizer_path = os.path.join(llama_path, model_size, "tokenizer.model")

        assert os.path.exists(os.path.join(llama_path, model_size)), f"Model {model_size} does not exist"
        assert os.path.exists(state_dict), f"Model {model_size} does not exist"
        assert os.path.exists(params_file), f"Model {model_size} does not exist"
        assert os.path.exists(tokenizer_path), f"Missing tokenizer in {llama_path}"
        
        with open(params_file, "r") as f:
            params = json.load(f)
        
        model_args = dict(
            max_seq_len=2048,
            max_batch_size=1,
            **params
        )
        model_args.update(kwargs)
        model_args = ModelArgs(**model_args)
        
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = self.tokenizer.n_words
        model_args.device = 'cuda'
        
        # TODO: Currently only support either using 'accelerate' to scale to multiple gpus
        # or use quantization without 'accelerate'.
        if not quantize:
            with init_empty_weights():
                torch.set_default_dtype(torch.half)
                ctx_tok = default_quantize.set(quantize)
                self.model = Transformer(model_args)
                default_quantize.reset(ctx_tok)
            torch.set_default_dtype(torch.float)
            
            self.model = load_checkpoint_and_dispatch(
                self.model,
                state_dict,
                device_map=device_map,
                no_split_module_classes=["EncoderBlock"]
            )
        
        else:
            torch.set_default_dtype(torch.half)
            ctx_tok = default_quantize.set(quantize)
            self.model = Transformer(model_args)
            default_quantize.reset(ctx_tok)
            key_to_dim = {
                "w1": 0,
                "w2": -1,
                "w3": 0,
                "wo": -1,
                "wq": 0,
                "wk": 0,
                "wv": 0,
                "output": 0,
                "tok_embeddings": -1,
                "ffn_norm": None,
                "attention_norm": None,
                "norm": None,
                "rope": None,
            }
            torch.set_default_dtype(torch.float)

            # load the state dict incrementally, to avoid memory problems
            for i, ckpt in enumerate([state_dict]):
                print(f"Loading checkpoint {i}: {ckpt}")
                checkpoint = torch.load(ckpt, map_location="cpu")
                for parameter_name, parameter in self.model.named_parameters():        
                    short_name = parameter_name.split(".")[-2]
                    if key_to_dim[short_name] is None and i == 0:
                        parameter.data = checkpoint[parameter_name]
                    elif key_to_dim[short_name] == 0:
                        size = checkpoint[parameter_name].size(0)
                        parameter.data[size * i : size * (i + 1), :] = checkpoint[
                            parameter_name
                        ]
                    elif key_to_dim[short_name] == -1:
                        size = checkpoint[parameter_name].size(-1)
                        parameter.data[:, size * i : size * (i + 1)] = checkpoint[
                            parameter_name
                        ]
                    del checkpoint[parameter_name]
                del checkpoint
        
        self.model.cuda()
        self.generator = LLaMA(self.model, self.tokenizer)

    def generate(self, 
                 prompts: List[str], 
                 temperature: float = 0.9,
                 top_p: float = 0.9,
                 max_gen_len: int = 256,
                 repetition_penalty: float = 1.0,
                 stop_ids: List[int] = None,
                 stop_words: List[str] = None,
                 ) -> Tuple[List[str], Dict]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompts (List[str]): List of prompts, where each prompt is a string.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.9.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.95.
            max_gen_size (int):  Maximum length of the generated text sequence.
            repetition_penalty (float, optional): Repetition penalties can be used to counteract the model's tendency to repeat prompt text verbatim and/or get stuck in a loop. Defaults to 1.0.
            stop_words (List[str]): Stop words to stop the generation process.
            stop_ids (List[int]): Stop ids/tokens to stop the generation process.

        Returns:
            Tuple[List[str], Dict]: Generated text with original prompts and stats dictionary including 'num_input_tokens', 'num_generated_tokens', 'total_seconds' and 'token/s'.
            
        """
        start_time = time.time()
        results, stats = self.generator.generate(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_ids=stop_ids,
            stop_words=stop_words
        )
        end_time = time.time()
        stats['total_seconds'] = end_time - start_time
        stats['token/s'] = max(stats['num_generated_tokens']) / stats['total_seconds']
        
        return results, stats
