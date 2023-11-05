import time
import json
import torch
from typing import Tuple, List, Dict
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm import tqdm
from pathlib import Path
import os
from llama import ModelArgs, Tokenizer, Transformer, LLaMA

class LLaMAInference:
    def __init__(self, llama_path: str, model_size: str, device_map='auto', **kwargs) -> None:
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
        
        with init_empty_weights():
            torch.set_default_tensor_type(torch.HalfTensor)
            model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        
        self.model = load_checkpoint_and_dispatch(
            model,
            state_dict,
            device_map=device_map,
            no_split_module_classes=["EncoderBlock"]
        )
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
