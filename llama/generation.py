# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import torch
from typing import List

class LLaMA:
    def __init__(self, model: 'Transformer', tokenizer: 'Tokenizer') -> None:
        """
        Initialize LLaMa instance.

        Args:
            model (Transformer): Transformer model.
            tokenizer (Tokenizer): Tokenizer.
            
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def _should_stop(self, 
                     tokens: List[List[int]], 
                     prompt_tokens: List[List[str]],
                     stop_ids: List[int],
                     stop_words: List[str]
                     ) -> bool:
        """
        With given 'stop_ids' and 'stop_words', determine where the generation should stop.

        Args:
            tokens (List[List[int]]): Encoded tokens.
            prompt_tokens (List[List[str]]): Decoded prompt tokens and generated tokens.
            stop_ids (List[int]): A list of tokens to determine whether we should stop the generation.
            stop_words (List[str]): A list of words to determine whether we should stop the generation.

        Returns:
            bool: True for stop ALL generation process, else False.
            
        """
        if stop_ids is not None:
            do_stop = [False for _ in range(len(tokens))]
            for i, (t, p) in enumerate(zip(tokens, prompt_tokens)):
                g = t[len(p):].tolist()
                for stop_id in stop_ids:
                    if stop_ids in g:
                        do_stop[i] = True
                        break
            
            if all(do_stop):
                return True
        
        if stop_words is not None:
            do_stop = [False for _ in range(len(tokens))]
            for i, (t, p) in enumerate(zip(tokens, prompt_tokens)):
                t = t.clone()
                g = t[len(p):]
                g[g == self.tokenizer.pad_id] = self.tokenizer.eos_id
                g = g.tolist()
                d = self.tokenizer.decode(g)
                for stop_word in stop_words:
                    if stop_word in d:
                        do_stop[i] = True
        
            if all(do_stop):
                return True
            
        return False
    
    def generate(self, 
                 prompts: List[str],
                 max_gen_len: int,
                 temperature: float = 0.9,
                 top_p: float = 0.95,
                 stop_words: List[str] = None,
                 stop_ids: List[int] = None,
                 repetition_penalty: float = 1.0
                 ) -> List[str]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompts (List[str]): List of prompts, where each prompt is a string.
            max_gen_len (int):  Maximum length of the generated text sequence.
            stop_words (List[str]): Stop words to stop the generation process.
            stop_ids (List[int]): Stop ids/tokens to stop the generation process.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.9.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.95.
            repetition_penalty (float, optional): Repetition penalties can be used to counteract the model's tendency to repeat prompt text verbatim and/or get stuck in a loop. Defaults to 1.0.

        Returns:
            List[str]: Generated text with the original prompts.
            
        """
        args = self.model.args
        batch_size = len(prompts)
        assert batch_size <= args.max_batch_size, (batch_size, args.max_batch_size)
        
        prompt_tokens = [self.tokenizer.encode(prompt, bos=True, eos=False)
                         for prompt in prompts]
        num_input_tokens = [len(t) for t in prompt_tokens]
        min_prompt_size = min(num_input_tokens)
        max_prompt_size = max(num_input_tokens)
        
        total_len = min(args.max_seq_len, max_prompt_size + max_gen_len)
        
        tokens = torch.full((batch_size, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos: cur_pos], prev_pos)
            if repetition_penalty != 1.0:
                logits_new = logits.clone()
                for batch in range(batch_size):
                    for token in set(tokens[batch].tolist()):
                        if logits[batch, token] < 0:
                            logits_new[batch, token] = logits[batch, token] * repetition_penalty
                        else:
                            logits_new[batch, token] = logits[batch, token] / repetition_penalty
                logits = logits_new
            
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            
            if self._should_stop(tokens, prompt_tokens, stop_ids, stop_words):
                break
        
        tokens[tokens == self.tokenizer.pad_id] = self.tokenizer.eos_id
        decoded = []
        num_generated_tokens = []
        
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[:len(prompt_tokens[i]) + max_gen_len]
            # cut to eos token if any
            try:
                num_generated_tokens.append(t.index(self.tokenizer.eos_id)-len(prompt_tokens[i]))
                t = t[:, t.index(self.tokenizer.eos_id)]
            except ValueError:
                num_generated_tokens.append(max_gen_len)
            decoded.append(self.tokenizer.decode(t))
        
        return decoded, dict(num_input_tokens=num_input_tokens, num_generated_tokens=num_generated_tokens)


def sample_top_p(probs, top_p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        top_p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, dim=-1, index=next_token)
    
    return next_token