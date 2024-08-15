import numpy as np
import numpy.typing as npt

import hashlib

from typing import NamedTuple

from datetime import datetime # For LLM self-knowledge

from llama_cpp import Llama, LogitsProcessor, LogitsProcessorList

import argparse

def hash_custom(lst, key):
    def radix_expand(n, base):
        digits = []
        while n > 0:
            digits.append(n % base)
            n = n // base
        return digits
    def expand_list(lst, base):
        res = list()
        for item in lst:
            res += radix_expand(item, base)
        return res
    m = hashlib.sha256()
    history = bytes(expand_list(lst, 256))
    m.update(history)
    m.update(key.encode("utf-8"))
    return m.digest()

class WaterMarkConfig(NamedTuple):
    temperature: float
    key: str
    n_vocab: int
    window_m: int
    holdout: int
    ep: float = 0.000001

class WaterMarkingLogitsProcessor(LogitsProcessor):
    def __init__(self, config: WaterMarkConfig):
        self.config = config
        self.last_m = []
        self.native_rng = np.random.default_rng()
        self.first_time = True
        self.enable = True
    
    def __call__(self, input_ids: npt.NDArray[np.intc], scores: npt.NDArray[np.single]) -> npt.NDArray[np.single]:
        if self.first_time:
            self.first_time = False
            self.prompt_tokens = len(input_ids)
        if len(scores) != self.config.n_vocab:
            raise ValueError(f"score array len is {len(scores)}")
        # Find the recent history window
        outputs = input_ids[self.prompt_tokens:] # Cut the eval prompt
        self.last_m = outputs[-self.config.window_m:]
        
        active_rng = self.native_rng
        if len(outputs) >= self.config.holdout and self.enable:
            # Use history window + secret key => hash => psuedo-random sampling
            # For Initial segment, sample in a normally random way (so skip)
            digest = hash_custom(self.last_m, self.config.key)
            active_rng = np.random.default_rng(seed=[x for x in digest])
        # Gumbell sampling trick
        u = active_rng.random(self.config.n_vocab).astype(dtype=np.float32)
        r = np.clip(u, self.config.ep, 1-self.config.ep)
        g = -1.0 * np.log(-1.0 * np.log(r))
        return (scores/(self.config.temperature) + g)

class CustomLLMGeneration:
    def __init__(self, llm, prompt_template, logit_processors):
        self.llm = llm
        self.prompt_template = prompt_template
        self.logit_processors = logit_processors
    
    def generate(self, prompt):
        compiled_prompt = self.prompt_template.format(query=prompt, today=datetime.now().strftime("%d %b, %Y"))
        tokens = self.llm.tokenize(compiled_prompt.encode("utf-8"))
        for token in self.llm.generate(tokens, temp=0.0, repeat_penalty=1.0, logits_processor=LogitsProcessorList(self.logit_processors)):
            if token in [128007, 128008, 128009, 128001]:
                break
            #print(f"T: {token}", flush=True)
            print(self.llm.detokenize([token]).decode("utf-8"), end='', flush=True)

prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: {today}

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

if __name__ == '__main__':
    main_desc = "Educational demostration of the classical Gumbell trick based LLM Watermarking method."
    parser = argparse.ArgumentParser(description=main_desc)
    
    # Global arguments
    
    global_model_options = parser.add_argument_group("Model Options")
    global_model_options.add_argument("-m", "--model", required=True, help="Path to LLM model file")
    global_model_options.add_argument("--temp", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    global_model_options.add_argument("--ctx", type=int, default=2000, help="Context length (default: 2000)")
    global_model_options.add_argument("--ngl", type=int, default=-1, help="Number of layers to offload to GPU (default: all, i.e. -1)")
    
    global_watermark_options = parser.add_argument_group("Watermarking Options")
    global_watermark_options.add_argument("-s", "--secret", required=True, help="Secret key for the watermarking algorithm")
    global_watermark_options.add_argument("-w", "--window", required=True, type=int, help="Window length for watermarking")
    global_watermark_options.add_argument("-H", "--holdout", required=True, type=int, help="Generate at least this many tokens before beginning watermarking")
    
    # Subparsers
    subparsers = parser.add_subparsers(dest="mode", title='Main command', description="", help="Select one of the following", metavar='mode', required=True)
    
    chat_mode_parser = subparsers.add_parser("chat", help="Chat mode", description=main_desc + " Chat mode: Interactively chat with a LLM with watermarking enabled.")
    chat_mode_parser.add_argument("-a", "--arg1", help="argument 1 for mode 1")
    chat_mode_parser.add_argument("-b", "--arg2", help="argument 2 for mode 1")
    
    batch_mode_parser = subparsers.add_parser("batch", help="Batch mode", description=main_desc + " Batch mode: the program will automatically generate the output artifacts to either stdout or save to files depending on your setting, then exit.")
    batch_mode_parser.add_argument("-n", type=int, default=1, help="Number of examples to generate at once (default: 1)")
    batch_mode_parser.add_argument("-f", "--file", help="Prefix for output file names (default: output to stdout)")
    batch_mode_parser.add_argument("-p", "--prompt", required=True, help="Prompt for LLM")
    batch_mode_parser.add_argument("--no-baseline", action="store_false", help="Should we also generate a baseline without watermarking for each example? Default true, add this switch to disable.")
    
    # Parse arguments
    args = parser.parse_args()
    if args.mode == "chat":
        print("Not implemented yet")
    elif args.mode == "batch":
        print(args.model) #"C:\\Users\\user\\Desktop\\LLM_Model_Collections\\bartowski-Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf"
        print(args.temp)
        print(args.ctx)
        print(args.ngl)
        print(args.secret)
        print(args.window)
        print(args.holdout)
        
        print(args.n)
        print(args.file)
        print(args.prompt)
        print(args.no_baseline) # Careful that it is flipped
        llm = Llama(
              model_path=args.model,
              n_ctx=args.ctx,
              n_gpu_layers=args.ngl
        )
        
        #prompt = "Write an essay on the role of information technology in international supply chain."
        prompt = args.prompt
        
        my_conf = WaterMarkConfig(temperature=args.temp, key=args.secret, n_vocab=llm.n_vocab(), window_m=args.window, holdout=args.holdout)
        
        my_logit_proc = WaterMarkingLogitsProcessor(config=my_conf)
        
        mainLLM = CustomLLMGeneration(llm, prompt_template, [my_logit_proc])
        
        mainLLM.generate(prompt)
