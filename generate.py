import numpy as np
import numpy.typing as npt

import random
from datetime import datetime
#random.seed(datetime.now().timestamp())
#random.random()

# rng = np.random.default_rng(seed=42)
# rng.random(6)
# rng = np.random.default_rng(seed=[x for x in m.digest()])

import hashlib
#m = hashlib.sha256()
#m.update(b"")
#m.digest()

from llama_cpp import Llama, LogitsProcessor, LogitsProcessorList

def expand_int(n):
    c = n % 256
    tmp = n // 256
    b = tmp % 256
    a = tmp // 256
    return [a, b, c]

def expand_list(lst):
    res = list()
    for item in lst:
        res += expand_int(item)
    return res

def hash_custom(lst, key):
    m = hashlib.sha256()
    history = bytes(expand_list(lst))
    m.update(history)
    m.update(key.encode("utf-8"))
    return m.digest()

class WaterMarkingLogitsProcessor(LogitsProcessor):
    def __init__(self, temperature, key, n_vocab, ep, window_m, holdout):
        self.temp = temperature
        self.key = key
        self.n_vocab = n_vocab
        self.ep = ep
        self.window_m = window_m
        self.holdout = holdout
        self.last_m = []
        self.rng = np.random.default_rng()
        self.first_time = True
        #self.original_random_seed = datetime.now().timestamp()
        self._phasetwo = False
    
    def __call__(self, input_ids: npt.NDArray[np.intc], scores: npt.NDArray[np.single]) -> npt.NDArray[np.single]:
        if self.first_time:
            self.first_time = False
            self.prompt_tokens = len(input_ids)
        if len(scores) != self.n_vocab:
            raise ValueError(f"score array len is {len(scores)}")
        # Find the recent history window
        outputs = input_ids[self.prompt_tokens:] # Cut the eval prompt
        self.last_m = outputs[-self.window_m:]
        #if len(self.last_m) >= self.window_m:
        if len(outputs) >= self.holdout:
            if not self._phasetwo:
                print("[[DEBUG: ok]]")
                self._phasetwo = True
            # Use history window + secret key => hash => psuedo-random sampling
            # For Initial segment, sample in a normally random way (so skip)
            digest = hash_custom(self.last_m, self.key)
            self.rng = np.random.default_rng(seed=[x for x in digest])
        # Gumbell sampling trick
        u = self.rng.random(self.n_vocab).astype(dtype=np.float32)
        r = np.clip(u, self.ep, 1-self.ep)
        g = -1.0 * np.log(-1.0 * np.log(r))
        return (scores/(self.temp) + g)


llm = Llama(
      model_path="C:\\Users\\user\\Desktop\\LLM_Model_Collections\\bartowski-Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf",
      #chat_format="llama-2",
      n_ctx=2048,
      n_gpu_layers=-1
)



prompt = b"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Write an essay on the role of information technology in international supply chain.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

tokens = llm.tokenize(prompt)

my_logit_proc = WaterMarkingLogitsProcessor(temperature=0.7, key="helloworld", n_vocab=llm.n_vocab(), ep=0.000001, window_m=6, holdout=30)

for token in llm.generate(tokens, temp=0.0, repeat_penalty=1.0, logits_processor=LogitsProcessorList([my_logit_proc])):
    #for token in llm.generate(tokens, temp=0.7, repeat_penalty=1.0):
    if token in [128007, 128008, 128009, 128001]:
        break
    #print(f"T: {token}", flush=True)
    print(llm.detokenize([token]).decode("utf-8"), end='', flush=True)





