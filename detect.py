import argparse

import numpy as np
import hashlib

from tokenizers import Tokenizer


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

def new_hash_custom(lst, key):
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

if __name__ == '__main__':
    main_desc = "Educational demostration of the classical Gumbell trick based LLM Watermarking method. This will detect if the text is likely generated by our watermarked LLM. (Warning: be aware of its limitation and DO NOT blindly trust its result)"
    parser = argparse.ArgumentParser(description=main_desc)
    
    # Global arguments    
    global_watermark_options = parser.add_argument_group(title="Watermarking Options", description="You MUST supply the exact same param as when you run the watermarked LLM to get correct result")
    global_watermark_options.add_argument("-s", "--secret", required=True, help="Secret key for the watermarking algorithm")
    global_watermark_options.add_argument("-w", "--window", required=True, type=int, help="Window length for watermarking")
    #global_watermark_options.add_argument("-H", "--holdout", required=True, type=int, help="Generate at least this many tokens before beginning watermarking")
    
    global_analysis_options = parser.add_argument_group(title="Analysis Options")
    global_analysis_options.add_argument("--hf-tok", required=True, help="ID of a Huggingface repo (Format: <user/org handle>/<repo name>) containing the tokenizer of the LLM model.")
    global_analysis_options.add_argument("-f", "--file", help="Source of text to be analyzed. If None, will read from stdin, otherwise read from the file specified.")
    global_analysis_options.add_argument("-o", "--out-np", help="(Optional) If specified, will serialize all relevant numpy data to a file for your further analysis, such as loading it in a Jupyter Notebook for visualization.")
    
    args = parser.parse_args()
    
    # "Xenova/Meta-Llama-3.1-Tokenizer"
    t = Tokenizer.from_pretrained(args.hf_tok)
    
    key = args.secret
    file_name = args.file
    #ep = 0.00000001
    ep = 0.000001
    
    window_m = args.window
    
    with open(file_name, mode="r", encoding="utf-8") as f:
        suspect = f.read()
    
    o = t.encode(suspect)
    #o.ids
    
    nv = t.get_vocab_size()
    #128256
    
    elems = []
    
    for idx, tok in enumerate(o.ids):
        if idx < window_m:
            continue
        hist = o.ids[idx-window_m:idx]
        digest = new_hash_custom(hist, key)
        rng = np.random.default_rng(seed=[x for x in digest])
        u = rng.random(nv).astype(dtype=np.float32)
        r = np.clip(u, ep, 1-ep)
        elems.append(1.0 - r[tok])
    
    scores = -1.0 * np.ma.log(np.array(elems, dtype=np.float32))
    avg_s = np.ma.average(scores)
    
    print(f"Average score is: {avg_s}")
    # Save file
    np.savez("np_dump.npz", token_ids=o.ids, scores=scores.filled(fill_value=np.nan), avg=avg_s)
