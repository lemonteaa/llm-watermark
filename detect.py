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


t = Tokenizer.from_pretrained("Xenova/Meta-Llama-3.1-Tokenizer")

key = "helloworld"

file_name = "marked1edit.txt"

#ep = 0.00000001
ep = 0.000001

window_m = 6

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
    digest = hash_custom(hist, key)
    rng = np.random.default_rng(seed=[x for x in digest])
    u = rng.random(nv).astype(dtype=np.float32)
    r = np.clip(u, ep, 1-ep)
    elems.append(1.0 - r[tok])

scores = -1.0 * np.ma.log(np.array(elems, dtype=np.float32))
avg_s = np.ma.average(scores)

print(f"Average score is: {avg_s}")

