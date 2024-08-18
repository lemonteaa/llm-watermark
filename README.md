# LLM Watermarking demo

> **Important Notice**: Watermarking can be a socially controversial topic with arguments supporting or objecting to their use and deployment. The situation is complex, involving intersection of technology with society and their complex interaction, unintended second order effect, etc. At any rate, try to recognize that technical measures in a vacuum, without considering the surrounding socio-historical context, is often a bad idea, and that Watermarking, at its current state, is not a magical cure-all. One way to use this repo is to experiment with it yourself to see what watermarking can do and how/when will it produces false positives/negatives. Due to the possibility of false positive - some watermarking methods try to control it to a very low rate or use cryptographic style signature etc, but this is dependent on the specific method used - please reconsider before blindly trusting their results. This repo is educational only and do not imply the author's stance, thank you.

Currently implement the classic version of the Gumbell method.

## Dependencies

- `llama-cpp-python`
- Huggingface's `tokenizers`

## Theory background

### Gumbell sampling/reparametrization trick

Let $l_i$ be the logit vector that is output from a LLM to predict the next token, each number is a logit score assigned to the $i$-th token in the vocabulary $\mathcal{V}$. In basic sampling, we transform it into a probability distribution over the vocab as $p_i = \text{softmax}(l_i)$, then randomly sample the next token according to this distribution.

Let $g_i$ be a vector of i.i.d $\text{Gumbell}(0, 1)$ distributed random variables. (Check wiki page for definition, Gumbell is related to *extreme value theory* in statistics) Interestingly, choosing the next token simply by greedy sampling over a biased logits, $\text{argmax}(l_i + g_i)$ is equivalent to the softmax sampling above (!).

### Watermarking LLM

(TODO: add ACL reference)

Rephrased in a way that hopefully non-researcher can understand...

Given a LLM, run it in a modified manner to produce outputs that are subtly different. Ideally the modification is agnostic to which brand of LLM model you're using, so it's like adding a wrapper around it. And it should also have reasonably negligible performance impact (both in terms of speed and in terms of the model's "intelligence"). Separately, there should be a detection program that can run without access to the original LLM.

Academically, the proposed four attributes to evaluate a watermarking scheme is:
- Detection
- Quality
- Robustness
- Security

## Algorithm

### Generation

- Basic idea from steagnogrphy - a branch of cryptography where you try to inject message in some data while escaping detection by your adversary (which in this case is the user of LLM).
- Recall one-time pad encryption => In the Gumbell sampling trick, we may regard the randomness $g_i$ as kind of like the secret key (fuzzy idea only, may not be technically accurate)
- What we'll actually do: Generate $g_i$ as a pseudo-random stream derived from a secret key (like in stream based cipher)
- Consequences: 1. Without knowledge of secret key, would be indistinguashable from actually iid Gumbell random variables which is the LLM's original distribution (hence "Quality" is not impacted/ called "distortion-free" in literature). 2. But if you know the secret key, $g_i$ is deterministically known, so even without knowing anything about $l_i$, simply by piror, we know the logit bias/some tokens have a higher starting point and is more likely to be chosen on the outset.
- But how to choose the secret key? Since we'll generate many token => need a key-scheduling method
- Naive: same secret at all position (violate statistical independence across token, no longer distortion free/the pattern may get caught by user)
- Naive: secret key a fixed function of the token position (i.e. fixed key $k_1, k_2, \cdots$) (Ok is statistically independent this time, but if user just shift the text by one token, the same statistical independence would also mean it's no longer detected, hence not robust)
- Idea: key is a cryptographic hash of the tokens in a sliding window of the previous $m$ tokens together with your own secret, $k_t = \text{SHA256}(w_{t-m}, w_{t-m+1}, \cdots, w_{t-1}, S)$, where $w_t$ are the generated tokens and $S$ is your secret.
- Some implementation details: Once we start doing this, the text generation is effectively deterministic, so we should first generate some tokens at the beginning without watermarking to give it some randomness before starting to change $g_i$ to pseudo-random. If you don't do this, you will notice that given the same input, the output will always be the same.

### Detection

- Scan through the tokenized text, and for each token, recompute the key at that position using the formula above (i.e. sliding window plus secret) to regenerate $g_i$. Then we compute a per token statistics: $s_t = -\log(1 - \zeta_t[w_t])$.
- Details: A $\text{Gumbell}(0, 1)$ distribution can be sampled by the inverse CDF method as $g_i = -\log(- \log(\zeta_i))$, where $\zeta_i \sim \text{Uniform}(0, 1)$. So this is where the $\zeta_i$ above comes from. Notice this transform is monotone, so higher $\zeta_i$ implies higher $g_i$ and vice versa. The square bracket in the formula denotes the value of the zeta vector at the coordinate of the selected token $w_t$.
- Explanation: A rough intuition is that if the text is not watermarked/doesn't comes from the LLM, then $g_i$ (and hence $\zeta_i$) would be effectively statistically independent from the text, so the statistics will have a null hypothesis expected value. On the other hand, if it is watermarked, per remarks above, higher logit bias tokens are more likely to be selected, so when we turn around and examine the logit bias value of the selected token, they'd tends to be higher than average too, which is what this test capture.
- Average this statistics over all token in the text to reduce variance, then do the usual hypothesis testing stuff.

### Robustness

What if the user edit the text or copy/paste/move text around? Provided the editing is not too much (formalized by edit distance - see wiki page), we can still detect it somewhat. To see why, you may think through each scenario of "change one token/x contiguous tokens", "cut a segment of x contiguous tokens and insert it at another position" etc.

The key (pun not intended) lies in examining how editing affect the sliding windows. You should find out that the effect of the kinds of edits above are *local* - they may "corrupt" the sliding window only of themselves plus some token within the range of the sliding window at the edit boundary. After that other tokens still have an intact sliding window, and that's all you need to derive the per position key $k_i$ correctly, so those are not affected.

