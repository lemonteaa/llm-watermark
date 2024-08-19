# Some math

## Proof of Gumbell re-parametrization

The main technique is the one used for Extreme Value Theory (EVT) to derive probability distribution of the max of n random variables where you'd convert to and forth pdf vs cdf.

Recall that the pdf of a $\text{Gumbell}(0, 1)$ is $e^{-(x + e^{-x})}$ and the corresponding cdf is $e^{-e^{x}}$. Without loss of generality, suppose the first token is selected, then we have $l_0 + g_0 \geq l_i + g_i$ for all $i \neq 0$. This is if and only if $g_0 \geq (l_i - l_0) + g_i$.

Roughly speaking (ignoring issues of continuous vs discrete random variable), we have

$$
\begin{eqnarray}
\Pr( g_0 \geq (l_i - l_0) + g_i \forall i \neq 0 ) &=& \int \Pr(g_0 = z) \Pr(z \geq (l_i - l_0) + g_i \forall i \neq 0 ) dz \\
&=& \int \Pr(g_0 = z) \prod_{i \neq 0} \Pr(z \geq (l_i - l_0) + g_i) dz
\end{eqnarray}
$$

Due to independence of the $g_i$. Then we substitute the pdf and cdf and compute:

$$
\begin{eqnarray}
& \int \Pr(g_0 = z) \prod_{i \neq 0} \Pr(z \geq (l_i - l_0) + g_i) dz \\
=& \int e^{-(z + e^{-z})} \prod_{i \neq 0} e^{-e^{-(z - (l_i - l_0))}} dz \\
=& \int \exp \left(- \left( z + e^{-z} + \sum_{i \neq 0} e^{-(z - (l_i - l_0))} \right) \right) dz \\
=& \int \exp \left( -z -e^{-z} \left( 1 + \sum_{i \neq 0} e^{l_i - l_0} \right) \right) dz \\
=& \int \exp \left( -z - \frac{1}{p_0} e^{-z} \right) dz \\
=& p_0
\end{eqnarray}
$$

Where $p_0 = \frac{e^{l_0}}{\sum_i e^{l_i}}$. (The last integral with $p_0$ can be manually checked)
