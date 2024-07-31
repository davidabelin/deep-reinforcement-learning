

$\frac{1}{T}\sum^T_t\left(1-\frac{1}{N}\right)$

$\min\left\{R_{t}^{\rm future}\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)},R_{t}^{\rm future}{\rm clip}_{\epsilon}\!\left(\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}\right)\right\}$

$\frac{1}{T}\sum^T_t \min\left\{R_{t}^{\rm future},R_{t}^{\rm future} \right\}$


the ${\rm clip}_\epsilon$ function is implemented as 

* ${\rm clip}_{\epsilon}\!\left(\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right)$

${\pi_{\theta'}(a_t|s_t)}$ / ${\pi_{\theta}(a_t|s_t)}$


the ${\rm clip}_\epsilon$ function is implemented in pytorch as ```torch.clamp(ratio, 1-epsilon, 1+epsilon)```