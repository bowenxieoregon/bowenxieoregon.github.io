---
date: '2026-01-04'
draft: true
title: 'Diffusion Models for Continuous and Discrete Data'
description: 'A unified view of diffusion models: from the continuous DDPM formulation to discrete diffusion for language and graphs.'
tags: ['Diffusion Models', 'Generative Models', 'Deep Learning', 'Theory']
math: true
comments: true
---

Diffusion models have emerged as a powerful framework for generative modeling, achieving state-of-the-art results in image synthesis, audio generation, and more recently, discrete domains like text and molecular design. This post explores the theoretical foundations of diffusion models for both continuous and discrete data.

## The Core Idea

At its heart, a diffusion model learns to **reverse a corruption process**. We gradually add noise to data until it becomes pure noise, then train a neural network to reverse this process step by step.

$$
\text{Data} \xrightarrow{\text{forward (noise)}} \text{Pure Noise} \xrightarrow{\text{reverse (denoise)}} \text{Generated Data}
$$

## Continuous Diffusion (DDPM)

### Forward Process

Given data $\mathbf{x}_0 \sim q(\mathbf{x})$, we define a forward Markov chain that gradually adds Gaussian noise:

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

where $\beta_t$ is a noise schedule. A key insight is that we can sample $\mathbf{x}_t$ directly from $\mathbf{x}_0$:

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})
$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$.

### Reverse Process

The reverse process is parameterized by a neural network $\epsilon_\theta$:

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})
$$

### Training Objective

The simplified loss function is:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon}\left[ \|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)\|^2 \right]
$$

where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ is the noise added to create $\mathbf{x}_t$.

## Discrete Diffusion

For discrete data (text, graphs, categorical variables), we need different corruption processes.

### Discrete Forward Process

Instead of Gaussian noise, we use **transition matrices**. For data with $K$ categories:

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \text{Cat}(\mathbf{x}_t; \mathbf{Q}_t \mathbf{x}_{t-1})
$$

where $\mathbf{Q}_t \in \mathbb{R}^{K \times K}$ is a transition matrix.

### Common Corruption Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Uniform** | Transition to any token with equal probability | Text generation |
| **Absorbing** | Transition to a special [MASK] token | Masked language models |
| **Discretized Gaussian** | Embed tokens, add noise, round back | Preserves similarity |

### Absorbing State Diffusion

A popular choice for language modeling:

$$
\mathbf{Q}_t = \begin{pmatrix} 1-\beta_t & \beta_t & 0 & \cdots \\ 0 & 1 & 0 & \cdots \\ \vdots & \vdots & \ddots & \vdots \end{pmatrix}
$$

Tokens either stay the same or become [MASK].

## Connecting Continuous and Discrete

Recent work shows these frameworks are deeply connected:

1. **Score matching** in continuous space ↔ **Ratio matching** in discrete space
2. Both can be viewed through the lens of **stochastic differential equations (SDEs)**
3. The ELBO derivation follows similar structure

### Unified View

$$
\mathcal{L} = \mathbb{E}\left[ D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) \right]
$$

This objective works for both continuous (with Gaussian KL) and discrete (with categorical KL) cases.

## Practical Considerations

### For Continuous Data
- Use cosine or linear noise schedules
- Predict noise $\epsilon$ or score $\nabla \log p$
- Classifier-free guidance for conditional generation

### For Discrete Data
- Absorbing state works well for text
- Consider using continuous embeddings + rounding
- Parallel decoding possible (unlike autoregressive)

## Code Example

```python
import torch
import torch.nn.functional as F

def q_sample_continuous(x_0, t, noise_schedule):
    """Sample x_t given x_0 for continuous diffusion."""
    alpha_bar = noise_schedule.alpha_bar[t]
    noise = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
    return x_t, noise

def q_sample_discrete(x_0, t, transition_matrix):
    """Sample x_t given x_0 for discrete diffusion."""
    Q_bar = transition_matrix.get_Qt_bar(t)  # Cumulative transition
    probs = Q_bar[x_0]  # Get transition probabilities
    x_t = torch.multinomial(probs, num_samples=1)
    return x_t
```

## Key Papers

1. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
2. **Score SDE**: Song et al., "Score-Based Generative Modeling through SDEs" (2021)
3. **D3PM**: Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces" (2021)
4. **Discrete Flow**: Campbell et al., "A Continuous Time Framework for Discrete Denoising Models" (2022)

## Conclusion

Diffusion models provide a flexible framework for generative modeling across data types. The key insight — learning to reverse a corruption process — applies equally to images, audio, text, and graphs. Understanding both continuous and discrete formulations opens doors to hybrid approaches and novel applications.

---

*What aspects of diffusion models would you like me to dive deeper into? Leave a comment below!*
