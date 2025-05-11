# LLM inference economics from first principles

The main product LLM companies offer these days is access to their models via API, and the key question that will determine the profit margins they can enjoy is the inference cost structure. In this text we will explain where the cost of serving/hosting LLMs comes from, how many tokens can be produced by a GPU, and why this is the case. We will build a (simplified) world model of LLM inference arithmetics, based on the popular open-source model-LLama 3.3. The goal is to develop an accurate intuition regarding LLM inference.

The topic of LLM inference economics has far-reaching implications beyond technical considerations. As AI capabilities rapidly advance, inference efficiency directly shapes both industry economics and accessibility. For AI labs, token production costs fundamentally determine profit margins and the cost of generating synthetic training data -more efficient inference means higher returns on a fixed investment in hardware that can fuel further research and development cycles. For users, lower token costs democratize access to these powerful tools, potentially transforming AI from a premium resource into an everyday utility available for even routine tasks. Understanding these cost structures isn't merely academic-it provides insight into one of the key economic forces that will shape AI development in the coming years as we approach increasingly capable systems.

The primary cost behind a generated token boils down to the cost of compute-you need to buy or rent a GPU. In both cases, there is a fixed cost associated with running a GPU per hour. Each GPU can produce a limited number of tokens in an hour. The number of tokens produced per hour divided by the cost of hardware per hour will tell you the unit cost of generating a single token. This is how most of the LLM providers price their API offerings, and this will be the model we will explore.

## Model parameters and hardware requirements

As a basis for our inference economics analysis, we will use [LLama 3.3 70B](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md). Even today it is still one of the most popular open-source models and an architecture around which [a big portion of the industry standardized](https://huggingface.co/models?other=llama). There are numerous model fine-tunes of the Llama weights, and while these models will produce different outputs, because they share the same model architecture, they require exactly the same compute resources to run them. Hence, we consider Llama a good candidate to provide a real-world example that will be representative but, at the same time, quite simple to grasp.

LLMs store their "knowledge" in parameters-essentially the weights that define the model's behavior. These parameters require memory to store and compute resources to process. Generally, the more parameters a model has, the greater its resource demands, but so are its potential capability on downstream tasks. Llama 3.3 70B has around 70 billion parameters, which is where its name comes from.

A so-called decoder-only transformer model, like Llama, usually consists of the following components:

- one input embedding layer-converting tokens, or words, into vector representations.
- multiple transformer layers, each layer containing some parameters for the self-attention part and some for the MLP part.
- lanugage modeling (LM) head-the final layer

We assume the reader has a basic understanding of these concepts; hence, we will not be providing the deep intuitions behind them. If you are unfamiliar with the transformer architecture, stop here and check out [one](https://jalammar.github.io/illustrated-transformer/) [of](https://www.youtube.com/watch?v=kCc8FmEb1nY&t) [these](https://youtu.be/wjZofJX0v4M) amazing tutorials.

```jsx
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
	...
	"head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 8192,
  ...
  "intermediate_size": 28672,
  "max_position_embeddings": 131072,
	...
  "model_type": "llama",
  "num_attention_heads": 64,
  "num_hidden_layers": 80,
  "num_key_value_heads": 8,
  "torch_dtype": "bfloat16",
  ...
  "vocab_size": 128256
}
```

Fig. 1: config of llama 3.3 that will be using throughput this text as a reference model. [https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/config.json](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/config.json)

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image.png)

Fig. 2 Llama architecture: [https://devopedia.org/images/article/483/6445.1717130942.jpg](https://devopedia.org/images/article/483/6445.1717130942.jpg)

Now, let's break down the model's parameter count step by step, verifying that the claimed 70 billion parameters hold up. To do so, let’s start by looking at the Llama model config in Fig. 1. We can see there are multiple keys and values. Keys, such as `hidden_size` informing us about the sizes of specific parts of the model. We can use them to calculate the total model size. To grasp the high-level overview of the architecture, take a look at Fig. 2. You can see all three parts we described above, and the graph also shows some implementation details that we will dive into in the next section.

For input embedding layer, we need to convert every possible token position into its vector representation. Hence we have:

$$
\text{Input Embedding} = \text{hidden\_size} \times \text{vocab\_size}
$$

parameters.

Then we have `N` transformer layers (see Fig. 2). Each of these layers has typically many millions parameters that we will need to store in memory in order to run the model. Their sizes can be calculated with hyper parameters from the config.json above: :

$$
\text{w\_q} = \text{hidden\_size} \times \text{head\_size} \times \text{num\_attention\_heads} = \text{hidden\_size} \times \text{hidden\_size}
$$

$$
\text{w\_k} = \text{hidden\_size} \times \text{head\_size} \times \text{num\_key\_value\_heads} = \text{hidden\_size} \times \frac{\text{hidden\_size}}{8} ^*
$$

$$
\text{w\_v} = \text{hidden\_size} \times \text{head\_size} \times \text{num\_key\_value\_heads} = \text{hidden\_size} \times \frac{\text{hidden\_size}}{8} ^*
$$

$$
\text{w\_o} = \text{hidden\_size} \times \text{hidden\_size}
$$

$$
\text{w\_1} = \text{hidden\_size} \times \text{intermediate\_size} = \text{hidden\_size} \times \text{hidden\_size} \times 3.5^*
$$

$$
\text{w\_2} = \text{hidden\_size} \times \text{intermediate\_size} = \text{hidden\_size} \times \text{hidden\_size} \times 3.5^*
$$

$$
\text{w\_3} = \text{intermediate\_size} \times \text{hidden\_size} = 3.5^* \times \text{hidden\_size} \times \text{hidden\_size}
$$

$$
\text{2 $\times$ RMS norm} = 2 \times \text{hidden\_size} \text{ parameters}
$$


\*The `w_v` and `w_k` being 1/8th the size of the `w_q` is something Llama architecture specific. This is due to the Llama team using a technique called [Group Query Attention](https://arxiv.org/pdf/2305.13245) in which the model has fewer K and V heads than the total attention heads. You can verify this by looking at `num_key_value_heads` in the hyperparameters from the model config. The model `intermediate_size` being `3.5x` the hidden size is as well a Llama architecture-specific value. These were chosen by the Llama team, and we take them at face value, also to simplify our calculations.

Bringing us to a total of

$$
\text{Total per transformer block} = \text{hidden\_size}^2 + \frac{\text{hidden\_size}^2}{8} + \frac{\text{hidden\_size}^2}{8} + 3.5 \times \text{hidden\_size}^2 + 3.5 \times \text{hidden\_size}^2 + 3.5 \times \text{hidden\_size}^2 + \text{hidden\_size}^2 + 2 \times \text{hidden\_size} = 12.75 \times \text{hidden\_size}^2 + 2 \times \text{hidden\_size}
$$

per transformer block.

Finally, we apply a last RMS Norm before feeding the representation into the LM head, which converts vectors into token logits.

$$
\text{RMS Norm} = \text{hidden\_size}
$$

$$
\text{LM Head} = \text{hidden\_size} \times \text{vocab\_size}
$$

Summing up all of these parameters we obtain:

$$
\text{Total parameters} = \text{vocab\_size} \times \text{hidden\_size} + \text{num\_hidden\_layers} \times (12.75 \times \text{hidden\_size}^2 + 2 \times \text{hidden\_size}) +  \text{hidden\_size} + \text{hidden\_size} \times \text{vocab\_size}
$$

We can find the values of each of these, `vocab_size` , `hidden_size` , `num_hidden_layers`  in the config in Fig. 1. Substituting these values into the equation we will get:

$$
\text{Total parameters} =  128256 \times 8192 + 80 \times (12.75 \times 8192^2 + 2 \times 8192) + 8192 + 8192 \times 128256 =  70,553,706,496
$$

Each parameter is a floating-point number in a bfloat16 format-e.g., 0.22312, -4.3131. Storing each of these numbers takes 16 bits which is 2 bytes of memory. Given that we have a total of `70,553,706,496` parameters to store, we will need `141,107,412,992` bytes or `141GB` just to store the model weights in GPU memory.

Note that 141GB is more memory than there is on the most common data center GPUs, such as the Nvidia A100 or H100. Each of these GPUs comes with only 80GB of total memory (we refer to this memory interchangeably as HBM, high bandwidth memory, or global memory). Hence, for serving models, we usually use multiple cards for a single instance of a model. In practice, for more optimal model serving, we want to use more, 4 or even 8 such GPUs. Let’s now just take it at face value, and we will elaborate on why this is the case in the later part of this text.

## Compute and memory bound

When looking at the specification of a GPU, you should be paying most attention to two metrics:

- Compute: measured in FLOPS - how many floating-point operations (addition and multiplication) a GPU can do in a second *
- Memory bandwidth: how many bytes can be loaded from the global memory in a second.

These two factors dictate how quickly you can process computations; they affect the speed of a single feedforward operation and determine the generation speed (measured in tokens per second, tps) and ultimately define your cost per token.

\* Please be aware that FLOPS and FLOPs mean different things. **FLOPs** (small s) is the plural of floating-point operations not considering time at all but **FLOPS** (captital S) means floating-point operations that happen within a second

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%201.png)

Fig. 3: Compute and memory bandwidth A100 cards vs H100 cards [https://www.databricks.com/blog/coreweave-nvidia-h100-part-1](https://www.databricks.com/blog/coreweave-nvidia-h100-part-1)

A computer program (such as running an LLM) can be characterized by its arithmetic intensity. Arithmetic intensity is a concept that describes the ratio of computational operations, such as addition or multiplication (measured in FLOPs), to memory accesses (measured in bytes). A higher arithmetic intensity indicates that the program performs more computations per unit of data fetched from memory, which typically leads to better utilization of the processor's computational capabilities and reduced bottlenecking on memory bandwidth. LLM inference has a very low compute intensity because it involves repeatedly accessing large model weights from memory with relatively few computations per byte fetched. As you'll see in the following, LLM inference has both a phase that's heavily compute-bound (very high arithmetic intensity) and a heavily memory-bound (very low arithmetic intensity).

For A100:

- FLOPS: 3.12 * 10^14 floating point operations can be performed in a per second
- Memory: 2.03 * 10^12 bytes can be loaded from global memory (HBM) per a second

For H100:

- FLOPS: $9.89 \times 10^14$ floating point operations can be performed in a per second
- Memory: $3.35 \times 10^12$ bytes can be loaded from global memory (HBM) per a second

As you'll see in the next parts of this text, LLM inference has both a phase that's heavily compute-bound (very high arithmetic intensity) and a heavily memory-bound (very low arithmetic intensity). The majority of the wall clock time is spent in the memory-bound phase, so the goal of efficient LLM inference is to maximize the utilization of the GPUs' compute capacity during the memory-bound phase. So increasing the arithmetic intensity for the memory-bound phase represents a fundamental optimization target that directly translates to improved inference economics.

In the context of LLMs, in order to generate a token, we need to load the entire model with all parameters from global memory (HBM) (we utilize the memory bandwidth) and calculate the intermediate activations (we use the compute). The ratio between compute and memory utilization is crucial in determining what can be optimized and how to enjoy better inference economics. In the next part we will go more in depth on the two phases of LLM inference:

- prompt processing or so called pre-fill phase
- token-by-token or so called decoding phase

The end-to-end latency of an LLM request depends critically on the efficiency of both these two phases.

## FLOPs in matrix multiplication

Before delving into the two phases, let's clarify how we count floating point operations (FLOPs) in matrix multiplication.

When multiplying matrices `A` (shape `m×n`) and `B` (shape `n×o`), we produce matrix `C = A @ B` (shape `m×o`). The computation involves:

- Taking each row from `A` and each column from `B`
- Computing their dot product to fill each element of `C`

For a single dot product between vectors of length n:

- We perform `n` multiplications
- Followed by `n-1` additions
- Resulting in `n + n-1 = 2n-1` operations total

Since we need to compute this for every element in our result matrix `C` (`m×o` elements):

- Total FLOPs = `(2n-1) × m × o ≈ 2mno`

For simplicity in this post, we'll use `2mno` as our FLOP count for matrix multiplication.

## Prompt processing/prefill phase

The first phase of generating text with LLMs is prompt processing. In this phase an LLM is presented with a list of input (prompt) tokens, and we try to predict our first new token. The duration of this phase is what the API providers present as “latency” or “time to first token" (TTFT) (See Fig. 4).

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%202.png)

Fig. 4: You can see how latency, or time to first token, is reported in OpenRouter: [https://openrouter.ai/deepseek/deepseek-r1](https://openrouter.ai/deepseek/deepseek-r1)

This phase is heavily **compute bound**, which is good; we utilize most of the compute we have available on our GPU. Let’s estimate the flops of a single forward pass to see why it is the case.

Let’s manually count the FLOPs in the model during processing `S` tokens. For reference, we include the diagram of the Llama architecture (see Fig. 2).

## Embedding Layer

**FLOPs**:

- **Lookup operation**: Embedding lookups involve retrieving vectors from the embedding matrix and are considered to have negligible FLOPs since they involve memory access rather than arithmetic computations.

## Self-Attention (Per Layer)

### RMS Norm

```python
def rms_norm(x, weight, eps=1e-6):
    # Calculate the root mean square
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    # Normalize and scale with the weight parameter
    return x / rms * weight
```

$$ \begin{aligned} \text{FLOPS}_{\text{RMSNorm}} &= S \times d \times 1 \text{ (square)} \ &+ S \times d \text{ (mean)} \ &+ S \times 1 \text{ (add epsilon)} \ &+ S \times 1 \text{ (sqrt)} \ &+ S \times d \text{ (division)} \ &+ S \times d \text{ (multiplication)} \end{aligned} $$

Simplifying the expression:

$$ \begin{aligned} \text{FLOPS}_{\text{RMSNorm}} &= S \times d \times 4 + S \times 2 \ &\approx 4 \times S \times d \end{aligned} $$


## Query Projection

#### Shapes: 
- Input: $\mathbf{X} \in \mathbb{R}^{S \times \text{hidden\_size}}$
- Weight: $\mathbf{W}_Q \in \mathbb{R}^{\text{hidden\_size} \times \text{hidden\_size}}$

#### FLOPS:

$
\begin{aligned}
\text{FLOPS}_{\text{Query}} &= 2 \times S \times \text{hidden\_size} \times \text{hidden\_size}\\
&= 2 \times S \times \text{hidden\_size}^2
\end{aligned}
$

## Keys and Values Projections

*As we explained before, the 1/8th part is Llama architecture specific due to multi-query attention; see the "Model parameters and hardware requirements" paragraph*

#### Shapes:
- Input: $\mathbf{X} \in \mathbb{R}^{S \times \text{hidden\_size}}$
- Key Weight: $\mathbf{W}_K \in \mathbb{R}^{\text{hidden\_size} \times \text{hidden\_size}/8}$
- Value Weight: $\mathbf{W}_V \in \mathbb{R}^{\text{hidden\_size} \times \text{hidden\_size}/8}$

#### FLOPS:
$$
\begin{aligned}
\text{FLOPS}_{\text{Key+Value}} &= 2 \times 2 \times S \times \text{hidden\_size} \times \text{hidden\_size}/8\\
&= \frac{1}{2} \times S \times \text{hidden\_size}^2
\end{aligned}
$$

## Rotary Positional Embedding (RoPE)

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```
*Fig. 6: RoPE implementation*

#### Shapes:
- Query: $\mathbf{q} \in \mathbb{R}^{S \times \text{head\_dim} \times \text{num\_attention\_heads}}$
- Key: $\mathbf{k} \in \mathbb{R}^{S \times \text{head\_dim} \times \text{num\_attention\_heads}}$
- Cosine: $\mathbf{cos} \in \mathbb{R}^{S \times \text{head\_dim}}$
- Sine: $\mathbf{sin} \in \mathbb{R}^{S \times \text{head\_dim}}$

For each element in $\mathbf{q}$ and $\mathbf{k}$, the following operations are performed:

- **Multiplications**: 2 per tensor
  - Multiply $\mathbf{q}$ with $\mathbf{cos}$
  - Multiply the rotated version of $\mathbf{q}$ ($\text{rotate\_half}(\mathbf{q})$) with $\mathbf{sin}$
- **Additions**: 1 per tensor
  - Add the two results to get the embedded $\mathbf{q}$

Since these operations are performed on both $\mathbf{q}$ and $\mathbf{k}$, the total per element is:

- **Total operations per element**: 3 FLOPs per tensor $\times$ 2 tensors = **6 FLOPs**

#### FLOPS:

$$
\begin{aligned}
\text{FLOPS}_{\text{RoPE}} &= S \times \text{head\_dim} \times \text{num\_attention\_heads} \times 6 \\
&= S \times \text{hidden\_size} \times 6
\end{aligned}
$$

## Q × K^T

*We assume the naive attention implementation. In practice, with algorithms like flash attention, we calculate it iteratively to save memory.*

#### Shapes:
- Query: $\mathbf{Q} \in \mathbb{R}^{S \times \text{num\_attention\_heads} \times \text{head\_dim}}$
- Key: $\mathbf{K} \in \mathbb{R}^{S \times \text{num\_attention\_heads} \times \text{head\_dim}}$
- Transposed Key: $\mathbf{K}^T$ (after appropriate reshaping and transposition)
- Result: $\mathbf{QK}^T \in \mathbb{R}^{\text{num\_attention\_heads} \times S \times S}$

#### FLOPS:

For each attention head:

$$\text{FLOPS per head} = 2 \times S \times \text{head\_dim} \times S = 2S^2 \times \text{head\_dim}$$

For all attention heads:

$$\text{FLOPS total} = 2S^2 \times \text{head\_dim} \times \text{num\_attention\_heads} = 2S^2 \times \text{hidden\_size}$$

*Note: This quadratic dependence on sequence length ($S^2$) is why attention becomes expensive for long sequences.*

## Softmax

*It is kind of hard to estimate the FLOPs for Softmax. We approximate softmax as 5 FLOPs per element:*

#### Shapes:
- Input: $\mathbf{A} \in \mathbb{R}^{S \times S \times \text{num\_attention\_heads}}$
- Output: $\mathbf{A}_{\text{softmax}} \in \mathbb{R}^{S \times S \times \text{num\_attention\_heads}}$

#### FLOPS:

$$ \text{FLOPS}_{\text{softmax}} = 5 \times S \times S \times \text{num\_attention\_heads} = 5S^2 \times \text{num\_attention\_heads} $$

*This is a simplified approximation of the actual operations in softmax (exponentiation, sum, division).*

## Attention Output $(Q @ K^T) @ V$ 

#### Shapes:
- Attention Scores: $\mathbf{A}_{\text{softmax}} \in \mathbb{R}^{S \times S \times \text{num\_attention\_heads}}$
- Value Matrix: $\mathbf{V} \in \mathbb{R}^{S \times \text{head\_size} \times \text{num\_attention\_heads}}$
- Output: $\mathbf{O} \in \mathbb{R}^{S \times \text{hidden\_size}}$

#### FLOPS:

For each head:
$$\text{FLOPS}_{\text{attn output per head}} = 2 \times S \times S \times \text{head\_size} = 2S^2 \times \text{head\_size}$$

For all heads:
$$\text{FLOPS}_{\text{attn output total}} = 2S^2 \times \text{head\_size} \times \text{num\_attention\_heads} = 2S^2 \times \text{hidden\_size}$$

## O-Projection

#### Shapes:
- Input: $\mathbf{O} \in \mathbb{R}^{S \times \text{hidden\_size}}$
- Weight Matrix: $\mathbf{W}_O \in \mathbb{R}^{\text{hidden\_size} \times \text{hidden\_size}}$
- Output: $\mathbf{O}_{\text{proj}} \in \mathbb{R}^{S \times \text{hidden\_size}}$

#### FLOPS:

$$\text{FLOPS}_{\text{O-projection}} = 2 \times S \times \text{hidden\_size} \times \text{hidden\_size} = 2S \times \text{hidden\_size}^2$$
## Total FLOPs for Self-Attention

**RMS Norm:** $4S \times \text{hidden\_size}$

**Query projection:** $2S \times \text{hidden\_size}^2$

**Keys and values projections:** $0.5S \times \text{hidden\_size}^2$

**Positional Embedding (RoPE)**: $6S \times \text{hidden\_size}$

**Q @ K^T** (across all heads): $2S^2 \times \text{hidden\_size}$

**Softmax** (across all heads): $5S^2 \times \text{num\_attention\_heads}$

**Attention Output (Q @ K^T) @ V**: $2S^2 \times \text{hidden\_size}$

**O-Projection**: $2S \times \text{hidden\_size}^2$

**Total FLOPs** = $10S \times \text{hidden\_size} + 4.5S \times \text{hidden\_size}^2 + 4S^2 \times \text{hidden\_size} + 5S^2 \times \text{num\_attention\_heads}$

## MLP (Per Layer)

```python
class SwiGLU(nn.Module):
    ...
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

### Gate W1

#### Shapes: 
- Input: $\mathbf{X} \in \mathbb{R}^{S \times \text{hidden\_size}}$
- Weight: $\mathbf{W}_1 \in \mathbb{R}^{\text{hidden\_size} \times \text{intermediate\_size}}$ 
- Where $\text{intermediate\_size} = 3.5 \times \text{hidden\_size}$ (Llama specific)

#### FLOPS:

$$
\begin{aligned}
\text{FLOPS}_{\text{Gate W1}} &= 2 \times S \times \text{hidden\_size} \times 3.5 \times \text{hidden\_size}\\
&= 7S \times \text{hidden\_size}^2
\end{aligned}
$$

### Up W2

#### Shapes:
- Input: $\mathbf{X} \in \mathbb{R}^{S \times \text{hidden\_size}}$
- Weight: $\mathbf{W}_2 \in \mathbb{R}^{\text{hidden\_size} \times \text{intermediate\_size}}$
  - Where $\text{intermediate\_size} = 3.5 \times \text{hidden\_size}$ (Llama specific)

#### FLOPS:

$$
\begin{aligned}
\text{FLOPS}_{\text{Up W2}} &= 2 \times S \times \text{hidden\_size} \times 3.5 \times \text{hidden\_size}\\
&= 7S \times \text{hidden\_size}^2
\end{aligned}
$$

### Swish/SiLU Activation

#### Shapes:
- Input: $\mathbf{X}_{\text{gate}} \in \mathbb{R}^{S \times \text{intermediate\_size}}$
  - Where $\text{intermediate\_size} = 3.5 \times \text{hidden\_size}$

#### FLOPS:
*We approximate the activation function as 5 FLOPs per element*

$$
\begin{aligned}
\text{FLOPS}_{\text{Swish}} &= 5 \times S \times \text{intermediate\_size}\\
&= 5S \times 3.5 \times \text{hidden\_size}\\
&= 17.5S \times \text{hidden\_size}
\end{aligned}
$$

### Element-wise Multiplication

#### Shapes:
- First Input: $\text{SiLU}(\mathbf{X}_{\text{gate}}) \in \mathbb{R}^{S \times \text{intermediate\_size}}$
- Second Input: $\mathbf{X}_{\text{up}} \in \mathbb{R}^{S \times \text{intermediate\_size}}$
- Where $\text{intermediate\_size} = 3.5 \times \text{hidden\_size}$

#### FLOPS:

$$
\begin{aligned}
\text{FLOPS}_{\text{Element-wise Mult}} &= S \times \text{intermediate\_size}\\
&= S \times 3.5 \times \text{hidden\_size}\\
&= 3.5S \times \text{hidden\_size}
\end{aligned}
$$

### Down W3

#### Shapes:
- Input: $\mathbf{X}_{\text{combined}} \in \mathbb{R}^{S \times \text{intermediate\_size}}$
  - Where $\text{intermediate\_size} = 3.5 \times \text{hidden\_size}$
- Weight: $\mathbf{W}_3 \in \mathbb{R}^{\text{intermediate\_size} \times \text{hidden\_size}}$

#### FLOPS:

$$
\begin{aligned}
\text{FLOPS}_{\text{Down W3}} &= 2 \times S \times \text{intermediate\_size} \times \text{hidden\_size}\\
&= 2 \times S \times 3.5 \times \text{hidden\_size} \times \text{hidden\_size}\\
&= 7S \times \text{hidden\_size}^2
\end{aligned}
$$

### Total FLOPS MLP

$$
\begin{aligned}
\text{Total FLOPS}_{\text{MLP}} &= \text{FLOPS}_{\text{Gate W1}} + \text{FLOPS}_{\text{Up W2}} + \text{FLOPS}_{\text{Swish}} + \text{FLOPS}_{\text{Element-wise Mult}} + \text{FLOPS}_{\text{Down W3}}\\
&= 7S \times \text{hidden\_size}^2 + 7S \times \text{hidden\_size}^2 + 17.5S \times \text{hidden\_size} + 3.5S \times \text{hidden\_size} + 7S \times \text{hidden\_size}^2\\
&= 21S \times \text{hidden\_size}^2 + 21S \times \text{hidden\_size}\\
&\approx 21S \times \text{hidden\_size}^2
\end{aligned}
$$

## LM Head

During the inference, we only care about the next token prediction for the last token in our sequence. All of the other "next tokens" we already know, as they are part of the input prompt.

#### Shapes:
- Input: $\mathbf{X} \in \mathbb{R}^{1 \times \text{hidden\_size}}$
- Weight: $\mathbf{W}_{\text{LM}} \in \mathbb{R}^{\text{hidden\_size} \times \text{vocab\_size}}$
- Output: $\mathbf{Logits} \in \mathbb{R}^{1 \times \text{vocab\_size}}$

#### FLOPS:

$$
\begin{aligned}
\text{FLOPS}_{\text{LM Head}} &= 2 \times 1 \times \text{hidden\_size} \times \text{vocab\_size}\\
&= 2 \times \text{hidden\_size} \times \text{vocab\_size}
\end{aligned}
$$

## Total FLOPs in a Llama Model

Total FLOPs in the Llama model is a product of the number of FLOPs per transformer block times the number of blocks, plus the FLOPs for the LM head.

### Transformer Block

#### Components
- **Attention:** $10S \times \text{hidden\_size} + 4.5S \times \text{hidden\_size}^2 + 4S^2 \times \text{hidden\_size} + 5S^2 \times \text{num\_attention\_heads}$
- **MLP:** $21S \times \text{hidden\_size}^2$

#### Total Per Block
$10S \times \text{hidden\_size} + 25.5S \times \text{hidden\_size}^2 + 4S^2 \times \text{hidden\_size} + 5S^2 \times \text{num\_attention\_heads}$

### LM Head
$2 \times \text{hidden\_size} \times \text{vocab\_size}$

### Total FLOPs Calculation

#### Formula
$\text{Total FLOPs} = \text{num\_hidden\_layers} \times (10S \times \text{hidden\_size} + 25.5S \times \text{hidden\_size}^2 + 4S^2 \times \text{hidden\_size} + 5S^2 \times \text{num\_attention\_heads}) + 2 \times \text{hidden\_size} \times \text{vocab\_size}$

#### Example: Llama 3.3 70B
For Llama 3.3 70B with:
- $\text{hidden\_size} = 8192$
- $\text{vocab\_size} = 128256$
- $\text{num\_attention\_heads} = 64$
- $\text{num\_hidden\_layers} = 80$
- $S = 2048$ tokens

$$
\begin{align}
\text{Total FLOPs} &= 80 \times (10 \times 2048 \times 8192 + 25.5 \times 2048 \times 8192^2 + 4 \times 2048^2 \times 8192 + 5 \times 2048^2 \times 64)\\
&+ 2 \times 8192 \times 128256\\
&= 80 \times 3.6436 \times 10^{12} + 2.1013 \times 10^9\\
&= 2.9149 \times 10^{14} + 2.1013 \times 10^9\\
&\approx 2.9149 \times 10^{14}\\
&= 291.49 \times 10^{12}\\
&= 291.49 \text{ TFLOPs}
\end{align}
$$

`291 TFLOPs` is roughly the order of magnitude of FLOPs available in a modern GPU. For example, with H100 cards, it would theoretically take roughly `291/989 = 0.29s` to process a prompt of 2048 tokens.
As a reminder, to load the model from global memory, we need to load `141 GB` worth of parameters. The memory bandwidth of a modern GPU is around `3350 GB/s`, meaning that in theory it will take `141/3350 = 0.04s` to load the entire model from global memory - roughly 7x faster than the time needed for all of the computations.
This demonstrates that in the pre-fill phase we are much more bound by the available compute than by the memory bandwidth. This is a desirable situation, as we want to utilize all of the existing compute resources.

# the decode phase

This first forward pass for doing the prefill is computationally very expensive. We can eliminate doing large parts of that computation over and over again by introducing a special cache. This cache is called KV cache because it stores the key and value matrices for each token position.

In the attention mechanism, we calculate attention relationships between all tokens in the sequence. The key insight is that at step `S+1`, we've already calculated the attention between all of the first `S` tokens during the pre-fill phase. We can store these intermediate values in memory (the "cache") and only calculate new attention values involving the most recently generated token.

This optimization works elegantly with matrix operations:

### During Pre-fill (S tokens):
$$
\begin{align}
\mathbf{Q} &\in \mathbb{R}^{S \times d} \\
\mathbf{K}^T &\in \mathbb{R}^{d \times S} \\
\mathbf{V} &\in \mathbb{R}^{S \times d} \\
\end{align}
$$
Where $d$ is the hidden dimension.

### The attention scores and outputs are computed as:
$
\begin{align}
\text{Score Matrix} &= \mathbf{Q} \cdot \mathbf{K}^T \in \mathbb{R}^{S \times S} \\
\text{Attention Output} &= \text{softmax}(\text{Score Matrix}) \cdot \mathbf{V} \in \mathbb{R}^{S \times d}
\end{align}
$

### During Token-by-Token Generation (token S+1):
For generating the next token, we only need to compute:
$$
\begin{align}
\mathbf{Q}_{new} &\in \mathbb{R}^{1 \times d} \text{ [Only for the new token]} \\
\mathbf{K}^T_{cache} &\in \mathbb{R}^{d \times S} \text{ [From cache]} \\
\mathbf{K}^T_{new} &\in \mathbb{R}^{d \times 1} \text{ [For the new token]} \\
\mathbf{K}^T_{full} &= [\mathbf{K}^T_{cache} \; | \; \mathbf{K}^T_{new}] \in \mathbb{R}^{d \times (S+1)} \\
\mathbf{V}_{cache} &\in \mathbb{R}^{S \times d} \text{ [From cache]} \\
\mathbf{V}_{new} &\in \mathbb{R}^{1 \times d} \text{ [For the new token]} \\
\mathbf{V}_{full} &= \begin{bmatrix} \mathbf{V}_{cache} \\ \mathbf{V}_{new} \end{bmatrix} \in \mathbb{R}^{(S+1) \times d}
\end{align}
$$

The new attention calculation becomes:
$$
\begin{align}
\text{Score Vector} &= \mathbf{Q}_{new} \cdot \mathbf{K}^T_{full} \in \mathbb{R}^{1 \times (S+1)} \\
\text{Attention Output} &= \text{softmax}(\text{Score Vector}) \cdot \mathbf{V}_{full} \in \mathbb{R}^{1 \times d}
\end{align}
$$

The key efficiency gain comes from:
1. Reusing $\mathbf{K}^T_{cache}$ and $\mathbf{V}_{cache}$ from previous calculations
2. Only computing new key-value projections for the latest token
3. Reducing the attention calculation from $O(S^2)$ to $O(S)$ for each new token


![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%203.png)

Fig. 8: We need only to calculate the query, key, and a value for the new token.

KV caching reduces the total FLOPs by a factor of approximately S for all parts of the forward pass:

- In self-attention, we only compute attention for the new token against all previous tokens
- In the MLP and LM head components, we only process the new token
- The LM head remains the same, but it is such a small fraction of the overall computations that we will skip it in our calculations.

For example, with a 2048-token context:

- During pre-fill: ~291 TFLOPs total
- For generating token 2049: ~291/2048 ≈ 0.14 TFLOPs

On an H100 GPU (989 TFLOP/s), this would take only:

$$
\frac{0.14 \text{ TFLOP}}{989  \text{ TFLOP/s}} \approx 0.00014  \text{ seconds}
$$

This is approximately `2048` times faster than the pre-fill phase in terms of pure computation, but remember that we still need to load the entire model parameters (`141 GB`) from the global memory, and that now we need to also load the KV cache.

KV cache memory footprint can be easily calculated as

$$
kv \ cache = 2 \times \text{bytes\_per\_param} \times \text{num\_hidden\_layers} \times \text{head\_size} \times \text{num\_key\_value\_heads} \times \text{sequence\_length}
$$

For Llama 3.3 70B with 2048 tokens using BF16 precision, this amounts to:

$$
kv \ cache = 2 \times 2 \times 80 \times 128 \times 8 \times 2048 = 640\text{ MB}
$$

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%204.png)

Fig. 9: How the memory footprint of KV cache increases with sequence length for Llama 3.3 70B. Note that the x-axis uses a logarithmic scale (powers of 2), which visually compresses the exponential growth. In reality, the memory usage is growing superlinearly with sequence length, not linearly as might be incorrectly inferred from the graph's appearance.

While 640 MB may sound not that significant, this number scales linearly with the batch size (we elaborate on batches later) and with the sequence length (see Fig. 9). This is also the main reason why, at longer sequence lengths, the token-by-token generation\* is slower than at shorter sequence lengths-on top of the model weights, the KV cache also needs to be loaded from global memory, increasing processing time for each token produced.

Model weights, plus KV cache, are roughly `141 + 0.6 ≈ 142GB` so it takes `142/3350 = 0.04s` to load them from the global memory. We calculated above that it only takes `0.00014s` to do all computations (assuming 100% compute utilization) - so it takes two orders of magnitude more time to load the model weights than to do the actual computations. This is what we mean by the token-by-token phase of using LLMs being memory bound. We are primarily limited by the time memory transfer takes, not by the speed of compute.

**This is one of the insights we hope you take out of reading this article - the token-by-token phase is memory bound; the amount of available compute throughput is of secondary importance, as we massively underutilize the compute resources anyway while waiting for weights to be loaded.**

\* we use the terms token-by-token phase, decode phase and generation phase interchangeably

# Scaling with the input length

One of the main challenges in LLM serving is understanding how input prompt length affects end-to-end performance. Sequence length impacts both the prefill stage and the token-by-token decoding phase, though in fundamentally different ways.

The prompt processing phase exhibits $O(N^2)$ computational complexity; as the sequence length grows, the processing time will grow quadratically*. We derived the FLOPs before as

- `2 S² hidden_size` for `score matrix Q @ Kᵗ`
- `2 S² hidden_size` for `(Q @ Kᵗ) @ V`.

This is especially relevant for longer sequence lengths, where the time to the first token will quadratically increase with the length of the input sequence. You can experience this, e.g., when using the long-context Gemini that, as of May 2025, can take up to 2M tokens, but you can wait up to 2 minutes for the first token of the answer to be generated. The intuition you should have developed here is that as the input length keeps increasing, a larger and larger percentage of the total time of processing a request will be spent in prompt processing - the compute-bound part (see Fig. 10).

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%205.png)

Fig. 10: As the prompt length increases, the compute required, and hence the time, increases quadratically, occupying an ever-increasing portion of the total processing time of a request. Please note: this is just a visualization to give you some intuitions; it is not actually based on real-world observations.

During the token-by-token phase, the relationship between the generation speed and the sequence length is less straightforward.

FLOP-wise compute is scaling linearly; the new token needs to attend to all of the tokens from the past, or `2 S hidden_size` for `Q @ Kᵗ` and `2 S hidden_size` for `(Q @ Kᵗ) @ V`, but as we already know, FLOPs are not that relevant for the token-by-token case because we are primarily memory bound. What is much more relevant is the size of the data that we need to load from the global memory.

As a reminder, for each forward pass, we need to load the entire model weights from the global memory; on top of this, we also need to load the KV cache from global memory. As we showed in the section above, as we increase the KV cached sequence length, it will occupy linearly more and more memory or by factor `S` in `2 x bytes_per_param x num_hidden_layers x head_size x num_key_value_heads x S.`

Initially the size of the KV cache will be negligible compared to the model size that we need to load, but as we increase the size of the processed prompt, it will occupy an increasingly big portion of the memory (see Fig. 11). Note that if we decide to process larger batches (we discuss this in detail in a later section), the size of the KV cache will grow linearly w.r.t. batch size, as we need to cache the keys and values independently for all examples from the batch. Then at some point the size of the KV cache will overtake the size of the model itself.

The intuition to develop here is that for small batches and short sequences, the sequence length has minimal impact on throughput because loading model weights dominates the memory bandwidth utilization. However, as either batch size or sequence length increases, loading the KV cache takes an amount of time to load for each token, eventually surpassing the time consumed by loading of the model weights themselves.

This transition creates two distinct performance regimes: in the so-called model-dominated regime (short sequences/small batches), throughput remains relatively stable despite increasing sequence length. Once we enter the KV-cache-dominated regime, generation speed begins to degrade in proportion to sequence length. This is largely irrelevant for short sequence lengths but is a significant issue at very long sequence lengths (in the order of tens of thousands of tokens). The additional time loading the KV cache takes scales linearly w.r.t. sequence length.

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%206.png)

Fig. 11: KV cache scaling with the increasing sequence length. At 128k tokens for Llama 3.3 70B, the KV cache for a single sentence will amount to 40GB.

*In the naive implementations of the attention mechanism memory would also scale quadratically when we realize the `SxS` score matrix, but [flash attention](https://github.com/Dao-AILab/flash-attention) replaces these naive implementations. Flash attention is calculating `(Q @ Kᵗ) @ V` in an iterative way, where memory required is kept at `O(N)`.

# Multi GPU inference

As you might have noted, the 141 GB we need to store the Llama 3.3 70B parameters is more than what we have available on a single Nvidia H100 GPU. H100 cards come with 80GB of HBM memory. We would need a minimum of two to store the model in memory; however, in practice, we would probably like to use more for the KV cache. If we have more memory available, we will be able to allocate a higher proportion of it to the KV cache and a smaller proportion to the model weights, allowing us to run larger batches. We would also linearly increase the available memory bandwidth, at the cost of an increased overhead in cross-GPU communication, though.

Using just two GPUs for Llama 3.3 70B would result in only having a tiny amount of memory available for KV cache because the model weights already take up 88% (`141/160=88%`) of that memory, leaving only 19GB of memory (`160-141=19GB`) available for KV cache (in practice even less than that because we can't use 100% of GPU memory but are limited to around 95%). We wouldn’t be able to run large batches nor long sequence lengths; this would be very inefficient. We touch on this in a later section, but being able to run larger batches is the key to enjoying the good inference economics.

GPU servers almost always come in deployments of 4 or 8 GPUs per node, so using 3 GPUs would be wasteful because that would lead to one GPU in the server being entirely unused in a lot of circumstances. Hence, we jump from 2 to 4 GPUs for a single model instance right away.

Let’s assume then we will run the Llama 3.3 70B on 4 H100 cards. There are two main ways to run large-scale AI models on multiple GPUs:

- Pipeline parallel
- Tensor parallel

Both offer different tradeoffs between throughput and latency. Let’s explore them briefly.

## Pipeline parallelism

In the pipeline parallel (PP) setting, we split the model along the layers axis, meaning that each GPU will have a fraction of all layers in the model. E.g., in the case of Llama 3.3 70B with 80 hidden layers served on 4 GPUs, GPU:0 will host the first 20 layers, GPU:1 the next 20, and so on.

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%207.png)

Fig. 12: Pipeline parallelism with continuous batching visualization [https://docs.graphcore.ai/projects/tf-model-parallelism/en/latest/pipelining.html](https://docs.graphcore.ai/projects/tf-model-parallelism/en/latest/pipelining.html)

The upside of such an approach is the very limited communications between the devices. We only need to broadcast activations from one device to the next one 3 times for a single forward pass.

We can also do pipelining - GPU:0 processes batch 0 and is passing it to GPU:1, then batch 1 is coming in, and GPU:0 can process batch 1 while GPU:1 is processing batch 0, etc. (see Fig. 12). This setting minimizes the stall time of each device, ensuring the maximum throughput; however, it comes at a price - at a single point in time, we only have access to 1/4 of the available compute and memory bandwidth per batch. So generation time is significantly slower than if we were to use all 4 GPUs at the same time.

In practice, orchestrating efficient overlapping batching can be quite challenging; hence, for the remaining part of this text, we will focus on analyzing the far more common tensor parallel setting.

## Tensor parallelism

The mainstream parallelism strategy, used by a vast majority of LLM inference providers, is tensor parallelism (TP). In TP, individual neural network layers are split across multiple GPUs, harnessing the combined compute power and memory bandwidth of all devices. This approach significantly shortens per-layer inference time but introduces important trade-offs that must be carefully considered:

1. **Communication Overhead**: In regular intervals, e.g., twice per transformer block, the execution must synchronize across GPUs, introducing a significant delay (in the order of milliseconds) per synchronization event. This overhead varies significantly based on interconnect technology (NVLink, PCIe, etc.) and network topology.

2. **Sequential Batch Processing**: Unlike pipeline parallelism, TP requires all GPUs to process the same batch simultaneously. A new batch cannot begin until the current one completes, reducing throughput efficiency under dynamic workloads.

The most efficient parallelization strategy is to have a so-called column-wise split linear layer (we split by the column dimension) followed by the row-wise layer (split across the row dimension). Such a layout minimizes the synchronization to only one sync every two MLP layers.

**Mathematical Intuition**:

For a weight matrix `W₁` split column-wise across 2 GPUs:

$$
\mathbf{W}_1 = [\mathbf{W}_{1,1} \quad \mathbf{W}_{1,2}], \quad \mathbf{W}_{1,i} \in \mathbb{R}^{d_{\text{in}} \times \frac{d_{\text{hid}}}{2}}
$$

Each GPU computes its partial output independently (no communication):

$$
\mathbf{h}_1 = \mathbf{x} \mathbf{W}_{1,1}, \quad \mathbf{h}_2 = \mathbf{x} \mathbf{W}_{1,2}
$$

The hidden layer activation becomes:

$$
\mathbf{h} = [\mathbf{h}_1 \quad \mathbf{h}_2]
$$

No communication is needed here because each GPU has all the necessary
data. For the subsequent row-wise split in `W₂`:

$$
\mathbf{W}_2 = \begin{bmatrix}\mathbf{W}_{2,1} \\\mathbf{W}_{2,2}\end{bmatrix}, \quad \mathbf{W}_{2,i} \in \mathbb{R}^{\frac{d_{\text{hid}}}{2} \times d_{\text{out}}}
$$

Each GPU computes:

$$
\mathbf{y}_1 = \mathbf{h}_1 \mathbf{W}_{2,1}, \quad \mathbf{y}_2 = \mathbf{h}_2 \mathbf{W}_{2,2}
$$

The final output requires an all-reduce sum; in other words, we need to synchronize between the devices:

$$
\mathbf{y} = \mathbf{y}_1 + \mathbf{y}_2
$$

This layout we can apply to the transformer block, reducing it to only two synchronizations per transformer block (see Fig. 13 and Fig. 14).

- Self-Attention: Heads are processed independently, with synchronization only during the output projection (`o_proj`).
- MLP: The up-projection (`w1`, `w3`) is split column-wise and the down-projection (`w2`) is split row-wise, and the sync is only executed after the down-projection.

```python
layer_tp_plan = {
    # by default ColwiseParallel input layouts is replicated
    # and RowwiseParallel output layouts is replicated
    "attention.wq": ColwiseParallel(),
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}
```

Fig. 13: The Colwise → Rowwise layout for a transformer layer from that is used as an example in torch documentation https://pytorch.org/tutorials/intermediate/TP_tutorial.html

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%208.png)

Fig. 14: The Colwise → Rowwise split we see in a transformer block.

Correctly estimating the extra overhead from the coms is quite complicated. In theory, we would need to take into account the two following factors:

1. **Message passing latency**: Typically 3-5 μs depending on hardware
2. **Data transfer time**: Based on interconnect bandwidth

In an ideal scenario with modern NVLink connections, we could estimate:

$$
\text{Overhead} \approx \frac{\text{params} \times \text{comms/layer} \times \text{hidden size} \times \text{number of layers} \times (N-1)/N}{\text{bandwidth}} = \\ \frac{2 \times 2 \times 8192 \times 80 \times 3/4}{450 \times 10^9} \approx 4\ \mu\text{s}
$$

The total overhead of 8 or 9 µs would be awesome. However, in practice, it gets much more complicated. During the sync barrier, the compute graph is stalled. We have a constant overhead of a few ms when the GPUs are idling while waiting for the sync to be finished. This extra "tax" is of the main reasons preventing us from utilizing the full memory bandwidth we have available across all the GPUs. Accurately modeling the overhead is quite challenging. As we'll demonstrate in the next sections, the gap between theoretical and actual performance can be quite substantial, requiring empirical measurement for accurate system modeling.

# Batching - the key to good economics

As we showed before, during the token-by-token phase, we are primarily memory bound-meaning the main limitation in terms of tokens/s we can get is how fast our GPU is able to load the model weights from the global memory. To generate every single token, we always need to load the entire model.

There is an obvious optimization we could introduce to improve the economics of our operation: run larger batches. Batches are a natural improvement because we can load the model weights once, but we can use the loaded weights to do inference with multiple items from our batch at the same time (and so serve different customers simultaneously).

Increasing the batch size increases the compute usage linearly - we have `k` times more multiplications to do, but it only marginally changes the memory bandwidth used (only for loading the KV cache), so it's an easy way to increase the compute intensity for our otherwise heavily memory-bound algorithm (to make it less memory-bound). Since the extra memory for the KV cache is significantly smaller than the memory needed for the model, it only adds a small overhead, but it linearly increases the number of produced tokens. We produce twice as many tokens for a batch of 2 and 16x as many tokens with a batch of sixteen.

**This is the core message of this post and the main intuition we hope you take away from reading this:** As we grow the batch size, we can effectively share the time to load the model from high bandwidth memory (HBM), aka our cost of loading the model is split across an increasing number of clients, enjoying the **economies of scale** and decreasing the per-request cost. **Having sufficient demand and continuously serving big batches is the key to running a profitable LLM inference business; if you can't support large batches, your cost per token will balloon, making your operation unprofitable*.** (greetings to the Groq team)

\*One thing to note is that there is a limit to this model. As we approach really long sequences or really big batches, as we will see in our experiments, the memory footprint of the KV cache starts to slowly overtake the memory footprint of the model itself (see Fig. 15). When this happens, the cost of loading the model will become increasingly irrelevant to the total time of loading data from the global memory.

"Luckily" for us, this situation also has its limit-the memory limit of a GPU node; in the case of H100 cards, it will be 8 × H100 = 8 × 80GB = 640GB. Note how for a batch of 8 at the full context length of Llama, we are already nearly there.

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%209.png)

Fig. 15: KV cache scaling - comparison between different batch sizes. Note how the KV cache scales linearly with the batch size.

# Throughput: theory vs practice

After all of the theoretical introductions, let’s try to combine all that we learned so far to estimate the LLM throughput. We will:

- Develop a theoretical performance model based on the GPU spec sheet
- Compare it with a real-world throughput of Llama 3.3 70B on 4 H100 cards
- Explain the discrepancies between theoretical and actual performance

The time to produce a response to a prompt is a combination of the pre-fill time and the decode time times the number of decode tokens. The more output tokens we produce, the smaller share of time will be spent in the prefill phase. The prefill time is primarily compute-bound, and the time for token-by-token is primarily memory-bound.

Since prefill is so heavily compute-bound, we can estimate the time by dividing the number of floating-point operations during prefill by the total effective FLOPS across all of the GPUs, plus the extra latency from cross-GPU communication time.

The decode time is mainly memory bound, but as we increase the batch size, the compute component will become increasingly important. We will calculate both and take the more significant factor. We also spend some small amount of time in coms.

The simple modeling script is based on what we have discussed above. We take in the model size and its architecture and estimate the throughput we should get given the hardware characteristics.

```python
def decode_time(self, sequence_length, batch_size=1, generated_tokens=1):
        """Estimate the decode time for generating tokens after the prefill"""
        total_decode_time = 0
      
        # Current sequence length starts at prompt length
        current_sequence_length = sequence_length
      
        # For each token to generate
        for _ in range(generated_tokens):
            # Get the time breakdown for this token
            breakdown = self.decode_time_breakdown(current_sequence_length, batch_size)
          
            # Add to total time
            total_decode_time += breakdown["total_time"]
          
            # Increment sequence length for next token
            current_sequence_length += 1
      
        return total_decode_time
  
    def total_inference_time(self, input_tokens, output_tokens, batch_size=1):
        """Calculate the total time for processing a batch of sequences"""
        # Prefill phase time
        prefill = self.prefill_time(input_tokens, batch_size)
      
        # Decode phase time
        decode = self.decode_time(input_tokens, batch_size, output_tokens)
      
        return prefill + decode
  
    def calculate_total_throughput(self, input_tokens, output_tokens, batch_size=1):
        """Calculate throughput in tokens per second for the entire batch"""
        # Calculate total time
        time_seconds = self.total_inference_time(input_tokens, output_tokens, batch_size)
      
        # Calculate throughput (total tokens generated per second)
        total_tokens = batch_size * output_tokens
        throughput = total_tokens / time_seconds if time_seconds > 0 else 0
      
        return throughput
```

The entire script can be found [here](https://github.com/tugot17/tokenomics-from-first-principles/blob/blogpost-code/throughput_simulations/tokenomics_model.py).

For a Llama 3.3 70B, with 2035 tokens in and 300 tokens out, we will get these estimates:

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%2010.png)

Fig. 16: The throughput estimated using first principles. We can observe what we discussed before: as the batch size increases, the extra memory that needs to be loaded for the KV cache and the speed per request decrease. 

Let’s first look at the estimated model performance under the different batch sizes. As we derived before, batch size is the single most relevant statistic regarding the tokenomics. Look how the throughput scales nearly linearly with the batch size (see Fig. 16). This is the single most important message you should get out of this text. The key to the good tokenomics is running the models at a large batch size, which is caused by the LLM inference being memory bound, meaning we want to share the cost of loading the model into the memory across as many requests as possible. As the KV cache size is approaching the model size, the total throughput gains are diminishing.

While the total throughput is increasing with the batch size, the per-request speed is decreasing, and the slowdown is accelerating as we increase the batch size due to the increased memory footprint of the KV cache. At some point, with massive batches, the per-request experience will be massively degraded. This means that even if we can support larger batches, e.g., due to the massive demand as DeepSeek experienced in February 2025, we might not want to do so because of the poor token generation speed each user will experience. In general, the variance in speed you experience on OpenRouter (see Fig. 17) can be largely attributed to the current demand, aka the batch size.
![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%2011.png)

Fig. 17: Throughput variance throughout the day from OpenRouter. The variance can be somehow explained by the varying demand for the service throughout the day, resulting in varying batch sizes.

It is also part of the reason why the [batch API](https://platform.openai.com/docs/guides/batch) is so much cheaper. In cases where speed is not of utmost importance, an LLM provider can just run massive batches, enjoying economy of scale, with individual requests being handled pretty slowly but processed at a higher profit. There are more nuances to this, e.g., the parallelism strategy (pipeline parallelism offers less cross-device communications overhead), that we consider beyond the scope of this text. We just wanted to give you a real-world example of the real impact of batch size on the price of a generated token.

Now let’s compare the results we get from our model to the actual LLM inference performance. For this, we run a vLLM inference server (a popular LLM serving framework) with Llama 3.3 70B on 4 H100 cards connected with NVLink. The result is quite underwhelming.

```python
{
  "metadata": {
    "timestamp": "2025-03-19T10:39:02.527962",
    "model": "llama-3.3-70bb",
    "dataset_key": "aime",
    "api_base": "http://localhost:8000/v1",
    "batch_sizes": [
      1,
      2,
      4,
      8,
      16,
      32,
      64,
      128,
      256,
      512
    ],
    "num_runs": 2,
    "max_tokens": 300,
    "temperature": 0.5,
    "seed": 42,
    "description": ""
  },
  "results": {
    "1": {
      "avg_input_tokens": 2035.0,
      "avg_output_tokens": 300.0,
      "elapsed_time": 5.565227147541009,
      "tokens_per_second_in_batch": 53.906158417337465,
      "avg_tokens_per_second": 53.90945989049379
    },
    "2": {
      "avg_input_tokens": 2035.0,
      "avg_output_tokens": 300.0,
      "elapsed_time": 5.83159231254831,
      "tokens_per_second_in_batch": 102.88852564309295,
      "avg_tokens_per_second": 51.53340431177088
    },
    "4": {
      "avg_input_tokens": 2035.0,
      "avg_output_tokens": 300.0,
      "elapsed_time": 6.580546063836664,
      "tokens_per_second_in_batch": 182.50572803685048,
      "avg_tokens_per_second": 45.84464409068055
    },
    "8": {
      "avg_input_tokens": 2035.0,
      "avg_output_tokens": 300.0,
      "elapsed_time": 7.507091326988302,
      "tokens_per_second_in_batch": 319.7977776715139,
      "avg_tokens_per_second": 40.3821037511306
    },
    "16": {
      "avg_input_tokens": 2035.0,
      "avg_output_tokens": 300.0,
      "elapsed_time": 8.966848284006119,
      "tokens_per_second_in_batch": 535.3051675015347,
      "avg_tokens_per_second": 34.09418974017362
    },
    "32": {
      "avg_input_tokens": 2035.0,
      "avg_output_tokens": 300.0,
      "elapsed_time": 12.410703025874682,
      "tokens_per_second_in_batch": 773.5258824927948,
      "avg_tokens_per_second": 24.93387577170966
    },
    "64": {
      "avg_input_tokens": 2035.0,
      "avg_output_tokens": 300.0,
      "elapsed_time": 19.927982787950896,
      "tokens_per_second_in_batch": 964.1459849127925,
      "avg_tokens_per_second": 15.679987913632873
    },
    "128": {
      "avg_input_tokens": 2035.0,
      "avg_output_tokens": 296.5546875,
      "elapsed_time": 36.80827986204531,
      "tokens_per_second_in_batch": 1036.7109688230596,
      "avg_tokens_per_second": 8.525344841719471
    },
    "256": {
      "avg_input_tokens": 2035.0,
      "avg_output_tokens": 299.48046875,
      "elapsed_time": 66.85448617395014,
      "tokens_per_second_in_batch": 1146.853722836619,
      "avg_tokens_per_second": 5.213941231526505
    },
    "512": {
      "avg_input_tokens": 2035.0,
      "avg_output_tokens": 299.056640625,
      "elapsed_time": 129.60231457301416,
      "tokens_per_second_in_batch": 1182.0893502232486,
      "avg_tokens_per_second": 3.262364097179704
    }
  }
}
```

Fig. 18: The results of a benchmark we run on Llama 3.3 70B TP=4 for different batch sizes. Note that we keep the input tokens and output tokens fixed. To make the model not stop too early, we run it with random weights and just add `max_tokens` param to them.

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%2012.png)

Fig. 19: Model vs. the reality. While the shape is somehow similar, the exact values are not due to the real-world constraints.

While the general sigmoid-like shape is quite similar, the actual values are very much different. We go from around ~60% of theoretical estimated performance in small batches to around 40%. Where does this difference come from?

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%2013.png)

Fig. 20: As we increase the batch size the discrepancies between estimated throughput and the measured one increase.

Well, this is the problem at the core of GPU optimization - in practice, it is actually extremely hard to properly estimate the wall-time performance of the GPU application.  There are many compounding factors here, making the picture more blurry. To mention a few things that are likely contributing to the discrepancy in the observed results vs. reality:

- In our model we assumed 100% memory and compute utilization. Theoretical peak performance metrics like TFLOPS and memory bandwidth are never fully achieved in practice. GPUs typically achieve only 50-70% of their theoretical compute capacity due to
  - Kernel launch overhead and synchronization costs
  - Memory access patterns that don't perfectly align with cache architectures
  - Warp divergence and other GPU-specific inefficiencies
  - Instruction mix that isn't optimal for utilizing all GPU units
  - … and a lot of other factors
- We assumed very little overhead from the cross-device communication; as we explored previously, this is not necessarily the case in practice. In practice, we have tons of other factors that contribute to the potential extra latency, such as
  - Having to sync the CUDA graph across devices for cross-GPU communication
  - Synchronization barriers that force all GPUs to wait for the slowest one
  - internal buffers and management overhead
  - potential suboptimal (non-coalesced) memory access patterns, especially at larger sizes, with KV caches being stored in random pages of the VRAM memory
  - and other factors we don’t mention here
- For simplicity of calculating FLOPs, we assumed a very naive implementation of the attention mechanism. In practice, everyone is using something like Flash attention. Properly estimating the time and FLOPs involved is quite challenging and complicated and outside the scope of this text.
- Overhead coming from the practical implementation of the LLM serving engine, such as paged attention
- Extra overhead from using Python, PyTorch, and the vLLM framework itself

We tried to account for some of the above and include them in the extended simulation model. The details can be found in the [code](https://github.com/tugot17/tokenomics-from-first-principles/blob/blogpost-code/throughput_simulations/advanced_tokenomics_model.py), but TL;DR; we assumed decreased compute and memory utilization values, the extra latency from coms, and a few other factors, like extra memory overhead increasing exponentially with the batch size.

![throughput_comparison_llama-3-3-70b_4x_2035in_300out.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/throughput_comparison_llama-3-3-70b_4x_2035in_300out.png)

Fig. 21: Updated model vs. reality. Now the two lines are much closer. We achieve this by adding the estimated overhead to the model. We run it for 2k tokens input and 300 tokens output.

While far from perfect, it kind of works. It generalizes pretty well to other sizes; e.g., it works for the Llama 3.1 8B run on TP1.

![throughput_comparison_llama-3-1-8b_1x_2035in_300out.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/throughput_comparison_llama-3-1-8b_1x_2035in_300out.png)

Fig. 22: The estimation we run seems to work quite well for other model sizes. E.g., here we run for Llama 3.1 8B, 2k tokens input 300 tokens output. The two lines are pretty close.

However, when tried with different batches of different lengths, the differences are more significant, suggesting that the model is far, far from perfect. E.g., we tried estimating the throughput for long context model performance, with 16k tokens in and 1000 tokens out. Due to excessive memory consumption, we kept this setting at a batch size of 8. In such a case, the model failed to correctly predict the final throughput, suggesting that our model is far from perfect.

![throughput_comparison_llama-3-3-70b_4x_16035in_1000out.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/throughput_comparison_llama-3-3-70b_4x_16035in_1000out.png)

![throughput_comparison_llama-3-1-8b_1x_16035in_1000out.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/throughput_comparison_llama-3-1-8b_1x_16035in_1000out.png)

Fig. 23: Unfortunately, for different input/output configurations (16k tokens input, 2k tokens output in this case), the model estimates the throughput less accurately, more so for Llama 3.3 70B and less so for Llama 3.1 8B.

How challenging it is to correctly predict a model's throughput is another thing that we hope you can take out of reading this text. Accurately estimating it would require a series of very detailed profiling steps, such as profiling the memory access patterns across multiple model sizes, batch sizes, and prompt length combinations; delving deeply into the exact implementation details of (paged) attention implementation; implementation details of the batch scheduling; estimating how different shapes affect the compute and memory utilization; and dozens of different experiments. We deem an accurate estimation across different settings challenging to a degree that it might not be feasible to do so in practice. If you want accurate results, you are probably better off just measuring the real-world results.

# From tokens to dollars - estimating tokenomics

As you might know, the pricing of various LLM providers is token-based: you will pay a specific price per million input and output tokens. Some providers use the same price for input and output tokens; others have two distinct prices for input and output tokens.

As you might know, the pricing of various LLM providers is token-based: you will pay a specific price per million input and output tokens. Some providers use the same price for input and output tokens; others have two distinct prices for input and output tokens.

To summarize what we've learned so far:

- The time of prefill is quadratically dependent on the sequence length. As the input size grows, it will occupy an increasingly big percentage of the request processing time.
- The time spent generating a single token grows linearly as the context length grows. KV cache will gradually become a more and more substantial percentage of the total data loaded from the global memory (alongside the model parameters). 
- Since the time of generating a single token is can be well approximated by the cost of loading the model weights once from global memory, this time grows linearly with the number of generated tokens.

What should be apparent from the description above is that estimating a fair market price for the input tokens is a non-trivial task. Increased input quadratically increases the cost of prefill, but for standard use cases, prefill is only a minority of the time the GPU spends on processing the request. Then, depending on the batch size and context length and the proportion of these two to the model size, it will affect the throughput.

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%2014.png)

Fig. 24: Market prices for input and output tokens of Llama 3.3 70B [https://openrouter.ai/meta-llama/llama-3.3-70b-instruct](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct)

While estimating a universal proportion of cost between the input and output tokens that generalizes well across all possible shapes of input and output is quite complicated, estimating the cost for a specific input and output config is much simpler. We can just:

- Measure the execution wall time.
- Assume a fixed ratio `γ` between input and output token costs (e.g., `γ=0.3`). There is no deep reasoning behind choosing this particular value of `γ` we just need to choose some value.
- Calculate the per-token cost `β` by solving the following equation:

$$
\gamma \times \beta \times \text{input tokens} + \beta \times \text{output tokens} = time
$$

$$
\beta = \frac{time}{(\gamma \times \text{input tokens} + \text{output tokens})} = \frac{GPU \ time \ cost}{(\gamma \times \text{input tokens} + \text{output tokens})}
$$

For example, with 2,035 input tokens, 300 output tokens, a batch size of 16, and a runtime of 8.96 second - as we measure in one of the experiments mentioned in the previous section:

$$
\text{Cost per second } 4 \times A100s= \frac{4 \times \$2.5}{3600} = \$0.0028
$$

$$
\beta = \frac{\$0.0028 \times 8.96}{16 \times (0.3 \times 2035 + 300)} = \$0.00000172
$$

Which equals approximately \$1.72 per million output tokens and \$0.51 per million input tokens.

We apply the same calculations to the different batch sizes based on the experiments we did run in the previous section. As you can see, there is a dramatic price reduction with increasing batch size. The reader should verify that the reduction is directly proportional to the increased total throughput from running larger batches.

```markdown
| Batch Size | Cost Per 1M Input | Cost Per 1M Output | Tokens/Second |
|------------|------------------|-------------------|---------------|
| 1          | $5.09            | $16.98            | 419.57        |
| 2          | $2.67            | $8.90             | 800.81        |
| 4          | $1.51            | $5.02             | 1,419.34      |
| 8          | $0.86            | $2.86             | 2,488.31      |
| 16         | $0.51            | $1.71             | 4,166.46      |
| 32         | $0.35            | $1.18             | 6,020.61      |
| 64         | $0.28            | $0.95             | 7,453.62      |

```

Table 1: Estimated cost per 1M tokens for different batch sizes, assuming that the batch is 2035 tokens input and 300 tokens output. The pricing is estimated via the method described in this paragraph.

Obviously, the above is just a single data point from a single experiment for a single pair of input and output tokens. If we change the proportions of input and output tokens, the picture will be very different. To get a more realistic picture, we could, for example, run a Monte Carlo simulation- gathering multiple data points for different input and output configs, calculating the pricing for each example, removing the outliers (e.g., random slower executions due to external factors), and calculating the final price as an average of the median of these samples.

The above strategy still suffers from making very strong assumptions, e.g., we only benchmark the performance of rectangularly shaped inputs-all elements of the batch have the same number of input tokens and the same number of output tokens. We don’t mix the prefill phase with the decode phase-we submit an entire batch at the same time, since all of its elements have the same shape; we expect the prefill to take more or less the same amount of time, after which we only do the decoding part. This is a very strong assumption that is not necessarily going to be the case in the real world. We also hardcoded the value of `γ`, which is not necessarily fair and optimal for the settings we will be serving.

We’ll pause our pricing model discussion here, and we hope to do a deep dive into more sophisticated pricing strategies in the future text. There are multiple pricing strategies one could use; another interesting angle is estimating the minimal number of users of the LLM model under which an inference is profitable.

Luckily for us, at the end of the day, you can easily verify if you are running a profitable operation. You just add up the profits you made from charging users for input and output tokens, and you verify if this sums up to more than what you paid for renting a GPU for an hour.

# Summary

We really hope this text to be a founding block for you to build an accurate world model of LLM inference economics, using Llama 3.3 70B as an example. We start explaining what parameters are present in an LLM, what it means for a model to have 70B parameters, and how the memory is consumed. Then we give you a brief introduction to compute- and memory bandwidth performance and what it means to be compute- or memory-bound, and we explain the concept of FLOP and how many FLOPs are in a matrix multiplication.

We introduce the concept of the prefill phase and the token-by-token phase. We break down FLOPs during different parts of a forward pass, and we show how the prefill is primarily compute-bound. We then explain how different the token-by-token phase is. We introduce the concept of kv-cache; we go again through a forward pass and show how, thanks to the kv-caching, it is now far less dependent on compute (times `S` less FLOPs), hence how it becomes primarily memory bound. We show how, with the increasing input length, the KV cache occupies an increasingly big portion of the memory load time. We then briefly mention different parallelization strategies, and we describe how extra latency is added when running a multi-GPU setting.
We follow this by introducing the concept of batching. We explain why, since we are primarily memory bound, we can radically improve the economics of our operation by running larger batches. This is the core message of this text and the intuition with which we hope you leave after reading it. We then build a simplified throughput model from first principles and compare its performance to a real-world Llama 3.3 70b run via vLLM. We show the difference between the theoretical performance and a real one, and we give a brief explanation of where the extra overhead is coming from. We show how inaccurate the theoretical model is, which we hope lets you build an intuition about the challenges of predicting real-world performance by a bunch of heuristics.

Lastly, we discuss the challenges of establishing pricing and a fair pricing ratio between input and output tokens. We present a simplified cost model that, while not fully accurate, enables you to build a simple heuristic to price the input and output tokens.

Readers should realize why running on more than the minimal number of GPUs is actually highly beneficial to the economics. More GPUs enable higher efficiency through btter caching that give a better unit cost per token. With additional GPUs, the same model weights occupy proportionally less of the total memory, allowing more space for KV cache and consequently supporting larger batch sizes. Since throughput scales nearly linearly with batch size, this directly translates to better economics. Each additional GPU also contributes its memory bandwidth to the system, further improving the token generation rate since we are memory-bound in the token-by-token phase.

When evaluating hardware for LLM inference, readers should understand that memory size is not the only important factor - memory bandwidth is equally, if not more, critical. Since token-by-token generation is primarily memory-bound, they should always ask, "What is the memory speed?" as this determines how fast models can run. For example, NVIDIA L40S GPUs offer 48GB of memory but with a bandwidth of only 864 GB/s (compared to 3350 in H100 cards), resulting in very slow inference. Similarly, the Apple Mac Studio with M3 Ultra has 512GB of unified memory but only 819 819GB/s of memory bandwidth (see Fig. 25), limiting its LLM inference capabilities despite the large memory pool.

![image.png](Tokenomics%20from%20first%20principles%201a415b90a59f809c8386ccebd52f8b42/image%2015.png)

Fig. 25: Comparison of memory characteristics between Mac MacStudio and an Nvidia GPU. While Mac has a large memory on paper, it is pretty slow compared to an Nvidia GPU, making it far less suitable to serve LLMs.

Readers should also realize why running models on edge devices will always be relatively expensive on a per-token basis. When running on a consumer end device, we are always running batch-size=1. We can’t enjoy the economies of scale from sharing the cost of model load between multiple users. We always bear the entire device and energy cost ourselves. This, combined with the likely suboptimal characteristics of the on-edge hardware and slow memory, will result in high costs such as electricity and hardware depreciation.


The above model is just the tip of the iceberg; we don't discuss any of the possible optimizations, such as speculation models, hardware-specific optimizations, or quantization techniques. The goal of this text is for you to build some basic intuitions on LLM inference. We hope you understand why there are two phases, why one is compute-bound and the other one is memory-bound, and why we hope you developed some intuitions about the relationship between the number of input and output tokens and the request processing time.


To replicate the experiments, find the instructions in: https://github.com/tugot17/tokenomics-from-first-principles
