# MFU in inference

We are running 4xH100s connected via NVLink. Each has 989 TFLOPS, so 4x989 = 3956 TFLOPS. I'm experimenting with Llama 3.3 70B served in bf16. I'm running a bunch of experiments, with varying batch sizes, number of input and output tokens.


## Observations

We are running a vLLM instance with 4 H100s and observe the elapsed time for different settings. 

### Prefill

```json
"results": {

"1": {

"avg_input_tokens": 2083.0,

"avg_output_tokens": 1.0,

"elapsed_time": 0.1892281991119186,

"tokens_per_second_in_batch": 5.284841043993375,

"avg_tokens_per_second": 5.295112720160771

},

"2": {

"avg_input_tokens": 2083.0,

"avg_output_tokens": 1.0,

"elapsed_time": 0.3499017124995589,

"tokens_per_second_in_batch": 5.721281363361239,

"avg_tokens_per_second": 2.94235686032967

},

"4": {

"avg_input_tokens": 2083.0,

"avg_output_tokens": 1.0,

"elapsed_time": 0.6514690578915179,

"tokens_per_second_in_batch": 6.139989131269306,

"avg_tokens_per_second": 1.9338770167336958

},

"8": {

"avg_input_tokens": 2083.0,

"avg_output_tokens": 1.0,

"elapsed_time": 1.2957249943477411,

"tokens_per_second_in_batch": 6.175886017021812,

"avg_tokens_per_second": 1.3235872234474921

},

"16": {

"avg_input_tokens": 2083.0,

"avg_output_tokens": 1.0,

"elapsed_time": 2.9849893894667425,

"tokens_per_second_in_batch": 5.493216350352828,

"avg_tokens_per_second": 0.6665480914772839

},

"32": {

"avg_input_tokens": 2083.0,

"avg_output_tokens": 1.0,

"elapsed_time": 5.853053971659392,

"tokens_per_second_in_batch": 5.510580241448484,

"avg_tokens_per_second": 0.3604449790517612

}

}
```

### Decode

```json
"results": {

"1": {

"avg_input_tokens": 138.0,

"avg_output_tokens": 900.0,

"elapsed_time": 16.02985040983185,

"tokens_per_second_in_batch": 56.14525257503265,

"avg_tokens_per_second": 56.14682869115928

},

"2": {

"avg_input_tokens": 138.0,

"avg_output_tokens": 900.0,

"elapsed_time": 16.137532979249954,

"tokens_per_second_in_batch": 111.54121279346015,

"avg_tokens_per_second": 55.775693386555375

},

"4": {

"avg_input_tokens": 138.0,

"avg_output_tokens": 900.0,

"elapsed_time": 16.529554434120655,

"tokens_per_second_in_batch": 217.7917144922433,

"avg_tokens_per_second": 54.4594341095503

},

"8": {

"avg_input_tokens": 138.0,

"avg_output_tokens": 900.0,

"elapsed_time": 16.876225371845067,

"tokens_per_second_in_batch": 426.63568667504876,

"avg_tokens_per_second": 53.354807958591195

},

"16": {

"avg_input_tokens": 138.0,

"avg_output_tokens": 868.75,

"elapsed_time": 17.624156017322093,

"tokens_per_second_in_batch": 788.6902491295602,

"avg_tokens_per_second": 51.142646605513434

},

"32": {

"avg_input_tokens": 138.0,

"avg_output_tokens": 846.0625,

"elapsed_time": 19.289324533659965,

"tokens_per_second_in_batch": 1403.5742906785429,

"avg_tokens_per_second": 45.975410822577565

}

}
```

## MFU Prefill
For a prefill of 2048 tokens I estimate ~296 TFLOP

$$\begin{align}
\text{MFU}_{\text{prefill}} &= \frac{\text{Computational Work Done}}{\text{Theoretical Maximum Computation}} \times 100\% \\
&= \frac{\text{Batch Size} \times \text{Tokens per sequence} \times \text{FLOP per token}}{\text{Elapsed time} \times \text{Theoretical peak FLOPS}} \times 100\%
\end{align}$$

$$
\begin{align}
\text{MFU}_{\text{prefill, bs=32}} &= \frac{32 \times 2083 \times \frac{296 \text{ TFLOP}}{2048 \text{ tokens}}}
{5.853 \text{ s} \times 3956 \text{ TFLOPS}} \times 100\% \\[10pt]
&= \frac{32 \times 2083 \times 0.14453125 \text{ TFLOP/token}}
{5.853 \text{ s} \times 3956 \text{ TFLOPS}} \times 100\% \\[10pt]
&= \frac{9,632.85 \text{ TFLOP}}
{23,150.67 \text{ TFLOP}} \times 100\% \\[10pt]
&= 41.61\%
\end{align}
$$

Where:
- 32 is the batch size
- 2083 is the average number of input tokens per sequence
- 296 TFLOP/2048 tokens ≈ 0.14453125 TFLOP/token is the computational cost per token
- 5.853 seconds is the elapsed time for processing the entire batch
- 3956 TFLOPS is the theoretical peak performance of 4×H100 GPUs


| Batch Size | Total Tokens | Elapsed Time (s) | FLOPS Used (TFLOPS) | MFU (%) |
|------------|--------------|------------------|---------------------|---------|
| 1          | 2,083        | 0.189            | 1,590.98            | 40.22%  |
| 2          | 4,166        | 0.350            | 1,720.82            | 43.50%  |
| 4          | 8,332        | 0.651            | 1,848.49            | 46.73%  |
| 8          | 16,664       | 1.296            | 1,858.78            | 46.99%  |
| 16         | 33,328       | 2.985            | 1,613.72            | 40.79%  |
| 32         | 66,656       | 5.853            | 1,645.96            | 41.61%  |


## MFU Decode

For a single token in decode I estimate 0.14TFLOP

$$
\begin{align}
\text{MFU}_{\text{decode}} &= \frac{\text{Computational Work Done per Second}}{\text{Theoretical Maximum Computation per Second}} \times 100\% \\
&= \frac{\text{Tokens per Second} \times \text{FLOP per token}}{\text{Theoretical peak FLOPS}} \times 100\%
\end{align}
$$

$$
\begin{align}
\text{MFU}_{\text{decode, bs=32}} &= \frac{1403.57 \text{ tokens/s} \times 0.14 \text{ TFLOP/token}}
{3956 \text{ TFLOPS}} \times 100\% \\[10pt]
&= \frac{196.50 \text{ TFLOPS}}
{3956 \text{ TFLOPS}} \times 100\% \\[10pt]
&= 4.97\%
\end{align}
$$

Where:

- 1403.57 tokens/s is the tokens per second in batch for BS=32
- 0.14 TFLOP/token is the computational cost per token during decode
- 3956 TFLOPS is the theoretical peak performance of 4×H100 GPUs

| Batch Size | Tokens/sec | FLOPS Used (TFLOPS) | MFU (%) |
|------------|------------|---------------------|---------|
| 1          | 56.15      | 7.86                | 0.20%   |
| 2          | 111.54     | 15.62               | 0.39%   |
| 4          | 217.79     | 30.49               | 0.77%   |
| 8          | 426.64     | 59.73               | 1.51%   |
| 16         | 788.69     | 110.42              | 2.79%   |
| 32         | 1403.57    | 196.50              | 4.97%   |

