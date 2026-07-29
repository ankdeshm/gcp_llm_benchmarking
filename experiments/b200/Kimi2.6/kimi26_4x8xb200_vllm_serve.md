
# Performance Benchmarking Report: Kimi-K2.6 on NVIDIA B200 (A4 High)

## 1. Model Overview

* **Model Name:** Kimi-K2.6 (Registry: `moonshotai/Kimi-K2.6`)
* **Source:** [Hugging Face - moonshotai/Kimi-K2.6](https://hf.co/moonshotai/Kimi-K2.6)
* **Architecture Highlights:**
* **Mixture-of-Experts (MoE):** The model is built on an expansive MoE architecture containing **1 Trillion total parameters** distributed across 384 specialized experts.
* **Expert Parallelism & Activation:** During inference, the MoE router selects only 8 experts (plus 1 shared expert) per token. This means that despite the 1T scale, only **32 Billion parameters are activated per token**. This allows for frontier-level reasoning with significantly reduced compute latency.
* **Massive Context Window:** Engineered to ingest up to **262,144 tokens (256K)** natively, making it highly effective for document-heavy analysis and long-horizon autonomous tasks.

**Hardware & Memory Justification:**
Deploying a 1 Trillion parameter model requires immense memory. Even utilizing advanced quantization formats (like INT4 or FP8), the sheer size of the model weights alone consumes hundreds of gigabytes of VRAM.

* **Model Weights:** A 1T parameter model in FP8 precision requires approximately **1,000 GB** of VRAM just to load the weights into memory.
* **KV Cache & Context:** To support high concurrency at extended context windows (e.g., 30K to 256K tokens), the PagedAttention KV Cache requires hundreds of additional gigabytes of memory overhead to prevent swapping or Out-of-Memory (OOM) failures.

**Why an 8x B200 Node is Mandatory:**
A single NVIDIA B200 GPU typically maxes out at 192 GB of memory. To securely hold a ~1,000 GB model plus allocate sufficient space for a heavy KV Cache, the deployment *requires* a tightly coupled 8-GPU node (providing ~1.5 TB of total VRAM). Running this via Expert Parallelism and Tensor Parallelism across all 8 GPUs ensures the 1T parameters are partitioned successfully, allowing the 32B active parameters to execute at the high speeds demonstrated in this benchmark.

---

## 2. Infrastructure Setup

The benchmark was executed on a tightly coupled Google Kubernetes Engine (GKE) cluster designed for extreme high-performance computing (HPC).

* **Compute Nodes:** 4
* **GPUs per Node:** 8x NVIDIA B200
* **Total GPUs:** 32
* **Total VRAM:** ~1,500 GB high-bandwidth memory per node (~6 TB cluster total)
* **Networking:** GPU-Direct enabled via NCCL plugins for high-bandwidth, low-latency cross-node communication.

---

## 3. Deployment Configuration (YAML Highlights)

The deployment relies on a specific Kubernetes Deployment YAML (attached separately `kimi26_b200_maxlen16k.yaml`). Key configurations driving this performance include:

* **Topology:** 4 Replicas (exactly 1 replica scheduled per 8-GPU node via node affinity).
* **Resource Allocation:** Each `vllm-server` container is allocated 80 CPUs, 800Gi of RAM, and all 8 NVIDIA GPUs.
* **Initialization:** Uses an `nccl-plugin-installer` init-container to map fast networking libraries, and a `model-downloader` to cache the massive model weights to a local SSD prior to server startup.
* **vLLM Engine Arguments:**
* `--tensor-parallel-size 8`: Splits the model layers across all 8 GPUs on the node.
* `--enable-expert-parallel`: Activates MoE-specific parallelism for the Kimi architecture.
* `--kv-cache-dtype fp8`: Compresses the KV cache to 8-bit precision, doubling the theoretical context capacity of the VRAM.
* `--max-model-len 32768`: The hard safety limit for context length (Input + Output). *Note: Requests exceeding this are rejected with a 400 Bad Request to prevent OOM crashes.*



---

## 4. Benchmarking Methodology

To ensure exact reproducibility and avoid common third-party schema issues (e.g., `400 Bad Request` errors), we utilized the native `vllm bench serve` CLI tool. Crucially, this tool was executed **directly inside an active vLLM server pod**. This approach bypasses local network port-forwarding bottlenecks, guarantees that all necessary PyTorch/vLLM dependencies are natively present, and allows direct communication with the cluster's internal load balancer.

### Execution Steps

**Step 1: Identify an active vLLM server pod**

```bash
kubectl get pods -n dynamo-cloud
# (Pick any pod)

```

**Step 2: Execute the Benchmark Scripts**
We used a bash loop via `kubectl exec` to automatically iterate through workloads. The tests were broken into three architectural profiles to test different memory and compute bottlenecks:

1. **Prefill-Heavy (Summarization/RAG):** Massive input, short output (16K/1K and 30K/1K).
2. **Balanced (Chat/Code Gen):** Equal input and output (8K/8K and 15K/15K).
3. **Decode-Heavy (Reasoning):** Short input, massive output (1K/16K and 1K/30K).

**The Complete Test Script:**

```bash
kubectl exec -it kimi-2.6-standard-75f86cf4fd-6qp4m -n dynamo-cloud -- bash -c '
# Format: "ISL OSL Concurrency Prompts TestName"
tests=(
  # --- PREFILL-HEAVY SWEEP ---
  "16000 1000 64 128 16k_Context_Conc_064"
  "16000 1000 128 128 16k_Context_Conc_128"
  "16000 1000 256 256 16k_Context_Conc_256"
  "30000 1000 8 16 30k_Context_Conc_008"
  "30000 1000 16 32 30k_Context_Conc_016"
  "30000 1000 32 64 30k_Context_Conc_032"
  "30000 1000 64 128 30k_Context_Conc_064"
  "30000 1000 128 128 30k_Context_Conc_128"
  "30000 1000 256 256 30k_Context_Conc_256"

  # --- BALANCED SWEEP ---
  "8000 8000 32 64 Balanced_08k_ISL_08k_OSL_Conc_032"
  "8000 8000 64 128 Balanced_08k_ISL_08k_OSL_Conc_064"
  "15000 15000 32 64 Balanced_15k_ISL_15k_OSL_Conc_032"
  "15000 15000 64 128 Balanced_15k_ISL_15k_OSL_Conc_064"

  # --- DECODE-HEAVY SWEEP ---
  "1000 16000 16 32 Decode_Heavy_01k_ISL_16k_OSL_Conc_016"
  "1000 16000 32 64 Decode_Heavy_01k_ISL_16k_OSL_Conc_032"
  "1000 30000 8 16 Decode_Heavy_01k_ISL_30k_OSL_Conc_008"
  "1000 30000 16 32 Decode_Heavy_01k_ISL_30k_OSL_Conc_016"
)

for test in "${tests[@]}"; do
  read -r isl osl conc prompts name <<< "$test"
  
  echo "RUNNING: $name | ISL=${isl} | OSL=${osl} | Concurrency=${conc}"
  
  vllm bench serve \
    --backend openai-chat \
    --model moonshotai/Kimi-K2.6 \
    --host kimi-26-standard-service \
    --port 8000 \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len ${isl} \
    --random-output-len ${osl} \
    --num-prompts ${prompts} \
    --max-concurrency ${conc} \
    --request-rate inf \
    --trust-remote-code
    
  sleep 15 # KV Cache Cooldown
done
'

```

---

## 5. Summary & Comparison Tables

The following tables highlight the hardware's resilience. Throughout all sweeps, **zero Out-of-Memory (OOM) crashes or failures occurred**, verifying that the KV Cache and PagedAttention mechanisms are scaling gracefully across the 32-GPU deployment.

### A. Prefill-Heavy (Summarization & RAG)

*Tests compute bounds during massive context ingestion.*

| Workload (ISL / OSL) | Concurrency | Peak Tok/s | TTFT (Mean) | ITL (Mean) |
| --- | --- | --- | --- | --- |
| **16K / 1K** | 64 | 3,936.00 | 7.58 s | 23.31 ms |
| **16K / 1K** | 128 | 6,239.00 | 15.41 s | 32.98 ms |
| **16K / 1K** | 256 | **9,388.00** | 27.99 s | 52.26 ms |
| **30K / 1K** | 32 | 2,334.00 | 8.01 s | 19.90 ms |
| **30K / 1K** | 128 | 5,538.00 | 29.07 s | 46.70 ms |
| **30K / 1K** | 256 | **7,457.00** | 52.84 s | 81.82 ms |

### B. Balanced (Chat & Data Translation)

*Tests memory fragmentation and active block swapping.*

| Workload (ISL / OSL) | Concurrency | Peak Tok/s | TTFT (Mean) | ITL (Mean) |
| --- | --- | --- | --- | --- |
| **8K / 8K** | 32 | 2,604.00 | 2.27 s | 13.19 ms |
| **8K / 8K** | 64 | **4,156.00** | 3.91 s | 16.86 ms |
| **15K / 15K** | 32 | 2,350.00 | 4.36 s | 14.70 ms |
| **15K / 15K** | 64 | **3,844.00** | 6.95 s | 18.57 ms |

### C. Decode-Heavy (Reasoning & Chain-of-Thought)

*Tests raw memory bandwidth limits during sustained generation.*

| Workload (ISL / OSL) | Concurrency | Peak Tok/s | TTFT (Mean) | ITL (Mean) |
| --- | --- | --- | --- | --- |
| **1K / 16K** | 16 | 1,570.00 | 0.29 s | 10.73 ms |
| **1K / 16K** | 32 | **2,590.00** | 0.52 s | 13.63 ms |
| **1K / 30K** | 8 | 878.00 | 0.18 s | 9.66 ms |
| **1K / 30K** | 16 | **1,596.00** | 0.30 s | 10.89 ms |

### Architectural Takeaways

1. **Linear MoE Scaling:** The Mixture-of-Experts routing remains remarkably efficient. Even when generating 15,000 sequential output tokens simultaneously across 64 streams (Balanced sweep), Inter-Token Latency remains sub-20ms.
2. **Bandwidth Resilience:** The decode-heavy sweeps showcase the sheer memory bandwidth of the B200s. Pushing 30,000 output tokens per request resulted in ITLs of ~10ms.
3. **Safe Queueing:** Under extreme saturation (Prefill-Heavy 30K/1K at 256 concurrency), the Time to First Token (TTFT) stretches to ~52 seconds, indicating the GPUs safely held the excess requests in the PagedAttention queue rather than terminating the process.

---

## 6. Next Steps: 64K+ Context Testing

To benchmark the model's performance at 64,000 tokens and beyond, the Kubernetes deployment must be updated to explicitly permit larger contexts.

A secondary YAML file (attached separately) modifies the `vllm-server` startup arguments to include `--max-model-len 65536`. Once applied, the same benchmark methodology detailed in Section 4 can be utilized to measure the absolute upper bounds of the KV Cache before hard OOM failures occur.

---

## Appendix: Raw Benchmark Outputs

### 16K & 30K Prefill-Heavy (Key Outputs)

```text
============ Serving Benchmark Result (16K/1K - Conc 256) ============
Successful requests:                     256       
Output token throughput (tok/s):         2978.71   
Mean TTFT (ms):                          27999.59  
Mean ITL (ms):                           52.26     

============ Serving Benchmark Result (30K/1K - Conc 256) ============
Successful requests:                     256       
Output token throughput (tok/s):         1660.98   
Mean TTFT (ms):                          52843.63  
Mean ITL (ms):                           81.82     

```

### Balanced Sweep (Raw Outputs)

```text
============ Serving Benchmark Result (8K/8K - Conc 32) ============
Successful requests:                     64        
Output token throughput (tok/s):         2054.19   
Mean TTFT (ms):                          2275.68   
Mean ITL (ms):                           13.19     

============ Serving Benchmark Result (8K/8K - Conc 64) ============
Successful requests:                     128       
Output token throughput (tok/s):         3467.07   
Mean TTFT (ms):                          3911.76   
Mean ITL (ms):                           16.86     

============ Serving Benchmark Result (15K/15K - Conc 32) ============
Successful requests:                     64        
Output token throughput (tok/s):         1922.53   
Mean TTFT (ms):                          4364.63   
Mean ITL (ms):                           14.70     

============ Serving Benchmark Result (15K/15K - Conc 64) ============
Successful requests:                     128       
Output token throughput (tok/s):         3271.61   
Mean TTFT (ms):                          6955.51   
Mean ITL (ms):                           18.57     

```

### Decode-Heavy Sweep (Raw Outputs)

```text
============ Serving Benchmark Result (1K/16K - Conc 16) ============
Successful requests:                     32        
Output token throughput (tok/s):         1352.49   
Mean TTFT (ms):                          292.32    
Mean ITL (ms):                           10.73     

============ Serving Benchmark Result (1K/16K - Conc 32) ============
Successful requests:                     64        
Output token throughput (tok/s):         2142.31   
Mean TTFT (ms):                          526.99    
Mean ITL (ms):                           13.63     

============ Serving Benchmark Result (1K/30K - Conc 8) ============
Successful requests:                     16        
Output token throughput (tok/s):         781.78    
Mean TTFT (ms):                          188.60    
Mean ITL (ms):                           9.66      

============ Serving Benchmark Result (1K/30K - Conc 16) ============
Successful requests:                     32        
Output token throughput (tok/s):         1317.51   
Mean TTFT (ms):                          302.74    
Mean ITL (ms):                           10.89     

```

**Hardware & Memory Justification:**
Deploying a 1 Trillion parameter model requires immense memory. Even utilizing advanced quantization formats (like INT4 or FP8), the sheer size of the model weights alone consumes hundreds of gigabytes of VRAM.

* **Model Weights:** A 1T parameter model in FP8 precision requires approximately **1,000 GB** of VRAM just to load the weights into memory.
* **KV Cache & Context:** To support 256 concurrent requests at a 16K context window (or pushing toward the 256K limit), the PagedAttention KV Cache requires hundreds of additional gigabytes of memory overhead to prevent swapping or Out-of-Memory (OOM) failures.


## 2. Infrastructure Setup

The benchmark was executed on a tightly coupled Google Kubernetes Engine (GKE) cluster designed for extreme high-performance computing (HPC).

* **Compute Nodes:** 4
* **GPUs per Node:** 8x NVIDIA B200
* **Total GPUs:** 32
* **Total VRAM:** ~1,500 GB high-bandwidth memory per node (~6 TB cluster total)
* **Networking:** GPU-Direct enabled via NCCL plugins for high-bandwidth, low-latency cross-node communication.

## 3. Deployment Configuration (YAML Highlights)

The deployment relies on a specific Kubernetes Deployment YAML (attached separately to this report). Key configurations driving this performance include:

* **Topology:** 4 Replicas (exactly 1 replica scheduled per 8-GPU node via node affinity).
* **Resource Allocation:** Each `vllm-server` container is allocated 80 CPUs, 800Gi of RAM, and all 8 NVIDIA GPUs.
* **Initialization:** Uses an `nccl-plugin-installer` init-container to map fast networking libraries, and a `model-downloader` to cache the massive model weights to a local SSD prior to server startup.
* **vLLM Engine Arguments:**
* `--tensor-parallel-size 8`: Splits the model layers across all 8 GPUs on the node.
* `--enable-expert-parallel`: Activates MoE-specific parallelism for the Kimi architecture.
* `--kv-cache-dtype fp8`: Compresses the KV cache to 8-bit precision, doubling the theoretical context capacity of the VRAM.
* `--max-model-len 32768`: The hard safety limit for context length (Input + Output). *Note: Requests exceeding this are rejected with a 400 Bad Request to prevent OOM crashes.*



## 4. Benchmarking Methodology

To ensure exact reproducibility and avoid common dependency or networking errors, we bypassed third-party tools (like `genai-perf`) and local port-forwarding tunnels. Instead, we executed the native `vllm bench serve` CLI tool directly from **inside one of the running vLLM server pods**.

Running the benchmark from within an active vLLM pod guarantees that all massive PyTorch/vLLM dependencies are natively present (preventing `ModuleNotFoundError`) and allows direct communication with the cluster's internal load balancer.

### Step-by-Step Execution

**Step 1: Identify an active vLLM server pod**
Retrieve the exact name of a running pod in your namespace:

```bash
kubectl get pods -n dynamo-cloud

```

*(Pick any pod name)*

**Step 2: Execute the Benchmark Script**
We wrap a bash loop inside a `kubectl exec` command to iterate through various concurrency levels and context lengths automatically.

**Crucial Flags Used:**

* `--backend openai-chat`: Formats the payload into the strict JSON schema required by `/v1/chat/completions`, preventing `400 Bad Request` errors.
* `--trust-remote-code`: Required to safely execute Kimi K2.6's custom Hugging Face tokenizer code.

Run the following script in your terminal to reproduce the benchmark:

```bash
kubectl exec -it kimi-2.6-standard-75f86cf4fd-6qp4m -n dynamo-cloud -- bash -c '
# Format: "ISL OSL Concurrency Prompts TestName"
tests=(
  # --- 16K CONTEXT SWEEP ---
  "16000 1000 64 128 16k_Context_Conc_064"
  "16000 1000 128 128 16k_Context_Conc_128"
  "16000 1000 256 256 16k_Context_Conc_256"

  # --- 30K CONTEXT SWEEP ---
  "30000 1000 8 16 30k_Context_Conc_008"
  "30000 1000 16 32 30k_Context_Conc_016"
  "30000 1000 32 64 30k_Context_Conc_032"
  "30000 1000 64 128 30k_Context_Conc_064"
  "30000 1000 128 128 30k_Context_Conc_128"
  "30000 1000 256 256 30k_Context_Conc_256"
)

for test in "${tests[@]}"; do
  read -r isl osl conc prompts name <<< "$test"
  
  echo ""
  echo "=========================================================================="
  echo "🔥 RUNNING: $name | ISL=${isl} | OSL=${osl} | Concurrency=${conc}"
  echo "=========================================================================="
  
  vllm bench serve \
    --backend openai-chat \
    --model moonshotai/Kimi-K2.6 \
    --host kimi-26-standard-service \
    --port 8000 \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len ${isl} \
    --random-output-len ${osl} \
    --num-prompts ${prompts} \
    --max-concurrency ${conc} \
    --request-rate inf \
    --trust-remote-code
    
  echo ""
  echo "Cooling down for 15 seconds to allow KV cache flush..."
  sleep 15
done
'

```

---

## 5. Summary & Comparison Tables

The metrics below highlight **Throughput** (how fast the cluster generates total text), **Time to First Token / TTFT** (how long it takes to process the massive input context), and **Inter-Token Latency / ITL** (the speed at which text generates after the first token).

### 16K Context Sweep (16,000 Input / 1,000 Output)

| Concurrency | Successful Requests | Peak Tokens/Sec | TTFT (Mean) | ITL (Mean) |
| --- | --- | --- | --- | --- |
| **64** | 128 / 128 | 3,936.00 tok/s | 7.58 s | 23.31 ms |
| **128** | 128 / 128 | 6,239.00 tok/s | 15.41 s | 32.98 ms |
| **256** | 256 / 256 | **9,388.00 tok/s** | 27.99 s | **52.26 ms** |

**16K Analysis:** The cluster handles 16K contexts flawlessly. At 256 concurrency, the cluster manages over 4.3 million active tokens in memory with **zero failures**. Inter-Token Latency remains at 52ms—roughly human reading speed—even under absolute maximum load.

### 30K Context Sweep (30,000 Input / 1,000 Output)

| Concurrency | Successful Requests | Peak Tokens/Sec | TTFT (Mean) | ITL (Mean) |
| --- | --- | --- | --- | --- |
| **8** | 16 / 16 | 810.00 tok/s | 2.85 s | 10.84 ms |
| **16** | 32 / 32 | 1,390.00 tok/s | 4.69 s | 14.05 ms |
| **32** | 64 / 64 | 2,334.00 tok/s | 8.01 s | 19.90 ms |
| **64** | 128 / 128 | 3,606.00 tok/s | 13.81 s | 30.66 ms |
| **128** | 128 / 128 | 5,538.00 tok/s | 29.07 s | 46.70 ms |
| **256** | 256 / 256 | **7,457.00 tok/s** | 52.84 s | **81.82 ms** |

**30K Analysis:** Pushing the context to 30K requires significantly more VRAM and prefill compute. The TTFT increases as expected due to the massive context ingestion, and queueing becomes visible at 256 concurrency (52 seconds TTFT). However, **zero OOM crashes occurred**, proving the PagedAttention blocks safely queued the 7.9 million active tokens. ITL remains highly performant at 81.8ms under extreme stress.

---

## 6. Next Steps: 64K+ Context Testing

To test the model's performance at 64,000 tokens and beyond, the Kubernetes deployment must be updated to explicitly permit larger contexts.

A secondary YAML file (attached separately) modifies the `vllm-server` startup arguments to include `--max-model-len 65536`. Once applied, the same benchmark script detailed in Section 4 can be executed by adjusting the loop definitions to `--random-input-len 64000` to measure the upper bounds of the B200's 1.5TB VRAM footprint.

---

## Appendix: Raw Benchmark Outputs

### 16K Raw Outputs (64, 128, 256 Concurrency)

```text
============ Serving Benchmark Result (16K - Conc 64) ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             64        
Output token throughput (tok/s):         1761.98   
Mean TTFT (ms):                          7587.25   
Mean ITL (ms):                           23.31     

============ Serving Benchmark Result (16K - Conc 128) ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             128       
Output token throughput (tok/s):         2350.91   
Mean TTFT (ms):                          15414.49  
Mean ITL (ms):                           32.98     

============ Serving Benchmark Result (16K - Conc 256) ============
Successful requests:                     256       
Failed requests:                         0         
Maximum request concurrency:             256       
Output token throughput (tok/s):         2978.71   
Mean TTFT (ms):                          27999.59  
Mean ITL (ms):                           52.26     

```

### 30K Raw Outputs (8, 16, 32, 64, 128, 256 Concurrency)

```text
============ Serving Benchmark Result (30K - Conc 8) ============
Successful requests:                     16        
Failed requests:                         0         
Maximum request concurrency:             8         
Output token throughput (tok/s):         515.90    
Mean TTFT (ms):                          2856.71   
Mean ITL (ms):                           10.84     

============ Serving Benchmark Result (30K - Conc 16) ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             16        
Output token throughput (tok/s):         771.68    
Mean TTFT (ms):                          4695.65   
Mean ITL (ms):                           14.05     

============ Serving Benchmark Result (30K - Conc 32) ============
Successful requests:                     64        
Failed requests:                         0         
Maximum request concurrency:             32        
Output token throughput (tok/s):         915.93    
Mean TTFT (ms):                          8018.15   
Mean ITL (ms):                           19.90     

============ Serving Benchmark Result (30K - Conc 64) ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             64        
Output token throughput (tok/s):         1254.28   
Mean TTFT (ms):                          13813.84  
Mean ITL (ms):                           30.66     

============ Serving Benchmark Result (30K - Conc 128) ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             128       
Output token throughput (tok/s):         1550.03   
Mean TTFT (ms):                          29071.18  
Mean ITL (ms):                           46.70     

============ Serving Benchmark Result (30K - Conc 256) ============
Successful requests:                     256       
Failed requests:                         0         
Maximum request concurrency:             256       
Output token throughput (tok/s):         1660.98   
Mean TTFT (ms):                          52843.63  
Mean ITL (ms):                           81.82     

```
