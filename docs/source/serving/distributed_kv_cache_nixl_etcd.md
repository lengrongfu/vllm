# NIXL ETCD Distributed KV Cache Transfer

This guide explains how to use NIXL with ETCD for distributed KV cache transfer in vLLM, enabling efficient coordination between multiple vLLM instances in distributed environments.

## Overview

NIXL (NVIDIA Inference Xfer Library) with ETCD support provides:

- **Service Discovery**: Automatic discovery of remote vLLM instances
- **Metadata Exchange**: Distributed metadata storage and synchronization
- **Coordination**: Reliable coordination between distributed nodes
- **Fault Tolerance**: Automatic cleanup and recovery mechanisms

## Prerequisites

### 1. ETCD Server

Install and run an ETCD server:

```bash
# Using Docker (recommended)
docker run -d \
  --name etcd \
  -p 2379:2379 \
  -p 2380:2380 \
  quay.io/coreos/etcd:v3.5.1 \
  etcd \
  --advertise-client-urls http://0.0.0.0:2379 \
  --listen-client-urls http://0.0.0.0:2379

# Or install natively on Ubuntu
sudo apt install etcd-server etcd-client
sudo systemctl start etcd
```

### 2. Python Dependencies

Install required Python packages:

```bash
pip install etcd3
```

### 3. NIXL Library

Ensure NIXL is installed with ETCD support:

```bash
# Install NIXL with ETCD dependencies
pip install nixl[etcd]
```

## Configuration

### Environment Variables

Configure NIXL ETCD coordination using environment variables:

```bash
# Enable ETCD coordination
export VLLM_NIXL_USE_ETCD=1

# ETCD server endpoints (comma-separated for cluster)
export VLLM_NIXL_ETCD_ENDPOINTS="http://localhost:2379"

# ETCD namespace for metadata storage
export VLLM_NIXL_ETCD_NAMESPACE="/vllm/nixl/agents"

# ETCD connection timeout (seconds)
export VLLM_NIXL_ETCD_TIMEOUT=10
```

### vLLM Configuration

Configure vLLM to use NIXL with ETCD:

```python
from vllm import LLM

# Producer instance (prefill)
producer_llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_transfer_config={
        "kv_connector": "NixlConnector",
        "engine_id": "prefill_engine_1",
        "kv_role": "kv_producer"
    }
)

# Consumer instance (decode)
consumer_llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_transfer_config={
        "kv_connector": "NixlConnector",
        "engine_id": "decode_engine_1",
        "kv_role": "kv_consumer"
    }
)
```

## Usage Examples

### Basic Distributed Setup

#### Terminal 1 - Prefill Instance (Producer)

```bash
export VLLM_NIXL_USE_ETCD=1
export VLLM_NIXL_ETCD_ENDPOINTS="http://localhost:2379"

python -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='facebook/opt-125m',
    kv_transfer_config={
        'kv_connector': 'NixlConnector',
        'engine_id': 'prefill_engine',
        'kv_role': 'kv_producer'
    }
)

prompts = ['The future of AI is']
sampling_params = SamplingParams(max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f'Prompt: {output.prompt}')
    print(f'Generated: {output.outputs[0].text}')
"
```

#### Terminal 2 - Decode Instance (Consumer)

```bash
export VLLM_NIXL_USE_ETCD=1
export VLLM_NIXL_ETCD_ENDPOINTS="http://localhost:2379"

python -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='facebook/opt-125m',
    kv_transfer_config={
        'kv_connector': 'NixlConnector',
        'engine_id': 'decode_engine',
        'kv_role': 'kv_consumer'
    }
)

prompts = ['The future of AI is bright because']
sampling_params = SamplingParams(max_tokens=50)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f'Prompt: {output.prompt}')
    print(f'Generated: {output.outputs[0].text}')
"
```

### Multi-Node Deployment

For multi-node deployments, ensure all nodes can access the ETCD cluster:

```bash
# Node 1 (Prefill)
export VLLM_NIXL_USE_ETCD=1
export VLLM_NIXL_ETCD_ENDPOINTS="http://etcd-server:2379"
export VLLM_NIXL_SIDE_CHANNEL_HOST="node1.example.com"

# Node 2 (Decode)
export VLLM_NIXL_USE_ETCD=1
export VLLM_NIXL_ETCD_ENDPOINTS="http://etcd-server:2379"
export VLLM_NIXL_SIDE_CHANNEL_HOST="node2.example.com"
```

### Tensor Parallel Support

NIXL ETCD coordination supports tensor parallel deployments:

```python
# Multi-GPU prefill instance
producer_llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    kv_transfer_config={
        "kv_connector": "NixlConnector",
        "engine_id": "prefill_tp4",
        "kv_role": "kv_producer"
    }
)

# Multi-GPU decode instance
consumer_llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=8,  # Heterogeneous TP supported
    kv_transfer_config={
        "kv_connector": "NixlConnector",
        "engine_id": "decode_tp8",
        "kv_role": "kv_consumer"
    }
)
```

## Monitoring and Debugging

### ETCD Monitoring

Monitor ETCD for registered agents:

```bash
# List all registered agents
etcdctl get --prefix "/vllm/nixl/agents/"

# Watch for changes
etcdctl watch --prefix "/vllm/nixl/agents/"

# Get specific agent metadata
etcdctl get "/vllm/nixl/agents/prefill_engine/rank_0"
```

### Logging

Enable detailed logging for debugging:

```bash
export VLLM_LOGGING_LEVEL=DEBUG
```

Check logs for ETCD coordination events:

```
INFO: Initialized NIXL ETCD coordinator: engine_id=prefill_engine
INFO: Registered NIXL agent with ETCD
DEBUG: Discovering engine decode_engine through ETCD
DEBUG: ETCD handshake completed in 0.15 seconds
```

## Troubleshooting

### Common Issues

1. **ETCD Connection Failed**
   ```
   RuntimeError: ETCD connection failed: [Errno 111] Connection refused
   ```
   - Ensure ETCD server is running and accessible
   - Check `VLLM_NIXL_ETCD_ENDPOINTS` configuration

2. **Agent Not Found**
   ```
   RuntimeError: Engine decode_engine not found in ETCD
   ```
   - Verify both instances are using the same ETCD endpoints
   - Check ETCD namespace configuration
   - Ensure target engine has registered successfully

3. **Timeout Waiting for Agents**
   ```
   RuntimeError: Timeout waiting for engine prefill_engine
   ```
   - Increase `VLLM_NIXL_ETCD_TIMEOUT`
   - Check network connectivity between nodes
   - Verify engine IDs are unique and correct

### Performance Optimization

1. **ETCD Cluster**: Use ETCD cluster for high availability
2. **Network**: Ensure high-bandwidth, low-latency network between nodes
3. **Caching**: ETCD metadata is cached locally to reduce lookup overhead

## Advanced Configuration

### ETCD Cluster Setup

For production deployments, use an ETCD cluster:

```bash
# ETCD cluster endpoints
export VLLM_NIXL_ETCD_ENDPOINTS="http://etcd1:2379,http://etcd2:2379,http://etcd3:2379"
```

### Custom Namespaces

Use different namespaces for different deployments:

```bash
# Development environment
export VLLM_NIXL_ETCD_NAMESPACE="/vllm/dev/nixl/agents"

# Production environment
export VLLM_NIXL_ETCD_NAMESPACE="/vllm/prod/nixl/agents"
```

### Security

For secure deployments, configure ETCD with TLS:

```bash
export VLLM_NIXL_ETCD_ENDPOINTS="https://etcd-server:2379"
# Additional TLS configuration may be needed in the ETCD coordinator
```

## Migration from Direct ZMQ

To migrate from direct ZMQ handshake to ETCD coordination:

1. Set up ETCD server
2. Add environment variables
3. No code changes required - ETCD is used automatically when enabled
4. Direct ZMQ is used as fallback if ETCD fails

The system gracefully falls back to direct ZMQ handshake if ETCD coordination fails, ensuring backward compatibility.
