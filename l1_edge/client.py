"""L1 edge client entry (registration + local training loop)."""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable

from .api_client import ApiClient
from .data_sandbox import DataSandbox
from .flower_client import start_flower_client
from .l3_chain_client import L3ChainClient, L3ChainConfig
from .local_trainer import LocalTrainer
from .resource_limiter import ResourceLimiter
from .resource_monitor import ResourceMonitor
from .utils import (
    compute_ash_id,
    decode_base64,
    encode_base64,
    get_gpu_serial,
    get_mac_hash,
    read_file_bytes,
    sign_payload,
)


def build_hardware_profile(monitor: ResourceMonitor) -> dict:
    profile = monitor.hardware_profile()
    return {
        "gpu_serial": get_gpu_serial(),
        "mac_hash": get_mac_hash(),
        "tflops": profile.get("tflops") or os.getenv("GPU_TFLOPS"),
        "vram_gb": profile.get("vram_gb") or os.getenv("GPU_VRAM_GB"),
        "cuda_version": profile.get("cuda_version") or os.getenv("CUDA_VERSION"),
        "rocm_version": profile.get("rocm_version") or os.getenv("ROCM_VERSION"),
        "gpu_name": profile.get("gpu_name"),
        "gpu_count": profile.get("gpu_count"),
        "cpu_cores": profile.get("cpu_cores"),
        "ram_gb": profile.get("ram_gb"),
    }


def register_with_l2(client: ApiClient | None, payload: dict) -> dict | None:
    if not client:
        print("[L1] skip L2 registration (no endpoint)")
        return None
    return client.post_json("/v1/nodes/register", payload)


def heartbeat(client: ApiClient | None, payload: dict) -> dict | None:
    if not client:
        return None
    return client.post_json("/v1/heartbeat", payload)


def fetch_task(client: ApiClient | None, payload: dict) -> dict | None:
    if not client:
        return None
    return client.post_json("/v1/tasks/next", payload)


def upload_gradients(client: ApiClient | None, payload: dict) -> dict | None:
    if not client:
        return None
    return client.post_json("/v1/gradients/upload", payload)


def submit_proof_to_l3(client: ApiClient | None, payload: dict) -> dict | None:
    if not client:
        print("[L1] skip L3 proof submit (no endpoint)")
        return None
    return client.post_json("/v1/proofs/submit", payload)


def retry_call(
    label: str,
    func: Callable[[dict], dict | None],
    payload: dict,
    attempts: int,
    backoff_seconds: float,
) -> dict | None:
    for attempt in range(1, attempts + 1):
        result = func(payload)
        if result and result.get("status") not in ("error", "rejected"):
            return result
        if attempt < attempts:
            sleep_for = backoff_seconds * attempt
            print(f"[L1] retry {label} attempt={attempt} sleep={sleep_for:.1f}s")
            time.sleep(sleep_for)
    return result


def poll_for_task(
    client: ApiClient | None,
    payload: dict,
    poll_interval: float,
    max_polls: int,
    attempts: int,
    backoff_seconds: float,
) -> dict | None:
    if not client:
        return None
    for idx in range(max_polls):
        task = retry_call("fetch_task", lambda p: fetch_task(client, p), payload, attempts, backoff_seconds)
        if not task:
            time.sleep(poll_interval)
            continue
        status = task.get("status")
        if status in (None, "ready"):
            return task
        if status in ("wait", "no_task"):
            time.sleep(poll_interval)
            continue
        return task
    return None

def main() -> None:
    parser = argparse.ArgumentParser(description="L1 edge client")
    parser.add_argument("--epoch-id", default=str(int(time.time())))
    parser.add_argument("--model-params", default=None)
    parser.add_argument("--sandbox-dir", default="./l1_edge/sandbox")
    parser.add_argument("--l2-endpoint", default=os.getenv("L2_ENDPOINT"))
    parser.add_argument("--l3-endpoint", default=os.getenv("L3_ENDPOINT"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY"))
    parser.add_argument("--signing-key", default=os.getenv("SIGNING_KEY"))
    parser.add_argument("--client-version", default=os.getenv("CLIENT_VERSION", "0.1.0"))
    parser.add_argument("--flower-address", default=os.getenv("FLOWER_ADDRESS"))
    parser.add_argument("--l3-rpc-url", default=os.getenv("L3_RPC_URL"))
    parser.add_argument("--l3-contract-address", default=os.getenv("L3_CONTRACT_ADDRESS"))
    parser.add_argument("--l3-abi-path", default=os.getenv("L3_CONTRACT_ABI_PATH"))
    parser.add_argument("--l3-private-key", default=os.getenv("L3_PRIVATE_KEY"))
    parser.add_argument("--l3-function", default=os.getenv("L3_FUNCTION", "submitContribution"))
    parser.add_argument("--l3-gas", type=int, default=int(os.getenv("L3_GAS", "0")))
    parser.add_argument("--l3-gas-price", type=int, default=int(os.getenv("L3_GAS_PRICE_WEI", "0")))
    parser.add_argument("--l3-chain-id", type=int, default=int(os.getenv("L3_CHAIN_ID", "0")))
    parser.add_argument("--cpu-limit", type=float, default=float(os.getenv("CPU_LIMIT", "85")))
    parser.add_argument("--mem-limit", type=float, default=float(os.getenv("MEM_LIMIT", "85")))
    parser.add_argument("--gpu-mem-limit", type=float, default=None)
    parser.add_argument("--dp", action="store_true", help="enable DP noise")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "sshleifer/tiny-gpt2"))
    parser.add_argument("--lora-r", type=int, default=int(os.getenv("LORA_R", "8")))
    parser.add_argument("--lora-alpha", type=int, default=int(os.getenv("LORA_ALPHA", "16")))
    parser.add_argument("--lora-dropout", type=float, default=float(os.getenv("LORA_DROPOUT", "0.05")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "3")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "2")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("LEARNING_RATE", "0.0002")))
    parser.add_argument("--task-type", default=os.getenv("TASK_TYPE", "CAUSAL_LM"))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("MAX_RETRIES", "3")))
    parser.add_argument("--retry-backoff", type=float, default=float(os.getenv("RETRY_BACKOFF", "1.5")))
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("POLL_INTERVAL", "5")))
    parser.add_argument("--poll-max", type=int, default=int(os.getenv("POLL_MAX", "12")))
    args = parser.parse_args()

    monitor = ResourceMonitor()
    hardware = build_hardware_profile(monitor)
    ash_id = compute_ash_id()

    if args.flower_address:
        trainer = LocalTrainer(
            use_dp=args.dp,
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            task_type=args.task_type,
        )
        start_flower_client(args.flower_address, trainer=trainer, epoch_id=args.epoch_id)
        return
    l2_client = ApiClient(args.l2_endpoint, api_key=args.api_key) if args.l2_endpoint else None
    l3_client = ApiClient(args.l3_endpoint, api_key=args.api_key) if args.l3_endpoint else None
    register_payload = {
        "ash_id": ash_id,
        "hardware": {
            "tflops": hardware.get("tflops"),
            "vram_gb": hardware.get("vram_gb"),
            "cuda_version": hardware.get("cuda_version"),
            "rocm_version": hardware.get("rocm_version"),
        },
        "client_version": args.client_version,
        "ts": int(time.time()),
    }
    register_payload["sig"] = sign_payload(register_payload, args.signing_key)
    register_result = retry_call(
        "register",
        lambda p: register_with_l2(l2_client, p),
        register_payload,
        args.max_retries,
        args.retry_backoff,
    ) or {}
    node_id = register_result.get("node_id") or ash_id

    sandbox = DataSandbox(base_dir=args.sandbox_dir)
    run_ctx = sandbox.create_run_dir(args.epoch_id)
    sandbox.write_audit(run_ctx.audit_log, "init", {"hardware": hardware})

    metrics = monitor.read()
    limiter = ResourceLimiter(
        cpu_percent=args.cpu_limit,
        mem_percent=args.mem_limit,
        gpu_mem_percent=args.gpu_mem_limit,
    )
    limit_result = limiter.check(metrics)
    sandbox.write_audit(run_ctx.audit_log, "resource_check", metrics | {"ok": limit_result.ok})
    if not limit_result.ok:
        print(f"[L1] resource limits exceeded: {limit_result.reason}")
        return
    retry_call(
        "heartbeat",
        lambda p: heartbeat(l2_client, p),
        {"node_id": node_id, "ts": int(time.time()), "metrics": metrics},
        args.max_retries,
        args.retry_backoff,
    )

    pending = sandbox.load_pending(run_ctx.pending_path)
    if pending:
        print(f"[L1] found pending items={len(pending)}")
    still_pending = []
    for record in pending:
        kind = record.get("kind")
        payload = record.get("payload") or {}
        if kind == "upload":
            result = retry_call(
                "upload_gradients",
                lambda p: upload_gradients(l2_client, p),
                payload,
                args.max_retries,
                args.retry_backoff,
            )
            if not result or result.get("status") == "error":
                still_pending.append(record)
        elif kind == "proof":
            result = retry_call(
                "submit_proof",
                lambda p: submit_proof_to_l3(l3_client, p),
                payload,
                args.max_retries,
                args.retry_backoff,
            )
            if not result or result.get("status") == "error":
                still_pending.append(record)
    if still_pending:
        sandbox.save_pending(run_ctx.pending_path, still_pending)
    else:
        sandbox.clear_pending(run_ctx.pending_path)

    task = poll_for_task(
        l2_client,
        {"node_id": node_id, "capabilities": register_payload["hardware"], "last_epoch_id": None},
        args.poll_interval,
        args.poll_max,
        args.max_retries,
        args.retry_backoff,
    )
    if task and task.get("status") not in (None, "ready"):
        print(f"[L1] task status={task.get('status')}")
        return
    epoch_id = task.get("epoch_id") if task else args.epoch_id
    model_params = (
        decode_base64(task["model_params_base64"]) if task and task.get("model_params_base64") else None
    )
    params = model_params if model_params is not None else read_file_bytes(args.model_params)
    trainer = LocalTrainer(
        use_dp=args.dp,
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        task_type=args.task_type,
    )
    result = trainer.train(params, epoch_id)
    sandbox.write_audit(run_ctx.audit_log, "train", result.metrics)

    print(f"[L1] gradients bytes={len(result.compressed_gradients)}")
    print(f"[L1] proof_hash={result.proof_hash}")
    upload_payload = {
        "node_id": node_id,
        "epoch_id": epoch_id,
        "model_hash": result.model_hash,
        "gradient_blob_base64": encode_base64(result.compressed_gradients),
        "metrics": result.metrics,
    }
    upload_result = retry_call(
        "upload_gradients",
        lambda p: upload_gradients(l2_client, p),
        upload_payload,
        args.max_retries,
        args.retry_backoff,
    )
    if not upload_result or upload_result.get("status") == "error":
        sandbox.save_pending(run_ctx.pending_path, [{"kind": "upload", "payload": upload_payload}])
    if args.l3_rpc_url:
        if not (args.l3_contract_address and args.l3_abi_path and args.l3_private_key):
            raise RuntimeError("L3 chain config missing (address/abi/private_key)")
        chain_client = L3ChainClient(
            L3ChainConfig(
                rpc_url=args.l3_rpc_url,
                contract_address=args.l3_contract_address,
                abi_path=args.l3_abi_path,
                private_key=args.l3_private_key,
                function_name=args.l3_function,
                gas=args.l3_gas or None,
                gas_price_wei=args.l3_gas_price or None,
                chain_id=args.l3_chain_id or None,
            )
        )
        tx_hash = None
        for attempt in range(1, args.max_retries + 1):
            try:
                tx_hash = chain_client.submit_contribution(node_id, epoch_id, result.proof_hash)
                break
            except Exception as exc:
                if attempt < args.max_retries:
                    sleep_for = args.retry_backoff * attempt
                    print(f"[L1] L3 submit failed: {exc} retry in {sleep_for:.1f}s")
                    time.sleep(sleep_for)
        if tx_hash:
            print(f"[L1] L3 tx_hash={tx_hash}")
        else:
            sandbox.save_pending(
                run_ctx.pending_path,
                [{"kind": "proof", "payload": {"node_id": node_id, "epoch_id": epoch_id, "proof_hash": result.proof_hash}}],
            )
    else:
        proof_payload = {"node_id": node_id, "epoch_id": epoch_id, "proof_hash": result.proof_hash}
        proof_result = retry_call(
            "submit_proof",
            lambda p: submit_proof_to_l3(l3_client, p),
            proof_payload,
            args.max_retries,
            args.retry_backoff,
        )
        if not proof_result or proof_result.get("status") == "error":
            sandbox.save_pending(
                run_ctx.pending_path,
                [{"kind": "proof", "payload": proof_payload}],
            )


if __name__ == "__main__":
    main()
