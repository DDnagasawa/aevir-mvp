"""L3 chain client (Web3) for proof submission."""

from __future__ import annotations

import json
from dataclasses import dataclass

try:
    from web3 import Web3  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Web3 = None


def _to_bytes32(hex_str: str) -> bytes:
    cleaned = hex_str[2:] if hex_str.startswith("0x") else hex_str
    if len(cleaned) != 64:
        raise ValueError("proof_hash must be 32 bytes hex")
    return bytes.fromhex(cleaned)


@dataclass
class L3ChainConfig:
    rpc_url: str
    contract_address: str
    abi_path: str
    private_key: str
    function_name: str = "submitContribution"
    gas: int | None = None
    gas_price_wei: int | None = None
    chain_id: int | None = None


class L3ChainClient:
    def __init__(self, config: L3ChainConfig):
        if Web3 is None:
            raise RuntimeError("web3 is required for L3 chain integration")
        self.w3 = Web3(Web3.HTTPProvider(config.rpc_url))
        with open(config.abi_path, "r", encoding="utf-8") as handle:
            abi = json.load(handle)
        self.contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(config.contract_address),
            abi=abi,
        )
        self.account = self.w3.eth.account.from_key(config.private_key)
        self.function_name = config.function_name
        self.gas = config.gas
        self.gas_price_wei = config.gas_price_wei
        self.chain_id = config.chain_id

    def submit_contribution(self, node_id: str, epoch_id: str, proof_hash: str) -> str:
        fn = getattr(self.contract.functions, self.function_name)
        args = [node_id, epoch_id, _to_bytes32(proof_hash)]
        tx = fn(*args).build_transaction(
            {
                "from": self.account.address,
                "nonce": self.w3.eth.get_transaction_count(self.account.address),
                "gas": self.gas or 300000,
                "gasPrice": self.gas_price_wei or self.w3.eth.gas_price,
                "chainId": self.chain_id or self.w3.eth.chain_id,
            }
        )
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        return tx_hash.hex()

