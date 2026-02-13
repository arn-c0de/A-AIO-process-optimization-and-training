"""Deterministic random seed derivation.

Seeds must be stable across platforms and Python versions, and must not depend
on values that are assigned after rendering (e.g., train/val/test split).
"""

import hashlib


def derive_sample_seed(run_seed: int, domain: str, index: int) -> int:
    """Derive deterministic sample seed from run parameters.

    Uses SHA256 to ensure reproducibility across platforms and Python versions.

    Args:
        run_seed: Master seed from config
        domain: Domain name (e.g., 'domain_A')
        index: Sample index within (run_id, domain)

    Returns:
        63-bit non-negative integer seed (safe for NumPy / PyTorch)
    """
    # Create unique string representation
    seed_string = f"{run_seed}|{domain}|{index}"

    # Hash and convert to integer
    hash_bytes = hashlib.sha256(seed_string.encode('utf-8')).digest()

    # Take first 8 bytes and convert to int (64-bit)
    seed = int.from_bytes(hash_bytes[:8], byteorder='big')

    # Ensure non-negative (NumPy accepts 0; avoid negative for other libs).
    seed = seed & 0x7FFFFFFFFFFFFFFF  # Mask to 63 bits

    return seed


def make_sample_id(run_id: str, domain: str, split: str, index: int) -> str:
    """Create human-readable sample ID.

    Args:
        run_id: Run identifier (e.g., 'run_0001')
        domain: Domain name (e.g., 'domain_A')
        split: Split name ('train', 'val', 'test')
        index: Sample index within (domain, split)

    Returns:
        Formatted ID string (e.g., 'run_0001/domain_A/train/000042')
    """
    return f"{run_id}/{domain}/{split}/{index:06d}"


def parse_sample_id(sample_id: str) -> tuple:
    """Parse sample ID back into components.

    Args:
        sample_id: Formatted ID string

    Returns:
        Tuple of (run_id, domain, split, index)
    """
    parts = sample_id.split('/')
    if len(parts) != 4:
        raise ValueError(f"Invalid sample ID format: {sample_id}")

    run_id, domain, split, index_str = parts

    try:
        index = int(index_str)
    except ValueError:
        raise ValueError(f"Invalid index in sample ID: {sample_id}")

    return run_id, domain, split, index
