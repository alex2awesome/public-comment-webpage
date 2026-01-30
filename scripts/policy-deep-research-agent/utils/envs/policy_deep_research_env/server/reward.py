"""Backward-compatible reward shim for the legacy OpenEnv package."""

from policy_src.policy_research_core.reward import RewardResult, compute_reward

__all__ = ["RewardResult", "compute_reward"]
