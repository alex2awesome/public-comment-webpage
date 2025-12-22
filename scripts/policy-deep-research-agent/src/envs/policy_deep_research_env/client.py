# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Policy Deep Research Env Environment HTTP Client.

This module provides the client for connecting to a Policy Deep Research Env Environment server
over HTTP.
"""

from typing import Dict

from openenv_core.http_env_client import HTTPEnvClient, StepResult

from .models import ResearchAction, ResearchObservation, ResearchState


class PolicyDeepResearchEnv(HTTPEnvClient[ResearchAction, ResearchObservation]):
    """
    HTTP client for the Policy Deep Research Env Environment.

    This client connects to a PolicyDeepResearchEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = PolicyDeepResearchEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.echoed_message)
        >>>
        >>> # Send a message
        >>> result = client.step(PolicyDeepResearchAction(message="Hello!"))
        >>> print(result.observation.echoed_message)
        >>> print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = PolicyDeepResearchEnv.from_docker_image("policy_deep_research_env-env:latest")
        >>> result = client.reset()
        >>> result = client.step(PolicyDeepResearchAction(message="Test"))
    """

    def _step_payload(self, action: ResearchAction) -> Dict:
        """Convert a ResearchAction to the JSON body expected by the server."""
        return {
            "type": action.type,
            "query": action.query,
            "paper_id": action.paper_id,
            "top_k": action.top_k,
            "filters": action.filters,
            "content": action.content,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ResearchObservation]:
        """Convert the FastAPI payload into a StepResult."""
        obs_payload = payload.get("observation", {})
        observation = ResearchObservation(**obs_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ResearchState:
        """Convert /state responses into ResearchState objects."""
        return ResearchState(**payload)
