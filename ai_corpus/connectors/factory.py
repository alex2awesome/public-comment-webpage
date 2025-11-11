"""
Factory helpers for instantiating connectors based on configuration.
"""

from __future__ import annotations

from typing import Dict, Optional

from ai_corpus.connectors.cppa_admt import CppaAdmtConnector
from ai_corpus.connectors.eu_haveyoursay import EuHaveYourSayConnector
from ai_corpus.connectors.eu_haveyoursay_keyword import EuHaveYourSayKeywordConnector
from ai_corpus.connectors.gov_uk import GovUkConnector
from ai_corpus.connectors.nist_airmf import NistAirmfConnector
from ai_corpus.connectors.nitrd_ai_rfi import NitrdAiRfiConnector
from ai_corpus.connectors.regulations_gov import RegulationsGovConnector


CONNECTOR_MAP = {
    "regulations_gov": RegulationsGovConnector,
    "nist_airmf": NistAirmfConnector,
    "nitrd_ai_rfi": NitrdAiRfiConnector,
    "cppa_admt": CppaAdmtConnector,
    "eu_have_your_say": EuHaveYourSayConnector,
    "eu_have_your_say_keyword": EuHaveYourSayKeywordConnector,
    "gov_uk": GovUkConnector,
}


def build_connector(
    name: str,
    config: Dict,
    global_config: Optional[Dict] = None,
):
    """
    Instantiate a connector by name.
    """
    connector_cls = CONNECTOR_MAP.get(name)
    if connector_cls is None:
        raise KeyError(f"Unknown connector '{name}'")
    return connector_cls(config=config, global_config=global_config or {})


__all__ = ["build_connector", "CONNECTOR_MAP"]
