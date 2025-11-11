"""
Factory helpers for instantiating connectors based on configuration.
"""

from __future__ import annotations

from typing import Dict, Optional

from ai_corpus.connectors.connecticut_eregulations import ConnecticutEregsConnector
from ai_corpus.connectors.cppa_admt import CppaAdmtConnector
from ai_corpus.connectors.eu_haveyoursay_playwright import EuHaveYourSayPlaywrightConnector
from ai_corpus.connectors.gov_uk import GovUkConnector
from ai_corpus.connectors.nist_airmf import NistAirmfConnector
from ai_corpus.connectors.nitrd_ai_rfi import NitrdAiRfiConnector
from ai_corpus.connectors.oregon_admin_orders import OregonAdminOrdersConnector
from ai_corpus.connectors.pa_dep_ecomment import PaDepEcommentConnector
from ai_corpus.connectors.regulations_gov import RegulationsGovConnector
from ai_corpus.connectors.utah_bulletin import UtahBulletinConnector
from ai_corpus.connectors.virginia_townhall import VirginiaTownhallConnector
from ai_corpus.connectors.washington_register import WashingtonRegisterConnector


CONNECTOR_MAP = {
    "connecticut_eregulations": ConnecticutEregsConnector,
    "regulations_gov": RegulationsGovConnector,
    "nist_airmf": NistAirmfConnector,
    "nitrd_ai_rfi": NitrdAiRfiConnector,
    "cppa_admt": CppaAdmtConnector,
    "eu_have_your_say_playwright": EuHaveYourSayPlaywrightConnector,
    "gov_uk": GovUkConnector,
    "virginia_townhall": VirginiaTownhallConnector,
    "oregon_admin_orders": OregonAdminOrdersConnector,
    "utah_bulletin": UtahBulletinConnector,
    "pa_dep_ecomment": PaDepEcommentConnector,
    "washington_register": WashingtonRegisterConnector,
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
