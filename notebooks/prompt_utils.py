"""Prompt and OpenAI helper utilities for the AI corpus notebook."""

from __future__ import annotations

import ast
import asyncio
import json
from typing import Callable, Iterable, List, Optional

from openai import AsyncOpenAI, OpenAI, PermissionDeniedError, InternalServerError
from tqdm.auto import tqdm

__all__ = [
    "async_client",
    "sync_client",
    "process_batch",
    "process_one",
    "CLEANER_PROMPT",
    "ANALYSIS_PROMPT",
    "AI_PROMPT",
    "ARGUMENTATION_PROMPT",
]

async_client = AsyncOpenAI()
sync_client = OpenAI()


def _is_json_like(candidate: str) -> bool:
    try:
        json.loads(candidate)
        return True
    except json.JSONDecodeError:
        pass
    try:
        ast.literal_eval(candidate)
        return True
    except (ValueError, SyntaxError):
        return False


def numbered_list_checker(output: str, expected_count: int = None) -> bool:
    """Ensure the model output is a numbered list with `expected_count` items."""
    parts = output.strip().split("\n\n")
    if expected_count is not None:
        if len(parts) != expected_count:
            return False
    for idx, part in enumerate(parts, start=1):
        expected_prefix = f"{idx}."
        if not part.strip().startswith(expected_prefix):
            return False
    return True


async def process_batch(
    texts: Optional[Iterable[str]] = None,
    prompt_template: Optional[str] = None,
    prompts: Optional[Iterable[str]] = None,
    *,
    concurrency: int = 500,
    check_json: bool = False,
    check_additional: Optional[List[Callable[[str], bool]]] = None,
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    model: str = "gpt-5-mini",
) -> List[str]:
    """Run an async batch of prompt completions.

    Provide either:
        - `texts` along with `prompt_template` (containing `{input_text}`), or
        - `prompts`, an iterable of already-formatted prompt strings.
    """
    if prompts is not None:
        prompt_list = list(prompts)
    else:
        if texts is None or prompt_template is None:
            raise ValueError("Provide either `prompts` or both `texts` and `prompt_template`.")
        texts = list(texts)
        prompt_list = [prompt_template.format(input_text=text) for text in texts]

    if not prompt_list:
        return []

    sem = asyncio.Semaphore(concurrency)
    results: List[Optional[str]] = [None] * len(prompt_list)

    async def call_api(idx: int, prompt_str: str) -> None:
        async with sem:
            for attempt in range(1, max_attempts + 1):
                delay = backoff_base * (2 ** (attempt - 1))
                try:
                    resp = await async_client.responses.create(
                        model=model,
                        input=prompt_str,
                        service_tier="flex",
                    )
                except InternalServerError as err:
                    if attempt == max_attempts:
                        payload = {
                            "error": "internal_server_error",
                            "detail": str(err),
                        }
                        request_id = getattr(getattr(err, "response", None), "headers", {}).get("x-request-id") if getattr(err, "response", None) else None
                        if request_id:
                            payload["request_id"] = request_id
                        results[idx] = json.dumps(payload)
                        return
                    await asyncio.sleep(delay)
                    continue
                except PermissionDeniedError as err:
                    if attempt == max_attempts:
                        results[idx] = json.dumps(
                            {
                                "error": "permission_denied",
                                "detail": getattr(err, "message", "policy_blocked"),
                            }
                        )
                        return
                    await asyncio.sleep(delay)
                    continue
                except Exception as err:  # noqa: BLE001
                    if attempt == max_attempts:
                        results[idx] = json.dumps(
                            {
                                "error": err.__class__.__name__,
                                "detail": str(err),
                            }
                        )
                        return
                    await asyncio.sleep(delay)
                    continue

                cleaned = resp.output_text
                passes_json = (not check_json) or _is_json_like(cleaned)
                passes_extra = True
                if check_additional:
                    passes_extra = all(check(cleaned) for check in check_additional)
                if passes_json and passes_extra:
                    results[idx] = cleaned
                    return
                if attempt == max_attempts:
                    results[idx] = None
                    return
                await asyncio.sleep(delay)

    tasks = [asyncio.create_task(call_api(i, prompt)) for i, prompt in enumerate(prompt_list)]

    with tqdm(total=len(prompt_list)) as bar:
        for coro in asyncio.as_completed(tasks):
            await coro
            bar.update(1)

    return [r if r is not None else "" for r in results]


def process_one(
    text: str = None,
    prompt_template: str = None,
    prompt: str = None,
    *,
    model: str = "gpt-5-mini",
) -> str:
    """Run a synchronous prompt completion."""
    if prompt == None:
        prompt = prompt_template.format(input_text=text)
    resp = sync_client.responses.create(
        model=model,
        input=prompt,
    )
    return resp.output_text


CLEANER_PROMPT = """You are a diligent text cleaner. You receive raw text, parsed from HTML on the internet that 
  mixes true comments with online boilerplate: navigation menus, cookie notices, language selectors, legal disclaimers, and other boilerplate. 
  Return only the substantive article body as plain text paragraph(s).

  Rules:

  - Keep the comment's original sentences and order; no summaries or paraphrasing.
  - Strip everything that is not part of the main article (headers, footers, “Accept cookies”, “Login”, language
    lists, contact info, disclaimers, etc.).
  - If the article has a title, keep the title on the first line. Do not invent titles.
  - Preserve paragraph breaks from the original content when they clearly belong to the article; otherwise collapse
    to single spaces.
  - If you cannot find meaningful comment content, reply with NO COMMENT FOUND.

  Input:

  {input_text}

  Output:
"""

ANALYSIS_PROMPT = """You are a capable legal and public policy assistant. 

I am trying to categorize different comments to RFCs based on who the identify of the person is who wrote the comment. 
Please select the most likely label from this list. If there is no label on this list that captures the group/person, please say so 
and categorize the person with another label.

<categories>
 - tech_platforms: Large technology companies building AI models, cloud infrastructure, or hardware;
    they focus on innovation pace, interoperability, and the operational burdens of new rules.
  - industry_associations: Trade groups and lobbying coalitions representing sectors like
    advertising, finance, or telecom that coordinate member positions and argue for harmonized,
    business-friendly standards.
  - civil_society: Advocacy nonprofits and NGOs centered on civil liberties, digital rights, consumer
    protection, or equity; they emphasize transparency, accountability, and safeguarding the public.
  - academic_labs: University researchers and interdisciplinary centers studying AI, ethics, law, and
    policy who offer evidence-based analysis and highlight research or oversight needs.
  - policy_thinktanks: Independent institutes crafting policy proposals on technology governance,
    economics, and societal impacts; they translate scholarship into actionable regulatory
    frameworks.
  - enterprise_users: Corporations outside the tech sector—finance, healthcare, retail, manufacturing
    —deploying AI in operations and concerned about compliance costs and practical guidance.
  - startups_smbs: Small and mid-sized firms innovating with AI or heavily affected by it; they
    typically ask for clear, proportionate requirements to avoid stifling early-stage growth.
  - education_sector: Schools, universities, and education coalitions interested in student privacy,
    academic integrity, and how AI tools reshape teaching and administration.
  - labor_organizations: Unions and professional guilds representing workers whose roles may be
    automated or reshaped, advocating for worker protections, training, and fair labor practices.
  - legal_compliance: Law firms, privacy consultants, and standards bodies advising on regulatory
    obligations; they seek clarity, safe harbors, and workable enforcement mechanisms.
  - public_agencies: Municipal, state, federal, or international government entities sharing lessons
    from public-sector AI deployments and coordinating policy approaches.
  - individual_citizens: Concerned residents, grassroots groups, or small community organizations
    submitting personal or local perspectives on AI’s benefits and risks.
  - healthcare_stakeholders: Hospitals, medical associations, biotech firms, and health IT vendors
    focused on patient safety, data protection, and sector-specific guidance.
  - security_defense: Cybersecurity companies, national-security NGOs, and defense analysts
    addressing adversarial misuse, critical-infrastructure protection, and resilience.
  - media_organizations: Newsrooms and journalism associations exploring AI’s impact on information
    integrity, misinformation, and newsroom workflows.
</categories>

<comment>
{input_text}
</comment>

Respond with a JSON in the following format:
{{
'label': <label>,
'reason': <reason>
}}

Your response:
"""

AI_PROMPT = """You are a capable legal and public policy assistant. 

I am trying to categorize different comments to RFCs. Is the comment about artificial intelligence (AI) legislation? 
Answer with just a Yes/No and explain why.

<comment>
{input_text}
</comment>

Your response:
"""

ARGUMENTATION_PROMPT = """You are a capable legal and public policy assistant. 

I am trying to categorize different comments to RFCs to identify the argument that the writer is making about AI legislation.
Please summarize the KIND of argument with a high-level label and a brief description, in the form:
<argument_type>: <brief_description>

Avoid talking too much about the specifics. If multiple different arguments are being made, please separate them by newlines ('\\n')

<comment>
{input_text}
</comment>

Your response:
"""
