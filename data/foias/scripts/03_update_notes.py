#!/usr/bin/env python3
"""
Step 3: Append filing guidance notes.

Appends comprehensive, jurisdiction-specific filing guidance
to the Notes column for all 110 rows. Each entry includes:
  - Exact request language template referencing known AI platforms
  - Export mechanism instructions for the agency
  - Pitfalls to avoid based on precedent
  - Framing strategy to keep all-employee scope without being "too broad"
"""
from helpers import load_tracker, save_tracker, build_jur_map, append_notes

wb, ws = load_tracker("foia_tracker_detailed_v3.xlsx")
jur_map = build_jur_map(ws)

count = 0

# ============================================================
# FEDERAL AGENCIES (5 U.S.C. § 552)
# ============================================================

federal_base = """FRAMING STRATEGY: Federal FOIA has no "too broad" doctrine — agencies must process or negotiate scope. Name specific platforms and record types to make processing feasible.

KNOWN PITFALL: Federal agencies average 6-12 months. Request "expedited processing" if you can claim "urgency to inform the public" (5 U.S.C. §552(a)(6)(E)). Also request fee waiver as media/educational requester.

REQUEST TEMPLATE:
"Pursuant to the Freedom of Information Act, 5 U.S.C. § 552, I request the following records:

1. All generative AI conversation logs — including user prompts/inputs and AI-generated outputs/responses — created by [AGENCY] employees using any of the following platforms: {platforms}. Time period: January 1, 2024 to present.

2. Administrative usage logs or reports showing the number of users, sessions, and queries for each generative AI platform deployed at [AGENCY] during this period.

3. For ChatGPT Enterprise accounts: the data export available through the admin console (ZIP archive containing conversation.html files and metadata JSON). For Microsoft Copilot: the admin-accessible conversation history logs available through the M365 admin center or Purview compliance portal.

This request encompasses all [AGENCY] employees and contractors with access to these tools. I am not requesting the tools' training data or source code — only the conversation logs between users and AI systems.

I request a fee waiver pursuant to 5 U.S.C. § 552(a)(4)(A)(iii) as disclosure is in the public interest and I am a representative of the news media / educational institution [adjust as applicable]. I also request expedited processing as this information is urgently needed to inform the public about government use of AI technology, a matter of current public interest and debate.

If this request is denied in whole or in part, please cite the specific exemption(s) and release all segregable portions."
"""

if append_notes(ws, jur_map, "GSA (USAi.gov)", federal_base.format(
    platforms="USAi.gov (GSA's government-wide AI platform providing Chat, API, and Console access to models from OpenAI, Google, Amazon, Meta, Microsoft, and xAI), ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: USAi.gov is the central federal AI platform — requesting its logs is novel (no prior FOIA found). GSA may argue platform-level aggregate data isn't an agency record. Counter: individual employee conversation histories ARE agency records under 44 U.S.C. §3301. Reference GSA's FY 2025-2027 AI compliance plan acknowledging AI governance obligations."): count += 1

if append_notes(ws, jur_map, "HHS", federal_base.format(
    platforms="ChatGPT Enterprise (deployed department-wide September 9, 2025 via GSA OneGov Strategy, FISMA moderate ATO), HHSGPT (internal HHS AI tool), Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: HHS had the largest federal AI expansion: 7 to 116 use cases (2023-2024) per GAO. ChatGPT Enterprise deployed dept-wide Sept 2025. No prior FOIA for chat logs exists — this would be a novel request. Specify ChatGPT Enterprise and HHSGPT separately. HHS FOIA Officer: Arianne Perkins. File via HHS PAL portal (requests.publiclink.hhs.gov)."): count += 1

if append_notes(ws, jur_map, "OPM", federal_base.format(
    platforms="ChatGPT (GPT-5 access confirmed), Microsoft Copilot Chat (GPT-4o), and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: Small agency = potentially faster processing. No prior FOIA for AI chat logs found. File via OPM Public Access Link (foia.opm.gov)."): count += 1

if append_notes(ws, jur_map, "House of Representatives", """NOTE: FOIA does not apply to Congress (legislative branch exempt). No legal mechanism to compel disclosure of AI chat logs.

ALTERNATIVE APPROACH: File with the Clerk of the House or Government Accountability Office (which IS subject to FOIA for some records). Congressional Research Service may also be a target. Otherwise, rely on voluntary disclosure, press inquiries, or Congressional oversight hearings.

KNOWN DEPLOYMENT: Up to 6,000 Microsoft 365 Copilot licenses for 1-year pilot. ChatGPT also used.

STRATEGY: Since FOIA doesn't apply, consider: (1) contacting individual members' offices directly citing public interest, (2) requesting through GAO FOIA for any reports/assessments of Congressional AI use, (3) tracking Congressional Record/hearing testimony for voluntary disclosures."""): count += 1

if append_notes(ws, jur_map, "Senate", """NOTE: FOIA does not apply to Congress (legislative branch exempt). No legal mechanism to compel disclosure of AI chat logs.

ALTERNATIVE APPROACH: Same as House — file with GAO, rely on voluntary disclosure. Senate Sergeant at Arms approved ChatGPT Enterprise, Google Gemini Chat, and Microsoft Copilot Chat (March 2026).

STRATEGY: Very recent approval (March 2026) means logs are just beginning to accumulate. Track Senate Rules Committee for any disclosure policies. Consider press inquiries to individual Senators' offices."""): count += 1

# DHS and components
if append_notes(ws, jur_map, "DHS (Department of Homeland Security)", federal_base.format(
    platforms="DHSChat (internal GenAI chatbot deployed December 2024, 19,000+ HQ staff, 10 operating agencies), ChatGPT Enterprise, Claude 2, BingChat, DALL-E2, Grammarly, Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: STRONGEST FEDERAL TARGET. DHS-OpenAI contract explicitly states 'outputs from the model are considered federal records.' Use this language in your request: 'Per DHS's own contractual terms with OpenAI, outputs from generative AI models are federal records subject to FOIA.' CJ Ciaramella's second request (Oct 2025) is still pending — DHS cited 'voluminous amount of separate and distinct records,' confirming records exist. File via DHS FIRST portal (first.dhs.gov) — REQUIRED since Jan 2026. Specify DHSChat by name."): count += 1

if append_notes(ws, jur_map, "ICE (Immigration and Customs Enforcement)", federal_base.format(
    platforms="DHSChat (DHS-wide internal AI chatbot), Stella AI Chatbot (internal cybersecurity AI), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: DHS component — DHS-OpenAI contract language applies ('outputs are federal records'). Cite this in your request. File via DHS SecureRelease portal. No prior FOIA for ICE AI chat logs found."): count += 1

if append_notes(ws, jur_map, "USCBP (US Customs and Border Protection)", federal_base.format(
    platforms="chatCBP (custom internal AI chatbot, production since May 2025), DHSChat (DHS-wide), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: DHS component — cite DHS-OpenAI contract. chatCBP is a named internal tool — reference it specifically. File via CBP SecureRelease portal. Online-only since Jan 2026."): count += 1

if append_notes(ws, jur_map, "USCIS (US Citizenship and Immigration Services)", federal_base.format(
    platforms="DHSChat (DHS-wide internal AI chatbot), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: DHS component — cite DHS-OpenAI contract. File via FIRST portal (first.uscis.gov) — REQUIRED since Jan 2026."): count += 1

if append_notes(ws, jur_map, "TSA (Transportation Security Administration)", federal_base.format(
    platforms="DHSChat (DHS-wide internal AI chatbot, deployed Dec 2024), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: DHS component — cite DHS-OpenAI contract. File via FOIA.gov only — TSA no longer accepts hard copy. Large workforce."): count += 1

if append_notes(ws, jur_map, "FEMA (Federal Emergency Management Agency)", federal_base.format(
    platforms="DHSChat (DHS-wide), proprietary GenAI tool for budget responses, ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: DHS component — cite DHS-OpenAI contract. FEMA piloting AI for hazard mitigation. File via DHS FOIA form. Online-only since Jan 2026."): count += 1

if append_notes(ws, jur_map, "SEC (Securities and Exchange Commission)", federal_base.format(
    platforms="ChatGPT (provisionally approved), Bing Chat (provisionally approved), Claude 2 (provisionally approved), DALL-E2 (provisionally approved), Grammarly (provisionally approved), Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: STRONG PRECEDENT — Sungho Park's FOIA completed Jan 2025 with records released. SEC has already produced AI chat records once. Cite this precedent in your request. Independent agency — file via SEC FOIA Services (sec.gov/foia-services). List all 5 provisionally approved tools by name."): count += 1

if append_notes(ws, jur_map, "DOJ (Department of Justice)", federal_base.format(
    platforms="ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool, including any AI tools used in FOIA processing (e.g., MITRE FOIA Assistant)"
) + "\nSPECIFIC NOTES: Bergin's investigation found agencies using AI in FOIA processing but claiming 'no responsive documents.' File specifically for employee chat logs, NOT for AI-in-FOIA (which Bergin is already pursuing). File via FOIA STAR portal (doj-foia.entellitrak.com). Large dept — consider filing to specific components (OIP, FBI, DEA, ATF) separately."): count += 1

if append_notes(ws, jur_map, "CMS (Centers for Medicare & Medicaid Services)", federal_base.format(
    platforms="CMS Chat (internal GenAI chatbot deployed December 2024, 100% employee access), ChatGPT Enterprise (HHS-wide deployment), Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: CMS Chat has 100% employee access — 4,700+ trained through 'AI Ignite' program. No prior FOIA for chat logs — novel request. Reference CMS Chat by name. File via HHS PAL portal selecting CMS."): count += 1

if append_notes(ws, jur_map, "DOE (Department of Energy)", federal_base.format(
    platforms="ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, PermitAI, and any other generative AI chatbot tool deployed under DOE's Generative AI Policy"
) + "\nSPECIFIC NOTES: DOE has the most comprehensive published AI governance framework (AI Strategy, Compliance Plan, GenAI Policy, GenAI Reference Guide). Reference these in your request to demonstrate the agency's own acknowledgment of AI governance obligations. No prior FOIA for chat logs. File via energy.gov FOIA request form. Consider national labs separately."): count += 1

if append_notes(ws, jur_map, "EPA (Environmental Protection Agency)", federal_base.format(
    platforms="GovChat (customized GenAI chatbot for EPA employees), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: GovChat is EPA's named internal AI tool — reference it specifically. File via EPA PAL portal (foiapublicaccessportal.epa.gov). 10 regional offices — consider whether to target HQ or specific regions."): count += 1

if append_notes(ws, jur_map, "FDA (Food and Drug Administration)", federal_base.format(
    platforms="Elsa (GenAI tool deployed agency-wide), ChatGPT Enterprise (HHS-wide), Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: Elsa is FDA's named internal AI tool. HHS sub-agency — HHS ChatGPT Enterprise applies. Call (301) 796-3900 before filing for guidance. File via accessdata.fda.gov FOIA portal."): count += 1

if append_notes(ws, jur_map, "NIH (National Institutes of Health)", federal_base.format(
    platforms="ChIRP (Chatbox for Intramural Research Programs), ChatGPT Enterprise (HHS-wide), Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: ChIRP is NIH's named internal AI tool. 27 Institutes/Centers each have FOIA coordinators — consider targeting specific ICs or HQ-wide. File via securefoia.nih.gov."): count += 1

if append_notes(ws, jur_map, "CDC (Centers for Disease Control and Prevention)", federal_base.format(
    platforms="ChatCDC (built on Microsoft Azure OpenAI), ChatGPT Enterprise (HHS-wide), Microsoft 365 Copilot, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: CDC is a pioneer in federal GenAI adoption. ChatCDC is a strong named target. File via CDC FOIA PAL (foia.cdc.gov/app/Home.aspx)."): count += 1

if append_notes(ws, jur_map, "NIST (National Institute of Standards and Technology)", federal_base.format(
    platforms="ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool"
) + "\nSPECIFIC NOTES: NIST develops the AI Risk Management Framework and runs the AI Innovation Lab — ironic if they can't produce their own AI records. Small workforce = potentially faster. File via FOIA.gov (select Commerce/NIST)."): count += 1

if append_notes(ws, jur_map, "USDA (Department of Agriculture)", federal_base.format(
    platforms="USDA IT Support GenAI Chatbot (pilot, built with Microsoft technologies), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool deployed via USDA GenAI sandboxes"
) + "\nSPECIFIC NOTES: USDA has GenAI sandboxes available to all sub-agencies. Large dept with many components — OIA handles HQ FOIA, sub-agencies have separate processes. File via USDA PAL portal (securefoia.usda.gov)."): count += 1

# Remaining federal agencies with generic template
for fed_agency, platforms_str in [
    ("AMS (Agricultural Marketing Service)", "USDA IT Support GenAI Chatbot, ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool available via USDA GenAI sandboxes"),
    ("APHIS (Animal and Plant Health Inspection Service)", "USDA IT Support GenAI Chatbot, ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool available via USDA GenAI sandboxes"),
    ("FS (US Forest Service)", "Axon+ Copilot Studio Chatbot, ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool available via USDA GenAI sandboxes"),
    ("FSIS (Food Safety and Inspection Service)", "ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool available via USDA GenAI sandboxes"),
    ("FWS (Fish and Wildlife Service)", "ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool deployed under DOI AI initiatives"),
    ("FCC (Federal Communications Commission)", "ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool"),
    ("NHTSA (National Highway Traffic Safety Administration)", "Google Gemini (DOT-wide deployment), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"),
    ("NSF (National Science Foundation)", "ServiceNow GenAI 'Now Assist', ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"),
    ("NTIA (National Telecommunications and Information Administration)", "ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"),
    ("FDIC (Federal Deposit Insurance Corporation)", "AI Coding Assistant, GenAI chatbot tools, ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"),
    ("FAA (Federal Aviation Administration)", "Azure OpenAI (GPT-3.5 Turbo, GPT-4, Whisper), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"),
    ("MSHA (Mine Safety and Health Administration)", "DOL-wide AI tools (DOL AI Center GenAI Assistant), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"),
    ("OSHA (Occupational Safety and Health Administration)", "DOL-wide AI tools (DOL AI Center GenAI Assistant), HoloLens AR AI, ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool"),
]:
    if append_notes(ws, jur_map, fed_agency, federal_base.format(platforms=platforms_str) + f"\nSPECIFIC NOTES: Sub-agency — file via parent department portal if no dedicated FOIA portal. Reference specific named tools above."): count += 1

# ============================================================
# US STATES
# ============================================================

def state_template(statute_cite, statute_name, platforms, specific_notes, pitfalls=""):
    base = f"""FRAMING STRATEGY: Name specific AI platforms and record types. State public records laws generally don't have a "too broad" doctrine the way some local agencies claim — but specificity about the WHAT (not the WHO) prevents foot-dragging.

{f'KNOWN PITFALL: {pitfalls}' if pitfalls else ''}

REQUEST TEMPLATE:
"Pursuant to {statute_name} ({statute_cite}), I request the following records:

1. All generative AI conversation logs — including user prompts/inputs and AI-generated outputs/responses — created by state employees using any of the following platforms: {platforms}. Time period: January 1, 2024 to present.

2. Administrative usage logs or dashboards showing the number of state employees with access to, and usage statistics for, each generative AI platform deployed statewide during this period.

3. For ChatGPT Enterprise accounts: the data export available through the admin console (ZIP archive containing conversation.html files and metadata JSON with timestamps). For Microsoft 365 Copilot: the admin-accessible conversation history logs available through the M365 admin center or Microsoft Purview compliance portal. For Google Gemini: the admin console activity logs.

This request encompasses all state employees and contractors with access to these tools across all departments and agencies. I am not requesting the tools' training data or source code — only the conversation logs between users and AI systems.

I request a fee waiver as this information will contribute significantly to public understanding of government operations. If any records are withheld, please cite the specific statutory exemption and release all segregable portions."

{specific_notes}"""
    return base

if append_notes(ws, jur_map, "Massachusetts", state_template(
    "M.G.L. c.66, § 10(b)", "the Massachusetts Public Records Law",
    "ChatGPT Enterprise (deployed across executive branch per Feb 2026 announcement — first state to do so), Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: MA is the first state to deploy ChatGPT across the entire executive branch (Feb 2026). This means logs are being actively generated right now. File to Secretary of State's Supervisor of Records for enforcement if denied. EOTSS GenAI Policy (eff. 01/31/2025) governs use but doesn't explicitly address retention — use this: 'The EOTSS Enterprise GenAI Policy establishes governance requirements for AI use; conversation logs generated under this policy framework constitute public records under M.G.L. c.66.'",
    pitfalls="MA can be slow (6-12 mo typical). Supervisor of Records can order compliance within 10 days."
)): count += 1

if append_notes(ws, jur_map, "California - CDT (Poppy)", state_template(
    "Cal. Gov. Code § 6253(c)", "the California Public Records Act",
    "Poppy (California's state-approved vendor-agnostic GenAI platform where 'information never leaves California's trusted environment'), ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: Poppy is CA's centralized GenAI platform — name it specifically. CA is notoriously slow on PRA (8-18 mo). File via CDT webform (cdt.ca.gov/public-records-request/). EO N-12-23 directed GenAI guidelines but didn't explicitly address records retention — argue that Poppy conversations are 'writing' under CPRA's broad definition regardless.",
    pitfalls="CA agencies routinely delay. Consider filing with multiple agencies simultaneously. CDT may redirect to individual agencies."
)): count += 1

if append_notes(ws, jur_map, "Minnesota", state_template(
    "Minn. Stat. § 13.03, subd. 3", "the Minnesota Government Data Practices Act",
    "Microsoft Copilot Chat (approved for state employees via MNIT, green shield version for non-public data), ChatGPT, Google Gemini, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: STRONG PRECEDENT — Minneapolis PD produced 37,000 Copilot items despite claiming 'ChatGPT is not a city application.' This proves Copilot data is extractable and producible in MN. Specify Copilot Chat specifically (only tool approved for non-public data per MNIT). Expect redactions per MN Data Practices Act §13.43 (personnel data). TAIGA (Transparent AI Governance Alliance) partnership with MNIT = documented governance framework. File to specific agency's 'responsible authority' per statute.",
    pitfalls="MN Data Practices Act classifies some data as 'not public' — expect redactions for personnel data, security data. But conversation content about government business should be public."
)): count += 1

if append_notes(ws, jur_map, "New York", state_template(
    "N.Y. Pub. Off. Law § 89(3)", "the New York Freedom of Information Law (FOIL)",
    "ITS AI Pro (powered by Google Gemini — NOT Copilot), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: NYS-P24-001 explicitly requires 'data retention settings that follow federal and state standards' for AI systems — cite this in your request. 2025 Comptroller audit (2023-S-50) found agencies lacked AI testing procedures — cite as evidence that AI governance gaps make FOIL oversight necessary. File via Open FOIL NY portal (ny.gov/programs/open-foil-ny). Note: ITS AI Pro uses Gemini, not Copilot — name the right platform. RAISE Act (eff. March 19, 2026) adds additional transparency requirements.",
    pitfalls="Large state workforce = potentially voluminous response. Consider targeting specific agencies (ITS, DMNA, DOCCS) rather than all-state."
)): count += 1

if append_notes(ws, jur_map, "Pennsylvania", state_template(
    "65 P.S. § 67.901", "the Pennsylvania Right-to-Know Law",
    "ChatGPT Enterprise (175-employee pilot across 14 agencies), Microsoft 365 Copilot, and any other generative AI chatbot tool",
    """SPECIFIC NOTES: CRITICAL — BAD PRECEDENT EXISTS. PA OOR ruled Feb 2026 (WITF case) that AI chat logs may be exempt as 'notes and working papers used solely for that official's own personal use' and 'internal, predecisional deliberations.' Philadelphia also rejected Canelon's request as 'insufficiently specific.'

COUNTER-STRATEGY: (1) Request AI outputs that were incorporated into official communications, reports, policies, or decisions — these can't be 'personal working papers.' (2) Specify exact departments from the 14-agency pilot. (3) Narrow date range. (4) Cite that every AI chat remains 'presumptively public' unless agency proves exemption per the OOR's own analysis. (5) Add: 'This request specifically includes AI-generated content that was used, referenced, or incorporated into any official agency communication, report, policy document, or decision, which cannot constitute personal working papers.'

FILE to each agency's Open Records Officer directly. Appeal denials to OOR within 15 business days.""",
    pitfalls="OOR Feb 2026 ruling is the biggest obstacle. Frame request to target AI outputs used in official business, not personal drafts."
)): count += 1

if append_notes(ws, jur_map, "Connecticut", state_template(
    "Conn. Gen. Stat. § 1-206(a)", "the Connecticut Freedom of Information Act",
    "ChatGPT, Microsoft 365 Copilot, and any other generative AI chatbot tool (small-scale trials with ~260 employees trained)",
    "SPECIFIC NOTES: CT has an independent FOI Commission that can ORDER disclosure — strong enforcement mechanism. Reference Policy AI-01 Responsible AI Framework (Feb 2024) and SB 1103 requiring AI inventory and impact assessments. Request the public AI inventory itself as well. Small-scale deployment (~260 employees) means response should be manageable. File directly to agency — each has own contact.",
    pitfalls="Small deployment may mean few responsive records. FOI Commission is a strong appeal mechanism."
)): count += 1

if append_notes(ws, jur_map, "New Jersey", state_template(
    "N.J.S.A. 47:1A-5(i)", "the New Jersey Open Public Records Act (OPRA)",
    "NJ AI Assistant (15,000+ state employees, 300,000+ sessions, 1M+ prompts as of Feb 2026, hosted on state infrastructure), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: NJ has the nation's FIRST state-level AI/ML records retention guidance (DORES/Treasury guidelines). Use this in your request: 'Per NJ Treasury DORES Guidelines for Developing Retention and Disposition Policies for AI/ML Systems, documentary records from AI/ML systems are subject to retention scheduling.' NJ AI Assistant has 1M+ prompts from 15,000+ employees — MASSIVE volume. Very tight 7-day OPRA deadline works in your favor. AG says non-residents can file. File via nj.gov/opra online form.",
    pitfalls="7-day deadline is tight — agency may claim 'extraordinary circumstances' for extension. 1M+ prompts = huge volume, agency may seek to narrow scope."
)): count += 1

if append_notes(ws, jur_map, "Colorado", state_template(
    "C.R.S. § 24-72-203(3)(b)", "the Colorado Open Records Act (CORA)",
    "Google Gemini Advanced (150-person pilot across 18 agencies), Microsoft 365 Copilot, and any other generative AI chatbot tool. NOTE: Free ChatGPT is PROHIBITED on state devices per OIT directive",
    "SPECIFIC NOTES: OIT prohibits free ChatGPT — so any AI use should be through approved enterprise tools (Gemini Advanced pilot). Reference the Statewide GenAI Policy requiring OIT risk assessment for all GenAI efforts. File to agency records custodian. Guidance at sos.state.co.us. Denver, Douglas Co, Arapahoe Co are all GovAI participants.",
    pitfalls="OIT website returns 403 errors — specific retention language not publicly verifiable. Agency may claim GenAI policy is not a public record."
)): count += 1

if append_notes(ws, jur_map, "Oregon", state_template(
    "ORS 192.329(2)–(5)", "the Oregon Public Records Law",
    "Microsoft Copilot (approved by EIS for state employees, Sept 2024), ChatGPT Enterprise, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: EIS Interim AI Guidelines v1.5 (Dec 2024) require AI-generated content to be 'clearly labeled as such.' Reference this: 'Per EIS Interim AI Guidelines v1.5, AI-generated content used in official state capacity must be labeled — these labeled records are subject to Oregon Public Records Law.' Portland Police returned no responsive docs to Scott's Oct 2025 request. File to agency records officer. Appeals to state AG.",
    pitfalls="Portland PPB had no responsive docs — try state-level agencies which have formal Copilot deployments."
)): count += 1

if append_notes(ws, jur_map, "Texas", state_template(
    "Tex. Gov't Code § 552.221", "the Texas Public Information Act",
    "Microsoft 365 Copilot (deployed to TxDOT's 9,400+ employees), ChatGPT Enterprise, Google Gemini, and any other generative AI chatbot tool",
    """SPECIFIC NOTES: STRONGEST STATE PRECEDENT. Fort Worth completed with full ChatGPT records. TX AG Decision OR2026-006497 authorized FULL DISCLOSURE without redactions — cite this: 'Per Texas Attorney General Open Records Decision OR2026-006497, ChatGPT conversation logs are subject to full disclosure under the Texas Public Information Act.'

TxDOT reported 1,931 ChatGPT users, 4,527 Copilot accounts, 136,664 files. 912 files released in batches. TSLAC guidance (June 2024) explicitly states AI-generated records should be classified and retained identically to human-generated records.

File to agency Public Information Officer/Coordinator. TX AG rulings are binding precedent for all state agencies.""",
    pitfalls="Large agencies (TxDOT) may have massive volumes requiring phased production. Budget for potential copying costs."
)): count += 1

if append_notes(ws, jur_map, "Illinois", state_template(
    "5 ILCS 140/3", "the Illinois Freedom of Information Act",
    "ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool deployed under DoIT AI Policy v2",
    "SPECIFIC NOTES: DoIT AI Policy v2 (eff. April 1, 2025) governs all agency AI use. HB 3773 requires records of AI system use in employment decisions to be retained for FOUR YEARS — cite if relevant. Chicago Mayor's Office returned no responsive docs (Scott, Oct 2025). CPD request still pending. File to agency FOIA officer. Appeals to Public Access Counselor (PAC).",
    pitfalls="Chicago-level requests may get 'no responsive docs.' State-level agencies more likely to have enterprise deployments."
)): count += 1

if append_notes(ws, jur_map, "Florida", state_template(
    "Fla. Stat. § 119.07(1)(a)", "the Florida Public Records Act (Sunshine Law)",
    "ChatGPT Enterprise, Microsoft 365 Copilot, AHCA internal AI model, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: FL has one of strongest sunshine laws — constitutional right of access. But NO state-level AI use policy found, and FL Supreme Court had no responsive ChatGPT docs (Park, Oct 2024). SB 1118 may create new data center exemptions — FILE BEFORE this takes effect. Miami-Dade County has its own AI policy (Directive 231203) — better target than state. File to agency custodian of records — no central portal.",
    pitfalls="No state AI policy means agencies may claim no records exist. Target agencies known to use AI (AHCA). Miami-Dade is the strongest FL target."
)): count += 1

if append_notes(ws, jur_map, "Utah", state_template(
    "Utah Code § 63G-2-204", "the Government Records Access and Management Act (GRAMA)",
    "Google Gemini (enterprise-wide deployment to 15,000-16,000 state employees), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool deployed under DTS Enterprise GenAI Policy 4000-0008",
    "SPECIFIC NOTES: Very AI-forward state — 15,000-16,000 employees have Gemini access. File via centralized GRAMA portal (openrecords.utah.gov). Reference DTS Enterprise GenAI Policy 4000-0008 in your request. Utah AI Policy Act (SB 149, 2024) establishes governance framework.",
    pitfalls="Gemini admin export mechanisms may differ from ChatGPT. Specify Google Workspace admin console activity logs."
)): count += 1

if append_notes(ws, jur_map, "Georgia", state_template(
    "O.C.G.A. § 50-18-71(b)(1)(A)", "the Georgia Open Records Act",
    "ChatGPT, Google Gemini, Anthropic Claude, Microsoft Copilot, and any other generative AI chatbot tool",
    """SPECIFIC NOTES: STRONG LEGAL BASIS. SS-23-002 'AI Responsible Use' explicitly requires agencies to 'Keep records. Save prompts, outputs, and who reviewed the content.' Cite this in your request: 'Per GTA Standard SS-23-002, agencies are required to maintain records of AI tool usage including purpose, inputs, outputs, and actions taken based on AI-generated results. I request all such records.'

Also cite 'Red Light, Green Light' guidelines (Aug 2025) requiring agencies to label AI outputs with tool name, prompt, and reviewer identity. These labeled records are subject to the Georgia Open Records Act.

Atlanta PD returned no responsive docs (Scott, Oct 2025). Try state agencies and city administration/IT rather than PD.""",
    pitfalls="SS-23-002 mandates record-keeping but some agencies may not have implemented it yet. Atlanta PD had no records."
)): count += 1

if append_notes(ws, jur_map, "North Carolina", state_template(
    "N.C.G.S. § 132-6(a)", "the North Carolina Public Records Act",
    "Microsoft Copilot for M365 (statewide pilot per Feb 2025 policy), ChatGPT Enterprise, Google Gemini, and any other generative AI chatbot tool",
    """SPECIFIC NOTES: STRONG LEGAL FRAMEWORK. UNC School of Government legal analysis (March 2026) confirmed both AI prompts ('records made by officials') and AI outputs ('records received') are public records under N.C.G.S. Chapter 132. Cite: 'Per UNC School of Government analysis and N.C.G.S. Chapter 132, AI prompts and outputs constitute public records regardless of physical form or characteristics.'

Also cite Gray Media Group, Inc. v. City of Charlotte (290 N.C. App. 384, 2023): records on third-party servers (like ChatGPT/Copilot platforms) qualify as public records if agency has 'actual or constructive possession.'

State Copilot for M365 Policy (Feb 26, 2025) governs usage. Retention per NC DNCR disposition schedules based on content. Raleigh (Kelly Kauffman, Sept 2025) returned no responsive docs. File to agency custodian.""",
    pitfalls="Some agencies may claim AI outputs are 'transitory records' or 'drafts.' Counter with UNC legal analysis that records used in official capacity aren't transitory."
)): count += 1

if append_notes(ws, jur_map, "Ohio", state_template(
    "Ohio Rev. Code § 149.43(B)", "the Ohio Public Records Act",
    "ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool deployed under DAS Policy IT-17",
    "SPECIFIC NOTES: DAS IT-17 explicitly states 'only data that is public record should be entered by state employees into generative AI systems.' Use this: 'Per DAS Policy IT-17, data entered into generative AI systems is restricted to public records. Therefore, all AI conversation logs contain only public record data and are fully disclosable.' IT-17 also requires GenAI outputs to be annotated with 'at least the Generative AI technology used and a description of how it was used' — these annotations are records. File to agency records custodian. Appeals via writ of mandamus.",
    pitfalls="Ohio's 'only public record data' restriction actually helps your request — it means the content should be fully releasable."
)): count += 1

if append_notes(ws, jur_map, "Michigan", state_template(
    "MCL § 15.235(2)", "the Michigan Freedom of Information Act",
    "AWS AI Code Generator (for DTMB developers), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: DTMB AI Guidelines state AI tools are 'already governed by existing State of Michigan policies, standards, and procedures' — argue that existing FOIA framework applies without question. DTMB hired Chief Privacy Officer (Jan 2025) for AI governance. External tools require Authority to Operate via MiSAP — request ATO records as well to identify which tools are approved. File to agency FOIA coordinator.",
    pitfalls="General DTMB Policy 0910.02 governs records retention — no AI-specific schedule. Agency may claim AI logs aren't 'records' under MCL 15.232."
)): count += 1

if append_notes(ws, jur_map, "Wisconsin", state_template(
    "Wis. Stat. § 19.35(4)(a)", "the Wisconsin Public Records Law",
    "DSPS Maverick AI (Google Cloud + MTX), Microsoft 365 Copilot, ChatGPT Enterprise, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: DET AUP (eff. March 10, 2025) explicitly states communications using state systems 'create public records that must be retained and produced if requested.' Cite this. The AUP also states content generated using AI 'may constitute a public record' — argue it clearly DOES when used for government business. DET recommends 'living AI Tool Registry' with retention schedules. Request the registry itself. File to legal custodian of records. Appeals to AG DOJ.",
    pitfalls="'May constitute a public record' is hedging language — some agencies may argue AI drafts that aren't used don't qualify."
)): count += 1

if append_notes(ws, jur_map, "New Hampshire", state_template(
    "RSA 91-A:4, IV", "the New Hampshire Right-to-Know Law",
    "Lexis+ AI (DOJ legal research), Microsoft 365 Copilot, ChatGPT, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: RSA 5-D 'Use of AI by State Agencies' (eff. July 1, 2024) requires GenAI content disclosure unless human-reviewed. Cite: 'Per RSA 5-D, generative AI content must be accompanied by disclosure that it was generated by AI — these disclosure records and the underlying AI conversations are subject to RSA 91-A.' Kauffman's mass filing to 20+ NH cities (Dec 2025): only Nashua provided records. Most cities had no responsive docs. Target state agencies (subject to RSA 5-D) rather than municipalities. File to agency custodian. Appeals to Office of Right to Know.",
    pitfalls="Small state with early-stage AI deployment — expect low volume of responsive records. Most municipal-level requests returned nothing."
)): count += 1

# New states added in this tracker
if append_notes(ws, jur_map, "Washington", state_template(
    "RCW 42.56", "the Washington Public Records Act",
    "Microsoft 365 Copilot, ChatGPT Enterprise, Google Gemini, and any other generative AI chatbot tool deployed under EO 24-01 (Jan 30, 2024)",
    """SPECIFIC NOTES: HIGHEST PRIORITY STATE. WA PRA is one of the strongest in the US (5-30 day typical response). WA Secretary of State issued two advice sheets (June 2024) declaring AI inputs/outputs are public records. 6 WA cities already in tracker with proven precedents (Kent 16 days, Bellingham/Everett thousands of pages, Spokane 9 sessions, Seattle multiple requests).

Cite WA SOS advice sheets: 'Per Washington Secretary of State guidance (June 2024), generative AI inputs and outputs created by government employees constitute public records under RCW 42.56.'

EO 24-01 directed AI framework development. File to each state agency's public records officer. WA has strong penalties for non-compliance.""",
    pitfalls="State agencies may be slower than cities. Multiple successful city-level precedents exist."
)): count += 1

if append_notes(ws, jur_map, "Missouri", state_template(
    "RSMo 610.023", "the Missouri Sunshine Law",
    "Google Gemini (per EO 26-02 directing AI framework), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: 3-DAY DEADLINE is one of the tightest in the country. Penalties: up to $5,000 for purposeful violations (RSMo 610.027) + attorney fees. EO 26-02 (Jan 2026) directs AI framework. Kansas City PD rejected Kauffman's request claiming ChatGPT histories are 'not a record we compile or maintain.' St. Louis Metro PD said 'Chat GPT is not authorized on department devices.' Target state agencies with known Gemini/Copilot deployments rather than local PD. File to custodian — no centralized portal.",
    pitfalls="Local PDs rejected as 'not a record we compile.' State agencies may have more formal AI deployments. 3-day deadline gives strong leverage."
)): count += 1

if append_notes(ws, jur_map, "Virginia", state_template(
    "Code of Virginia § 2.2-3704", "the Virginia Freedom of Information Act",
    "ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool deployed under EO 30 (Jan 18, 2024) and VITA AI Policy Standard",
    "SPECIFIC NOTES: Governor Youngkin's EO 30 established AI governance. VITA AI Policy Standard requires retention of 'the specific AI technologies and tools utilized.' Cite: 'Per VITA AI Policy Standard, agencies shall retain records of specific AI technologies utilized — conversation logs from these tools constitute such records under Virginia FOIA.' Civil penalties: $500-$2,000 first offense; $2,000-$5,000 subsequent (§ 2.2-3714). Active Chief Data Officer. File to FOIA officer at each public body.",
    pitfalls="5-12 working day response time. VITA policy requires retention of AI tool records but may not mean individual chat logs."
)): count += 1

if append_notes(ws, jur_map, "Indiana", state_template(
    "IC 5-14-3-3", "the Indiana Access to Public Records Act (APRA)",
    "ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool deployed under AI Policy v1.1 (Dec 2024) via OCDO",
    "SPECIFIC NOTES: No fees for electronic records is very favorable for requesting large volumes of AI chat exports. Active Public Access Counselor for enforcement. AI Policy v1.1 (Dec 2024) via Office of Chief Data Officer addresses data governance for AI. File to specific agency. Phone/email/mail all acceptable.",
    pitfalls="1-2 weeks for simple requests, 2-6 weeks for complex. Early-stage AI deployment may mean limited records."
)): count += 1

if append_notes(ws, jur_map, "Montana", state_template(
    "Montana Constitution Art. II, Sec. 9", "the Montana Right to Know (Constitutional right)",
    "ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: CONSTITUTIONAL right of access (one of ~6 states). Strong judicial enforcement leverage. HB 178 (signed May 2025) requires govt entities to disclose AI use in public-facing contexts. Commissioner of Political Practices documented ChatGPT use. File via centralized Office of Public Information Request portal. 8 free hours of staff time.",
    pitfalls="Small state with early-stage deployment. Constitutional right provides strong leverage for appeal."
)): count += 1

if append_notes(ws, jur_map, "Nebraska", state_template(
    "Neb. Rev. Stat. § 84-712", "the Nebraska Public Records Statutes",
    "ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool deployed under NITC AI Policy section 8-609 (Nov 2024)",
    "SPECIFIC NOTES: Historically ranked #1 FOI law nationally by BGA/IRE. 8 free hours of staff time + no attorney fees = very cost-effective. NITC AI policy section 8-609 (Nov 2024) explicitly references record-keeping. Cite: 'Per NITC AI Policy section 8-609, AI tools are subject to state governance requirements — conversation logs generated under this policy constitute public records under Neb. Rev. Stat. § 84-712.' File to custodian of specific record.",
    pitfalls="4-10 business day response. Small state but strong legal framework."
)): count += 1

if append_notes(ws, jur_map, "Maryland", state_template(
    "Md. Code Ann., Gen. Prov. § 4-201", "the Maryland Public Information Act (MPIA)",
    "ChatGPT Enterprise (12,500+ active GenAI users statewide), Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool deployed under Governor Moore's EO 01.01.2024.02 and AI Governance Act SB818 (July 2024)",
    "SPECIFIC NOTES: 12,500 active GenAI users = substantial volume. Governor's AI EO + AI Governance Act SB818 establish governance framework. Todd Feathers reported Baltimore findings. Cite: 'Per Executive Order 01.01.2024.02 and AI Governance Act SB818, the State has deployed generative AI tools to 12,500+ employees under a governance framework — conversation logs from these tools are public records under the MPIA.' File to specific agency — many have online portals.",
    pitfalls="10-30 day response. Baltimore (city) already returned 'no responsive docs' to Feathers. State agencies more likely to have enterprise deployments."
)): count += 1

if append_notes(ws, jur_map, "Tennessee", state_template(
    "TCA § 10-7-503", "the Tennessee Public Records Act",
    "ChatGPT Enterprise (1,000 licenses, spring 2025), Microsoft 365 Copilot, and any other generative AI chatbot tool deployed under Enterprise AI Policy 200-POL-007 and GenAI Policy ISC 345",
    "SPECIFIC NOTES: 1,000 ChatGPT Enterprise licenses = documented deployment. Enterprise AI Policy 200-POL-007 (approved 9/26/2025) + GenAI Policy ISC 345 govern usage. Constitutional basis for open records. Office of Open Records Counsel (OORC) provides free guidance. Cite: 'Per Enterprise AI Policy 200-POL-007 and GenAI Policy ISC 345, the State has deployed 1,000 ChatGPT Enterprise licenses — conversation logs from these accounts are public records under TCA § 10-7-503.' File to Public Record Request Coordinator (PRRC) at each agency.",
    pitfalls="7-21 business day response. Nashville has ISM-20 AI policy — consider filing to Nashville separately."
)): count += 1

# ============================================================
# US CITIES
# ============================================================

def city_template(statute_cite, statute_name, platforms, specific_notes, pitfalls=""):
    base = f"""FRAMING STRATEGY: Cities are where 'too broad' rejections are most common. Counter by naming exact platforms and describing the export mechanism. 'All employees' scope is fine — the key is specificity about the RECORD, not the PEOPLE.

{f'KNOWN PITFALL: {pitfalls}' if pitfalls else ''}

REQUEST TEMPLATE:
"Pursuant to {statute_name} ({statute_cite}), I request the following records:

1. All generative AI conversation logs — including user prompts/inputs and AI-generated outputs/responses — created by city employees using any of the following platforms: {platforms}. Time period: January 1, 2024 to present.

2. Administrative usage logs or reports showing the number of employees with access to, and usage statistics for, each generative AI platform deployed by the city during this period.

3. For ChatGPT Enterprise accounts: the data export available through the admin console (ZIP archive containing conversation.html files and metadata JSON). For Microsoft 365 Copilot: the conversation history logs available through the M365 admin center or Purview compliance portal.

This request encompasses all city employees and contractors with access to these tools. I am not requesting the tools' training data or source code — only the conversation logs between users and AI systems.

If any records are withheld, please cite the specific statutory exemption and release all segregable portions."

{specific_notes}"""
    return base

# WA cities (strongest precedents)
for wa_city, wa_platforms, wa_notes, wa_pitfalls in [
    ("Seattle, WA",
     "ChatGPT Enterprise, Microsoft Copilot, and any other generative AI chatbot tool approved under POL-211 (formerly POL-209)",
     """SPECIFIC NOTES: PROVEN PRECEDENT — multiple successful requests. Cite POL-211 (formerly POL-209):
'Per City of Seattle AI Policy POL-211, §6.1: All records generated by GenAI vendors may be considered public records and must be disclosed upon request. §6.2: All GenAI solutions shall support retrieval and export of all prompts and outputs. §6.3: Employees must maintain records of inputs, prompts, and outputs.'

Rose Terse completed April 2023 (2 installments), Education & Early Learning completed Oct 2024 (9 files from 5 employees). File via seattle.gov/public-records. WARNING: SPD backlog is 4,500+ open requests — expect 6-12 month wait for police-specific requests. Non-police departments may be faster.""",
     "SPD backlog is massive. File to non-police departments for faster response. Request both ChatGPT exports AND Copilot logs separately."),

    ("Kent, WA",
     "ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: FASTEST KNOWN COMPLETION — Rose Terse obtained 6 files in ~16 days (July 2023). ChatGPT histories from 5 officers released, fees waived. Kent PD had 'no effective policy in place.' File via kentwa.gov/PublicRecords or email publicrecords@kentwa.gov.",
     "No known pitfalls — excellent track record."),

    ("Spokane, WA",
     "AI Language Translator, ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: Rose Terse obtained 9 ChatGPT sessions (completed March 2024, 8 months). No ChatGPT policies existed. Requester had to follow up about incomplete conversations — verify completeness. File via my.spokanecity.org/administrative/public-records.",
     "8-month completion time. Verify completeness of responsive records."),

    ("Bellingham, WA",
     "ChatGPT (extensively used — thousands of pages of logs obtained), Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: STRONGEST CITY PRECEDENT — Nate Sanford/KNKX obtained thousands of pages. Bellingham was 'fastest and most responsive' of ~12 WA cities. MAJOR SCANDAL (Jan 2026): City staffer used ChatGPT to draft biased $2.7M contract requirements — 16 requirements matched ChatGPT output verbatim. City now has heightened awareness of AI records obligations. File via cob.org/gov/public-records.",
     "Post-scandal, city may be more cautious/thorough in producing records. Strong precedent in your favor."),

    ("Everett, WA",
     "ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: PROVEN PRECEDENT — Nate Sanford/KNKX obtained thousands of pages. Everett was among 'fastest and most responsive.' Reporting prompted editorial calling for cautious AI governance. File via everettwa.mycusthelp.com GovQA portal.",
     "No known pitfalls — excellent track record."),

    ("Tacoma, WA",
     "ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: Rose Terse filed July 2023 — Tacoma PD returned 'No Responsive Documents.' GenAI Guidelines published by IT Dept. Try non-police departments (IT, planning, communications) which may have AI deployments. File via tacoma.gov/government/departments/public-records.",
     "PD had no responsive docs. Target non-police departments."),

    ("Bellevue, WA",
     "Govstream.ai AI Smart Assistant (built on Microsoft infrastructure), Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: Govstream.ai is Bellevue's named internal AI tool — reference it specifically. File via bellevuewa.gov/city-government/departments public records page.",
     "No known FOIA precedent. Named tool provides specificity."),
]:
    if append_notes(ws, jur_map, wa_city, city_template("RCW 42.56.520", "the Washington Public Records Act", wa_platforms, wa_notes, wa_pitfalls)): count += 1

# California cities
for ca_city, ca_platforms, ca_notes, ca_pitfalls in [
    ("San Francisco, CA",
     "Microsoft 365 Copilot Chat (GPT-4o, deployed to all ~30K employees), ChatGPT Enterprise, and any other generative AI chatbot tool",
     """SPECIFIC NOTES: Reference Chapter 22J (Ordinance No. 288-24) and GenAI Guidelines (July 2025): 'Per SF GenAI Guidelines, content entered into or generated by GenAI tools may constitute public records subject to disclosure under CPRA and the Sunshine Ordinance.' Use the Chapter 22J AI Use Inventory (data.sfgov.org) to identify which departments actively use AI. Copilot deployed to all ~30K employees = massive volume.

Todd Feathers (Aug 2024) to SFPD: No Responsive Documents. Patrick O'Doherty (Jan 2026) for SFPD AI policies: AWAITING. Target non-police departments. File via sanfrancisco.nextrequest.com (NextRequest portal).""",
     "SFPD had no responsive docs. CA is slow (3-6 mo). Use Chapter 22J inventory to target departments with known AI use."),

    ("San Jose, CA",
     "ChatGPT Teams (89 licenses, $35K+), Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: CIO Khaled Tawfik is GovAI board chair — city is AI-forward. 89 ChatGPT Teams licenses documented. Cite: 'The City procured 89 ChatGPT Teams licenses — conversation logs from these accounts are public records under CPRA.' File via sanjoseca.govqa.us (GovQA portal) or email PublicRecords@sanjoseca.gov.",
     "CA agencies routinely delay. Known procurement gives specificity."),

    ("Oakland, CA",
     "ChatGPT (confirmed via Oaklandside investigation Oct 2025, including '311 chatbots to ChatGPT-written emails'), Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: Oaklandside confirmed AI use including ChatGPT-written emails. ITD Interim AI Security Guidelines (Dec 2024) require citations when AI is used in public records. Cite: 'Per ITD Interim AI Security Guidelines, staff must include citations when AI is used in reports, memos, or other public records — I request all such AI-generated content and the underlying conversation logs.' File via oaklandca.nextrequest.com.",
     "Confirmed AI use but no direct FOIA precedent for chat logs. Citation requirement creates paper trail."),

    ("San Diego, CA",
     "ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool ($360K budgeted for AI, CIO Jonathan Behnke is GovAI board member)",
     "SPECIFIC NOTES: $360K budgeted for AI. CIO is GovAI board member. File via sandiego.gov/communications/public-records.",
     "CA agencies routinely delay. Budget allocation proves AI deployment."),

    ("Long Beach, CA",
     "Holly AI platform (hiring, Oct 2024 pilot), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: Holly AI platform for hiring is documented. File via longbeach.gov/cityclerk or longbeachca.govqa.us GovQA portal.",
     "Holly AI may be HR-specific — request broader employee AI tools as well."),

    ("Alameda County, CA",
     "GPT-4 Board Conversational AI Assistant (summarizes Board of Supervisors agendas), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: GPT-4 Board AI Assistant is a named public-facing tool. File via alamedacountyca.nextrequest.com.",
     "Named tool provides specificity. File via NextRequest portal."),

    ("San Mateo County, CA",
     "Innovaccer Healthcare AI Platform (July 2024), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: Healthcare AI platform documented. File via smcgov.org/ceo/public-records-act-request form.",
     "Healthcare AI may not have conversational logs in the traditional sense."),

    ("Placer County, CA",
     "Microsoft Copilot ($45K budgeted FY 2024-2025), ChatGPT Enterprise, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: $45K budgeted for Copilot. Dedicated AI analyst = high likelihood of structured AI use. File via placercounty.nextrequest.com.",
     "Budget allocation proves Copilot deployment. Dedicated AI analyst suggests organized records."),
]:
    if append_notes(ws, jur_map, ca_city, city_template("Cal. Gov. Code § 6253(c)", "the California Public Records Act", ca_platforms, ca_notes, ca_pitfalls)): count += 1

# Texas cities
for tx_city, tx_platforms, tx_notes, tx_pitfalls in [
    ("Houston, TX",
     "ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: No confirmed GenAI tool deployments for city employees, but TX precedent is strong: Fort Worth completed with full ChatGPT records, TX AG Decision OR2026-006497 authorized full disclosure, TxDOT had 1,931 ChatGPT users. Cite TX AG ruling. File via houstontx.gov/pia.html.",
     "No confirmed AI deployment — may get 'no responsive docs.' TX AG ruling provides strong leverage if records exist."),

    ("Dallas, TX",
     "Hazel AI (procurement), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: Hazel AI for procurement is documented — first major TX city to use AI for procurement. TX AG Decision OR2026-006497 authorizes full disclosure. File via dallascityhall.com/government/citysecretary.",
     "Hazel AI is procurement-focused — may not have conversational employee logs. Request broadly."),

    ("Austin, TX",
     "Archistar AI (residential permit review, Oct 2024), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: Archistar AI for permits is documented. TX AG Decision OR2026-006497 authorizes full disclosure. File via austintexas.gov/services/submit-public-information-request.",
     "Archistar is specialized — also request general employee AI tools."),
]:
    if append_notes(ws, jur_map, tx_city, city_template("Tex. Gov't Code § 552.221", "the Texas Public Information Act", tx_platforms, tx_notes, tx_pitfalls)): count += 1

# Other US cities
if append_notes(ws, jur_map, "New York City", city_template(
    "N.Y. Pub. Off. Law § 89(3)", "the New York Freedom of Information Law (FOIL)",
    "ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool",
    """SPECIFIC NOTES: TWO PENDING REQUESTS — Joey Scott to Mayor's Office (deadline extended to April 10, 2026), Brandon Galbraith to OTI (Feb 2026). Jason Koebler/VICE obtained ChatGPT-related materials from NYC DOE in 15 months.

Largest US city workforce — file at a860-openrecords.nyc.gov (NYC OpenRecords portal). Target specific agencies: OTI (most likely to have AI tools), NYPD, DoITT. OTI GenAI Use Guidance and AI Principles exist but don't explicitly address retention.

No confirmed citywide employee GenAI deployment — but Copilot likely available through M365. If agency claims no records, request IT procurement records for AI tools as follow-up.""",
    "Pending requests suggest records may exist. 15-month Koebler timeline shows NYC is slow. Target OTI and DoITT specifically."
)): count += 1

if append_notes(ws, jur_map, "Washington, DC", city_template(
    "D.C. Code § 2-532(c)", "the DC Freedom of Information Act",
    "Microsoft Copilot Chat (under DC M365 environment), ChatGPT Enterprise, and any other generative AI chatbot tool",
    """SPECIFIC NOTES: WARNING — REJECTION PRECEDENT. Joey Scott's request to MPD was rejected as 'inadequate description,' 'too broad (4-month period),' and 'research rather than records access.'

COUNTER-STRATEGY: Be extremely specific. Name Microsoft Copilot Chat by name. Reference OCTO AI/ML Governance Policy requiring 'adequate auditing and logging mechanisms' — these logs are the records you're requesting. Narrow to specific departments. Add: 'I am requesting the auditing and logging records that OCTO AI/ML Governance Policy requires agencies to maintain for AI/ML technology usage. This is a request for specific, identifiable records, not research.'

File via myfoiadc.govqa.us (centralized DC FOIA portal). DC was first major US city to require responsible AI training — training records may also contain useful information about which tools are deployed.""",
    "MPD rejected as 'inadequate description.' Must be extremely specific about platforms, departments, date ranges. Reference OCTO's own logging requirements."
)): count += 1

if append_notes(ws, jur_map, "Kansas City, MO", city_template(
    "Mo. Rev. Stat. § 610.023", "the Missouri Sunshine Law",
    "Maya AI chatbot (named city deployment), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
    """SPECIFIC NOTES: WARNING — REJECTION PRECEDENT. KCPD rejected claiming ChatGPT histories are 'not a record we compile or maintain.' St. Louis Metro PD said 'Chat GPT is not authorized on department devices.'

COUNTER-STRATEGY: (1) Request Microsoft Copilot data from IT department — enterprise M365 logs are centrally managed. (2) Reference Maya AI chatbot by name. (3) Request IT department records of AI tool procurement/deployment. (4) Add: 'I request all conversation logs from any generative AI tool available through the city's Microsoft 365 environment, including but not limited to Microsoft Copilot, accessible via the M365 admin center.'

3-day Sunshine Law deadline gives strong leverage. File via kcmo.gov/i-want-to/submit-a-sunshine-request.""",
    "PD claimed ChatGPT is 'not a record we compile.' Request Copilot data from IT instead. 3-day deadline is very tight — may get extension but gives leverage."
)): count += 1

if append_notes(ws, jur_map, "Chicago, IL", city_template(
    "5 ILCS 140/3", "the Illinois Freedom of Information Act",
    "ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: Joey Scott (Oct 2025): Mayor's Office returned No Responsive Documents. CPD request still pending (narrowed to command staff). GenAI Roadmap recommends vetted platforms. File via chicago.gov/publicrecords. Target IT department and departments with known tech initiatives rather than Mayor's Office or PD.",
    "Mayor's Office had no responsive docs. CPD pending. Try IT/innovation departments."
)): count += 1

if append_notes(ws, jur_map, "Minneapolis, MN", city_template(
    "Minn. Stat. § 13.03, subd. 3", "the Minnesota Government Data Practices Act",
    "Microsoft Copilot (37,000+ items collected from PD alone), ChatGPT, and any other generative AI chatbot tool",
    """SPECIFIC NOTES: STRONG PRECEDENT — Joey Scott obtained 37,000 Copilot items from PD despite city claiming 'ChatGPT is not a City of Minneapolis application.' Two batches of redacted Copilot records released (Dec 2025, Jan 2026).

KEY LESSON: City doesn't track ChatGPT but DOES have Copilot data. Request Copilot specifically: 'All Microsoft 365 Copilot conversation histories accessible through the city's M365 admin center or Microsoft Purview compliance portal.' Expect redactions per MN Data Practices Act §13.43.

File via minneapolismn.gov/government/government-data. Good pairing with MN state + St. Paul requests.""",
    "City claims ChatGPT isn't a city app. Request COPILOT data — 37,000 items proves it's available. Expect redactions per DPA §13.43."
)): count += 1

if append_notes(ws, jur_map, "Boston, MA", city_template(
    "M.G.L. c.66, § 10(b)", "the Massachusetts Public Records Law",
    "ChatGPT Enterprise, Microsoft 365 Copilot, Google Gemini, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: No city-provisioned enterprise GenAI tool confirmed. File via boston.gov/public-records or bostonma.govqa.us GovQA portal. If no responsive docs for chat logs, request IT procurement records for AI tools.",
    "No confirmed AI deployment — may get 'no responsive docs.' File for IT procurement records as fallback."
)): count += 1

if append_notes(ws, jur_map, "St. Paul, MN", city_template(
    "Minn. Stat. § 13.03, subd. 3", "the Minnesota Government Data Practices Act",
    "Axon Auto-Transcribe (police AI transcription), Microsoft 365 Copilot, ChatGPT, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: Axon Auto-Transcribe is documented for police. Minneapolis precedent (37,000 Copilot items) shows MN cities have extractable AI data. Request Copilot data specifically. Good pairing with Minneapolis and MN state requests. File via stpaul.gov/departments/city-clerk.",
    "Axon is specialized — also request general Copilot/ChatGPT logs."
)): count += 1

if append_notes(ws, jur_map, "Portland, OR (TriMet)", city_template(
    "ORS 192.329(2)–(5)", "the Oregon Public Records Law",
    "ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: Joey Scott (Oct 2025) to Portland Police Bureau: No Responsive Documents. Transit agency may have different use cases. Reference EIS Interim AI Guidelines requiring AI output labeling. File via trimet.org/publicrecords or trimet.govqa.us. Try city administration/IT rather than police.",
    "PPB had no responsive docs. TriMet is a transit agency — different AI use cases possible."
)): count += 1

if append_notes(ws, jur_map, "Denver, CO", city_template(
    "C.R.S. § 24-72-203(3)(b)", "the Colorado Open Records Act (CORA)",
    "Microsoft Copilot Chat (for city/county staff), ChatGPT Enterprise, Google Gemini, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: Denver is a combined city/county. Microsoft Copilot Chat for city/county staff documented. OIT prohibits free ChatGPT statewide. GovAI participant. File to specific department CORA officer.",
    "No known FOIA precedent for AI chat logs. Copilot deployment provides specificity."
)): count += 1

if append_notes(ws, jur_map, "Atlanta, GA", city_template(
    "O.C.G.A. § 50-18-71(b)(1)(A)", "the Georgia Open Records Act",
    "Axon body-camera AI report summarization, ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: Joey Scott (Oct 2025) to Atlanta PD: No Responsive Documents (same-day closure). Reference state SS-23-002 requiring agencies to save prompts, outputs, and reviewer identity. Try city IT/administration departments rather than PD. Axon is specialized police tool. File via Open Records Officer Search Portal.",
    "PD returned no responsive docs same-day. Target non-police departments. Cite state SS-23-002."
)): count += 1

if append_notes(ws, jur_map, "Tempe, AZ", city_template(
    "A.R.S. § 39-121.01(D)(1)", "the Arizona Public Records Law",
    "ChatGPT (OpenAI, confirmed in use by city employees for writing/research), Microsoft 365 Copilot, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: ChatGPT use confirmed by city employees. File via tempe.justfoia.com/publicportal (JustFOIA portal).",
    "No known FOIA precedent. Confirmed ChatGPT use provides basis."
)): count += 1

if append_notes(ws, jur_map, "Tucson, AZ", city_template(
    "A.R.S. § 39-121.01(D)(1)", "the Arizona Public Records Law",
    "Axon Draft One (police body-cam report drafting), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
    "SPECIFIC NOTES: Axon Draft One for police body-cam reports is documented. File via tucsonaz.gov/Departments/Clerks/Public-Records.",
    "Axon is specialized — also request general employee AI tools."
)): count += 1

# Counties
for county, statute, statute_name, platforms, notes, pitfalls in [
    ("Cook County, IL", "5 ILCS 140/3", "the Illinois Freedom of Information Act",
     "Assessor's Office open-source ML models, ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: Assessor's Office uses open-source ML (publicly documented). File via cookcountyil.gov/service-groups/foia-requests.",
     "ML models may not produce conversational logs. Request Copilot/ChatGPT separately."),
    ("Miami-Dade County, FL", "Fla. Stat. § 119.07(1)(a)", "the Florida Public Records Act (Sunshine Law)",
     "AI tools per miamidade.gov/ai AI Roadmap (including Pawfect Match, 311 chatbot), ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: Has its own AI policy (Directive 231203) addressing data retention. Strong Sunshine Law. AI Roadmap at miamidade.gov/ai. File via miamidade.gov/global/publicrecords/search. STRONGEST FL TARGET given explicit AI policy.",
     "Best FL target. Has its own AI policy addressing retention."),
    ("Broward County, FL", "Fla. Stat. § 119.07(1)(a)", "the Florida Public Records Act (Sunshine Law)",
     "Microsoft Copilot, ChatGPT Enterprise, and any other generative AI chatbot tool (CIO Domenic DiLullo references Copilot and AI tools)",
     "SPECIFIC NOTES: CIO references Copilot. File via broward.org/OpenGovernment/prr.",
     "Limited confirmed deployment details."),
    ("Clark County, NV", "NRS § 239.0107", "the Nevada Public Records Act",
     "15-20 active AI pilots, ChatGPT Enterprise, Microsoft 365 Copilot, and any other generative AI chatbot tool",
     "SPECIFIC NOTES: 15-20 active AI pilots = substantial deployment. AI agents formally tested. File via clarkcountynv.gov/government/departments/public-records.",
     "Many pilots may not have persistent conversation logs. Request admin usage data as well."),
]:
    if append_notes(ws, jur_map, county, city_template(statute, statute_name, platforms, notes, pitfalls)): count += 1

# ============================================================
# INTERNATIONAL
# ============================================================

intl_notes = {
    "Sweden (10-15 agencies)": """FRAMING STRATEGY: Sweden has the world's strongest FOI since 1766. No ID needed. Email in English OK. Principle of public access (offentlighetsprincipen) covers all official documents.

REQUEST TEMPLATE (English OK):
"Under the principle of public access to official documents (offentlighetsprincipen, Tryckfrihetsförordningen 2 kap.), I request the following records from [AGENCY]:

All generative AI conversation logs — including user prompts/inputs and AI-generated outputs/responses — created by agency employees using ChatGPT, Microsoft 365 Copilot, Google Gemini, or any other generative AI chatbot tool. Time period: January 1, 2024 to present.

PM Kristersson has confirmed using ChatGPT and LeChat. I request records in electronic format."

SPECIFIC NOTES: File via handlingar.se (Alaveteli platform, free, covers all agencies). Can also email registrator@[agency].se directly. Response expected in days to weeks. PM's confirmed AI use establishes that records exist at the highest level.""",

    "United Kingdom (5-8 depts)": """FRAMING STRATEGY: UK FOI Act 2000 requires response within 20 working days. WhatDoTheyKnow platform makes filing trivial. Peter Kyle AI disclosure precedent exists.

REQUEST TEMPLATE:
"Under the Freedom of Information Act 2000, I request the following information from [DEPARTMENT]:

All generative AI conversation logs — including user prompts/inputs and AI-generated outputs/responses — created by department employees using Microsoft 365 Copilot (20,000 civil servant deployment across 12 departments, Sept-Dec 2024), ChatGPT, Google Gemini, or any other generative AI chatbot tool. Time period: January 1, 2024 to present.

I also request administrative usage statistics showing number of users and sessions."

SPECIFIC NOTES: 20,000 civil servants across 12 depts had Copilot access. Peter Kyle (DSIT Secretary) disclosed AI use. File via whatdotheyknow.com. Target Cabinet Office, DSIT, HMRC, Home Office specifically. 673-page Norwegian precedent shows Nordic/EU govts produce substantial records.""",

    "Canada (3-5 depts)": """FRAMING STRATEGY: PROXY NEEDED — Canadian ATIA requires Canadian citizens/residents or persons in Canada. Notoriously slow (6-18 mo). CANChat has 11,500 users across 20 departments.

REQUEST TEMPLATE (via proxy or if eligible):
"Under the Access to Information Act (RSC 1985 c.A-1), I request:

All generative AI conversation logs from [DEPARTMENT] employees using CANChat (Shared Services Canada platform, ~11,500 users across 20 departments), ChatGPT, Microsoft 365 Copilot, or any other generative AI tool. Time period: January 1, 2024 to present."

SPECIFIC NOTES: File via atip-aiprp.apps.gc.ca portal. Target TBS (Treasury Board Secretariat — manages CANChat), SSC, IRCC. CANChat is the named platform. Processing time 6-18 months. $5 CAD application fee.""",

    "Finland (3-5 agencies)": """FRAMING STRATEGY: English OK. Very strong Nordic FOI. Act on Openness of Government Activities (621/1999) applies.

REQUEST TEMPLATE (English OK):
"Under the Act on Openness of Government Activities (621/1999, § 14), I request from [AGENCY]:

All generative AI conversation logs created by agency employees using Microsoft 365 Copilot (Helsinki pilot 2024), ChatGPT, or any other generative AI chatbot tool. Time period: January 1, 2024 to present."

SPECIFIC NOTES: File by email to kirjaamo@[agency].fi. Target DVV (Digital and Population Data Services Agency), Vero (Tax Administration), Kela. Helsinki M365 Copilot pilot documented. Response expected in 2-6 weeks.""",

    "Norway (2-3 agencies)": """FRAMING STRATEGY: Very fast (1-4 weeks). Offentleglova (2006) provides strong access rights. einnsyn.no is a searchable public records archive.

REQUEST TEMPLATE (English OK):
"Under Offentleglova (2006), § 29(1), I request from [AGENCY]:

All generative AI conversation logs created by agency employees using Ayfie AI platform (mandated for municipalities), ChatGPT, Microsoft 365 Copilot, or any other generative AI chatbot tool. Time period: January 1, 2024 to present."

SPECIFIC NOTES: 673-page precedent from prior AI records request — demonstrates substantial records production. Ayfie AI platform mandated for municipalities. Check einnsyn.no for existing public records. File by email to postmottak@[agency].dep.no for ministries. Target Digitaliseringsdirektoratet, Datatilsynet.""",

    "Denmark (2-3 agencies)": """FRAMING STRATEGY: Offentlighedsloven (§ 36, stk. 2) governs. Danish government has 'Borge' GenAI editor assistant.

REQUEST TEMPLATE (English OK):
"Under Offentlighedsloven, § 36, stk. 2, I request from [AGENCY]:

All generative AI conversation logs from 'Borge' (GenAI editor assistant, launched Feb 2025) and any other AI chatbot tool used by agency employees. Time period: January 1, 2024 to present."

SPECIFIC NOTES: Borge is a named GenAI tool. File by email to [ministry]@[ministry].dk. Target Digitaliseringsstyrelsen, Datatilsynet. Response expected in 2-6 weeks.""",

    "Estonia": """FRAMING STRATEGY: Small but very digital government. High signal-to-noise ratio. Avaliku teabe seadus (§ 18) governs.

REQUEST TEMPLATE (English OK):
"Under Avaliku teabe seadus, § 18, I request from [AGENCY]:

All generative AI conversation logs created by agency employees using ChatGPT, Microsoft 365 Copilot, or any other generative AI chatbot tool. Time period: January 1, 2024 to present."

SPECIFIC NOTES: Civil servants actively use AI for document summarization/translation. File by email to ria@ria.ee (IT Authority), mfm@fin.ee (Ministry of Finance). Response expected in 1-3 weeks. Very digital government = likely well-organized AI records.""",

    "Netherlands": """FRAMING STRATEGY: Wet open overheid (Woo) governs. Vlam-chat is the named internal chatbot.

REQUEST TEMPLATE (Dutch preferred but English may work):
"Under the Wet open overheid (Woo), Art. 4.4(1), I request from [AGENCY]:

All generative AI conversation logs from Vlam-chat (SSC-ICT internal chatbot on European open-source models) and any other AI chatbot tool used by agency employees. Time period: January 1, 2024 to present."

SPECIFIC NOTES: Vlam-chat is named internal tool. File in writing to Woo-contactpersoon at each agency. Target Ministerie van BZK (Interior), SSC-ICT. Response expected in 1-3 months.""",

    "Germany (federal)": """FRAMING STRATEGY: Informationsfreiheitsgesetz (IFG) governs. FragDenStaat.de is excellent filing platform (like WhatDoTheyKnow).

REQUEST TEMPLATE (German preferred):
"Unter dem Informationsfreiheitsgesetz (IFG), § 7(5), beantrage ich folgende Unterlagen von [BEHÖRDE]:

Alle Gesprächsprotokolle generativer KI — einschließlich Benutzereingaben und KI-generierter Antworten — erstellt von Behördenmitarbeitern unter Verwendung der Aleph Alpha F13 Plattform (seit Sept 2024, €1,50/Arbeitsplatz/Monat), ChatGPT, Microsoft 365 Copilot oder anderen generativen KI-Chatbot-Tools. Zeitraum: 1. Januar 2024 bis heute."

SPECIFIC NOTES: Aleph Alpha F13 platform deployed Sept 2024 at €1.50/workstation/month. File via fragdenstaat.de (free, excellent tracking). Target BMI (Interior), BMAS (Labor), BSI (IT Security). Response expected in 1-4 months.""",

    "France": """FRAMING STRATEGY: CRPA (Art. R*311-13) governs. Albert is the named sovereign AI chatbot. May need to file in French.

REQUEST TEMPLATE (French preferred):
"En application du Code des Relations entre le Public et l'Administration (CRPA), je demande communication des documents suivants :

Tous les journaux de conversation d'IA générative — y compris les entrées utilisateur et les réponses générées par l'IA — créés par les agents utilisant Albert (chatbot IA souverain développé par DINUM/Etalab), ChatGPT, Microsoft 365 Copilot ou tout autre outil chatbot IA. Période : 1er janvier 2024 à aujourd'hui."

SPECIFIC NOTES: Albert is France's sovereign AI chatbot for civil servants. File to PRADA officer at relevant ministry. Appeal to CADA (Commission d'accès aux documents administratifs) if denied. French AI models = unique data. Response expected in 1-4 months.""",

    "Australia (federal)": """FRAMING STRATEGY: FOI Act 1982 applies. No application fee since 2010. M365 Copilot whole-of-govt trial documented.

REQUEST TEMPLATE:
"Under the Freedom of Information Act 1982 (Cth), s. 15, I request from [AGENCY]:

All generative AI conversation logs created by APS staff using Microsoft 365 Copilot (whole-of-government trial, 5,000+ staff), ChatGPT, Google Gemini, or any other generative AI chatbot tool. Time period: January 1, 2024 to present."

SPECIFIC NOTES: 5,000+ APS staff in Copilot trial. No fee since 2010. File directly to agency FOI contact. Target DTA (Digital Transformation Agency), ATO, Home Affairs. Response expected in 1-3 months.""",

    "New Zealand": """FRAMING STRATEGY: PROXY NEEDED. OIA 1982 applies. ~90% on-time response rate. FYI.org.nz makes filing easy.

REQUEST TEMPLATE (via proxy or fyi.org.nz):
"Under the Official Information Act 1982, s. 12, I request from [AGENCY]:

All generative AI conversation logs created by employees using Microsoft Copilot Chat (deployed at Dept of Internal Affairs), ChatGPT, or any other generative AI chatbot tool. Time period: January 1, 2024 to present."

SPECIFIC NOTES: File via fyi.org.nz (Alaveteli platform). Target DIA (Internal Affairs — confirmed Copilot), MBIE, IRD. Response expected in 2-4 weeks. ~90% on-time.""",

    "EU Institutions": """FRAMING STRATEGY: PROXY NEEDED. Regulation (EC) 1049/2001 governs. GPT@EC is the named internal tool.

REQUEST TEMPLATE (via asktheeu.org):
"Under Regulation (EC) 1049/2001, Art. 6, I request from [INSTITUTION]:

All generative AI conversation logs from GPT@EC (Commission's internal multi-LLM GenAI tool, pilot Oct 2024) and any other AI chatbot tool used by institution employees. Time period: January 1, 2024 to present."

SPECIFIC NOTES: GPT@EC is the Commission's named internal AI tool. File via asktheeu.org (Alaveteli platform). Target European Commission, DIGIT (Informatics DG). EU institutions are notoriously slow (6-15 months). Confirmatory application to Secretary-General if denied.""",

    "Mexico": """FRAMING STRATEGY: Most sophisticated transparency portal in the world. LGTAIP (Art. 132) governs. No citizenship requirement.

REQUEST TEMPLATE (Spanish required):
"Con fundamento en la Ley General de Transparencia y Acceso a la Información Pública (LGTAIP), Art. 132, solicito:

Todos los registros de conversaciones de IA generativa — incluyendo entradas de usuario y respuestas generadas por IA — creados por empleados del [DEPENDENCIA] utilizando ChatGPT, Microsoft 365 Copilot, u otra herramienta chatbot de IA generativa. Período: 1 de enero de 2024 a la fecha."

SPECIFIC NOTES: IMSS voice recognition AI in 27 Family Medicine units documented. File via Plataforma Nacional de Transparencia (PNT). Target IMSS, SAT (Tax Administration), Coordinación de Estrategia Digital Nacional. Response expected in 1-3 months.""",

    "South Korea": """FRAMING STRATEGY: PROXY NEEDED. Korean language required. Official Information Disclosure Act governs. Seoul has its own LLM.

REQUEST TEMPLATE (Korean required, via proxy):
"공공기관의 정보공개에 관한 법률 제11조에 따라, [기관]의 다음 정보를 공개 청구합니다:

직원이 ChatGPT, Microsoft 365 Copilot, 서울시 Chatbot 2.0, 또는 기타 생성형 AI 챗봇 도구를 사용하여 생성한 모든 대화 기록. 기간: 2024년 1월 1일부터 현재까지."

SPECIFIC NOTES: Seoul has 'Chatbot 2.0' LLM-powered admin tool. File via open.go.kr portal. Response expected in 2-4 weeks. Need Korean-language proxy.""",
}

for jur, notes_text in intl_notes.items():
    if append_notes(ws, jur_map, jur, notes_text): count += 1

# ============================================================
# SAVE
# ============================================================
save_tracker(wb, "foia_tracker_detailed_v3.xlsx")
print(f"\nSaved. Updated Notes for {count} jurisdictions.")
