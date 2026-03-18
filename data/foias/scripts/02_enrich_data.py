#!/usr/bin/env python3
"""
Enrichment step 2: Apply retention policies, FOIA precedents, template notes,
and AI deployment updates to all jurisdictions.
Combined from original update_foia_tracker_v2.py and update_foia_tracker_v3.py.
"""

from helpers import load_tracker, save_tracker, build_jur_map, apply_updates, COL_RETENTION, COL_PRECEDENT, COL_TEMPLATE_NOTES, COL_AI_DEPLOY

wb, ws = load_tracker("foia_tracker_detailed_v3.xlsx")
jur_map = build_jur_map(ws)

# =============================================================================
# RETENTION POLICY UPDATES (col 23)
# v2 entries first, then v3 entries override where keys overlap
# =============================================================================

retention_updates = {
    # ---- v2 entries ----
    "Washington": (
        "YES — WA Secretary of State issued two advice sheets (June 2024): "
        "(1) 'Are Generative AI Interactions Public Records?' — YES, prompts and outputs meet the definition of 'writings' "
        "containing 'information relating to the conduct of government' under RCW 42.56.010. "
        "(2) 'How Long Do GenAI Records Need to Be Kept?' — retention depends on content and function, not format. "
        "Supported by Belenski v. Jefferson County (2015) precedent. "
        "WaTech EA-01-02-S policy governs responsible GenAI use. EO 24-01 (Jan 2024) directs GenAI guidelines. "
        "If a record exists when a PRA request is received, agencies must produce it regardless of retention status. "
        "| Sources: https://www.sos.wa.gov/sites/default/files/2025-02/advice-sheet-are-generative-ai-interactions-public-records-(june-2024).pdf | "
        "https://www.sos.wa.gov/sites/default/files/2025-02/advice-sheet-how-long-do-generative-ai-records-need-to-be-kept-(june-2024).pdf | "
        "https://mrsc.org/stay-informed/mrsc-insight/july-2024/public-records-and-ai"
    ),
    "Bellevue, WA": (
        "YES — City of Bellevue AI Policy (updated Oct 2025): "
        "'All records prepared, owned, used or retained by the city in AI technology for city business are public records' "
        "and must follow the city's retention schedule and be disclosed upon request. "
        "'All AI solutions and/or vendors approved for city use will be required to support retrieval and export of public records upon request.' "
        "One of the strongest municipal AI records policies in the US. "
        "| Source: https://bellevuewa.gov/ai-policy"
    ),
    "Tacoma, WA": (
        "YES — Generative AI ML Systems Guideline 4.1-GD-02 by IT Dept. "
        "Governs employee use of ChatGPT, Gemini, Copilot. Applies to all city employees and third-party contractors. "
        "Under WA SOS June 2024 guidance, these interactions are public records. "
        "| Source: https://www.cityoftacoma.org/government/city_departments/information_technology/generative_artificial_intelligence_guidelines"
    ),
    "San Jose, CA": (
        "YES — City Policy Manual 1.7.12; Generative AI Guidelines (23-page memo, July 2023). "
        "'Presume all interactions with Generative AI are PRA-able' (subject to CA Public Records Act). "
        "Employees must use city email for ChatGPT accounts to 'ensure proper retention of public records.' "
        "Mandatory form submission required each time staff use AI for work. "
        "'Presume anything you submit could end up on the front page of a newspaper.' "
        "| Source: https://www.sanjoseca.gov/your-government/departments-offices/information-technology/itd-generative-ai-guideline | "
        "https://www.governing.com/policy/san-jose-warns-employees-chatgpt-use-is-public-record"
    ),
    "Tempe, AZ": (
        "YES — Generative AI Use Guidelines (GitBook): STRONGEST municipal policy found. "
        "'Unless any recognized exception to the Arizona Public Records Law applies, content produced by generative AI is considered public record. "
        "This includes prompts and both draft and finalized content and images.' "
        "'Everything entered into Generative AI is subject to Public Records Requests.' "
        "Departments must determine retention requirements and document expected retention methods. "
        "| Source: https://tempe.gitbook.io/data-policy-and-governance/data-policy/generative-ai-use-guidelines"
    ),
    "Denver, CO": (
        "YES — 'Generative AI Transparency, Use, and Accountability Policy' POL0020153 (effective 01/01/2024). "
        "§4.1.4: 'Document the tool, processes, and data sets input into the Generative AI tool and the resulting output.' "
        "§4.1.5: Ensure outputs are explainable. §4.1.6: Create accountability mechanisms. "
        "Does NOT explicitly use 'public records' or 'records retention' language. CIO title expanded to include AI (2025). "
        "| Source: https://newspack-coloradosun.s3.amazonaws.com/wp-content/uploads/2024/09/Denver-Policy-17_AI-Transparency-Use-and-Accountability-2024.pdf"
    ),
    "Maryland": (
        "YES — Governor's EO 01.01.2024.02 (Jan 2024) + AI Governance Act SB818 (July 2024). "
        "DoIT Responsible AI Policy: 'State staff must follow all current procedures for records retention and disclosure when implementing AI systems.' "
        "Interim GenAI Guidance (May 2024) for all exec branch employees. "
        "Google Gemini extended to 43K employees; 12,500 active users. Anthropic Claude partnership (Nov 2025). "
        "Baltimore City EO: 'information entered into a generative AI system immediately becomes a public record.' "
        "| Sources: https://doit.maryland.gov/policies/ai/Pages/maryland-responsible-ai-policy.aspx | https://ai.maryland.gov/"
    ),
    "Tennessee": (
        "YES — Enterprise AI Policy 200-POL-007 (approved 9/26/2025) + GenAI Policy ISC 3.00 (approved 2/28/2024). "
        "Data Classification system (Public Record, Confidential, Restricted Access). "
        "ISC 3.00 requires validation of all GenAI inputs and outputs. "
        "Nashville Metro ISM-20 §1.4: 'Use of AI and GenAI shall comply with all applicable public records laws, "
        "including the Tennessee Public Records Act. This extends to all data requestable under that act.' "
        "| Sources: https://www.tn.gov/finance/sts/artificial-intelligence.html | "
        "https://www.nashville.gov/sites/default/files/2025-08/ISM-20-Artificial-Intelligence-and-Generative-Artificial-Intelligence-Use.pdf"
    ),
    "Missouri": (
        "PARTIAL — EO 26-02 (Jan 2026) directs AI framework. Google Gemini in government-secured tenant. "
        "MO SOS Communications Guidelines: electronic messages same retention as hard copy; "
        "auto-delete is prohibited; no communication should be automatically destroyed. "
        "Sunshine Law broadly defines 'public record' regardless of physical form (Ch. 109). "
        "| Source: https://www.sos.mo.gov/CMSImages/RecordsManagement/CommunicationsGuidelines.pdf"
    ),
    "Virginia": (
        "PARTIAL — VITA AI Policy Standard: VITA 'shall retain records of the specific AI use and approvals thereof.' "
        "AI registry tracks all 112+ use cases with model-level detail (inputs, outputs, algorithms, datasets). "
        "Agencies must secure data and define retention timeframes. "
        "Mandatory disclaimers when AI processes/produces decisions. "
        "Registry submissions and approvals are themselves FOIA-able. "
        "| Source: https://www.vita.virginia.gov/media/vitavirginiagov/it-governance/ea/pdf/Utilization-of-Artificial-Intelligence-by-COV-Policy-Standard.pdf"
    ),
    "Indiana": (
        "PARTIAL — AI Policy v1.1 (Dec 2024) via OCDO addresses data governance for AI. "
        "IARA guidance: AI meeting note outputs must be retained under existing schedules (IC 5-14-1.5-2.9, effective July 2025). "
        "Broad 'writing' definition under APRA likely covers AI chat logs. "
        "ChatGPT broadly permitted for state employees (exempt from full review). "
        "| Source: https://www.in.gov/mph/AI/ | https://www.in.gov/iara/"
    ),
    "Montana": (
        "PARTIAL — HB 178 (signed May 2025) requires govt entities to disclose AI use in public interfaces and materials. "
        "Constitutional Art. II, Sec. 9 'Right to Know' broadly covers 'documents' of all public bodies. "
        "No explicit AI retention rule but strongest constitutional basis of any state. "
        "Commissioner of Political Practices' documented ChatGPT use would be a strong first target. "
        "| Source: https://leg.mt.gov/bills/2025/billpdf/HB0178.pdf"
    ),
    "Nebraska": (
        "PARTIAL — NITC AI policy section 8-609 (Nov 2024). Policy explicitly references "
        "ChatGPT, DALL-E, Bing/Gemini, Adobe AI Assistant, Copilot as categories of AI. "
        "Requires agencies to consult with OCIO Security Risk Mitigation team before deploying AI. "
        "Public Records Statutes broadly define records. "
        "| Source: https://nitc.nebraska.gov/docs/8-609.pdf"
    ),
    "St. Paul, MN": (
        "PARTIAL — Police Dept Policy 236.06 'Generative AI' (effective Oct 17, 2025). "
        "Requires AI content reviewed as 'draft material only'; disclosure when work generated by AI. "
        "No records retention requirement for AI inputs/outputs. References MGDPA Ch. 13 for data classification. "
        "| Source: https://www.stpaul.gov/sites/default/files/2025-10/236.06%20Generative%20Artificial%20Intelligence%20(GenAI)%2010.17.25_1.pdf"
    ),
    "Long Beach, CA": (
        "PARTIAL — Generative AI Interim Guidance v1.3 (May 2025). "
        "'Any information entered into Gen AI may be subject to PRA requests.' "
        "Does NOT explicitly address records retention. 1,000+ employees have Copilot. "
        "| Source: https://www.longbeach.gov/globalassets/smart-city/media-library/documents/final---generative-ai-guidance-v1-3"
    ),
    "Boston, MA": (
        "PARTIAL — Interim Guidelines for Using Generative AI v1.1 (May 2023). "
        "Requires disclosure of AI usage including version and type of model. "
        "Warns 'you and the City will bear responsibility over its use in the public.' "
        "No explicit records retention clause. Updated guidelines forthcoming. "
        "| Source: https://www.boston.gov/sites/default/files/file/2023/05/Guidelines-for-Using-Generative-AI-2023.pdf"
    ),
    "Miami-Dade County, FL": (
        "PARTIAL — AI Administrative Order (Directive 231203) + AI Employee Policies. "
        "Must 'Always cite AI generated content' and 'Disclose use in public-facing documents.' "
        "Requires 'compliance with Miami-Dade County's public records policies and retention rules' "
        "but does NOT create AI-specific retention schedules. "
        "| Source: https://www.miamidade.gov/global/comms-tech/technology/ai/ai-employee-policies.page"
    ),
    "Alameda County, CA": (
        "PARTIAL — Generative AI Policy adopted Sept 1, 2024. "
        "Provides framework for all departments. Focuses on risk management. "
        "No public text explicitly declaring AI inputs/outputs as public records. "
        "| Source: https://itd.alamedacountyca.gov/generative-ai-policy/"
    ),
    "USCBP (US Customs and Border Protection)": (
        "PARTIAL — CBP Directive 1450-030 on AI Operations and Governance (Nov 2025). "
        "chatCBP deployed (May 2025) + DHSChat. 83 AI use cases. "
        "CBP proactively releasing extensive AI FOIA records (model cards, contracts, policies). "
        "| Source: https://www.cbp.gov/document/foia-record/artificial-intelligence-records"
    ),
    "CDC (Centers for Disease Control and Prevention)": (
        "PARTIAL — ChatCDC: enterprise GenAI chatbot deployed to ALL CDC staff (first federal agency). "
        "Azure OpenAI powered. Saved $3.7M (527% ROI). Internal GenAI guidance exists. "
        "No specific chat log retention schedule published. Falls under HHS AI strategy (Dec 2025). "
        "| Source: https://www.cdc.gov/data-modernization/php/ai/cdcs-vision-for-use-of-artificial-intelligence-in-public-health.html"
    ),
    "TSA (Transportation Security Administration)": (
        "PARTIAL — TSA Contact Center Virtual Assistant (public-facing, GenAI/NLP/RAG), "
        "TSA Answer Engine (internal tool for field staff to query SOPs), "
        "LLM-based training dialogue simulation. Covered by DHS governance. "
        "| Source: https://www.dhs.gov/ai/use-case-inventory/tsa"
    ),
    "NSF (National Science Foundation)": (
        "PARTIAL — AI Strategy published. GenAI tools: ServiceNow GenAI (FedRAMP High), "
        "Microsoft Copilot M365 (pilot), AWS CodeWhisperer/Bedrock, open-source models (Llama, BART). "
        "Developing AI chatbot for grant opportunities. No chat log retention published. "
        "| Source: https://www.nsf.gov/policies/ai"
    ),
    "United Kingdom (5-8 depts)": (
        "PARTIAL — Generative AI Framework for HMG (Jan 2024). "
        "ICO confirmed: 'official information held in non-corporate channels (private email, WhatsApp, Signal, AI chats) "
        "is subject to FOI requests if it relates to public authority business.' "
        "Platform used is irrelevant — content determines disclosability. "
        "Records retention: 2-20 years depending on type. "
        "MAJOR PRECEDENT: Peter Kyle's full ChatGPT history was disclosed via FOI. "
        "| Source: https://www.gov.uk/government/publications/foi2025-00038-peter-kyles-chatgpt-history"
    ),
    "Canada (3-5 depts)": (
        "YES — Treasury Board Guide on Use of Generative AI requires: "
        "(1) document decisions to deploy GenAI, (2) document steps for accuracy of outputs, "
        "(3) identify retention and disposal requirements for AI documentation, "
        "(4) comply with Directive on Automated Decision-Making. "
        "Public AI Register catalogues government AI systems. "
        "| Source: https://regulations.ai/regulations/RAI-CA-NA-GUGATXX-2024"
    ),
    "Norway (2-3 agencies)": (
        "YES (via ruling) — County Governor (Sept 23, 2025) ruled AI chat logs ARE public documents: "
        "(1) AI chat logs qualify as 'documents' under offentleglova §4, "
        "(2) They can be 'case documents,' "
        "(3) They are NOT exempt as internal documents when shared externally. "
        "Result: 673 pages of ChatGPT logs ordered disclosed (Tromso municipality). "
        "| Source: https://www.vaar.law/en/nar-ki-kladden-blir-et-offentlig-dokument/"
    ),
    "Netherlands": (
        "PARTIAL — Algorithm Register (algoritmeregister) launched Dec 2022. "
        "Registration of high-risk algorithms mandatory from 2025. "
        "WOO (Open Government Act, May 2022) provides access framework. "
        "Dutch DPA flagging gap between rules on paper and enforcement. "
        "| Source: https://algoritmes.overheid.nl/en"
    ),
    "Australia (federal)": (
        "YES — National Archives of Australia guidance (Jan 2025): "
        "'AI-generated content created or received by Australian Government agencies constitutes Commonwealth records "
        "for the purposes of the Archives Act 1983 and must be managed accordingly.' "
        "Victoria PROV AI Technologies & Recordkeeping Policy (March 2024). "
        "PM&C disclosed AI testing documents via FOI (FOI-2023-156). "
        "| Source: https://www.naa.gov.au/information-management/manage-information-assets/types-information/information-management-records-created-using-artificial-intelligence-ai-technologies"
    ),
    "New Zealand": (
        "PARTIAL — Public Service AI Framework (Jan 2025) + Responsible AI Guidance for GenAI (Feb 2025). "
        "Requires risk assessments, human oversight, logging, and auditability. "
        "Algorithm Charter for Aotearoa NZ (2020). "
        "Multiple OIA responses disclosing AI tools (NZDF, IRD, MoD). "
        "| Source: https://www.digital.govt.nz/standards-and-guidance/technology-and-architecture/artificial-intelligence/public-service-artificial-intelligence-framework"
    ),
    "EU Institutions": (
        "PARTIAL — EC internal guidelines: 'Staff shall never directly replicate the output of a generative AI model in public documents.' "
        "GPT@EC (internal GenAI tool) launched Oct 2024. "
        "EDPS AI meeting with EU institution Secretaries-General. "
        "Commission policy C(2024) 380 (Jan 24, 2024). "
        "ICCL complaint to Ombudsman after ChatGPT use exposed in public documents. "
        "| Source: https://commission.europa.eu/system/files/2024-01/EN%20Artificial%20Intelligence%20in%20the%20European%20Commission.PDF"
    ),
    "France": (
        "PARTIAL — 2016 Digital Republic Law (CRPA) requires: "
        "(1) inform citizens when algorithms used in individual decisions, "
        "(2) publish essential rules underlying algorithms. "
        "CNAF algorithm source code eventually published Jan 2026 after legal battle. "
        "| Source: https://www.defenseurdesdroits.fr/algorithmes-intelligence-artificielle-et-services-publics-2024"
    ),
    "Germany (federal)": (
        "PARTIAL — IFG covers official documents. KI Market Surveillance Act (KIMÜG, Dec 2024) for EU AI Act. "
        "BfDI guidance for public authorities on data protection in AI. "
        "Multiple FragDenStaat IFG requests about ChatGPT in government. "
        "| Source: https://fragdenstaat.de/en/request/chatgpt-in-behoerden/"
    ),
    "Sweden (10-15 agencies)": (
        "PARTIAL — Offentlighetsprincipen (constitutional public access principle). "
        "IMY + DIGG joint guidelines on GenAI in public sector (Jan 2025). "
        "AI interactions by officials on official matters would in principle constitute official documents. "
        "PM Kristersson's regular ChatGPT use sparked debate (Aug 2025). "
        "| Source: https://www.imy.se/en/news/national-guidelines-for-generative-ai-in-public-administration-are-launched/"
    ),

    # ---- v3 entries (override v2 where keys overlap) ----
    # --- STATES (from Agent 3 - batch 2) ---
    "Massachusetts": "PARTIAL — EOTSS 'Enterprise Use and Development of Generative AI Policy' (eff. 01/31/2025). EO 629 established AI Strategic Task Force. Requires GenAI Inventory and AI Sandbox. Addresses data protection and risk review. No explicit language declaring AI inputs/outputs as public records or setting AI-specific retention schedules. General M.G.L. c.66, s.10 applies. Gov. Healey announced MA as first state to deploy ChatGPT across executive branch (Feb 2026). Sources: mass.gov/policy-advisory/enterprise-use-and-development-of-generative-ai-policy",

    "California - CDT (Poppy)": "PARTIAL — EO N-12-23 (Sept 2023) directed agencies to issue GenAI guidelines. Poppy platform uses 'state-approved, vendor-agnostic infrastructure' where 'information never leaves California's trusted environment.' No explicit language found declaring AI inputs/outputs as public records subject to CPRA or setting AI-specific retention schedules. General CPRA applies. Sources: genai.ca.gov/poppy/, cdt.ca.gov/technology-innovation/artificial-intelligence-community/genai-executive-order/",

    "Minnesota": "PARTIAL — MNIT 'Public AI Services Security Standard' developed with TAIGA (Transparent AI Governance Alliance). Only Microsoft Copilot Chat (green shield) approved for non-public data. Employees must comply with MN Government Data Practices Act. MnDOT Standard IT-003 also governs AI use. No explicit language declaring AI prompts/outputs as government records with specific retention schedules. Sources: mn.gov/mnit/government/policies/security/ai-standard/",

    "New York": "YES — NYS-P24-001 'Acceptable Use of AI Technologies' (ITS) explicitly requires 'data retention settings that follow the requirements of federal and state standards; ensuring the accuracy of data put into the AI system and the AI system's outputs; and disposal of the data once the purpose of using the data has been fulfilled.' Routine audits of AI outputs mandated. RAISE Act (signed Dec 19, 2025, eff. March 19, 2026) requires frontier model developers to retain safety protocols for model life + 5 years. 2025 Comptroller audit (2023-S-50) found 'none of the agencies required or developed specific procedures to test AI systems.' Sources: its.ny.gov/system/files/documents/2026/01/nys-p24-001-acceptable-use-of-ai.pdf",

    "Pennsylvania": "PARTIAL — OA 'Commonwealth Use of Public Generative AI Policy' requires 'Generative AI use must be disclosed even if it was only used to generate a portion of the content.' HOWEVER, OOR ruled Feb 2026 (WITF Right-to-Know request) that AI chat logs may be exempt as 'notes and working papers used solely for that official's or employee's own personal use' and 'internal, predecisional deliberations.' Legal experts note every AI chat remains 'presumptively public' unless agency proves exemption. No specific AI retention schedule found. Sources: witf.org/2026/02/18/public-access-to-pennsylvania-officials-ai-conversations-may-be-limited-after-agency-ruling/",

    "New Jersey": "YES — Nation's first state-level AI/ML records retention guidance. NJ Treasury DORES 'Guidelines for Developing Retention and Disposition Policies for AI/ML Systems' recommends AI/ML Governing Board working with DORES/RMS and State Records Committee. Documentary records from AI/ML systems can be managed at item or entity level with risk-based retention scheduling. Also: Joint Circular 23-OIT-007 (2023) and 25-OIT-001 (2025). 15,000+ state employees use NJ AI Assistant (300,000+ sessions, 1M+ prompts as of Feb 2026), hosted on state infrastructure. Sources: nj.gov/treasury/revenue/rms/pdf/BackgroundandGuidelinesonRetentionandDispositionPolicies.pdf",

    "Oregon": "PARTIAL — EIS Interim AI Guidelines v1.5 (Dec 17, 2024) per EO 23-26. AI-generated content 'should be clearly labeled as such, and details of its review and editing process should be provided.' Restricted to Level 1/2 data per Information Asset Classification Policy. Decision support AI requires 'detailed documentation describing training, testing, results, and validation steps.' AI Advisory Council Final Recommended Action Plan (Feb 2025). No explicit retention schedules for AI inputs/outputs. Sources: oregon.gov/eis/cyber-security-services/Documents/Interim_AI_Guidelines_V1.5",

    "Texas": "YES — TSLAC (June 2024) guidance explicitly states AI-generated records should be classified same as human-generated records on retention schedules. AI treated as 'assistive technology'; final product classified based on 'administrative purpose and function the record serves.' SB 1964 (AI Ethics Code) requires agencies to 'document how each AI system aligns with the code's ethical principles' and 'maintain records of AI system behavior, decisions and interventions over time.' DIR must inventory all agency AI systems. Also: HB 2818 (AI Division at DIR), TRAIGA (HB 149, signed 2025). Sources: tsl.texas.gov/slrm/blog/2024/06/classifying-ai-generated-records/",

    "Florida": "UNKNOWN — No state-level executive order or DMS policy on state employee GenAI use identified. Proposed 'Citizen Bill of Rights for AI' (SB 482) pending in legislature. SB 1118 addresses data center public records exemptions. DeSantis stated FL is 'using AI to find government waste' but no formal AI use policy issued. Miami-Dade County has its own AI policy (Directive 231203) addressing data retention. Sources: flgov.com/eog/news/press/2025/governor-ron-desantis-announces-proposal-citizen-bill-rights-artificial",

    "Utah": "PARTIAL — DTS Enterprise Generative AI Policy 4000-0008 provides guidance for executive branch employees. Utah AI Policy Act (SB 149, 2024). 15,000-16,000 state employees now have Gemini access. Policy designed to 'promote use of generative AI while protecting safety, privacy, and intellectual property rights.' Full policy text regarding GRAMA compliance and AI retention not publicly accessible. Sources: dts.utah.gov/policies/enterprise-generative-ai-policy",

    "Ohio": "PARTIAL — DAS Administrative Policy IT-17 'Use of AI in State of Ohio Solutions' (2023) explicitly states 'only data that is public record should be entered by state employees into generative AI systems.' Requires human verification for AI-driven decisions with legal/financial/HR/regulatory impact. Must annotate GenAI outputs with 'at least the Generative AI technology used and a description of how it was used.' No specific retention schedule for AI inputs/outputs. Sources: digitalgovernmenthub.org/examples/it-17-use-of-artificial-intelligence-in-state-of-ohio-solutions/",

    "Michigan": "PARTIAL — DTMB 'Adoption and Usage of AI Guidelines and Responsibilities.' AI tools 'already governed by existing State of Michigan policies, standards, and procedures.' External tools require Authority to Operate via MiSAP. Users must review and edit AI responses (HITL). DTMB hired Chief Privacy Officer (Jan 2025) for AI governance. No explicit AI-specific retention schedules. General FOIA and DTMB Policy 0910.02 apply. Sources: michigan.gov/dtmb/-/media/Project/Websites/dtmb/Law-and-Policies/Governance/",

    "Wisconsin": "PARTIAL — DET Acceptable Technology Use Policy (eff. March 10, 2025) includes 'Public Records and Records Retention' section stating communications using state systems 'create public records that must be retained and produced if requested.' Recommends 'living AI Tool Registry' with owner, purpose, data flows, and retention schedules. Content generated using AI 'may constitute a public record and be subject to a records retention schedule as required by applicable law.' EO #211 (Aug 2023) created Governor's Task Force on Workforce and AI. Sources: det.wi.gov/Documents/AUP",

    "New Hampshire": "PARTIAL — RSA 5-D 'Use of AI by State Agencies' (eff. July 1, 2024). GenAI content 'must be accompanied by disclosure that the content was generated by AI' unless human-reviewed. Irreversible AI recommendations require human review. DoIT maintains AI technologies policy. Annual report to Governor/Speaker/Senate President. RSA 5-D does NOT contain specific records retention provisions for AI inputs/outputs. General RSA 8-B records management applies. Sources: gc.nh.gov/rsa/html/I/5-D/5-D-mrg.htm",

    # --- STATES (from Agent 1 - batch 1, enrichments) ---
    "Connecticut": "PARTIAL — Policy AI-01 Responsible AI Framework (Feb 1, 2024) implementing SB 1103 (June 2023). Agencies must submit documentation and initial impact assessment before implementing AI, with ongoing yearly assessments. DAS must inventory all agency AI systems publicly including 'name, description, training data used, output, and financial impact.' No standalone section declaring AI inputs/outputs are public records. Operates through documentation, impact assessment, and inventory requirements. Sources: portal.ct.gov/-/media/OPM/Fin-General/Policies/CT-Responsible-AI-Policy-Framework-Final-02012024.pdf",

    "Illinois": "PARTIAL — DoIT 'Policy on Acceptable and Responsible Use of AI' v2 (eff. April 1, 2025). Agencies must define oversight roles for AI system's full lifecycle. Must maintain 'transparent communication channels in writing with stakeholders.' Separately, IL HB 3773 requires records of AI system use, impact assessments, and employment decisions assisted by AI to be retained for FOUR YEARS. DoIT policy focuses on governance rather than records management. Sources: doit.illinois.gov/content/dam/soi/en/web/doit/documents/support/policies/2021/20250401-DoIT-AI%20Policy-v2",

    "North Carolina": "YES — Public Records Act (N.C.G.S. Chapter 132) applies to AI prompts and outputs per UNC School of Government legal analysis (March 2026). Both AI prompts ('records made by officials') and outputs ('records received') are public records 'regardless of physical form or characteristics.' Records on third-party servers qualify per Gray Media Group, Inc. v. City of Charlotte (290 N.C. App. 384, 2023) — 'actual or constructive possession' suffices. State Copilot for M365 Policy (Feb 26, 2025). Retention governed by NC DNCR disposition schedules based on content. Sources: canons.sog.unc.edu/blog/2026/03/11/the-intersection-of-artificial-intelligence-and-the-public-records-act/",

    "Georgia": "YES — SS-23-002 'AI Responsible Use' standard + 'Red Light, Green Light' guidelines (Aug 1, 2025). Explicit record-keeping mandate: 'Keep records. Save prompts, outputs, and who reviewed the content. These may be subject to public records laws.' Agencies must label AI outputs with 'tool name, the prompt, and who reviewed the content.' Must 'maintain a record of AI tool usage, including the purpose, inputs, outputs, and any actions taken based on AI-generated results.' Misuse reports under Georgia Open Records Act O.C.G.A. 50-18-72. Sources: ai.georgia.gov/blog/2025-08-01/red-light-green-light-new-state-guidelines-using-genai",

    "Colorado": "PARTIAL — OIT prohibits free ChatGPT on state devices ('accepting terms would violate state law'). Statewide GenAI Policy requires 'all GenAI efforts and use cases, including those by third-party vendors, must go through OIT to conduct a risk assessment.' Specific records retention language within GenAI policy not publicly accessible (oit.colorado.gov returns 403). Sources: oit.colorado.gov/standards-policies-guides/guide-to-artificial-intelligence/free-chatgpt-prohibited",

    # --- CITIES (from Agent 1) ---
    "San Francisco, CA": "PARTIAL — Chapter 22J SF Admin Code 'AI Tools' (Ordinance No. 288-24) requires publicly available AI tool inventory answering 22 standardized questions. GenAI Guidelines (July 8, 2025) state: 'Content entered into or generated by GenAI tools may constitute public records and may be subject to disclosure under the California Public Records Act (CPRA), as well as the City's public access and records retention requirements, including the Sunshine Ordinance.' Must provide 'Direct Notice' disclosing AI tool use. No AI-specific retention periods. Sources: media.api.sf.gov/documents/July2025-GenAI-Guidelines.pdf",

    "Oakland, CA": "PARTIAL — ITD Interim AI Security Guidelines (Dec 2024) from AI Working Group (AIWG). Staff must 'include citations when AI is used in any reports, memos, or other public records.' Must get permission before entering city data into AI tools. Must use official City email for commercial AI accounts. Data governance pilot prevents sensitive data access by AI. No explicit standalone declaration that AI inputs/outputs must be retained as public records, but citation requirement creates documentation trail. Sources: oaklandca.gov/files/assets/city/v/1/information-technology/documents/itd-interim-ai-security-guidelines-2024-final.pdf",

    "New York City": "PARTIAL — OTI Preliminary Use Guidance: GenAI (May 2024) + AI Principles & Definitions (March 2024) + NYC AI Action Plan (Oct 2023) + Annual Progress Report (Oct 2024). Guidance focuses on 'cybersecurity, privacy, trust and transparency' but no explicit language declaring AI inputs/outputs are government records requiring retention. Dept of Records has Supplemental Retention Schedule (Sept 2024) with no AI-specific category. Aspen Policy Academy project (2025) recommended OTI update guidance with 'guiding questions' framework. Sources: nyc.gov/assets/oti/downloads/pdf/New-York-City-Generative-AI-Use-Guidance.pdf",

    "Washington, DC": "NO — OCTO AI/ML Governance Policy requires written agency director approval for AI data use and 'adequate auditing and logging mechanisms.' But policy 'contains no explicit language addressing AI-generated records, AI training data, or outputs from AI tools.' Separate Data and Records Retention Policy (eff. Feb 22, 2021; revised May 10, 2024) also contains no AI-specific language. DC was first major US city to require responsible AI training for government workforce. Sources: octo.dc.gov/page/aiml-governance-policy, octo.dc.gov/publication/data-and-records-retention-policy",

    "Seattle, WA": "YES — POL-209 (Nov 3, 2023), superseded by broader POL-211 (eff. 12/31/2024). §6.1: 'All records generated, used, or stored by Generative AI vendors or solutions may be considered public records and must be disclosed upon request.' §6.2: 'All Generative AI solutions and/or vendors approved for City use shall be required to support retrieval and export of all prompts and outputs.' §6.3: City employees must 'maintain, or be able to retrieve upon request, records of inputs, prompts, and outputs in a manner consistent with the City's records management and public disclosure policies.' Retention based on content, not format. Sources: mrsc.org/stay-informed/mrsc-insight/july-2024/public-records-and-ai, seattle.gov/tech/data-privacy/the-citys-responsible-use-of-artificial-intelligence",

    # --- FEDERAL (from Agent 4) ---
    "GSA (USAi.gov)": "PARTIAL — Operates USAi.gov, government-wide AI platform with Chat, API, and Console access to models from OpenAI, Google, Amazon, Meta, Microsoft, xAI. References Privacy Policy and Rules of Behavior. Platform is 'designed for maximum security and privacy' with FedRAMP-compliant providers. No explicit public statement that USAi.gov chat logs are retained as federal records. GSA references OMB M-25-21 in FY 2025-2027 AI compliance plan. Sources: usai.gov, gsa.gov/technology/government-it-initiatives/artificial-intelligence",

    "HHS": "PARTIAL — Deployed ChatGPT Enterprise department-wide (Sept 9, 2025) via GSA's OneGov Strategy ($1/agency OpenAI offering). Received FISMA moderate ATO. Prohibited uses: sensitive PII, classified materials, export-controlled info, PHI. GAO found HHS had largest AI expansion: 7 use cases (2023) to 116 (2024). No public records retention policy for AI chat logs identified. Also operates HHSGPT. Sources: fedscoop.com/hhs-rolls-out-chatgpt-across-department/",

    "DHS (Department of Homeland Security)": "YES — DHSChat launched Dec 2024 for 19,000+ HQ staff, piloted across 10 operating agencies. DHS-OpenAI contract amendment EXPLICITLY states 'outputs from the model are considered federal records' — strongest federal agency statement on AI outputs as federal records. Data 'is not used to train external models.' Whether inputs/prompts also qualify as records remains unclear. Previously approved ChatGPT, Claude 2, BingChat, DALL-E2, Grammarly for publicly available info. Sources: fedscoop.com/chatgpt-meet-chatdhs-homeland-security-ai-bot/, fedscoop.com/home/generative-ai-could-raise-questions-for-federal-records-laws/",

    "DOJ (Department of Justice)": "PARTIAL — Published Compliance Plan for OMB M-24-10 (Oct 2024) and AI Strategy (Dec 2020). Maintains AI Use Case Inventory. No specific records retention policy for AI chat logs identified. Sources: justice.gov/ai",

    "CMS (Centers for Medicare & Medicaid Services)": "PARTIAL — Deployed 'CMS Chat' (Dec 2024), 100% employee access. 80% rated it highly. 4,700+ employees trained through 'AI Ignite' program. Also used AI to fight fraud, saving $2B (per COO Kim Brandt, Feb 2026). No specific records retention policy for CMS Chat identified. Sources: nextgov.com/artificial-intelligence/2026/02/cms-built-waitlist-its-ai-chatbot-and-drove-momentum-official-says/411369/",

    "SEC (Securities and Exchange Commission)": "PARTIAL — Provisionally approved ChatGPT, Bing Chat, Claude 2, DALL-E2, Grammarly for use. No specific records retention policy for AI chat logs identified beyond general Federal Records Act obligations. Sources: sec.gov/ai",

    "DOE (Department of Energy)": "PARTIAL — Most comprehensive published AI governance documentation framework among federal agencies: AI Strategy 2025, AI Compliance Plan 2025, Generative AI Policy, GenAI Reference Guide, FY25-28 Enterprise Data Strategy, annual AI Use Case Inventory. Released PermitAI for federal environmental permitting. Specific retention provisions for AI chat logs could not be confirmed (GenAI Policy returned 404). Sources: energy.gov/ai",

    "EPA (Environmental Protection Agency)": "PARTIAL — Published AI Compliance Plan (2025) and AI Strategy (2025) per EO 14179 and OMB M-25-21. Focus on maintaining annual AI use case inventory and risk management. No specific records retention for AI chat logs identified. Sources: epa.gov/ai",

    "ICE (Immigration and Customs Enforcement)": "PARTIAL — As DHS component, DHS-level policies apply including DHS-OpenAI contract language that 'outputs are federal records.' No ICE-specific AI records retention policy found.",

    "NIST (National Institute of Standards and Technology)": "PARTIAL — Develops AI Risk Management Framework, runs Center for AI Standards and Innovation (CAISI), AI Innovation Lab, Zero Drafts Pilot Project. These are external-facing standards programs. No specific internal AI records retention policy found. Sources: nist.gov/artificial-intelligence",

    # --- v3 entries for keys also in v2 (overriding v2's Spokane, Bellingham, Everett) ---
    "Spokane, WA": "YES — ADMIN 5300-24-09 (Generative AI Policy): "
        "'Only designated City personnel with proper training shall utilize Generative AI to develop business products.' "
        "'AI-based records must be retrievable by the City Clerk's Office, Police Records Unit, and by all other public disclosure personnel.' "
        "| Source: https://static.spokanecity.org/documents/opendata/policies/2025/admin-5300-24-09.pdf",

    "Bellingham, WA": "IN DEVELOPMENT — Draft AI policy (Aug 2025), expected adoption before end of 2025. "
        "Currently 'permissive approach' — staff can use Copilot, ChatGPT broadly. No explicit retention policy. "
        "Draft policy itself was largely written using ChatGPT (7 of 10 principles copied from ChatGPT output). "
        "ChatGPT logs were successfully obtained via PRA request, confirming de facto public records treatment. "
        "WA SOS June 2024 guidance applies: AI interactions are public records. "
        "| Sources: https://www.knkx.org/government/2025-08-27/washington-state-everett-bellingham-government-officials-embrace-artificial-intelligence-chatgpt-policies-catching-up",

    "Everett, WA": "IN DEVELOPMENT — Provisional IT guidelines (June 2024); formal policy modeled on GovAI Coalition template pending mayoral approval. "
        "All AI-generated material 'released to external audiences for public policy decision making should be clearly labeled.' "
        "Copilot is designated primary platform; ChatGPT requires special exemption. "
        "ChatGPT logs were successfully obtained via PRA, confirming de facto public records treatment. "
        "WA SOS June 2024 guidance applies: AI interactions are public records. "
        "| Sources: https://www.cascadepbs.org/news/2025/08/washington-cities-ai-policies-play-catch-up-as-officials-embrace-new-tech/",

    "OPM": "UNKNOWN — No FOIA request found targeting OPM AI/ChatGPT/Copilot records.",
}

# =============================================================================
# PRECEDENT UPDATES (col 24)
# v2 entries first, then v3 entries override where keys overlap
# =============================================================================

precedent_updates = {
    # ---- v2 entries (excluding keys overridden by v3 below) ----
    "United Kingdom (5-8 depts)": (
        "YES — STRONGEST INTERNATIONAL PRECEDENT. Chris Stokel-Walker (New Scientist) filed FOI for "
        "Science Secretary Peter Kyle's ChatGPT history. DSIT disclosed Kyle's full ChatGPT logs: "
        "speeches, policy explanations (antimatter, quantum), podcast recommendations, AI adoption analysis. "
        "Multiple follow-on FOI requests filed (FOI2025-00038, -00120, -00211, -00257, -00262). "
        "Broader request for ALL govt employee records refused under s.12 (cost exemption, >24 hrs work). "
        "| Sources: https://www.gov.uk/government/publications/foi2025-00038-peter-kyles-chatgpt-history | "
        "https://theconversation.com/why-a-journalist-could-obtain-a-ministers-chatgpt-prompts-and-what-it-means-for-transparency-252269"
    ),
    "Norway (2-3 agencies)": (
        "YES — MAJOR PRECEDENT. Tromso municipality used ChatGPT for school restructuring proposal; "
        "fabricated research citations found. After PwC investigation, journalists requested chat logs. "
        "County Governor ruled (Sept 23, 2025): AI chat logs ARE public documents under offentleglova. "
        "673 pages of ChatGPT logs ordered disclosed. "
        "| Source: https://www.vaar.law/en/nar-ki-kladden-blir-et-offentlig-dokument/"
    ),
    "Denmark (2-3 agencies)": (
        "YES — Akademikerbladet filed aktindsigt for Tax Administration (Skatteforvaltningen) AI systems. "
        "Confirmed 23 AI systems in use/development. Refused to disclose purpose of 14 systems, data types of 7, names of 4 models. "
        "Legal experts criticized lack of transparency. "
        "| Source: https://dm.dk/akademikerbladet/aktuelt/ai/2024/skattestyrelsen-moerklaegger-ai-overvaagning-af-danskerne/"
    ),
    "Germany (federal)": (
        "YES — Multiple FragDenStaat IFG requests: "
        "(1) 'ChatGPT in Behorden' (Jan 2026): Schleswig-Holstein State Chancellery, response received Feb 2026. "
        "(2) BfDI assessment of ChatGPT blocking. (3) Hessian DPA questionnaire sent to OpenAI. "
        "| Source: https://fragdenstaat.de/en/request/chatgpt-in-behoerden/"
    ),
    "France": (
        "YES — CNAF algorithm battle: La Quadrature du Net + 14 organizations (incl. Amnesty International France) "
        "challenged CNAF's algorithmic risk-scoring for benefits. Filed before Conseil d'État. "
        "CNAF eventually published algorithm source code (Jan 2026). "
        "| Source: https://basta.media/discrimination-opacite-des-associations-attaquent-en-justice-algorithme-Caf"
    ),
    "Australia (federal)": (
        "YES — PM&C FOI-2023-156 disclosed AI testing documents: "
        "ChatGPT proof-of-concept chatbot for PM&C Enterprise Agreement info (Feb 2023). "
        "No cybersecurity review had been undertaken. Testing 'no longer active.' "
        "| Source: https://www.pmc.gov.au/sites/default/files/foi-logs/FOI-2023-156.pdf"
    ),
    "New Zealand": (
        "YES — Multiple OIA responses on AI: "
        "(1) NZDF OIA-2025-5581: AI tools disclosed. (2) IRD: approved tools (Copilot Studio, Snowflake Cortex, Genesys), "
        "usage guidelines, licensing, privacy impact assessments. (3) Ministry of Defence MOD-OIA-096. "
        "| Source: https://www.nzdf.mil.nz/assets/Uploads/DocumentLibrary/OIA-2025-5581-AI-tools.pdf"
    ),
    "EU Institutions": (
        "YES — Access to documents request exposed EC ChatGPT use: response contained 'utm_source=chatgpt.com' links. "
        "ICCL filed formal complaint to European Ombudsman (Nov 14, 2025). "
        "| Source: https://www.iccl.ie/news/european-commission-breaches-own-ai-guidelines-by-using-chatgpt-in-public-documents/"
    ),
    "Mexico": (
        "YES — 45 transparency requests revealed at least 14 federal institutions using AI. "
        "SAT uses statistical learning models for detecting fraudulent companies. "
        "Problem: absence of public documentation about criteria, parameters, and audits. "
        "| Source: https://latinamericanpost.com/science-technology/artificial-intelligence-en/mexicos-government-loves-ai-but-rules-lag-in-practice/"
    ),
    "USCBP (US Customs and Border Protection)": (
        "YES — CBP proactively released extensive AI FOIA records: "
        "Minimum Practices for Safety-Impacting AI (Jan 2026), AI Operations Directive (Nov 2025), "
        "Illicit Trade Model Cards (July 2025), vendor contracts (DataMinr, BabelStreet, Fivecast ONYX). "
        "chatCBP deployed May 2025. "
        "| Source: https://www.cbp.gov/document/foia-record/artificial-intelligence-records"
    ),

    # ---- v3 entries (override v2 where keys overlap) ----
    # --- CITIES (from Agent 2 - extensive MuckRock research) ---
    "Seattle, WA": "YES — MULTIPLE REQUESTS: (1) Rose Terse (April 28, 2023) to SPD: COMPLETED. Two installments released (June 16 & Aug 4, 2023), $1.25/installment. MuckRock: muckrock.com/foi/seattle-69/chatgpt-chat-history-145174/ (2) Todd Feathers (Aug 9, 2024) to SPD: ABANDONED — closed Aug 28, 2025 due to non-receipt of $1.25 check. SPD cited backlog of 2,000+ requests. MuckRock: muckrock.com/foi/seattle-69/agency-generative-ai-prompts-171511/ (3) Rose Terse (Nov 12, 2025) to SPD re Chief Barnes: AWAITING, est. April 17, 2026 (4,500+ backlog). (4) Rose Terse (Dec 16, 2025) to City Council re CM Hollingsworth AI images: COMPLETED March 4, 2026 — two ZIP files released. Council member 'used a free account so the prompts were not saved.' (5) Todd Feathers (Aug 12, 2024) to Education & Early Learning: COMPLETED Oct 4, 2024 — 9 files from 5 employees, most had not used GenAI.",

    "Kent, WA": "YES — COMPLETED in ~16 days. Rose Terse (MuckRock, July 24, 2023). 6 files released Aug 10, 2023: ChatGPT histories from officers Wesson, Stansfield, O'Reilly, Doherty, Hemmen + one exemption log. Hemmen's records partially redacted for non-city business. Key finding: 'There is no effective policy in place' regarding ChatGPT usage. Fees waived. MuckRock: muckrock.com/foi/kent-67/chatgpt-chat-history-kent-police-department-149817/",

    "Spokane, WA": "YES — COMPLETED (8 months). Rose Terse (MuckRock, July 24, 2023, completed March 26, 2024). 9 ChatGPT conversation sessions covering: Process Mapping Techniques, Police Records Systems Examples, Digital Forensic Evidence Mishandling, Cloud System WBS, OnBase, Planning Capital Projects, Local Government Budgeting, PTSD Prevention for Police. No ChatGPT policies or memos existed. Supplemental records provided after requester noted incomplete conversation. MuckRock: muckrock.com/foi/spokane-71/chatgpt-chat-history-spokane-police-department-149818/",

    "Tacoma, WA": "YES — FILED, No Responsive Documents. Rose Terse (MuckRock, July 24, 2023). MuckRock: muckrock.com/foi/tacoma-72/chatgpt-chat-history-tacoma-police-department-149819/",

    "Bellingham, WA": "YES — COMPLETED (major findings). Nate Sanford (KNKX/Cascade PBS) obtained thousands of pages of ChatGPT logs. Key findings: (1) Mayor Kim Lund's assistant used ChatGPT to draft letter to WA Dept of Commerce about Lummi Nation grant, prompting it to 'include some facts about violence in native communities.' (2) City's own draft AI policy was written with ChatGPT. (3) MAJOR SCANDAL (Jan 2026): City staffer used ChatGPT to draft biased contract requirements to 'favor VertexOne over Origin Smart City' for $2.7M utility billing contract — 16 requirements in final evaluation matrix matched ChatGPT output verbatim. Mayor initiated independent investigation. Sources: knkx.org/government/2026-01-05/city-of-bellingham-chatgpt-ai-contract-vendor, poynter.org/reporting-editing/2025/how-to-foia-chatgpt-logs-government-public-records/",

    "Everett, WA": "YES — COMPLETED. Nate Sanford (KNKX/Cascade PBS) obtained thousands of pages of ChatGPT logs. Filed to ~12 WA cities; Bellingham and Everett were 'the fastest and most responsive.' Reporting prompted editorials in both Bellingham and Everett newspapers calling for cautious AI governance. Sources: poynter.org/reporting-editing/2025/how-to-foia-chatgpt-logs-government-public-records/",

    "San Francisco, CA": "PARTIAL — Todd Feathers (Aug 9, 2024) to SFPD: No Responsive Documents. SFPD asked for clarification about which AI tools; after specifying ChatGPT and Axon's Draft One, found nothing. MuckRock: muckrock.com/foi/san-francisco-141/agency-generative-ai-prompts-171514/ Also: Patrick O'Doherty (Jan 2, 2026) filed for SFPD AI usage policies (vendor contracts, impact assessments, training materials): AWAITING RESPONSE. MuckRock: muckrock.com/foi/san-francisco-141/sfpd-ai-usage-policies-201130/",

    "Oakland, CA": "PARTIAL — Oaklandside investigation (Oct 17, 2025) confirmed Oakland is 'quietly incorporating AI into city work' including '311 chatbots to ChatGPT-written emails.' No MuckRock FOIA specifically for AI chat logs found. Source: oaklandside.org/2025/10/17/oakland-quietly-incorporating-ai-into-city-work/",

    "New York City": "PARTIAL — TWO PENDING: (1) Joey Scott (Oct 3, 2025) to Mayor's Office: AWAITING RESPONSE, deadline extended to April 10, 2026. MuckRock: muckrock.com/foi/new-york-city-17/ai-chatlogs-office-of-the-mayor-194104/ (2) Brandon Galbraith (Feb 5, 2026) to OTI re NYC AI Chatbot procurement/governance: AWAITING, est. March 13, 2026. MuckRock: muckrock.com/foi/new-york-city-17/nyc-ai-chatbot-governance-authorization-procurement-artifacts-204260/ Also: Jason Koebler/VICE (Feb 17, 2023) to NYC DOE re ChatGPT: COMPLETED May 30, 2024 (15 months).",

    "Washington, DC": "YES — REJECTED. Joey Scott (Oct 3, 2025) to Metropolitan Police: Deemed 'inadequate description' — MPD said request lacked specific incident descriptions, was too broad (4-month period), amounted to 'research rather than records access.' MuckRock: muckrock.com/foi/washington-48/ai-chatlogs-metropolitan-police-department-194108/",

    "Chicago, IL": "PARTIAL — Joey Scott (Oct 3, 2025): (1) Mayor's Office: No Responsive Documents. MuckRock: muckrock.com/foi/chicago-169/ai-chatlogs-mayors-office-194105/ (2) Chicago PD: AWAITING RESPONSE (Scott narrowed to command staff only). MuckRock: muckrock.com/foi/chicago-169/ai-chatlogs-chicago-police-department-194112/",

    "Minneapolis, MN": "YES — COMPLETED. Joey Scott (Oct 3, 2025). IT stated 'ChatGPT is not a City of Minneapolis application, so they cannot track or provide ChatGPT data.' HOWEVER, 37,000 Copilot items were collected. Two batches of redacted CoPilot records released (Dec 12, 2025 and Jan 2, 2026). Redactions per MN Data Practices Act. MuckRock: muckrock.com/foi/minneapolis-1607/ai-chatlogs-minneapolis-police-department-194109/",

    "Kansas City, MO": "YES — REJECTED. Kelly Kauffman (Sept 18, 2025) to KCPD: PD stated 'a record including chat histories of ChatGPT is not a record we compile or maintain.' MuckRock: muckrock.com/foi/kansas-city-331/chatgpt-usage-193232/ Also: Kauffman filed to Kansas City KS PD (Sept 18, 2025): No Responsive Documents after six deadline extensions, closed Dec 17, 2025.",

    "Atlanta, GA": "YES — FILED, No Responsive Documents. Joey Scott (Oct 3, 2025) to Atlanta PD: Same-day closure, no responsive documents. MuckRock: muckrock.com/foi/atlanta-325/ai-chatlogs-atlanta-police-department-194106/",

    "Portland, OR (TriMet)": "PARTIAL — Joey Scott (Oct 3, 2025) to Portland Police Bureau: No Responsive Documents. MuckRock: muckrock.com/foi/portland-166/ai-chatlogs-portland-police-bureau-194111/",

    "Houston, TX": "PARTIAL — Bradford William Davis filed 5 parallel ChatGPT requests to Fort Worth (same state) Oct 28, 2025. Fort Worth City Manager's Office COMPLETED with records from 4 employees. Fort Worth City Attorney's Office COMPLETED with ChatGPT data export (TX AG Decision OR2026-006497 authorized full disclosure). TxDOT reported 1,931 ChatGPT users, 4,527 Copilot accounts, 136,664 files — 912 released in batches. Precedent suggests TX agencies are responsive to AI records requests.",

    "Dallas, TX": "PARTIAL — Bradford William Davis filed to Fort Worth (same state) Oct 28, 2025. TX AG Decision OR2026-006497 authorized full disclosure of ChatGPT records without redactions. TxDOT released 912 files from 1,931 ChatGPT users. Strong TX precedent for AI record requests.",

    "Austin, TX": "PARTIAL — TX precedent strong: Fort Worth City Manager & City Attorney both completed with ChatGPT records. TX AG Decision OR2026-006497 authorizes full disclosure. TxDOT reported 1,931 ChatGPT users with 136,664 files, 912 released. Sources: MuckRock Fort Worth requests by Bradford William Davis (Oct 2025).",

    "Philadelphia, PA": "YES — REJECTED. Silvia Canelon (Dec 11, 2025) to Mayor's Office: Deemed 'insufficiently specific' — City argued scope lacked definition, citing PA precedent requiring requests to identify 'a discrete group of documents, either by type or recipient.' MuckRock: muckrock.com/foi/philadelphia-211/chatgpt-chat-history-2024-present-199720/ Also: PA OOR ruled (Feb 2026) AI chat logs may be exempt as 'notes and working papers.'",

    # --- FEDERAL AGENCIES (from Agent 4) ---
    "DHS (Department of Homeland Security)": "YES — TWO REQUESTS by CJ Ciaramella (MuckRock News): (1) 'DHS Public Affairs ChatGPT logs' (Sept 9, 2025): REJECTED. (2) 'DHS Public Affairs AI chat logs part deux' (Oct 1, 2025, #2026-HQFO-00031): Requested all prompts/logs for AI tools including image generation by DHS Office of Public Affairs Social Media team, June 1 - Oct 1, 2025. DHS acknowledged Nov 17, 2025 (delayed by govt shutdown), invoked 10-day extension citing 'voluminous amount of separate and distinct records.' AWAITING RESPONSE. MuckRock: muckrock.com/foi/united-states-of-america-10/dhs-public-affairs-ai-chat-logs-part-deux-193938/",

    "SEC (Securities and Exchange Commission)": "YES — COMPLETED. Sungho Park (Oct 22, 2024, #25-00248-FOIA) requested: ChatGPT chat logs of all SEC employees, emails concerning ChatGPT usage, training materials, communications, access records. SEC requested clarification (Dec 17, 2024); Park specified timeframe Nov 30, 2022 - Jan 7, 2025. CLOSED Jan 28, 2025 with responsive records released. MuckRock: muckrock.com/foi/united-states-of-america-10/chatgpt-related-data-175077/",

    "DOJ (Department of Justice)": "PARTIAL — Dillon Bergin (MuckRock, Jan 6, 2025) filed TWO requests to Office of Information Policy: (1) 'Auditing and testing of AI FOIA programs' — AWAITING. (2) 'Contracts and vendor materials of AI FOIA programs' — AWAITING. These target DOJ's AI use in FOIA processing, not employee chat logs. Bergin's broader investigation found CPSC shared 49 pages about MITRE FOIA Assistant tool; Commerce/Treasury/CPSC claimed no responsive docs despite reporting AI use. NARA estimated 38-month completion (mid-June 2028). Source: muckrock.com/news/archives/2025/may/07/how-federal-agencies-responded-to-our-requests-about-ai-use-in-foia/",

    "GSA (USAi.gov)": "UNKNOWN — No FOIA request found specifically targeting USAi.gov chat logs despite it being the government-wide AI platform.",

    "HHS": "UNKNOWN — No FOIA request found specifically for HHS ChatGPT Enterprise chat logs despite department-wide deployment (Sept 2025) with FISMA moderate ATO.",

    "CMS (Centers for Medicare & Medicaid Services)": "UNKNOWN — No FOIA request found targeting CMS Chat logs despite 100% employee access and 4,700+ trained through AI Ignite program.",

    "DOE (Department of Energy)": "UNKNOWN — No FOIA request found targeting DOE AI records despite having the most comprehensive published AI governance framework among federal agencies.",

    "EPA (Environmental Protection Agency)": "UNKNOWN — No FOIA request found targeting EPA AI records.",

    "OPM": "UNKNOWN — No FOIA request found targeting OPM AI/ChatGPT/Copilot records.",

    # --- States with new precedent data ---
    "Pennsylvania": "YES — BAD PRECEDENT. (1) Philadelphia Mayor's Office: Silvia Canelon (Dec 11, 2025) REJECTED as 'insufficiently specific.' (2) PA OOR ruling (Feb 2026, WITF Right-to-Know request): AI chat logs may be exempt as 'notes and working papers used solely for that official's own personal use' and 'internal, predecisional deliberations.' However, legal experts note every AI chat remains 'presumptively public' unless agency proves exemption. Sources: witf.org/2026/02/18/, penncapital-star.com/technology-information/right-to-know-in-pennsylvania-during-the-age-of-ai/",

    "New Hampshire": "PARTIAL — Kelly Kauffman (MuckRock, Dec 18, 2025) filed AI chat log requests to 20+ NH municipal clerks. Results: Nashua COMPLETED (records provided). Dover AWAITING. Concord/Salem/Claremont/Bow/Lebanon/Berlin REJECTED. Manchester/Keene/Troy/Peterborough/Somersworth/Belmont/Hopkinton/Gorham/Rochester/Hanover/Derry/Laconia: No Responsive Documents. Sources: MuckRock mass filing results.",

    "Florida": "PARTIAL — Sungho Park (Oct 22, 2024) filed to Florida Supreme Court for ChatGPT usage: No Responsive Documents after $257.12 in fees (refunded). MuckRock: muckrock.com/foi/florida-34/chatgpt-usage-for-florida-court-system-175079/",
}

# =============================================================================
# TEMPLATE ADAPTATION NOTES UPDATES (col 25)
# v2 entries first, then v3 entries override where keys overlap
# =============================================================================

template_updates = {
    # ---- v2 entries ----
    "Washington": (
        "Cite WA PRA (RCW 42.56) + EO 24-01 + WaTech EA-01-02-S policy. "
        "CRITICAL: Cite WA SOS June 2024 advice sheets confirming AI interactions are public records. "
        "File to WaTech for state-level platform logs. Also file to individual agencies. "
        "WA PRA is one of strongest — no exemption for 'burden.' No state-level FOIA filed yet = opportunity."
    ),
    "Bellevue, WA": (
        "Cite WA PRA + Bellevue AI Policy: 'All records prepared, owned, used or retained by the city in AI technology "
        "for city business are public records.' One of strongest municipal policies."
    ),
    "San Jose, CA": (
        "Cite CA PRA + City Policy Manual 1.7.12. City explicitly warns employees: "
        "'presume all interactions with Generative AI are PRA-able.' Mandatory AI use form submissions exist."
    ),
    "Tempe, AZ": (
        "Cite AZ Public Records Law + Tempe GenAI Use Guidelines: "
        "'content produced by generative AI is considered public record. This includes prompts and both draft and finalized content.' "
        "Strongest explicit municipal declaration in the US."
    ),
    "Tennessee": (
        "Cite TN Public Records Act TCA § 10-7-503 + TN Const. Art. I § 19. "
        "Cite Enterprise AI Policy 200-POL-007 and GenAI Policy ISC 3.00. "
        "For Nashville: Cite ISM-20 §1.4 (AI data subject to Public Records Act). "
        "1,000 ChatGPT Enterprise licenses. First-of-kind request in TN."
    ),
    "Maryland": (
        "Cite MPIA + Governor's EO 01.01.2024.02 + AI Governance Act SB818. "
        "Cite DoIT Responsible AI Policy (staff must follow records retention for AI). "
        "12,500 active GenAI users. Anthropic Claude partnership (Nov 2025). Target DoIT: pia@doIT.maryland.gov"
    ),
    "Missouri": (
        "Cite Sunshine Law Ch. 610 RSMo. 3-day deadline. "
        "Mention Google Gemini in secured tenant. MO SOS prohibits auto-delete of electronic messages. "
        "Target Office of Administration / ITSD for centralized AI records."
    ),
    "Virginia": (
        "Cite VA FOIA § 2.2-3700. Request BOTH: (a) AI registry submissions themselves, "
        "(b) Copilot Chat usage/session logs, (c) VITA's records of AI use approvals. "
        "Registry is FOIA-able. Budget for fees (~$25/hr)."
    ),
    "United Kingdom (5-8 depts)": (
        "Cite FOI Act 2000. Peter Kyle precedent proves ChatGPT logs are disclosable. "
        "ICO: platform (email, WhatsApp, AI) irrelevant — content determines disclosability. "
        "Broader requests may hit s.12 cost exemption (>24 hrs) — target specific ministers/departments."
    ),
    "Norway (2-3 agencies)": (
        "Cite offentleglova. Tromso precedent: County Governor ruled AI chat logs ARE public documents. "
        "673 pages disclosed. Strongest international legal precedent."
    ),

    # ---- v3 entries (override v2 where keys overlap) ----
    "Seattle, WA": "Cite WA PRA (RCW 42.56). Reference POL-209 §6.1-6.3 (now POL-211) requiring AI records disclosure. Note vendor export capability requirement (§6.2). Multiple successful precedents: Rose Terse completed 2023, Seattle Education completed 2024. Warn about SPD backlog (4,500+ open requests as of Nov 2025). Request ChatGPT data exports AND Copilot logs separately. MRSC analysis at mrsc.org/stay-informed/mrsc-insight/july-2024/public-records-and-ai confirms records status.",

    "Kent, WA": "Cite WA PRA (RCW 42.56). Fastest known completion (~16 days). Reference Rose Terse's successful 2023 request for ChatGPT histories from police. 6 files released, fees waived. Note Kent PD had 'no effective policy in place' — may still lack one. Target specific departments. Request both ChatGPT exports and Copilot data.",

    "Spokane, WA": "Cite WA PRA (RCW 42.56). Reference Rose Terse's successful 2023 request (completed in 8 months). 9 ChatGPT sessions obtained. No existing policies/memos. Note: requester had to follow up about incomplete conversations — verify completeness of responsive records.",

    "Bellingham, WA": "Cite WA PRA (RCW 42.56). Strong precedent: Nate Sanford/KNKX obtained thousands of pages. Bellingham was 'fastest and most responsive.' Request ChatGPT data exports specifically — contract manipulation scandal (Jan 2026) shows significant internal AI use. City now has heightened awareness of AI records obligations post-scandal.",

    "Everett, WA": "Cite WA PRA (RCW 42.56). Strong precedent: Nate Sanford/KNKX obtained thousands of pages. Everett was among 'fastest and most responsive' of ~12 WA cities. Good filing target.",

    "New York City": "Cite NY FOIL (Public Officers Law §87). Two pending requests (Joey Scott to Mayor's Office, Brandon Galbraith to OTI). Reference OTI GenAI Use Guidance and AI Principles & Definitions. NYC DOE request took 15 months (Koebler/VICE). Target specific agencies — OTI most likely to have logs. Note: Mayor's Office deadline extended to April 10, 2026.",

    "Washington, DC": "Cite DC FOIA (D.C. Code §2-531). WARNING: MPD rejected Scott's request as 'inadequate description' and 'research rather than records access.' Must be highly specific: name departments, date ranges, specific AI tools. Reference OCTO AI/ML Governance Policy requiring 'auditing and logging mechanisms.'",

    "Chicago, IL": "Cite IL FOIA (5 ILCS 140). Mayor's Office returned no responsive docs. CPD request (narrowed to command staff) still pending. Reference DoIT AI Policy v2. HB 3773 requires 4-year retention for employment AI decisions — cite if relevant.",

    "Minneapolis, MN": "Cite MN Government Data Practices Act (Ch. 13). Strong precedent: Scott obtained 37,000 Copilot items despite city claiming 'ChatGPT is not a city application.' Request COPILOT data specifically (city may not track ChatGPT). Expect redactions per §13.43. Two batches released Dec 2025 - Jan 2026.",

    "Kansas City, MO": "Cite MO Sunshine Law (Ch. 610). WARNING: KCPD rejected claiming ChatGPT histories are 'not a record we compile or maintain.' Request Microsoft Copilot data instead (city may have enterprise M365). Also request IT department records of AI tool procurement/deployment.",

    "Atlanta, GA": "Cite GA Open Records Act (O.C.G.A. 50-18-70). Reference SS-23-002 requiring agencies to save prompts, outputs, and reviewer identity. PD returned no responsive docs to Scott's Oct 2025 request. Try city administration/IT departments rather than PD.",

    "San Francisco, CA": "Cite CA PRA (Gov. Code §6250). Reference Chapter 22J and GenAI Guidelines (July 2025) explicitly noting content 'may constitute public records.' SFPD had no responsive docs (Feathers, 2024). Request from departments known to use AI per Chapter 22J inventory (data.sfgov.org/City-Management-and-Ethics/San-Francisco-AI-Use-Inventory-Chapter-22J-/9tj4-z32q).",

    "Oakland, CA": "Cite CA PRA (Gov. Code §6250). Reference ITD Interim AI Security Guidelines' citation requirement for AI use in public records. Oaklandside confirmed AI use in city operations (Oct 2025). Target IT department and agencies with known AI pilots.",

    "Portland, OR (TriMet)": "Cite OR Public Records Law (ORS 192.311). PPB returned no responsive docs. Reference EIS Interim AI Guidelines v1.5 requiring AI output labeling. Try city administration/IT rather than police.",

    "Philadelphia, PA": "Cite PA Right-to-Know Law (65 P.S. §67.101). WARNING: Bad precedent. Canelon's request REJECTED as 'insufficiently specific.' OOR ruled (Feb 2026) AI logs may be exempt as 'notes and working papers.' MUST specify narrow date range, specific departments, specific AI tools (ChatGPT, Copilot), and types of records sought. Each chat remains 'presumptively public' unless agency proves exemption.",

    "New Hampshire": "Cite NH Right-to-Know Law (RSA 91-A). Reference RSA 5-D requiring AI disclosure. Kauffman's mass filing (Dec 2025) to 20+ cities: most returned no responsive docs. Nashua provided records. Target state agencies (which are subject to RSA 5-D) rather than municipalities.",

    "Florida": "Cite FL Public Records Act (Ch. 119). No state-level AI use policy found. Florida Supreme Court had no responsive docs (Park, Oct 2024). SB 1118 may create new data center exemptions — file before this takes effect. Miami-Dade County has own AI policy (Directive 231203) — better target than state.",

    "Pennsylvania": "Cite PA Right-to-Know Law (65 P.S. §67.101). CRITICAL: OOR Feb 2026 ruling allows agencies to claim AI chat logs are exempt 'notes and working papers.' Counter by: (1) specifying AI outputs used in official communications/policies, (2) narrow date ranges, (3) specific departments, (4) citing that each chat is 'presumptively public.' Avoid broad 'all ChatGPT logs' requests.",

    "New Jersey": "Cite NJ OPRA (N.J.S.A. 47:1A-1). Reference DORES AI/ML retention guidelines (first state-level). 15,000+ employees use NJ AI Assistant — request from this platform specifically. Also request Copilot logs from enterprise M365.",

    "Texas": "Cite TX PIA (Gov. Code Ch. 552). Strong precedent: Fort Worth completed with TX AG Decision OR2026-006497 authorizing full disclosure. TxDOT had 1,931 ChatGPT users, 136,664 files. Reference TSLAC guidance that AI records = human records. Expect large volume of responsive records. Request ChatGPT data exports and Copilot histories.",

    "New York": "Cite NY FOIL (Public Officers Law §87). Reference NYS-P24-001 requiring 'data retention settings.' Comptroller audit (2023-S-50) found agencies lacked AI testing procedures — cite in request. Target ITS and specific agencies.",

    "Georgia": "Cite GA Open Records Act (O.C.G.A. 50-18-70). Reference SS-23-002 explicitly requiring agencies to save prompts, outputs, and reviewer identity. Cite 'Red Light, Green Light' guidelines. Strong legal basis for AI record requests.",

    "North Carolina": "Cite NC Public Records Act (N.C.G.S. Ch. 132). Reference UNC legal analysis (March 2026) confirming AI prompts and outputs are public records. Cite Gray Media Group v. City of Charlotte (290 N.C. App. 384, 2023) for records on third-party servers. Reference Copilot M365 Policy (Feb 2025).",

    "Houston, TX": "Cite TX PIA. Fort Worth precedent: TX AG ruled in favor of ChatGPT data disclosure (OR2026-006497). Harris County has GenAI Guidelines requiring clear identification of AI-generated content.",

    "Dallas, TX": "Cite TX PIA + Fort Worth precedent (TX AG OR2026-006497 — disclosure ordered). City Manager envisions Dallas as 'model city for AI.' CIO/CDO briefed Council March 2025.",

    "Austin, TX": "Cite TX PIA + Fort Worth precedent (TX AG OR2026-006497). Council Resolution Item 55 requires audits and human oversight. TX State Library: AI records classified same as non-AI records.",

    # Federal agencies
    "DHS (Department of Homeland Security)": "Cite FOIA (5 U.S.C. §552). Reference DHS-OpenAI contract: 'outputs are federal records.' Request DHSChat logs specifically (19,000+ HQ staff). Ciaramella's second request pending — 'voluminous' records cited. Specify DHS Office of Public Affairs or specific components. Note: DHS has acknowledged record existence through 'voluminous' response.",

    "SEC (Securities and Exchange Commission)": "Cite FOIA (5 U.S.C. §552). Strong precedent: Park's request completed Jan 2025 with records released. Specify ChatGPT, Bing Chat, Claude 2 (all provisionally approved). Include date range and specific divisions.",

    "DOJ (Department of Justice)": "Cite FOIA (5 U.S.C. §552). Bergin's requests for AI-in-FOIA still pending. For chat logs, file separate request specifically targeting employee ChatGPT/Copilot use (not AI in FOIA processing).",

    "GSA (USAi.gov)": "Cite FOIA (5 U.S.C. §552). No prior FOIA for USAi.gov logs — would be novel request. Reference USAi.gov platform specifically. GSA may argue platform-level data is not agency records.",

    "HHS": "Cite FOIA (5 U.S.C. §552). ChatGPT Enterprise deployed dept-wide Sept 2025 with FISMA moderate ATO. No prior FOIA for chat logs — novel request. Specify ChatGPT Enterprise and HHSGPT separately.",

    "CMS (Centers for Medicare & Medicaid Services)": "Cite FOIA (5 U.S.C. §552). Request CMS Chat logs specifically (100% employee access). 4,700+ trained through AI Ignite. No prior FOIA — novel request.",
}

# =============================================================================
# AI DEPLOYMENT UPDATES (col 6) — from v3 only
# =============================================================================

ai_deploy_updates = {
    "HHS": "ChatGPT Enterprise (dept-wide, Sept 2025, FISMA moderate ATO) + HHSGPT. GAO: 7→116 AI use cases (2023→2024). Largest federal AI expansion.",
    "CMS (Centers for Medicare & Medicaid Services)": "CMS Chat (Dec 2024, 100% employee access). AI Ignite training: 4,700+ employees. AI fraud detection: saved $2B.",
    "DHS (Department of Homeland Security)": "DHSChat (Dec 2024, 19,000+ HQ staff, 10 operating agencies). Previously approved: ChatGPT, Claude 2, BingChat, DALL-E2, Grammarly.",
    "DOE (Department of Energy)": "PermitAI for environmental permitting. Most comprehensive published AI governance framework: AI Strategy, Compliance Plan, GenAI Policy, GenAI Reference Guide, Enterprise Data Strategy.",
    "GSA (USAi.gov)": "USAi.gov: Government-wide AI platform with Chat, API, Console. Models from OpenAI, Google, Amazon, Meta, Microsoft, xAI. FedRAMP-compliant.",
    "Minneapolis, MN": "Microsoft Copilot (37,000+ items collected from PD alone). City states ChatGPT is 'not a city application.'",
    "New Jersey": "NJ AI Assistant: 15,000+ state employees, 300,000+ sessions, 1M+ prompts (as of Feb 2026). Hosted on state infrastructure.",
    "Massachusetts": "ChatGPT deployed across executive branch (Feb 2026 announcement). First state to do so.",
    "California - CDT (Poppy)": "Poppy: State-approved, vendor-agnostic GenAI platform. 'Information never leaves California's trusted environment.'",
    "Bellingham, WA": "ChatGPT used extensively by city staff. Thousands of pages of logs obtained via PRA. Used for official correspondence, policy drafting, contract evaluation.",
}

# =============================================================================
# APPLY ALL UPDATES
# =============================================================================

apply_updates(ws, jur_map, retention_updates, COL_RETENTION)
apply_updates(ws, jur_map, precedent_updates, COL_PRECEDENT)
apply_updates(ws, jur_map, template_updates, COL_TEMPLATE_NOTES)
apply_updates(ws, jur_map, ai_deploy_updates, COL_AI_DEPLOY)

save_tracker(wb, "foia_tracker_detailed_v3.xlsx")
