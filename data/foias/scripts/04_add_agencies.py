#!/usr/bin/env python3
"""
Step 4: Add 12 new federal agencies to foia_tracker_detailed_v3.xlsx with all 25 columns populated.
Also appends updated federal AI FOIA guidance to all federal rows' Notes.
"""
from helpers import load_tracker, save_tracker, add_agency_row, COL_CAT, COL_NOTES

wb, ws = load_tracker("foia_tracker_detailed_v3.xlsx")

# Current last row
last_row = ws.max_row  # Should be 111

# New agencies to add (in order)
new_agencies = [
    {
        "cat": "US-Fed", "tier": "Tier 2", "jurisdiction": "DOL (Department of Labor)",
        "country": "US", "level": "Federal",
        "ai_deploy": "EMPLOYEE-FACING: Chief AI Officer Mangala Kuppa (also CIO, promoted March 2026). AI-generated imagery for social media (DOL posters styled as historical artwork). No confirmed enterprise ChatGPT/Copilot/Gemini deployment publicly reported — DOL did not respond to FedScoop's 2024 GenAI survey. Sub-agencies OSHA and MSHA have DOL-wide AI tools. Sources: fedscoop.com/how-risky-is-chatgpt-depends-which-federal-agency-you-ask/",
        "employees": "~15,000-17,000",
        "proxy": "No",
        "filing": "ONLINE PORTAL: dol.secureocp.com\nEMAIL: foiarequests@dol.gov\nFAX: 202-693-5389\nMAIL: Office of the Solicitor, DMALS, 200 Constitution Ave NW, Room N-2420, Washington, DC 20210\nComponent-specific: OSHA: osha.foia@dol.gov; BLS: BLSFOIAServiceCenter@bls.gov; ETA: foiastatus.eta@dol.gov",
        "source_url": "https://www.dol.gov/general/foia",
        "cost": "$0.15/page; fee categories vary by requester type; default $25 limit; fee waivers available",
        "statute": "5 U.S.C. § 552",
        "deadline": "20 business days",
        "turnaround": "3-6+ months",
        "template": "", "govai": "", "status": "", "date_filed": "", "date_due": "", "date_received": "",
        "notes": "Decentralized FOIA structure — file with Office of the Solicitor for routing to all components. Chief AI Officer is Mangala Kuppa. DOL used AI to generate social media artwork (triggering Morisy FOIA). Sub-agencies OSHA and MSHA have documented AI tools.",
        "requirements": "Standard federal FOIA",
        "retention": "PARTIAL — No DOL-specific AI records retention policy found. Federal Records Act (44 U.S.C. 3101), NARA Bulletin 2023-04 (collaboration platform records), and OMB M-24-10 apply. NARA Bulletin 2023-02 covers 'chats' and 'electronic messaging systems' which arguably includes AI interactions. Sources: archives.gov/records-mgmt/bulletins/2023/2023-02",
        "precedent": "YES — PENDING. Michael Morisy filed 'Department of Labor AI-Generated Poster Prompts & Extras' (Nov 3, 2025, #2026-F-00378): Requested AI chat logs/prompt logs used to generate DOL social media art, all generated imagery, planning docs, vendor communications. AWAITING RESPONSE (4+ months, no reply as of March 2026). MuckRock: muckrock.com/foi/united-states-of-america-10/department-of-labor-ai-generated-poster-prompts-extras-196900/",
        "template_notes": "Cite FOIA (5 U.S.C. §552). DOL has decentralized FOIA — request routing to all components (OSHA, BLS, ETA, WHD, etc.). Name ChatGPT, Copilot, Gemini, and any custom AI chatbots. Cite Morisy precedent (#2026-F-00378) to show AI prompt logs are recognized as FOIA-able at DOL. DOL did not respond to FedScoop AI survey — request broadly. Specify ChatGPT admin console ZIP exports and M365 Purview compliance portal Copilot logs.",
    },
    {
        "cat": "US-Fed", "tier": "Tier 2", "jurisdiction": "DOT (Department of Transportation)",
        "country": "US", "level": "Federal",
        "ai_deploy": "EMPLOYEE-FACING: PHMSA proposed ChatGPT pilot for rulemaking (Sept 2023). DOT 'working to optimize resources and integrate AI' (Dec 2025). MITRE researched air traffic language AI for FAA. No confirmed enterprise-wide GenAI deployment — DOT did not respond to FedScoop's 2024 AI survey. Sources: fedscoop.com (PHMSA ChatGPT pilot, Sept 5, 2023)",
        "employees": "~55,000-58,000",
        "proxy": "No",
        "filing": "ONLINE: foia.gov (search DOT)\nMAIL: Department of Transportation, Office of the Secretary, 1200 New Jersey Ave SE, Washington, DC 20590\nNOTE: transportation.gov/foia returning 403 errors — use foia.gov\nRegulations: 49 CFR Part 7",
        "source_url": "https://www.transportation.gov/foia",
        "cost": "Standard federal; $0.10-$0.25/page; first 100 pages + 2 hrs search free for non-commercial",
        "statute": "5 U.S.C. § 552",
        "deadline": "20 business days",
        "turnaround": "4-8+ months",
        "template": "", "govai": "", "status": "", "date_filed": "", "date_due": "", "date_received": "",
        "notes": "DOT has multiple operating administrations (FAA, FHWA, FRA, FMCSA, PHMSA, etc.). FAA accounts for majority of employees. PHMSA ChatGPT pilot for rulemaking is documented. DOT website returning 403 errors — file via foia.gov. WARNING: Bradford William Davis's request has been completely ignored for 4+ months with no acknowledgement despite multiple follow-ups.",
        "requirements": "Standard federal FOIA",
        "retention": "PARTIAL — No DOT-specific AI records retention policy found. Federal Records Act, NARA Bulletins, OMB M-24-10 apply. 49 CFR Part 7 governs DOT FOIA. PHMSA's ChatGPT pilot for rulemaking implies some AI governance awareness.",
        "precedent": "YES — PENDING (IGNORED). Bradford William Davis filed 'Federal AI Usage | US DOT' (Oct 28, 2025): Requested ChatGPT chat histories from Office of General Counsel, Office of Litigation/Enforcement, and Office of Public Affairs, from Jan 21, 2025 to present. Included detailed ChatGPT export instructions. Status: AWAITING ACKNOWLEDGEMENT — agency has NOT responded despite 4+ MuckRock follow-ups over 4 months. MuckRock: muckrock.com/foi/united-states-of-america-10/federal-ai-usage-us-department-of-transportation-196228/",
        "template_notes": "Cite FOIA (5 U.S.C. §552). WARNING: DOT extremely non-responsive (Davis request ignored 4+ months). File via foia.gov for official tracking record. Specify all operating administrations (FAA, FHWA, FRA, FMCSA, PHMSA). Cite PHMSA ChatGPT pilot as evidence of AI usage. Be prepared to appeal and potentially litigate — contact DOT FOIA Public Liaison. Name ChatGPT, Copilot, Gemini.",
    },
    {
        "cat": "US-Fed", "tier": "Tier 1", "jurisdiction": "DOD (Department of Defense)",
        "country": "US", "level": "Federal",
        "ai_deploy": "EMPLOYEE-FACING: GenAI.mil (launched Dec 2025) — DOD's enterprise GenAI platform. 1.1 MILLION unique users. Available to all 3+ million DOD personnel. Defense Secretary Hegseth issued department-wide mandate for all employees to use it. Google Gemini first integrated; ChatGPT added Feb 2026. Agent Designer (March 2026) lets employees create custom AI assistants. GPT-4 on Azure Government Top Secret Cloud (May 2024) for classified environments. Legacy: NIPRGPT (sunset Dec 2025), CamoGPT (Army), AskSage. AI coding tools solicitation (Feb 2026) for 'tens of thousands' of developers. Sources: defensescoop.com/2026/02/02/military-branches-genai-mil-enterprise-ai-adoption/, defensescoop.com/2026/02/09/pentagon-adding-chatgpt-to-enterprise-generative-ai-platform/",
        "employees": "~950,000 civilian + 1.3M active duty + 800K Guard/Reserve = ~3M+ total",
        "proxy": "No",
        "filing": "ONLINE: foia.gov (search DOD components)\nMAIL (OSD): OSD/JS FOIA Requester Service Center, 1155 Defense Pentagon, Washington, DC 20301-1155\nNOTE: DOD FOIA is extremely decentralized — hundreds of components with separate FOIA offices. File with specific components.\nRegulations: 32 CFR Part 286",
        "source_url": "https://open.defense.gov/Transparency/FOIA.aspx",
        "cost": "Search: $45.80/hr (professional), $16.50/hr (clerical) after first 2 hrs; Duplication: $0.15/page after first 100; Min $25 threshold",
        "statute": "5 U.S.C. § 552",
        "deadline": "20 business days",
        "turnaround": "6-18+ months",
        "template": "", "govai": "", "status": "", "date_filed": "", "date_due": "", "date_received": "",
        "notes": "LARGEST AI DEPLOYMENT IN FEDERAL GOVERNMENT. GenAI.mil has 1.1M users with Hegseth mandate for all employees. Most complex FOIA environment — hundreds of components. File with CDAO (Chief Digital and AI Office) for enterprise platform data. U.S. Army CIO memo (Aug 26, 2025) is the most explicit federal statement that ALL AI prompts and outputs are official records subject to FOIA. WARNING: George Johnston's Cyber Command request REJECTED as too broad ('does not permit an organized, non-random search'). Must be specific about platforms, date ranges, offices.",
        "requirements": "Standard federal FOIA. May require classification review.",
        "retention": "PARTIAL — No explicit DOD-wide policy declaring GenAI.mil interactions as federal records. However, U.S. Army CIO Leonel Garciga signed memo (Aug 26, 2025) requiring: all AI prompts and outputs must be 'properly identified, retained and secured as official records'; personnel must 'capture and manage all aspects of the AI interaction'; FOIA responses must disclose AI use. This is the STRONGEST agency-level statement. DOD Directive 5015.02 governs records management. OMB M-24-10 applies. Sources: executivegov.com/articles/army-garciga-application-owners-ai-records-mgmt",
        "precedent": "PARTIAL — (1) Jason Koebler (Feb 17, 2023) to DOD Education Activity: No Responsive Documents (#23-F-00079). MuckRock: muckrock.com/foi/united-states-of-america-10/chatgpt-department-of-defense-education-activity-140687/ (2) George Johnston (May 6, 2023) to U.S. Cyber Command: REJECTED as too broad — USCYBERCOM said request 'does not permit an organized, non-random search.' MuckRock: muckrock.com/foi/united-states-of-america-10/chatgpt-145437/ KEY LESSON: DOD components will reject vague requests. Must provide specific search parameters.",
        "template_notes": "Cite FOIA (5 U.S.C. §552). CRITICAL: Name GenAI.mil specifically (1.1M users). Also name ChatGPT, Google Gemini for Government, CamoGPT, NIPRGPT (historical), AskSage, Agent Designer, M365 Copilot. Cite Army CIO Aug 2025 memo declaring AI interactions are official records. WARNING: Broad requests WILL be rejected (Johnston precedent). Target specific components: CDAO for platform data, each service branch separately. Request GenAI.mil admin console exports and usage analytics. For classified AI (GPT-4 on Top Secret cloud), expect Exemption 1 withholdings.",
    },
    {
        "cat": "US-Fed", "tier": "Tier 2", "jurisdiction": "Treasury (Department of the Treasury)",
        "country": "US", "level": "Federal",
        "ai_deploy": "EMPLOYEE-FACING: Confirmed using xAI Grok (testing, early 2026), OpenAI Codex (software engineers), Google Gemini (software engineers). 129 AI use cases in Treasury inventory (nearly doubled from 54 in 2024): NLP employee Q&A system, GenAI for IT service desk tickets, AI coding assistants. IRS accounts for 61 use cases. Draft solicitation (June 2025) for ChatGPT-like chatbot + coding assistant. Sources: fedscoop.com/treasury-irs-ai-use-case-inventory/ (Jan 29, 2026), fedscoop.com/treasury-signals-interest-in-ai-for-code-writing-chatbot-tools/ (June 9, 2025)",
        "employees": "~85,000-100,000 (IRS is ~75,000+)",
        "proxy": "No",
        "filing": "ONLINE PORTAL: foia.treasury.gov/app/Home.aspx\nEMAIL: FOIA@treasury.gov\nMAIL: Director FOIA and Transparency, 1500 Pennsylvania Ave NW, Washington, DC 20220\nPHONE: (202) 622-0930\nFAX: (202) 622-3895\nIRS: file separately via foia.gov\nOCC: foia-pal.occ.gov\nRegulations: 31 CFR Part 1, Subpart A",
        "source_url": "https://home.treasury.gov/footer/freedom-of-information-act",
        "cost": "Standard federal; first 100 pages + 2 hrs search free for non-commercial/media/educational",
        "statute": "5 U.S.C. § 552",
        "deadline": "20 business days",
        "turnaround": "2-6 months (Departmental); 6-12+ months (IRS)",
        "template": "", "govai": "", "status": "", "date_filed": "", "date_due": "", "date_received": "",
        "notes": "Treasury has confirmed on-record AI tool usage (Grok, Codex, Gemini). 129 AI use cases in inventory. Multi-bureau structure — IRS, FinCEN, OCC, BEP, Mint each have separate FOIA. IRS FOIA is extremely slow. Consider filing Departmental Offices separately from IRS. Robert Delaware's successful CFTC and CFPB requests (sister financial regulatory agencies) provide strong precedent.",
        "requirements": "Standard federal FOIA",
        "retention": "PARTIAL — No Treasury-specific AI records retention policy found. 129 AI use cases in inventory implies governance structure. Federal Records Act, NARA Bulletins, OMB M-24-10 apply. Treasury spokesperson confirmed Codex/Gemini/Grok usage on the record (FedScoop, March 2026).",
        "precedent": "UNKNOWN — No FOIA requests for AI/ChatGPT chat logs at Treasury Departmental Offices found on MuckRock. Matthew Petti filed at Treasury for 'ChatGPT Chat Histories and Guidance' (Nov 21, 2025, #2026-FOIA-00208) but this targeted FinCEN. AWAITING RESPONSE, est. completion May 28, 2026. Also filed at FinCEN separately (Nov 5, 2025): REJECTED. Relevant cross-agency: Robert Delaware's CFTC (completed May 2024, docs released) and CFPB (completed May 2024, redacted docs released) requests. MuckRock: muckrock.com/foi/united-states-of-america-10/chatgpt-chat-histories-and-guidance-198123/",
        "template_notes": "Cite FOIA (5 U.S.C. §552). Name Grok (xAI), Codex (OpenAI), Google Gemini, ChatGPT, M365 Copilot — Treasury spokesperson confirmed these on the record. Reference 129-item AI use case inventory. Cite CFTC and CFPB precedents (Robert Delaware, completed with docs released) since these are sister financial agencies. Request NLP employee Q&A system and IT service desk GenAI logs. File Departmental Offices and IRS separately (IRS is much slower).",
    },
    {
        "cat": "US-Fed", "tier": "Tier 2", "jurisdiction": "IRS (Internal Revenue Service)",
        "country": "US", "level": "Federal",
        "ai_deploy": "EMPLOYEE-FACING: 'Winnie' internal chatbot (IT self-service FAQ). DOGE/AI initiative using AI tools to interact with 1960s-era COBOL Individual Master File (IMF) system — House Democrats demanded answers Dec 18, 2025. 61 AI use cases in Treasury inventory (nearly half of Treasury total). Treasury draft solicitation (June 2025) for coding assistant (like GitHub Copilot/AWS CodeWhisperer) and conversational chatbot (like ChatGPT). No confirmed enterprise ChatGPT/Copilot deployment for general employees. Sources: fedscoop.com/irs-using-ai-cobol-doge-house-dems-letter/, fedscoop.com/treasury-signals-interest-in-ai-for-code-writing-chatbot-tools/",
        "employees": "~70,000-80,000 (post-DOGE cuts; was ~87,000)",
        "proxy": "No",
        "filing": "ONLINE PORTAL: foiapublicaccessportal.for.irs.gov\nMAIL (non-taxpayer): IRS GLDS Support Services, Stop 211, PO Box 621506, Atlanta, GA 30362-3006\nFAX: 877-807-9215 (policies); 877-891-6035 (taxpayer records)\nFOIA Public Liaison: 312-292-2929\nNOTE: IRS REQUIRES handwritten signature and identity verification (driver's license or notarized statement)",
        "source_url": "https://www.irs.gov/privacy-disclosure/irs-freedom-of-information",
        "cost": "Commercial: $0.20/page, $50/hr search/review. Media/education: $0.20/page (first 100 free). Others: $0.20/page (first 100 free), $50/hr (first 2 hrs free). No fee if total <$25.",
        "statute": "5 U.S.C. § 552",
        "deadline": "20 business days",
        "turnaround": "6-18+ months",
        "template": "", "govai": "", "status": "", "date_filed": "", "date_due": "", "date_received": "",
        "notes": "HIGH-PRIORITY TARGET due to DOGE/COBOL AI initiative and Congressional scrutiny (House Democrats Dec 2025 letter). IRS requires HANDWRITTEN SIGNATURE and identity verification. One of the largest FOIA backlogs in federal government. 'Winnie' chatbot is named internal tool. File separately from Treasury Departmental Offices. Strong public interest angle: AI tools being used to manage taxpayer data and modernize critical tax infrastructure.",
        "requirements": "Handwritten signature + identity verification (driver's license or notarized statement) REQUIRED",
        "retention": "PARTIAL — No IRS-specific AI records retention policy found. Federal Records Act (44 U.S.C. 3101) covers all records 'made or received by an agency in connection with the transaction of public business.' OMB M-24-10 applies (Treasury/IRS is a CFO Act agency). 61 AI use cases in Treasury inventory. Sources: fedscoop.com/treasury-irs-ai-use-case-inventory/",
        "precedent": "UNKNOWN — No FOIA requests specifically targeting IRS for AI/ChatGPT chat logs found on MuckRock. Matthew Petti's Treasury request (#2026-FOIA-00208, Nov 2025) is awaiting response. Robert Delaware's CFTC (completed, docs released) and CFPB (completed, redacted docs released) requests provide cross-agency precedent.",
        "template_notes": "Cite FOIA (5 U.S.C. §552). CRITICAL: Include handwritten signature and identity verification. Name 'Winnie' chatbot, ChatGPT, Copilot, Codex, Gemini. Reference DOGE/COBOL AI initiative and House Democrats' Dec 2025 letter demanding answers. Cite Petti Treasury precedent and Delaware CFTC/CFPB completed requests. Request admin console exports. One of slowest federal FOIA processors — file early, plan for appeal/litigation.",
    },
    {
        "cat": "US-Fed", "tier": "Tier 2", "jurisdiction": "EOIR (Executive Office for Immigration Review)",
        "country": "US", "level": "Federal",
        "ai_deploy": "UNKNOWN — No public reporting of ChatGPT, Copilot, Gemini, or custom GenAI tool deployment at EOIR. EOIR is primarily a judicial body (immigration courts, Board of Immigration Appeals). Technology infrastructure known to be outdated — long-struggled with case management modernization. DOJ parent agency has not been prominently featured in GenAI deployment reporting.",
        "employees": "~2,500-3,000",
        "proxy": "No",
        "filing": "ONLINE PORTAL: foia.eoir.justice.gov (EOIR PAL)\nALTERNATIVE: foia.gov\nEMAIL (inquiries only): EOIR.FOIARequests@usdoj.gov\nPHONE: 703-605-1297\nMAIL: Office of the General Counsel, Attn: FOIA Service Center, EOIR, 5107 Leesburg Pike, Suite 2150, Falls Church, VA 22041",
        "source_url": "https://www.justice.gov/eoir/freedom-information-act-foia",
        "cost": "Standard DOJ fees per 28 CFR Part 16; fee waivers available",
        "statute": "5 U.S.C. § 552",
        "deadline": "20 business days",
        "turnaround": "1-3 months (simple); 6-12+ months (complex)",
        "template": "", "govai": "", "status": "", "date_filed": "", "date_due": "", "date_received": "",
        "notes": "Under-scrutinized agency for AI. Strong public interest angle: whether AI tools are used in immigration court proceedings or case preparation — due process implications for respondents in removal proceedings. EOIR may redirect AI procurement questions to DOJ Justice Management Division or OCIO — file with both. Expect Exemption 5 (deliberative process) if AI used in adjudication. Small, judicial-focused agency — may have no enterprise GenAI tools.",
        "requirements": "Standard federal FOIA",
        "retention": "UNKNOWN — No EOIR-specific AI records retention policy found. DOJ as parent agency has not published AI-specific records retention directive. Federal Records Act and OMB M-24-10 apply.",
        "precedent": "UNKNOWN — No FOIA requests targeting EOIR for AI/ChatGPT/Copilot chat logs found on MuckRock. Untested target. Cross-agency precedents: Delaware CFTC/CFPB (completed), Park SEC (completed), Ciaramella DHS (pending).",
        "template_notes": "Cite FOIA (5 U.S.C. §552). File via EOIR PAL portal (foia.eoir.justice.gov). Also file with DOJ OCIO since AI procurement may be DOJ-level. Name ChatGPT, Copilot, Gemini, any DOJ/EOIR-specific AI tools. Frame around due process: whether AI is used in immigration adjudication. Expect Exemption 5 for adjudicative AI use. No handwritten signature required (unlike IRS).",
    },
    {
        "cat": "US-Fed", "tier": "Tier 2", "jurisdiction": "DOC (Department of Commerce)",
        "country": "US", "level": "Federal",
        "ai_deploy": "EMPLOYEE-FACING: Limited public information. NIST had 'No Responsive Documents' for ChatGPT account signups FOIA (Bouchtia, June 2025). Commerce approach to ChatGPT characterized as 'unclear' (FedScoop, Feb 2024). Census Bureau and others exploring Model Context Protocol (MCP) servers for AI access to public data (FedScoop, Feb 2026). NIST leads federal AI standards/framework work but unclear if NIST employees use GenAI tools internally. No confirmed enterprise deployment. Sources: fedscoop.com/how-risky-is-chatgpt-depends-which-federal-agency-you-ask/, fedscoop.com/federal-goverment-mcp-improve-ai-access-public-data/",
        "employees": "~46,000-48,000",
        "proxy": "No",
        "filing": "ONLINE PORTAL: foia-pal.commerce.gov\nALTERNATIVE: foia.gov\nNIST: foia@nist.gov, 100 Bureau Drive STOP 1710, Gaithersburg, MD 20899; (301) 975-4074\nNOTE: commerce.gov returning 403 errors — use PAL portal or foia.gov directly",
        "source_url": "https://www.commerce.gov/about/policies/foia",
        "cost": "Standard federal per 15 CFR Part 4; fee waivers available",
        "statute": "5 U.S.C. § 552",
        "deadline": "20 business days",
        "turnaround": "1-3 months (simple); 3-6+ months (multi-bureau)",
        "template": "", "govai": "", "status": "", "date_filed": "", "date_due": "", "date_received": "",
        "notes": "Multi-bureau agency: NIST, Census, NOAA, USPTO, ITA, BEA, NTIA. Commerce.gov FOIA pages returning 403 errors — use PAL portal directly. NIST is the federal AI standards body — ironic if they can't produce their own AI records. USPTO may be using AI for patent examination. NIST processed Bouchtia request in ~5 weeks (fast for federal). Consider filing targeted requests to specific bureaus.",
        "requirements": "Standard federal FOIA",
        "retention": "PARTIAL — No Commerce-specific AI records retention policy found. Commerce is CFO Act agency subject to OMB M-24-10. NIST leads federal AI Risk Management Framework but unclear if this translates to internal records policy. Federal Records Act applies.",
        "precedent": "PARTIAL — Nacim Bouchtia filed 'NIST ChatGPT Account Signups' (April 29, 2025, #DOC-NIST-2025-000338): Requested emails from OpenAI signup addresses to @nist.gov accounts. Status: No Responsive Documents (final response June 6, 2025). MuckRock: muckrock.com/foi/united-states-of-america-10/nist-chatgpt-account-signups-185866/ Suggests NIST employees did not have official OpenAI accounts as of mid-2025.",
        "template_notes": "Cite FOIA (5 U.S.C. §552). File via Commerce FOIA PAL (foia-pal.commerce.gov). Target specific bureaus: NIST (AI standards), USPTO (patent examination AI), Census (data analysis AI), NOAA (environmental modeling). Reference Bouchtia NIST precedent. Note NIST NRD suggests no official ChatGPT accounts — try Copilot and Gemini instead. USPTO may be particularly rich target for AI in patent examination.",
    },
    {
        "cat": "US-Fed", "tier": "Tier 2", "jurisdiction": "DOI (Department of the Interior)",
        "country": "US", "level": "Federal",
        "ai_deploy": "EMPLOYEE-FACING: 200+ AI use cases in inventory (FedScoop, Feb 2026). AI-powered document identification for litigation. RPA bots 'Bob, Bobby, and Oz' for procurement/contracting. RESTRICTIVE GenAI policy: employees 'may not disclose non-public data' in GenAI systems without agency authorization. No confirmed enterprise ChatGPT/Copilot/Gemini deployment. Exploring GenAI and blockchain but cautious approach. Sources: fedscoop.com/interior-modernization-emerging-tech-ai-inventory/ (Feb 18, 2026), fedscoop.com/how-risky-is-chatgpt-depends-which-federal-agency-you-ask/",
        "employees": "~67,000-70,000",
        "proxy": "No",
        "filing": "ONLINE PORTAL (preferred): securefoia.doi.gov\nALTERNATIVE: foia.gov\nMAIL: 1849 C Street NW, Washington, DC 20240\nBureau-specific: BIA: foia@bia.gov; BLM: blm_wo_foia@blm.gov; FWS: fwhq_foia@fws.gov; NPS: npsfoia@nps.gov; USGS: foia@usgs.gov; BOEM: BOEMFOIA@boem.gov\nAppeals: FOIA.Appeals@sol.doi.gov",
        "source_url": "https://www.doi.gov/foia",
        "cost": "Duplication: $0.15/page. Search: $27/hr (clerical), $48/hr (professional), $69/hr (managerial). FEES UNDER $50 AUTOMATICALLY WAIVED. Over $250: advance payment.",
        "statute": "5 U.S.C. § 552",
        "deadline": "20 business days",
        "turnaround": "1-2 months (simple); 3-12+ months (multi-bureau)",
        "template": "", "govai": "", "status": "", "date_filed": "", "date_due": "", "date_received": "",
        "notes": "200+ AI use cases but RESTRICTIVE GenAI policy requiring agency authorization. Fees under $50 automatically waived — cost-effective target. Multi-bureau: BLM, NPS, FWS, USGS, BIA, BOR, BOEM, BSEE. The authorization requirement means there SHOULD be records of GenAI authorization requests (granted or denied) even if chat logs are sparse. RPA bots named 'Bob, Bobby, and Oz.' Strong environmental/natural resource public interest angle.",
        "requirements": "Standard federal FOIA. Fees under $50 waived.",
        "retention": "PARTIAL — No DOI-specific AI records retention policy found. Has restrictive GenAI policy requiring agency authorization before employees use GenAI with non-public data — implies governance structure. 200+ AI use cases in inventory. Federal Records Act, OMB M-24-10 apply.",
        "precedent": "UNKNOWN — No FOIA requests targeting DOI for AI/ChatGPT/Copilot chat logs found on MuckRock. Untested target. Cross-agency precedents apply.",
        "template_notes": "Cite FOIA (5 U.S.C. §552). File via securefoia.doi.gov. UNIQUE OPPORTUNITY: DOI's restrictive GenAI policy requiring authorization means there should be records of authorization requests/approvals/denials — request these even if chat logs are sparse. Reference 200+ AI use cases in inventory. Request the full AI use case inventory. Fees under $50 automatically waived. Target specific bureaus: USGS (scientific analysis), BLM (land management), NPS (visitor services).",
    },
    {
        "cat": "US-Fed", "tier": "Tier 2", "jurisdiction": "USPS (U.S. Postal Service)",
        "country": "US", "level": "Federal",
        "ai_deploy": "EMPLOYEE-FACING: AI for logistics optimization, customer care, sentiment analysis, maintenance assistance, fraud detection, web risk analysis, augmented development, AI-assisted training. NLP customer service agent for passport calls. 30 edge AI applications for mail processing under consideration. CIO Pritha Mehra stated interest in GenAI (Jan 2024) but no confirmed enterprise ChatGPT/Copilot/Gemini deployment. Manages ~110 petabytes of data, processes 129B mail pieces/year. NOT subject to OMB M-24-10 (independent establishment). Sources: fedscoop.com/postal-service-ai-data-tech/, fedscoop.com/video/how-usps-is-pioneering-ai-integration-in-public-service/",
        "employees": "~640,000",
        "proxy": "No",
        "filing": "ONLINE PORTAL: pfoiapal.usps.com\nMAIL: Records Office, USPS, 475 L'Enfant Plaza SW, Room 1P830, Washington, DC 20260-1101\nNo email submission available",
        "source_url": "https://about.usps.com/who/legal/foia/welcome.htm",
        "cost": "Search: ~$30-50/hr; Duplication: $0.10/page; First 100 pages + 2 hrs search free for non-commercial. Per 39 CFR Part 265.",
        "statute": "39 U.S.C. § 410(c)(2) + 5 U.S.C. § 552",
        "deadline": "20 business days",
        "turnaround": "2-6 months (simple); 6-12+ months (complex)",
        "template": "", "govai": "", "status": "", "date_filed": "", "date_due": "", "date_received": "",
        "notes": "One of largest civilian employers (~640,000). USPS is an 'independent establishment' NOT subject to OMB M-24-10 — do not cite this memo. Cite 39 U.S.C. §410(c)(2) and Federal Records Act instead. USPS may claim certain AI procurement info is commercially sensitive under Exemption 4 or Postal Reorganization Act protections. No confirmed enterprise GenAI — expect possible 'no responsive documents.' Also consider filing with USPS Office of Inspector General for OIG-specific AI use.",
        "requirements": "Standard federal FOIA via 39 U.S.C. §410(c)(2)",
        "retention": "UNKNOWN — No USPS-specific AI records retention policy found. USPS is NOT subject to OMB M-24-10 as an independent establishment. Federal Records Act (44 U.S.C. Chapters 29-33) does apply. NARA has not issued AI-specific retention schedules.",
        "precedent": "UNKNOWN — No FOIA requests for AI/ChatGPT/Copilot chat logs at USPS found on MuckRock. Novel request target. Cross-agency precedents: Delaware CFPB/CFTC (completed with docs), Park SEC (completed), Ciaramella DHS (pending).",
        "template_notes": "Cite 39 U.S.C. §410(c)(2) (NOT standard 5 U.S.C. §552 alone). Do NOT cite OMB M-24-10 — USPS is not subject to it. Name ChatGPT, Copilot, Gemini, and any custom AI tools. Reference NLP customer service agent and AI logistics tools from public reporting. File via pfoiapal.usps.com. Be prepared for 'no responsive documents' if no enterprise GenAI deployed. Also request AI tool procurement/evaluation records.",
    },
    {
        "cat": "US-Fed", "tier": "Tier 2", "jurisdiction": "USACE (U.S. Army Corps of Engineers)",
        "country": "US", "level": "Federal",
        "ai_deploy": "EMPLOYEE-FACING: As DOD component, has access to GenAI.mil (1.1M DOD users), CamoGPT (Army-specific, 'thousands of daily users'), Ask Sage on cArmy (400 initial accounts, Sept 2024), NIPRGPT (sunset Dec 2025, 700K+ DOD users). CIO Dovarius Peoples discussed AI for 'predictive analysis, first call resolution, disaster response.' Sources: defensescoop.com/2026/01/27/army-camogpt-dod-genai-mil/, defensescoop.com/2024/09/10/army-generative-ai-capability-carmy-cloud/, fedscoop.com/advancing-government-innovation-with-private-ai/dovarius-peoples/",
        "employees": "~37,000-38,000",
        "proxy": "No",
        "filing": "ONLINE: eFOIA.usace.army.mil (may be experiencing connectivity issues)\nALTERNATIVE: foia.gov\nEMAIL: hq-foia@usace.army.mil\nMAIL: HQ USACE, ATTN: CECC-F (FOIA), 441 G Street NW, Washington, DC 20314-1000\nNOTE: USACE is decentralized — may need to file with specific districts/divisions",
        "source_url": "https://www.usace.army.mil/Information/Freedom-of-Information-Act/",
        "cost": "Per 32 CFR Part 286 (DOD rates); first 100 pages + 2 hrs search free for non-commercial; $0.15/page",
        "statute": "5 U.S.C. § 552",
        "deadline": "20 business days",
        "turnaround": "3-12+ months",
        "template": "", "govai": "", "status": "", "date_filed": "", "date_due": "", "date_received": "",
        "notes": "DOD component — Army CIO memo (Aug 2025) applies: ALL AI prompts and outputs are official records subject to FOIA. Cite this memo. Access to GenAI.mil, CamoGPT, Ask Sage. Decentralized — FOIA requests may need specific district targeting. CIO Dovarius Peoples publicly discussed AI for disaster response. WARNING: NIPRGPT was sunset Dec 2025 with 'no data export tool' — historical logs may be lost. Critical infrastructure mission (flood control, navigation, environmental remediation) provides strong public interest angle.",
        "requirements": "Standard federal FOIA per 32 CFR Part 286 and Army Regulation 25-55",
        "retention": "PARTIAL — Army CIO memo (Aug 26, 2025) is strongest agency-level statement: all AI interactions are official records, must be retained, FOIA-disclosable. DOD Directive 5015.02 governs records management. However, NIPRGPT sunset with no data export tool suggests gaps in historical records. OMB M-24-10 applies. Sources: executivegov.com/articles/army-garciga-application-owners-ai-records-mgmt",
        "precedent": "UNKNOWN — No FOIA requests targeting USACE for AI/ChatGPT/Copilot chat logs found on MuckRock. Cross-DOD: Koebler DoDEA (NRD), Johnston Cyber Command (rejected as too broad).",
        "template_notes": "Cite FOIA (5 U.S.C. §552). CITE Army CIO Aug 2025 memo: 'all user interactions with AI tools, including prompts and AI-generated content, must be properly identified, retained and secured as official records.' Name GenAI.mil, CamoGPT, Ask Sage, NIPRGPT (historical), ChatGPT, Copilot, Gemini. File with HQ (hq-foia@usace.army.mil) for enterprise data. Note NIPRGPT sunset may mean historical loss. Cite CIO Peoples' public AI statements.",
    },
    {
        "cat": "US-Fed", "tier": "Tier 2", "jurisdiction": "FEC (Federal Election Commission)",
        "country": "US", "level": "Federal",
        "ai_deploy": "UNKNOWN — No employee-facing GenAI tool deployments publicly identified. Very small agency (~340 employees) with limited IT budget. FEC's AI work focuses on REGULATING AI in campaign ads (Sept 2024 Interpretive Rule), not internal AI tool use. IT Strategic Plan FY 2025-2026 exists but not publicly accessible. Currently operating with only 2 of 6 commissioners (4 vacancies).",
        "employees": "~340",
        "proxy": "No",
        "filing": "EMAIL: FOIA@fec.gov\nFAX: (202) 219-3923\nMAIL (USPS): FEC, Attn: FOIA Requester Service Center, 1050 First Street NE, Washington, DC 20463\nMAIL (FedEx/UPS): Same address but ZIP 20002\nNo dedicated online portal — email, fax, or mail only",
        "source_url": "https://www.fec.gov/freedom-information-act/",
        "cost": "Per 11 CFR Part 4; notify if >$25; fee waivers available with written justification",
        "statute": "5 U.S.C. § 552",
        "deadline": "20 business days",
        "turnaround": "1-3 months",
        "template": "", "govai": "", "status": "", "date_filed": "", "date_due": "", "date_received": "",
        "notes": "Very small agency — may have no formal AI governance. Only 2 of 6 commissioners seated. Ironic angle: FEC regulates AI in campaign ads but its own AI governance is opaque. IT Strategic Plan FY 2025-2026 is best target for AI policy info. No online portal — email FOIA@fec.gov. Small FOIA volume = potentially faster processing. FOIA Public Liaison: Amber Smith, (202) 694-1437. Chief FOIA Officer: Gregory R. Baker.",
        "requirements": "Standard federal FOIA per 11 CFR Part 4",
        "retention": "UNKNOWN — No FEC-specific AI records retention policy found. FEC may or may not be subject to OMB M-24-10 (debatable for independent regulatory commissions). Federal Records Act applies. Small size and lack of confirmed AI deployment may mean no retention policy exists.",
        "precedent": "UNKNOWN — No FOIA requests for AI/ChatGPT/Copilot chat logs at FEC found on MuckRock. Novel request target. Cross-agency precedents apply.",
        "template_notes": "Cite FOIA (5 U.S.C. §552). Email FOIA@fec.gov (preferred method). Name ChatGPT, Copilot, Gemini, and any tools in FEC's M365 environment. Also request FEC IT Strategic Plan FY 2025-2026 for AI sections. Frame around irony: FEC regulates AI in campaign ads under Sept 2024 Interpretive Rule while its own AI governance is unknown. Small agency = faster processing but may get 'no responsive documents.'",
    },
    {
        "cat": "US-Fed", "tier": "Tier 2", "jurisdiction": "ED (Department of Education)",
        "country": "US", "level": "Federal",
        "ai_deploy": "EMPLOYEE-FACING: Microsoft Copilot — 15 documented use cases (document creation, data analysis, compliance, grants). ChatGPT Enterprise available via GSA OneGov deal ($1/year). Grammarly AI, Otter.AI (speech-to-text), AWS Bedrock, GSA USAi. Aidan Chatbot (FSA): 2.6M unique customers, 11M+ messages (public-facing). Multiple RAG chatbots for legislation comment categorization and internal Q&A. 35+ total AI use cases in inventory. Chief AI Officer: Gary Stevens. NOTE: Workforce cut >50% (from ~4,133 to ~2,000). AI Governance Board and AI Working Group may be functionally degraded. Sources: ed.gov/about/ed-overview/artificial-intelligence-ai-guidance, ed.gov/media/document/department-of-educations-response-m-24-10-actions-107919.pdf",
        "employees": "~2,000 (post-layoffs; was ~4,133)",
        "proxy": "No",
        "filing": "ONLINE PORTAL: doed.secureocp.com (requires Login.gov with 2FA)\nEMAIL: EDFOIAManager@ed.gov\nFAX: (202) 401-0920\nMAIL: U.S. Dept of Education, Office of the Deputy Secretary, FOIA Service Center, 400 Maryland Ave SW, LBJ 7W106A, Washington, DC 20202-4536\nPHONE: (202) 401-8365",
        "source_url": "https://www.ed.gov/about/ed-overview/required-notices/foia",
        "cost": "Commercial: $0.20/page + salary+16% for search/review. Media/education: $0.20/page (first 100 free). Others: $0.20/page (first 100 free), salary+16% (first 2 hrs free). No charge if <$5. No duplication charges for email attachments.",
        "statute": "5 U.S.C. § 552",
        "deadline": "20 business days (routinely missed)",
        "turnaround": "100-200+ days (54.49% backlog rate in FY2024; worse post-layoffs)",
        "template": "", "govai": "", "status": "", "date_filed": "", "date_due": "", "date_received": "",
        "notes": "CRITICAL CONTEXT: Workforce cut >50% (4,133 to ~2,000). 54.49% FOIA backlog rate in FY2024 — getting worse. 43 FOIA-related lawsuits in reporting period (constructive denials most common cause). FOIA Service Center lost contract support. Requires Login.gov with 2FA for portal. Chief FOIA Officer: Deborah O. Moore, Ph.D. Chief AI Officer: Gary Stevens. Uses Veritas (Clearwell) and Microsoft Purview for searches. 15 Copilot use cases documented. Strong framing angle: how is a drastically reduced workforce using AI tools? Are AI tools replacing functions of eliminated staff? Request expedited processing citing agency restructuring as matter of urgent public concern.",
        "requirements": "Login.gov with 2FA required for portal. Standard federal FOIA per 34 CFR Part 5.",
        "retention": "PARTIAL — No ED-specific policy explicitly declaring AI inputs/outputs as federal records. ED's M-24-10 compliance plan (Sept 2024, prepared by CAIO Gary Stevens) confirms ED developed 'interim procedural guidance addressing security and privacy concerns related to the use of generative AI' but does not address records retention. OMB M-24-10, M-25-21, Federal Records Act apply. Sources: ed.gov/media/document/department-of-educations-response-m-24-10-actions-107919.pdf",
        "precedent": "UNKNOWN — No FOIA requests targeting federal ED for AI/ChatGPT/Copilot chat logs found on MuckRock. Jason Koebler/VICE (Feb 2023) filed to ~60 STATE education departments (not federal ED). Sungho Park SEC request (completed Jan 2025) is cross-agency precedent. Sources: 404media.co/american-schools-were-deeply-unprepared-for-chatgpt-public-records-show/",
        "template_notes": "Cite FOIA (5 U.S.C. §552). File via doed.secureocp.com (Login.gov required). Name Microsoft Copilot (15 documented use cases), ChatGPT Enterprise, GSA USAi, Grammarly AI, Otter.AI, AWS Bedrock. Reference ED's own AI Use Case Inventory (35+ use cases). Request records from CAIO Gary Stevens, AI Governance Board, AI Working Group. Request ED's 'interim procedural guidance on generative AI' referenced in M-24-10 compliance plan. Request expedited processing: AI use at an agency undergoing 50%+ downsizing is a matter of urgent public concern.",
    },
]

# Write new rows using shared helper
for i, agency in enumerate(new_agencies):
    add_agency_row(ws, last_row + 1 + i, agency)

print(f"Added {len(new_agencies)} new agency rows (rows {last_row+1}-{last_row+len(new_agencies)})")

# ============================================================
# APPEND FEDERAL AI FOIA GUIDANCE TO ALL FEDERAL ROWS
# ============================================================
federal_guidance = """

---
UPDATED FEDERAL AI FOIA GUIDANCE (as of March 2026):

REGULATORY VACUUM: No unified federal determination exists on whether AI chat logs are federal records subject to FOIA.

KEY LEGAL ARGUMENTS TO INCLUDE IN ALL FEDERAL REQUESTS:
1. Federal Records Act (44 U.S.C. §3301): 'All recorded information, regardless of form, made or received by a Federal agency in connection with the transaction of public business' = federal record. AI chat logs clearly fit.
2. NARA Bulletin 2023-02: Expanded Capstone to cover 'chats' and 'other electronic messaging systems' — arguable that AI conversations qualify.
3. U.S. Army CIO memo (Aug 26, 2025): Most explicit agency statement — ALL AI prompts and outputs are 'official records' subject to FOIA. Cite as cross-agency precedent.
4. OMB M-25-21: Requires agencies to implement 'technical logging and monitoring controls' — the logs exist even if agencies don't call them 'records.'

KEY RISK — GSA/USAi.gov: GSA is explicitly NOT classifying USAi.gov chatbot conversations as federal records, even though prompts are logged (per FedScoop). This sets a bad precedent. Counter with Federal Records Act definition.

KEY RISK — Exemption 5: Agencies may claim AI outputs are 'predecisional deliberations' (PA already did this at state level with OOR Feb 2026 ruling). Counter: request AI outputs used in official communications/decisions, not just drafts.

RECOMMENDED LANGUAGE FOR ALL FEDERAL REQUESTS:
'Per the Federal Records Act (44 U.S.C. §3301), all recorded information made or received by a Federal agency in connection with the transaction of public business constitutes a federal record regardless of form. AI conversation logs — including prompts, outputs, and metadata — fall within this definition. See NARA Bulletin 2023-02 expanding records management to chats and electronic messaging systems, and the U.S. Army CIO memo of August 26, 2025, determining that all AI prompts and outputs are official records subject to FOIA.'

ACTIVE LITIGATION: Democracy Forward sued HUD, State, OPM, GSA, OMB (Oct 2025) over AI FOIA failures. CREW v. DOGE: court held DOGE subject to FOIA (Supreme Court stayed production).

ADDITIONAL PRECEDENTS: Robert Delaware obtained ChatGPT chat histories from CFPB (completed May 2024, redacted docs released) and CFTC (completed May 2024). MuckRock: muckrock.com/foi/united-states-of-america-10/cfpb-chatgpt-chat-histories-and-guidance-158792/ and muckrock.com/foi/united-states-of-america-10/cftc-chatgpt-chat-histories-and-guidance-163834/"""

# Append to all federal rows
fed_count = 0
for r in range(2, ws.max_row + 1):
    cat = ws.cell(r, COL_CAT).value
    if cat and "Fed" in str(cat):
        existing_notes = ws.cell(r, COL_NOTES).value or ""
        if "UPDATED FEDERAL AI FOIA GUIDANCE" not in existing_notes:
            ws.cell(r, COL_NOTES, existing_notes + federal_guidance)
            fed_count += 1

print(f"Appended federal AI FOIA guidance to {fed_count} existing federal rows")

# Save
save_tracker(wb, "foia_tracker_detailed_v3.xlsx")
print(f"Total rows: {ws.max_row} (was {last_row})")
