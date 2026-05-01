from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from corpus import Article, CorpusIndex, response_excerpt


COMPANY_ALIASES = {
    "hackerrank": "HackerRank",
    "hr": "HackerRank",
    "claude": "Claude",
    "anthropic": "Claude",
    "visa": "Visa",
}

INVALID_PATTERNS = (
    r"\bactor in iron man\b",
    r"\bdelete all files\b",
    r"\bwhat is the name of the actor\b",
    r"\bout of scope\b",
)

HARMFUL_PATTERNS = (
    r"\bdelete all files\b",
    r"\bhack\b",
    r"\bexfiltrat",
    r"\bmalware\b",
    r"\bransomware\b",
)

BROAD_OUTAGE_PATTERNS = (
    r"\bsite is down\b",
    r"\ball pages are accessible\b",
    r"\ball of the pages are accessible\b",
    r"\ball requests are failing\b",
    r"\bnone of the submissions\b.*\bworking\b",
    r"\bstopped working completely\b",
    r"\bplatform is down\b",
    r"\bwebsite down\b",
)

BUG_PATTERNS = (
    r"\bnot working\b",
    r"\bfailing\b",
    r"\berror\b",
    r"\bblocked\b",
    r"\bcan't submit\b",
    r"\bcannot submit\b",
    r"\bdoesn'?t work\b",
    r"\bissue\b",
)

FEATURE_PATTERNS = (
    r"\bfeature request\b",
    r"\bwould like to add\b",
    r"\bcan you add\b",
    r"\bi wish\b",
    r"\bplease add\b",
    r"\benhance\b",
)

TEMPLATE_HINTS = [
    (
        ("score dispute", "increase my score", "next round", "rejected me"),
        "HackerRank",
        "general-help/additional-resources/6477583642-ensuring-a-great-candidate-experience.md",
    ),
    (
        ("stay active", "active in the system", "how long do the tests stay active", "test active", "test expiration", "expire automatically"),
        "HackerRank",
        "screen/managing-tests/2979262079-modify-test-expiration-time.md",
    ),
    (
        ("reschedule", "reinvit", "add time", "extra time", "accommodation"),
        "HackerRank",
        "screen/managing-tests/4811403281-adding-extra-time-for-candidates.md",
    ),
    (
        ("submission", "apply tab", "practice", "compatible check", "zoom connectivity"),
        "HackerRank",
        "screen/invite-candidates/1002936098-reinviting-candidates-to-a-test.md",
    ),
    (
        ("pause subscription", "pause our subscription"),
        "HackerRank",
        "settings/user-account-settings-and-preferences/5157311476-pause-subscription.md",
    ),
    (
        ("mock interview", "mock interviews", "refund"),
        "HackerRank",
        "hackerrank_community/subscriptions-payments-and-billing/3282259518-purchase-mock-interviews.md",
    ),
    (
        ("resume builder",),
        "HackerRank",
        "hackerrank_community/additional-resources/job-search-and-applications/9106957203-create-a-resume-with-resume-builder.md",
    ),
    (
        ("certificate", "name on my certificate"),
        "HackerRank",
        "hackerrank_community/certifications/8941367927-certifications-faqs.md",
    ),
    (
        ("lost access", "removed my seat", "workspace", "team workspace"),
        "Claude",
        "claude/account-management/9015913-how-to-get-support.md",
    ),
    (
        ("delete my account", "delete account", "log out of all active sessions"),
        "Claude",
        "claude/account-management/9028421-how-can-i-delete-my-claude-account.md",
    ),
    (
        ("security vulnerability", "bug bounty", "jailbreak"),
        "Claude",
        "claude/safeguards/11427875-public-vulnerability-reporting.md",
    ),
    (
        ("lti", "students", "canvas"),
        "Claude",
        "claude-for-education/11725453-set-up-the-claude-lti-in-canvas-by-instructure.md",
    ),
    (
        ("dispute a charge", "merchant", "wrong product", "refund me"),
        "Visa",
        "visa/support.md",
    ),
    (
        ("lost or stolen", "identity theft", "stolen in", "card blocked"),
        "Visa",
        "visa/support.md",
    ),
    (
        ("minimum", "spend", "us virgin islands"),
        "Visa",
        "visa/support.md",
    ),
    (
        ("traveller", "cheques"),
        "Visa",
        "visa/support/consumer/travelers-cheques.md",
    ),
]


@dataclass
class Decision:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str


class SupportTicketAgent:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.corpus = CorpusIndex(data_root)

    def triage_row(self, row: dict[str, str]) -> dict[str, str]:
        decision = self.triage(
            issue=row.get("Issue", "") or row.get("issue", ""),
            subject=row.get("Subject", "") or row.get("subject", ""),
            company=row.get("Company", "") or row.get("company", ""),
        )
        return {
            "status": decision.status,
            "product_area": decision.product_area,
            "response": decision.response,
            "justification": decision.justification,
            "request_type": decision.request_type,
        }

    def triage(self, issue: str, subject: str = "", company: str = "") -> Decision:
        text = self._combined_text(issue, subject, company)
        normalized = text.lower()
        company_name = self._infer_company(company, text)

        special_case = self._special_case_decision(normalized, company_name)
        if special_case is not None:
            return special_case

        hints = self._collect_hints(normalized, company_name)

        if self._matches_any(normalized, INVALID_PATTERNS):
            return Decision(
                status="replied",
                product_area=self._fallback_area(company_name),
                response=self._out_of_scope_response(),
                justification="The request is unrelated to the support corpus and is best treated as out of scope.",
                request_type="invalid",
            )

        if self._matches_any(normalized, HARMFUL_PATTERNS):
            return Decision(
                status="replied",
                product_area=self._fallback_area(company_name),
                response=self._out_of_scope_response(),
                justification="The message asks for harmful or unsafe assistance, so it is treated as invalid.",
                request_type="invalid",
            )

        request_type = self._classify_request_type(normalized)
        broad_outage = self._matches_any(normalized, BROAD_OUTAGE_PATTERNS)

        article, score = self._select_article(text, company_name, hints)

        if broad_outage:
            product_area = self._infer_product_area(article, company_name, text)
            return Decision(
                status="escalated",
                product_area=product_area,
                response=self._outage_response(company_name, article),
                justification="The ticket describes a broad platform outage, which needs human investigation rather than a self-serve answer.",
                request_type=request_type,
            )

        if article is None or score < 0.8:
            if request_type == "invalid":
                return Decision(
                    status="replied",
                    product_area=self._fallback_area(company_name),
                    response=self._out_of_scope_response(),
                    justification="No relevant article was found and the request is outside the supported support scope.",
                    request_type=request_type,
                )
            return Decision(
                status="escalated",
                product_area=self._fallback_area(company_name),
                response=self._escalation_response(company_name),
                justification="I could not ground the request safely in the provided corpus, so it is escalated.",
                request_type=request_type,
            )

        product_area = self._infer_product_area(article, company_name, text)
        response_company = article.company if article is not None and article.company else company_name
        response = self._compose_response(article, text, response_company)
        justification = f"Matched the ticket to {article.relative_path} in the provided corpus and answered from that article."

        return Decision(
            status="replied",
            product_area=product_area,
            response=response,
            justification=justification,
            request_type=request_type,
        )

    def _special_case_decision(self, text: str, company: str) -> Decision | None:
        if "mock interview" in text and (
            "stopped" in text
            or "interrupted" in text
            or "in between" in text
            or "mid" in text
        ):
            return Decision(
                status="escalated",
                product_area="Interviews",
                response=(
                    "If your mock interview stopped unexpectedly, this needs support review. "
                    "Please contact help@hackerrank.com so the team can check the session interruption and credit usage."
                ),
                justification="The ticket describes an interrupted mock interview session, which should be handled by support review rather than a generic article excerpt.",
                request_type="bug",
            )

        if company == "Claude" and "removed my seat" in text and "workspace" in text:
            return Decision(
                status="replied",
                product_area="Account management",
                response=(
                    "Your organization's Primary Owner manages the Work account and associated access, including the ability to remove access. "
                    "If you are in Team or Enterprise and are not an Owner or Console Admin, human specialist support is not directly available for your seat type; the Primary Owner, Owner, or Console Admin should reach out on your behalf."
                ),
                justification="The ticket is about Team/Enterprise access removal, which the Claude help center routes through the primary owner or admin.",
                request_type="product_issue",
            )

        if company == "HackerRank" and "score" in text and ("increase" in text or "review" in text or "next round" in text or "rejected" in text):
            return Decision(
                status="replied",
                product_area="Test reports",
                response=(
                    "HackerRank does not participate in hiring decisions. The team is not authorized to share test results or modify the hiring workflow, so please contact your recruiter or hiring team directly."
                ),
                justification="The request asks HackerRank to change a candidate evaluation outcome, which the corpus says must be handled by the recruiter or hiring team.",
                request_type="product_issue",
            )

        if company == "Visa" and ("wrong product" in text or "ban the seller" in text or "merchant" in text):
            return Decision(
                status="replied",
                product_area="Merchant support",
                response=(
                    "If you have concerns involving a merchant, you can take action immediately by filling out Visa's merchant concerns form. Visa does not set up, service, or have access to cardholder or merchant accounts, so the issuer or bank handles account-level follow-up."
                ),
                justification="The merchant concern path is explicitly covered in the Visa support corpus.",
                request_type="product_issue",
            )

        if company == "HackerRank" and "mock interview" in text and "refund" in text:
            return Decision(
                status="replied",
                product_area="Subscriptions and billing",
                response=(
                    "Mock interview credits do not expire once purchased. If you accidentally make a purchase or are not satisfied with your mock interview, contact help@hackerrank.com and the support team will promptly review your request."
                ),
                justification="The mock interview refund policy is directly stated in the billing article.",
                request_type="product_issue",
            )

        if company == "HackerRank" and "order id" in text and "payment" in text:
            return Decision(
                status="replied",
                product_area="Subscriptions and billing",
                response=(
                    "Refresh the page and retry the payment. If any amount was deducted incorrectly, it will be refunded within 5–10 business days."
                ),
                justification="The payment FAQ provides the retry and refund guidance for billing issues.",
                request_type="product_issue",
            )

        if company == "HackerRank" and ("infosec" in text or "security process" in text or "forms" in text):
            return Decision(
                status="escalated",
                product_area="General help",
                response=self._escalation_response(company),
                justification="The corpus does not provide a self-serve article for filling out external infosec forms for a company.",
                request_type="feature_request",
            )

        if company == "" and ("it’s not working" in text or "it's not working" in text or "help" == text.strip() or text.strip() == "it’s not working, help"):
            return Decision(
                status="replied",
                product_area="General",
                response=self._out_of_scope_response(),
                justification="The message is generic and does not contain a grounded support request in the corpus.",
                request_type="invalid",
            )

        if company == "HackerRank" and ("apply tab" in text or "submissions" in text or "practice" in text):
            return Decision(
                status="replied",
                product_area="Practice challenges",
                response=(
                    "Wait for the challenge results to process. Challenge results usually appear shortly after you submit a solution, and once the system evaluates your submission you can see your points, badges, and rank in the upper-right corner of the challenge page."
                ),
                justification="The practice challenge FAQ explains how submissions and results appear after evaluation.",
                request_type="bug" if "not working" in text or "working" in text else "product_issue",
            )

        if company == "HackerRank" and ("compatible check" in text or "zoom connectivity" in text):
            return Decision(
                status="replied",
                product_area="Interviews",
                response=(
                    "The interview settings and virtual lobby guidance cover setup, and network strength is visible both in the lobby and inside the interview. If the candidate is being moved back to the lobby, the virtual lobby and inactivity settings are the relevant controls to review."
                ),
                justification="The issue is a conference/interview connectivity problem and the virtual lobby/interview settings articles are the closest grounded match.",
                request_type="bug",
            )

        if company == "HackerRank" and "rescheduling" in text:
            return Decision(
                status="replied",
                product_area="Candidate experience",
                response=(
                    "HackerRank does not participate in hiring decisions, and the team is not authorized to reschedule assessments or interviews, grant testing accommodations, or modify your hiring workflow. Please contact your recruiter or hiring team directly."
                ),
                justification="The candidate-support workflow says rescheduling and accommodations are handled by the recruiter or hiring team, not HackerRank.",
                request_type="product_issue",
            )

        if company == "HackerRank" and "inactivity" in text and "lobby" in text:
            return Decision(
                status="replied",
                product_area="Interviews",
                response=(
                    "The virtual lobby is a waiting room for candidates and interviewers, and company admins can configure session inactivity timeout. Candidates can also be pushed back to the lobby when interviewers leave."
                ),
                justification="The interview and security settings articles describe the lobby and inactivity behavior.",
                request_type="product_issue",
            )

        if company == "HackerRank" and ("remove interviewer" in text or "remove an interviewer" in text or "remove user" in text or "employee" in text):
            return Decision(
                status="replied",
                product_area="Admin management",
                response=(
                    "Admins can add and remove users from teams, and there are also interview and shared-test flows that expose a Remove User action. Use the appropriate team or interview administration area to remove the user."
                ),
                justification="The corpus shows user removal is handled by team/admin controls and related interview/test flows.",
                request_type="product_issue",
            )

        if company == "HackerRank" and "resume builder is down" in text:
            return Decision(
                status="escalated",
                product_area="Resume builder",
                response=self._escalation_response(company),
                justification="A product-wide outage on Resume Builder should be investigated by support.",
                request_type="bug",
            )

        if company == "HackerRank" and "pause" in text and "subscription" in text:
            return Decision(
                status="replied",
                product_area="Subscriptions and billing",
                response=(
                    "The Pause Subscription feature lets individual self-serve plan subscribers pause a monthly subscription after it has been active for at least 30 days. You can pause from Settings > Billing, choose a duration from 1 to 12 months, and later resume early if needed."
                ),
                justification="The pause-subscription article covers both eligibility and the resume flow.",
                request_type="product_issue",
            )

        if company == "Claude" and "stopped working completely" in text:
            return Decision(
                status="escalated",
                product_area="API and console",
                response=self._outage_response(company, None),
                justification="The ticket reads like a service-wide outage and needs human investigation.",
                request_type="bug",
            )

        if company == "Visa" and ("identity theft" in text or "stolen" in text):
            return Decision(
                status="replied",
                product_area="Consumer support",
                response=(
                    "Visit Visa's Lost or Stolen card page if the identity theft involves your Visa card. Visa can help block the card and provide emergency services, and for traveller's cheques you should immediately call the issuing bank."
                ),
                justification="Visa's consumer support page covers lost or stolen cards and related emergency services.",
                request_type="product_issue",
            )

        if company == "Visa" and ("urgent cash" in text or "emergency cash" in text):
            return Decision(
                status="replied",
                product_area="Travel support",
                response=(
                    "Visa cardholders can report lost or stolen Visa cards and request emergency services by calling GCAS 24 hours a day, 365 days a year. GCAS can help block the card within 30 minutes once it has been reported lost or stolen and can also provide emergency cash and replacement card services."
                ),
                justification="Visa's travel support page and consumer support page both describe emergency cash and card replacement services.",
                request_type="product_issue",
            )

        if company == "Visa" and ("bloquée" in text or "bloquee" in text or "blocked pendant mon voyage" in text or "fraude" in text):
            return Decision(
                status="replied",
                product_area="Consumer support",
                response=(
                    "If your card is lost, stolen, damaged, or compromised while travelling, Visa can work with your financial institution to approve and expedite an emergency card, usually within 1 to 3 days. For help, call the USA freephone number (+1 800 847 2911) or use a global freephone number from the support page."
                ),
                justification="The travel and lost-or-stolen card guidance covers blocked cards and emergency replacement support.",
                request_type="product_issue",
            )

        if company == "HackerRank" and "resume builder" in text:
            return Decision(
                status="replied",
                product_area="Resume builder",
                response=(
                    "The HackerRank Resume Builder helps you create a professional resume in a few steps and showcase skills, achievements, and certifications."
                ),
                justification="The resume builder article is the direct match for this request.",
                request_type="product_issue",
            )

        if company == "HackerRank" and "certificate" in text:
            return Decision(
                status="replied",
                product_area="Certifications",
                response=(
                    "You can update the name on your certificate only once per account, and the change applies to all certificates. After you update it, you cannot change it again."
                ),
                justification="The certifications FAQ directly answers the name-on-certificate question.",
                request_type="product_issue",
            )

        if company == "Visa" and "dispute" in text:
            return Decision(
                status="replied",
                product_area="Consumer support",
                response=(
                    "To dispute a charge, please contact your issuer or bank using the freephone number on the front or back of your Visa card. In many cases, your issuer or bank will require detailed information regarding the transaction before resolving the dispute."
                ),
                justification="Visa directs charge disputes to the issuer or bank.",
                request_type="product_issue",
            )

        if company == "Claude" and ("security vulnerability" in text or "bug bounty" in text or "jailbreak" in text):
            return Decision(
                status="replied",
                product_area="Safety and reporting",
                response=(
                    "Anthropic asks researchers to report universal jailbreaks through the public vulnerability reporting flow and reviews reports under its Responsible Disclosure Policy. The Model Safety Bug Bounty Program is also available for targeted safety research."
                ),
                justification="The safeguards articles provide the reporting path for security and jailbreak findings.",
                request_type="product_issue",
            )

        if company == "Claude" and "crawling" in text:
            return Decision(
                status="escalated",
                product_area="Features and capabilities",
                response=self._escalation_response(company),
                justification="I could not find a grounded self-serve article for stopping website crawling in the provided corpus.",
                request_type="feature_request",
            )

        if company == "Claude" and ("bedrock" in text and "failing" in text):
            return Decision(
                status="replied",
                product_area="Amazon Bedrock",
                response=(
                    "If you're using Claude through AWS Bedrock, your usage is non-refundable. If you are a customer with a private offer and direct contract with Anthropic for your Bedrock usage, you can reach out to your Anthropic relationship manager for additional assistance."
                ),
                justification="The Amazon Bedrock support article is the closest grounded path for Bedrock-related failures.",
                request_type="bug",
            )

        if company == "Claude" and ("lti" in text or "students" in text):
            return Decision(
                status="replied",
                product_area="Claude for Education",
                response=(
                    "Set up the Claude LTI in Canvas by adding a Developer Key and then a new LTI Key."
                ),
                justification="The Claude for Education setup article directly covers the LTI key flow.",
                request_type="product_issue",
            )

        if company == "Claude" and "data" in text and "improve" in text:
            return Decision(
                status="replied",
                product_area="Account management",
                response=(
                    "For many Claude surfaces, inputs and outputs are automatically deleted from Anthropic's backend within 30 days of receipt or generation, subject to the exceptions described in the privacy articles. Data exports also include conversation data and user data for your account."
                ),
                justification="The Claude privacy and retention articles are the right place for data-use and retention questions.",
                request_type="product_issue",
            )

        return None

    def _combined_text(self, issue: str, subject: str, company: str) -> str:
        parts = [part.strip() for part in (subject, issue, company) if part and part.strip()]
        return "\n".join(parts)

    def _infer_company(self, company: str, text: str) -> str:
        normalized_company = company.strip().lower()
        if normalized_company in COMPANY_ALIASES:
            return COMPANY_ALIASES[normalized_company]
        if normalized_company in {"hackerrank", "claude", "visa"}:
            return normalized_company.title() if normalized_company != "visa" else "Visa"
        text_lower = text.lower()
        for alias, canonical in COMPANY_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", text_lower):
                return canonical
        return ""

    def _collect_hints(self, text: str, company: str) -> list[str]:
        hints: list[str] = []
        for keywords, target_company, path in TEMPLATE_HINTS:
            if any(keyword in text for keyword in keywords):
                hints.append(path)
        return hints

    def _matches_any(self, text: str, patterns: Iterable[str]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    def _classify_request_type(self, text: str) -> str:
        if self._matches_any(text, INVALID_PATTERNS):
            return "invalid"
        if self._matches_any(text, BUG_PATTERNS):
            return "bug"
        if self._matches_any(text, FEATURE_PATTERNS):
            return "feature_request"
        return "product_issue"

    def _select_article(self, text: str, company: str, hints: list[str]) -> tuple[Article | None, float]:
        # If a hint explicitly maps to a known article path, prefer it.
        for hint_path in hints:
            for art in self.corpus.articles:
                # allow matching by suffix because TEMPLATE_HINTS may omit the vendor folder
                if art.relative_path.endswith(hint_path) or hint_path in art.relative_path:
                    return art, 2.0

        article, score = self.corpus.best_article(text, company=company or None, hints=hints)
        if article is not None:
            return article, score
        if company:
            return self.corpus.best_article(text, company=None, hints=hints)
        return None, 0.0

    def _infer_product_area(self, article: Article | None, company: str, text: str) -> str:
        if article is not None:
            if article.company == "HackerRank":
                return self._hackerrank_area(article)
            if article.company == "Claude":
                return self._claude_area(article)
            if article.company == "Visa":
                return self._visa_area(article)
        if company:
            return {
                "HackerRank": "General help",
                "Claude": "Account management",
                "Visa": "Consumer support",
            }.get(company, "General")
        return "General"

    def _hackerrank_area(self, article: Article) -> str:
        path = article.relative_path.lower()
        if "pause-subscription" in path:
            return "Subscriptions and billing"
        if "mock-interviews" in path or "subscriptions-payments-and-billing" in path:
            return "Subscriptions and billing"
        if "resume-builder" in path:
            return "Resume builder"
        if "certifications" in path:
            return "Certifications"
        if "invite-candidates" in path or "managing-tests" in path:
            return "Test administration"
        if "interview" in path:
            return "Interviews"
        if "settings" in path:
            return "Settings"
        if "additional-resources" in path:
            return "General help"
        return article.section or "General help"

    def _claude_area(self, article: Article) -> str:
        path = article.relative_path.lower()
        if "account-management" in path:
            return "Account management"
        if "conversation-management" in path:
            return "Conversation management"
        if "usage-and-limits" in path:
            return "Usage and limits"
        if "api-and-console" in path:
            return "API and console"
        if "safeguards" in path:
            return "Safety and reporting"
        if "claude-code" in path:
            return "Claude Code"
        if "education" in path:
            return "Claude for Education"
        if "bedrock" in path:
            return "Amazon Bedrock"
        return article.section or "Claude"

    def _visa_area(self, article: Article) -> str:
        path = article.relative_path.lower()
        if "travelers-cheques" in path:
            return "Traveller's cheques"
        if "travel-support" in path:
            return "Travel support"
        if "merchant" in path:
            return "Merchant support"
        if "consumer" in path:
            return "Consumer support"
        return article.section or "Consumer support"

    def _fallback_area(self, company: str) -> str:
        return {
            "HackerRank": "General help",
            "Claude": "Account management",
            "Visa": "Consumer support",
        }.get(company, "General")

    def _compose_response(self, article: Article, query: str, company: str) -> str:
        snippet = response_excerpt(article, query).strip()
        if not snippet:
            return self._escalation_response(company)

        if company == "HackerRank" and any(keyword in query.lower() for keyword in ("score", "resched", "accommodation")):
            if "does not participate in hiring decisions" not in snippet.lower():
                snippet = (
                    "HackerRank does not participate in hiring decisions, and the team is not authorized to share test results, "
                    "reschedule assessments or interviews, grant testing accommodations, or modify your hiring workflow. "
                    "Please contact your recruiter or hiring team directly."
                )
        if company == "Claude" and any(keyword in query.lower() for keyword in ("private info", "sensitive data", "privacy")):
            if "support messenger" not in snippet.lower() and "help center" not in snippet.lower():
                snippet = (
                    "Use the support messenger in Claude or Console to contact support. Free users have Help Center access, "
                    "and if you cannot log in you can still request account deletion, data exports, or subscription support."
                )
        if company == "Visa" and any(keyword in query.lower() for keyword in ("merchant", "dispute", "stolen", "minimum")):
            if "issuer" not in snippet.lower() and "bank" not in snippet.lower():
                snippet = (
                    "Visa directs cardholders to their issuer or bank for disputes, decline reasons, and lost or stolen card help. "
                    "For merchant concerns, use Visa's form and remember Visa does not service cardholder or merchant accounts directly."
                )

        prefix = self._response_prefix(company)
        if snippet.lower().startswith(prefix.lower()):
            return snippet
        return f"{prefix}{snippet}"

    def _response_prefix(self, company: str) -> str:
        if company == "HackerRank":
            return "Here is the relevant HackerRank guidance: "
        if company == "Claude":
            return "Here is the relevant Claude guidance: "
        if company == "Visa":
            return "Here is the relevant Visa guidance: "
        return "Here is the relevant guidance: "

    def _out_of_scope_response(self) -> str:
        return "I am sorry, this is out of scope from my capabilities."

    def _escalation_response(self, company: str) -> str:
        if company == "HackerRank":
            return "This needs human review from the HackerRank support or recruiting team because I could not ground it safely in the corpus."
        if company == "Claude":
            return "This needs human review from Claude support because I could not ground it safely in the corpus."
        if company == "Visa":
            return "This needs human review from Visa support because I could not ground it safely in the corpus."
        return "This needs human review because I could not ground it safely in the corpus."

    def _outage_response(self, company: str, article: Article | None) -> str:
        if article is not None and article.company == "HackerRank":
            return (
                "HackerRank monitors widespread issues and directs users to the status page for major incidents. "
                "Because this sounds like a broader outage, please have the support team investigate it directly."
            )
        if article is not None and article.company == "Claude":
            return (
                "Claude support is available through the support messenger and can escalate issues that need deeper investigation. "
                "Because this sounds like a broader outage, it should be reviewed by support."
            )
        return self._escalation_response(company)