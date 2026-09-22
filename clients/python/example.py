#!/usr/bin/env python3
"""Ask a running Sextant server three typed questions. SEXTANT_URL overrides the base URL."""
import os
from sextant_client import SextantClient, SextantError, choice, noul, score

client = SextantClient(os.environ.get("SEXTANT_URL", "http://127.0.0.1:8080"))
state = "Help! My payouts have been failing for 3 days. I need this fixed today or I will cancel."
questions = {
    "department": choice("Which team should handle this?", {
        "billing": "Payments, invoicing, refunds, payouts",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"}),
    "is_urgent": noul("Does this convey urgency?"),
    "frustration": score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
}
try:
    answers = client.systemone(state, questions)["answers"]
    print("department:", answers["department"]["choice"], answers["department"]["probabilities"])
    print("is_urgent: P(true) =", round(answers["is_urgent"]["noul"], 3))
    print("frustration: level", round(answers["frustration"]["score"], 2), "of", len(questions["frustration"]["criteria"]) - 1)
    client.systemone(state, {"bad": choice("Only one option", {"just_one": "not allowed"})})
except SextantError as e:
    print(f"server rejected a request: {e} [code={e.code}, status={e.status}]")
