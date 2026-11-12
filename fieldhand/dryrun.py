"""A scripted stand-in for the model, used only to test the plumbing without an API key.

It follows a fixed booking script for the "ants" scenario and reads real tool
results to fill in arguments, so the tools, the graph, the guard, and the
trace export all run for real. Its traces are labeled DRY RUN. Do not submit
them as a run of the agent.
"""

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class DryRunModel(BaseChatModel):
    loop_forever: bool = False  # True makes it call tools endlessly, to test the guard

    @property
    def _llm_type(self) -> str:
        return "fieldhand-dry-run"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _next(self, messages) -> AIMessage:
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        text = messages[last_human].content
        results = {m.name: json.loads(m.content) for m in messages[last_human:] if isinstance(m, ToolMessage)}
        n = len(results) + 1

        def call(say, name, args):
            return AIMessage(content=say, tool_calls=[{"name": name, "args": args, "id": f"dry_{n}"}])

        if self.loop_forever:
            return AIMessage(content="Checking the service area again.",
                             tool_calls=[{"name": "check_service_area", "args": {"zip_code": "30318"},
                                          "id": f"loop_{len(messages)}"}])

        zip_code = (re.search(r"\b(\d{5})\b", text) or [None, ""])[1]
        sqft = int((re.search(r"([\d,]{3,6})\s*sq", text) or [None, "0"])[1].replace(",", "") or 0)
        if "lookup_customer" not in results:
            return call("Checking whether this number belongs to an existing customer.", "lookup_customer", {})
        if "check_service_area" not in results:
            return call(f"Confirming {zip_code} is in the service area.", "check_service_area", {"zip_code": zip_code})
        if not results["check_service_area"].get("in_service_area"):
            return AIMessage(content="Sorry, that address is outside our service area. - Oakline Pest Co.")
        if "get_quote" not in results:
            return call("Pricing the job from the price book.", "get_quote",
                        {"pest": "ants", "property_type": "residential", "square_feet": sqft})
        quote = results["get_quote"]
        if "find_open_slots" not in results:
            return call("Finding open windows this week.", "find_open_slots",
                        {"service_code": quote["service_code"], "zip_code": zip_code, "days_ahead": 7})
        if "book_appointment" not in results:
            days = [d for d in results["find_open_slots"]["days"] if d.get("recommended", True)]
            wanted = [d for d in days if d["weekday"] in ("Thursday", "Friday")] or days
            day = wanted[0]
            morning = [w for w in day["open_windows"] if w < "12"] or day["open_windows"]
            name = (re.search(r"-\s*([A-Z][a-z]+ [A-Z][a-z]+)\s*$", text) or [None, "Customer"])[1]
            address = (re.search(r"(\d{2,6} [A-Za-z0-9 .]+?(?:St|Dr|Rd|Ave|Ln|Way|Blvd|Ct|Pl|Pkwy)\.?(?: [NS][EW])?)\b",
                                 text) or [None, ""])[1]
            return call(f"Booking {day['weekday']} {morning[0]}, the earliest recommended morning window.",
                        "book_appointment",
                        {"customer_name": name, "street_address": address, "zip_code": zip_code,
                         "service_code": quote["service_code"], "square_feet": sqft, "date": day["date"],
                         "time_window": morning[0], "notes": "Ants at kitchen baseboards and exterior. Kids and a dog."})
        b = results["book_appointment"]
        if not b.get("booked"):
            return AIMessage(content="Sorry, I couldn't book that window. Sam will text you shortly. - Oakline Pest Co.")
        return AIMessage(content=(
            f"You're booked: {b['weekday']} {b['date']}, {b['time_window']}, {b['service']}, ${b['price']}. "
            f"Confirmation {b['confirmation']}. Sam will go over prep and safety steps for your kids and dog "
            f"at the visit. - Oakline Pest Co."))
