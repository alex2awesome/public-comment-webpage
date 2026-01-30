# -*- coding: utf-8 -*-
"""
Standalone DSPy optimization agent with hot-swappable environments (parallel-safe)
and a dspy.Module that exposes a forward(...) method.

Key ideas:
- Tools are constructed ONCE and read the current environment from a ContextVar (parallel-safe).
- Each run sets CURRENT_ENV to its own LoggingEnv; no cross-talk between concurrent runs.
- OptimizeModule (dspy.Module) has forward(env, user_query, task_index=None) so you can keep your
  optimizer’s expected interface. It internally seeds the transcript and calls a ReAct submodule.
- DSPyOptimizeAgent wires up the LLM and provides a SolveResult-returning .solve(...) method.
"""

from __future__ import annotations

import os
import contextvars
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import json

import dspy
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)

def _to_jsonable(x: Any):
    """Best-effort conversion to JSON-serializable structures."""
    # Primitives
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    # Dicts (force string keys)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    # Lists / Tuples / Sets
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    # Pydantic-style
    if hasattr(x, "model_dump"):
        try:
            return _to_jsonable(x.model_dump())
        except Exception:
            pass
    # Dataclasses or generic objects
    if hasattr(x, "__dict__"):
        try:
            return _to_jsonable(vars(x))
        except Exception:
            pass
    # Fallback to string repr
    try:
        return str(x)
    except Exception:
        return "<unserializable>"

############################################################
###                 Context Vars (parallel-safe)         ###
############################################################

# The currently active environment (typically a LoggingEnv). Must be set per-run.
CURRENT_ENV: contextvars.ContextVar[Env] = contextvars.ContextVar("CURRENT_ENV")

############################################################
###                   Logging Env                         ###
############################################################

@dataclass
class StepEvent:
    name: str
    kwargs: Dict[str, Any]
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class Recorder:
    history: List[StepEvent] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)  # chat transcript
    info_agg: Dict[str, Any] = field(default_factory=dict)        # merged info
    total_cost: float = 0.0                                       # optional token $
    last_reward: float = 0.0
    last_done: bool = False

    @property
    def last(self) -> Optional[StepEvent]:
        return self.history[-1] if self.history else None

    def merge_info(self, new_info: Dict[str, Any]) -> None:
        # Shallow merge (like `{**info, **response.info.model_dump()}` in the reference)
        self.info_agg = {**self.info_agg, **new_info}


class LoggingEnv(Env):
    """
    Transparent proxy around an Env that:
      - records every step as StepEvent
      - builds a 'messages' transcript like the reference agent
      - maintains aggregated `info` dict and last reward/done
      - exposes helpers to seed/log messages and to pack a SolveResult
    """
    def __init__(self, inner_env: Env):
        self._inner = inner_env
        self.recorder = Recorder()

    # ---- Conversation helpers (agent should call these) ----
    def start_session(self, system_prompt: str, first_user_observation: str) -> None:
        """Seed messages with system prompt + first user observation (from reset)."""
        self.recorder.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_user_observation},
        ]

    def log_model_message(self, message: Dict[str, Any], cost: float = 0.0) -> None:
        """
        Append the model's message dict (e.g., {'role': 'assistant', 'content': 'Thought... Action: {...}'})
        and optionally accumulate token cost like the reference total_cost.
        """
        self.recorder.messages.append(message)
        self.recorder.total_cost += float(cost or 0.0)

    # ---- Env API passthroughs ----
    def reset(self, *args, **kwargs):
        return self._inner.reset(*args, **kwargs)

    def step(self, action: Action):
        # ----- 1) Log the assistant side (what we "say"/do) -----
        try:
            if action.name == RESPOND_ACTION_NAME:
                # Direct user-facing text (no prefixes in the transcript)
                content = action.kwargs.get(RESPOND_ACTION_FIELD_NAME, "")
                assistant_msg = {"role": "assistant", "content": content}
            else:
                # Tool call: show a canonical Action JSON (valid JSON if possible)
                action_json = json.dumps(
                    {"name": action.name, "arguments": action.kwargs},
                    ensure_ascii=False,
                )
                assistant_msg = {"role": "assistant", "content": f"Action:\n{action_json}"}
        except Exception as e:
            # Fallback: never let logging break the step
            assistant_msg = {
                "role": "assistant",
                "content": f"Action:\n{{\"name\": \"{action.name}\", \"arguments\": \"<unserializable>\"}}",
            }
        self.recorder.messages.append(assistant_msg)

        # ----- 2) Execute the action on the inner env -----
        resp = self._inner.step(action)

        # ----- 3) Record raw step details and aggregate info -----
        raw_info = getattr(resp, "info", None)
        if raw_info is None:
            info_dict: Dict[str, Any] = {}
        elif hasattr(raw_info, "model_dump"):
            info_dict = raw_info.model_dump()
        elif isinstance(raw_info, dict):
            info_dict = raw_info
        else:
            try:
                info_dict = dict(raw_info)  # type: ignore[arg-type]
            except Exception:
                info_dict = {"_info": str(raw_info)}

        event = StepEvent(
            name=action.name,
            kwargs=action.kwargs,
            observation=resp.observation,
            reward=resp.reward,
            done=resp.done,
            info=info_dict,
        )
        self.recorder.history.append(event)
        self.recorder.last_reward = resp.reward
        self.recorder.last_done = resp.done
        self.recorder.merge_info(info_dict)

        # ----- 4) Log the user-side observation -----
        obs_content = resp.observation
        if action.name != RESPOND_ACTION_NAME:
            obs_content = "API output: " + obs_content
        self.recorder.messages.append({"role": "user", "content": obs_content})

        return resp

    # Delegate any other attributes/methods to the inner env
    def __getattr__(self, attr):
        return getattr(self._inner, attr)

    # ---- Final packaging helper ----
    def build_solve_result(self) -> SolveResult:
        """
        Return a SolveResult(messages, reward, info) consistent with the reference.
        - messages: full transcript accumulated here
        - reward: last seen reward
        - info:   shallow-merged info over all steps
        """
        return SolveResult(
            messages=self.recorder.messages,
            reward=self.recorder.last_reward,
            info=self.recorder.info_agg,
        )

############################################################
###                   Tool Functions                      ###
############################################################

def make_tool_functions() -> SimpleNamespace:
    """
    Construct tool functions ONCE. At call time, each tool looks up the
    current environment from the CURRENT_ENV context var (parallel-safe).
    """

    def _step(name: str, kwargs: Dict[str, Any]) -> str:
        """Helper: call env.step and prefix the observation with 'API output:'."""
        env = CURRENT_ENV.get()
        action = Action(name=name, kwargs=kwargs)
        response = env.step(action)
        return f"API output: {response.observation}"

    # ---- Tools ----

    def book_reservation(
        user_id: str,
        origin: str,
        destination: str,
        flight_type: str,
        cabin: str,
        flights: List[Dict[str, Any]],
        passengers: List[Dict[str, Any]],
        payment_methods: List[Dict[str, Any]],
        total_baggages: int,
        nonfree_baggages: int,
        insurance: str,
    ) -> str:
        """
        Book a reservation.

        Parameters:
            user_id (str): The ID of the user to book the reservation, e.g., "sara_doe_496".
            origin (str): The IATA code for the origin city, e.g., "SFO".
            destination (str): The IATA code for the destination city, e.g., "JFK".
            flight_type (str): Type of flight. One of ["one_way", "round_trip"].
            cabin (str): Cabin class. One of ["basic_economy", "economy", "business"].
            flights (List[Dict[str, Any]]): Flight segments.
                Each includes:
                  - flight_number (str): e.g., "HAT001".
                  - date (str): "YYYY-MM-DD".
            passengers (List[Dict[str, Any]]): Passenger info.
                Each includes:
                  - first_name (str)
                  - last_name (str)
                  - dob (str): "YYYY-MM-DD".
            payment_methods (List[Dict[str, Any]]): Payment split across methods.
                Each includes:
                  - payment_id (str): e.g., "credit_card_7815826".
                  - amount (float)
            total_baggages (int): Total number of baggage items included.
            nonfree_baggages (int): Number of non-free baggage items.
            insurance (str): "yes" or "no" for travel insurance.

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step(
            "book_reservation",
            {
                "user_id": user_id,
                "origin": origin,
                "destination": destination,
                "flight_type": flight_type,
                "cabin": cabin,
                "flights": flights,
                "passengers": passengers,
                "payment_methods": payment_methods,
                "total_baggages": total_baggages,
                "nonfree_baggages": nonfree_baggages,
                "insurance": insurance,
            },
        )

    def calculate(expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Parameters:
            expression (str): A mathematical expression, e.g., "2 + 2".
                              Can include numbers, +, -, *, /, parentheses, and spaces.

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step("calculate", {"expression": expression})

    def cancel_reservation(reservation_id: str) -> str:
        """
        Cancel the whole reservation.

        Parameters:
            reservation_id (str): The reservation ID, e.g., "ZFA04Y".

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step("cancel_reservation", {"reservation_id": reservation_id})

    def get_reservation_details(reservation_id: str) -> str:
        """
        Get the details of a reservation.

        Parameters:
            reservation_id (str): The reservation ID, e.g., "8JX2WO".

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step("get_reservation_details", {"reservation_id": reservation_id})

    def get_user_details(user_id: str) -> str:
        """
        Get the details of a user, including their reservations.

        Parameters:
            user_id (str): The user ID, e.g., "sara_doe_496".

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step("get_user_details", {"user_id": user_id})

    def list_all_airports() -> str:
        """
        List all airports and their cities.

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step("list_all_airports", {})

    def search_direct_flight(origin: str, destination: str, date: str) -> str:
        """
        Search direct flights between two cities on a specific date.

        Parameters:
            origin (str): Origin airport IATA code, e.g., "JFK".
            destination (str): Destination airport IATA code, e.g., "LAX".
            date (str): Flight date in "YYYY-MM-DD".

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step(
            "search_direct_flight",
            {"origin": origin, "destination": destination, "date": date},
        )

    def search_onestop_flight(origin: str, destination: str, date: str) -> str:
        """
        Search one-stop flights between two cities on a specific date.

        Parameters:
            origin (str): Origin airport IATA code, e.g., "JFK".
            destination (str): Destination airport IATA code, e.g., "LAX".
            date (str): Flight date in "YYYY-MM-DD".

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step(
            "search_onestop_flight",
            {"origin": origin, "destination": destination, "date": date},
        )

    def send_certificate(user_id: str, amount: float) -> str:
        """
        Send a certificate to a user. Be careful!

        Parameters:
            user_id (str): The ID of the user to send the certificate, e.g., "sara_doe_496".
            amount (float): Certificate amount to send.

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step("send_certificate", {"user_id": user_id, "amount": amount})

    def think(thought: str) -> str:
        """
        Append a thought to the log (no external I/O).

        This tool does not obtain new information or change external state; it is
        intended for complex reasoning traces.

        Parameters:
            thought (str): A thought to record.

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step("think", {"thought": thought})

    def transfer_to_human_agents(summary: str) -> str:
        """
        Transfer the user to a human agent with a summary of the issue.

        Only use if the user explicitly requests a human agent, or the issue cannot
        be resolved with available tools.

        Parameters:
            summary (str): A concise summary of the user's issue.

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step("transfer_to_human_agents", {"summary": summary})

    def update_reservation_baggages(
        reservation_id: str,
        total_baggages: int,
        nonfree_baggages: int,
        payment_id: str,
    ) -> str:
        """
        Update the baggage information of a reservation.

        Parameters:
            reservation_id (str): The reservation ID, e.g., "ZFA04Y".
            total_baggages (int): The updated total number of baggage items included.
            nonfree_baggages (int): The updated number of non-free baggage items.
            payment_id (str): The payment ID stored in the user profile,
                              e.g., "credit_card_7815826", "gift_card_7815826", "certificate_7815826".

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step(
            "update_reservation_baggages",
            {
                "reservation_id": reservation_id,
                "total_baggages": total_baggages,
                "nonfree_baggages": nonfree_baggages,
                "payment_id": payment_id,
            },
        )

    def update_reservation_flights(
        reservation_id: str,
        cabin: str,
        flights: List[Dict[str, Any]],
        payment_id: str,
    ) -> str:
        """
        Update the flight information of a reservation.

        Parameters:
            reservation_id (str): The reservation ID, e.g., "ZFA04Y".
            cabin (str): Cabin class. One of ["basic_economy", "economy", "business"].
            flights (List[Dict[str, Any]]): Flight segments for the ENTIRE new reservation.
                Each includes:
                  - flight_number (str): e.g., "HAT001".
                  - date (str): "YYYY-MM-DD".
            payment_id (str): The payment ID stored in the user profile,
                              e.g., "credit_card_7815826", "gift_card_7815826", "certificate_7815826".

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step(
            "update_reservation_flights",
            {
                "reservation_id": reservation_id,
                "cabin": cabin,
                "flights": flights,
                "payment_id": payment_id,
            },
        )

    def update_reservation_passengers(
        reservation_id: str,
        passengers: List[Dict[str, Any]],
    ) -> str:
        """
        Update the passenger information of a reservation.

        Parameters:
            reservation_id (str): The reservation ID, e.g., "ZFA04Y".
            passengers (List[Dict[str, Any]]): List of passenger details.
                Each includes:
                  - first_name (str): e.g., "Noah".
                  - last_name (str): e.g., "Brown".
                  - dob (str): Date of birth, e.g., "1990-01-01".

        Returns:
            str: API output prefixed with "API output: ...".
        """
        return _step(
            "update_reservation_passengers",
            {"reservation_id": reservation_id, "passengers": passengers},
        )

    def respond(content: str) -> str:
        """
        Send a message directly to the user and return the user's reply.

        This function is used when the system needs to communicate with the user
        (e.g., provide information, ask for clarification, or continue a conversation).
        It sends the given `content` to the user and **returns whatever the user types back**.

        Parameters:
            content (str): The text to present to the user.
                           Example: "The current weather of San Francisco is 70F."

        Returns:
            str: The user's reply, prefixed with "User: ...".
        """
        env = CURRENT_ENV.get()
        action = Action(
            name=RESPOND_ACTION_NAME,
            kwargs={RESPOND_ACTION_FIELD_NAME: content},
        )
        response = env.step(action)
        return f"User: {response.observation}"

    # Expose as attributes on a simple namespace for ergonomic access
    return SimpleNamespace(
        book_reservation=book_reservation,
        calculate=calculate,
        cancel_reservation=cancel_reservation,
        get_reservation_details=get_reservation_details,
        get_user_details=get_user_details,
        list_all_airports=list_all_airports,
        search_direct_flight=search_direct_flight,
        search_onestop_flight=search_onestop_flight,
        send_certificate=send_certificate,
        think=think,
        transfer_to_human_agents=transfer_to_human_agents,
        update_reservation_baggages=update_reservation_baggages,
        update_reservation_flights=update_reservation_flights,
        update_reservation_passengers=update_reservation_passengers,
        respond=respond,
    )

############################################################
###                   DSPy Signature                      ###
############################################################

class AirlineAgentSignature(dspy.Signature):
    airline_policy: str = dspy.InputField(
        description="The policy of the airline. Contains details about what requests are acceptable and how to respond to them."
    )
    user_query: str = dspy.InputField(
        description="The user's request, can be a question or a command."
    )
    result_summary: str = dspy.OutputField(
        description="The summary of the results of the interaction and what actions were taken to satisfy the user's query."
    )

############################################################
###                  dspy.Module Wrapper                  ###
############################################################

class AirlineAgent(dspy.Module):
    """
    dspy.Module that:
      - is built ONCE with stable tools/signature
      - exposes forward(env, user_query, task_index=None)
      - sets CURRENT_ENV for this run, seeds transcript, and invokes ReAct
      - returns a SolveResult assembled by LoggingEnv
    """
    def __init__(self, airline_policy: str, max_iters: int = 30):
        super().__init__()
        self.airline_policy = airline_policy

        # Build tools ONCE (they read env from CURRENT_ENV at call time)
        tools = make_tool_functions()
        tools_list = [v for v in tools.__dict__.values() if callable(v)]

        # Build ReAct ONCE with stable signature/tooling surface
        self.react = dspy.ReAct(
            signature=AirlineAgentSignature,
            tools=tools_list,
            max_iters=max_iters,
        )

    def forward(
        self,
        env: Env,
        task_index: Optional[int] = None,
    ) -> SolveResult:
        """
        Run one episode on `env`:
          1) Wrap env with LoggingEnv and set CURRENT_ENV for this context.
          2) Reset env, seed transcript with airline policy + first observation.
          3) Invoke ReAct with (airline_policy, user_query=first observation).
          4) Return SolveResult from the LoggingEnv.
        """
        wrapped = LoggingEnv(env) if not isinstance(env, LoggingEnv) else env

        token = CURRENT_ENV.set(wrapped)
        try:
            first = wrapped.reset(task_index=task_index)
            wrapped.start_session(
                system_prompt=self.airline_policy,
                first_user_observation=first.observation,
            )

            # user_query is ignored in favor of the env's first observation (to mirror reference),
            # but passed in to keep API parity if you want to change this later.
            _ = self.react(
                airline_policy=self.airline_policy,
                user_query=first.observation,
            )

            return wrapped.build_solve_result()
        finally:
            CURRENT_ENV.reset(token)

############################################################
###                DSPy Optimization Agent                ###
############################################################

class DspyReActAgent(Agent):
    """
    Higher-level Agent that configures the LLM once and delegates to OptimizeModule.
    """
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,    # kept for parity; not used directly here
        temperature: float = 0.0,
        max_iters: int = 30,
    ) -> None:
        super().__init__()
        self.airline_policy = wiki
        self.tools_info = tools_info
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_iters = max_iters

        # Configure DSPy LLM once (optimizer-friendly)
        model_str = f"{provider}/{model}"
        llm = dspy.LM(
            model_str,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
            temperature=temperature,
        )
        dspy.configure(lm=llm)

        # Build the module ONCE (stable tools/signature)
        self.module = AirlineAgent(airline_policy=wiki, max_iters=max_iters)

    def solve(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: Optional[int] = None,  # not directly used by dspy.ReAct
    ) -> SolveResult:
        """
        Run one optimization/evaluation episode on `env` and return SolveResult.
        """
        return self.module(env=env, task_index=task_index)