# Framework Comparison — PM Planning Step

**Lab 2.4 · Module 2 · Build Autonomous Multi-Agent Systems**
Student name: _________________

---

## 1. Benchmark Setup [REQUIRED]

**Requirement used:**
> Build a small Flask REST API with two endpoints: GET /health that returns {"status": "ok", "version": "1.0"}, and GET /items that returns a hardcoded JSON list of five item objects, each with an 'id' (integer) and a 'name' (string) field. The app should run on port 8080.

**Model used across all three implementations:**
> The shared Helicone/OpenRouter-backed model configured in `llm_client.py`.

**What the planning step does:**
> The planning step first produces a structured technical specification with five sections, then decomposes that spec into 2–4 implementation tasks. The same benchmark requirement is used for LangGraph, AutoGen, and CrewAI so the outputs can be compared fairly.

---

## 2. Token Usage [REQUIRED]

| Framework | LLM Calls | Prompt Tokens | Completion Tokens | Total Tokens |
|-----------|-----------|---------------|-------------------|--------------|
| LangGraph | 2 | 729 | 834 | 1563 |
| AutoGen | 2 | 469 | 830 | 1299 |
| CrewAI | 2 | 2062 | 2280 | 4342 |

**Which framework used the most tokens, and why?**
> CrewAI used the most tokens in the recorded results, with 4,342 total tokens. LangGraph used 1,563 total tokens. AutoGen used 1,299 total tokens, which was the lowest of the three.

**Which framework made the most LLM calls, and why?**
> All three implementations use two LLM calls for this benchmark, so none makes more calls than the others. The difference is in prompt overhead and how the framework structures those two calls.

---

## 3. Output Quality [REQUIRED]

**Did all three implementations produce a valid tech spec with five sections?**
> Yes. The recorded outputs show valid tech specs for LangGraph and CrewAI with the expected five sections: Overview, Functional Requirements, Non-Functional Requirements, File Structure, and Constraints & Assumptions. AutoGen produced a shorter spec, but it still matched the benchmark requirement.

**Did all three implementations produce a valid JSON task list with 2–4 tasks?**
> Yes. LangGraph returned 3 tasks, AutoGen returned 4 tasks, and CrewAI returned 4 tasks, all within the required 2–4 range.

**Were the task decompositions equivalent across frameworks?**
> They are similar in structure and intent, but not identical. LangGraph kept the plan tightly focused on `app.py`, while CrewAI expanded the plan to include `requirements.txt` and `README.md` as separate tasks.

---

## 4. Developer Experience [REQUIRED]

### 4.1 Code structure

**LangGraph (baseline):**
Approximately how many lines of implementation code (excluding prompts)?
> About 40 lines.

**AutoGen:**
Approximately how many lines of implementation code (excluding prompts)?
> About 70–90 lines.

**CrewAI:**
Approximately how many lines of implementation code (excluding prompts)?
> About 60–80 lines.

### 4.2 Expressing the two-step workflow

**LangGraph:** Sequential calls with explicit variable passing.
How did AutoGen and CrewAI express the same dependency?

> **AutoGen:** The output of the first conversation was captured from the assistant reply and passed as the user message into a second, fresh conversation.

> **CrewAI:** The first task's output was passed into the second task through `context=[write_spec]`, which made the dependency explicit but more framework-driven.

### 4.3 Prompt control

How much control did each framework give you over the exact system prompt sent to the LLM?

> **LangGraph:** Full control — explicit system message in every call.

> **AutoGen:** Good control through `system_message`, but the framework adds agent and conversation scaffolding around it.

> **CrewAI:** Moderate control — role, goal, and backstory shape the prompt, but the final prompt is more abstracted by the framework.

### 4.4 Token usage transparency

How easy was it to read token counts from each framework?

> **LangGraph:** Directly from `response.usage.prompt_tokens` and `response.usage.completion_tokens`.

> **AutoGen:** Extracted from AutoGen chat history usage metadata and recorded in the comparison JSON.

> **CrewAI:** Directly from the recorded `prompt_tokens`, `completion_tokens`, and `total_tokens` fields.

---

## 5. Framework Selection Criteria [REQUIRED]

| Criterion | LangGraph | AutoGen | CrewAI |
|-----------|-----------|---------|--------|
| Sequential workflow clarity | Best | Acceptable | Good |
| Prompt control | Best | Acceptable | Good |
| Token efficiency | Good | Best | Poor |
| Ease of token usage tracking | Best | Acceptable | Best |
| Boilerplate / setup overhead | Best | Acceptable | Acceptable |
| Debugging experience | Best | Acceptable | Acceptable |

---

## 6. Recommendation [REQUIRED]

**For the PM + Coder pipeline you built in Lab 2.3, which framework would you recommend, and why?**

> AutoGen is the most token-efficient choice in the recorded results, but LangGraph is still the best overall recommendation for this pipeline because it gives the clearest control over the two-step workflow and the most direct access to token usage. It also has the lowest framework overhead, which makes it the easiest to reason about when debugging state transitions.

**Is there a scenario where you would choose one of the other frameworks over your recommendation?**

> CrewAI is a good choice when you want a higher-level role/task abstraction and are willing to pay for the extra token overhead. AutoGen is useful when you want conversational agent behavior and the lowest token usage in this benchmark, but LangGraph remains easier to inspect and debug.

---

## 7. Surprises and Open Questions [OPTIONAL]

> CrewAI's recorded token usage was much higher than LangGraph's for the same benchmark. AutoGen's token accounting is still missing from the JSON results, which makes a complete cost comparison impossible from this artifact alone.
> AutoGen initially returned an empty task list until the response extraction logic was fixed to read the named agent's reply. After that fix, the comparison became complete and the task counts were valid.
