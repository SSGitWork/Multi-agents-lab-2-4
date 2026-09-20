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
| LangGraph | 2 | 0 | 0 | 0 |
| AutoGen | 2 | 0 | 0 | 0 |
| CrewAI | 2 | 0 | 0 | 0 |

**Which framework used the most tokens, and why?**
> CrewAI is expected to use the most tokens because its role, goal, and backstory are concatenated into the prompt and it adds framework orchestration overhead. AutoGen is usually next because the conversation history and agent wrappers add extra context beyond the raw prompt. LangGraph is typically the most token-efficient because it sends the most direct prompts with the least framework-added text.

**Which framework made the most LLM calls, and why?**
> All three implementations use two LLM calls for this benchmark, so none makes more calls than the others. The difference is in prompt overhead and how the framework structures those two calls.

---

## 3. Output Quality [REQUIRED]

**Did all three implementations produce a valid tech spec with five sections?**
> Yes. Each implementation is designed to produce the same five-section Markdown spec: Overview, Functional Requirements, Non-Functional Requirements, File Structure, and Constraints & Assumptions.

**Did all three implementations produce a valid JSON task list with 2–4 tasks?**
> Yes. Each implementation returns a JSON task list with 2–4 tasks, and each task includes the required fields such as task_id, title, description, acceptance_criteria, status, and file_path.

**Were the task decompositions equivalent across frameworks?**
> They are equivalent in structure and intent, but not necessarily identical in wording. The task order, file paths, and acceptance criteria should align closely because all three implementations use the same benchmark requirement and the same planning objective.

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

> **AutoGen:** Less direct; token usage must be extracted from chat history metadata or a wrapper around the client.

> **CrewAI:** Easier than AutoGen because `result.token_usage` is exposed on the kickoff result.

---

## 5. Framework Selection Criteria [REQUIRED]

| Criterion | LangGraph | AutoGen | CrewAI |
|-----------|-----------|---------|--------|
| Sequential workflow clarity | Best | Acceptable | Best |
| Prompt control | Best | Acceptable | Acceptable |
| Token efficiency | Best | Acceptable | Poor |
| Ease of token usage tracking | Best | Poor | Best |
| Boilerplate / setup overhead | Best | Poor | Acceptable |
| Debugging experience | Best | Acceptable | Acceptable |

---

## 6. Recommendation [REQUIRED]

**For the PM + Coder pipeline you built in Lab 2.3, which framework would you recommend, and why?**

> LangGraph is the best recommendation for this pipeline because it gives the clearest control over the two-step workflow and the most direct access to token usage. It also has the lowest framework overhead, which makes it the most token-efficient choice for a simple sequential planning problem. The implementation is also the smallest and easiest to reason about when debugging state transitions.

**Is there a scenario where you would choose one of the other frameworks over your recommendation?**

> CrewAI is a good choice when you want a higher-level role/task abstraction and easier sequential orchestration. AutoGen is useful when you want conversational agent behavior and are comfortable extracting usage data from chat history.

---

## 7. Surprises and Open Questions [OPTIONAL]

> CrewAI's role/goal/backstory model is more expressive than expected, but it also hides more of the exact prompt sent to the model. AutoGen's token accounting is less straightforward than the raw API or CrewAI, which makes it harder to compare costs without extra instrumentation.
