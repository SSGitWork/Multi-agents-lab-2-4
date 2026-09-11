# Framework Comparison — PM Planning Step

**Lab 2.4 · Module 2 · Build Autonomous Multi-Agent Systems**
Student name: _________________

---

## How to complete this document

1. Run `python run_comparison.py` to generate `comparison_results.json`.
2. Fill in each section below using the measured data and your observations.
3. Every section marked **[REQUIRED]** must be completed for full credit.
4. Aim for 2–4 sentences per section. Be specific — reference actual
   token counts, line counts, or code patterns you observed.

---

## 1. Benchmark Setup [REQUIRED]

**Requirement used:**
> _(paste the BENCHMARK_REQUIREMENT string here)_

**Model used across all three implementations:**
> gpt-4o (routed through Helicone)

**What the planning step does:**
> _(1–2 sentences: describe what the two LLM calls produce)_

---

## 2. Token Usage [REQUIRED]

Fill in from `comparison_results.json` or the console output of
`python run_comparison.py`.

| Framework  | LLM Calls | Prompt Tokens | Completion Tokens | Total Tokens |
|------------|-----------|---------------|-------------------|--------------|
| LangGraph  |           |               |                   |              |
| AutoGen    |           |               |                   |              |
| CrewAI     |           |               |                   |              |

**Which framework used the most tokens, and why?**
> _(your analysis — consider system messages, conversation history,
> framework overhead prompts, number of API calls)_

**Which framework made the most LLM calls, and why?**
> _(your analysis)_

---

## 3. Output Quality [REQUIRED]

**Did all three implementations produce a valid tech spec with five sections?**
> _(yes/no/partially — explain any differences)_

**Did all three implementations produce a valid JSON task list with 2–4 tasks?**
> _(yes/no/partially — explain any differences)_

**Were the task decompositions equivalent across frameworks?**
> _(compare task titles, file paths, acceptance criteria across the three outputs)_

---

## 4. Developer Experience [REQUIRED]

### 4.1 Code structure

**LangGraph (baseline):**
Approximately how many lines of implementation code (excluding prompts)?
> ___

**AutoGen:**
Approximately how many lines of implementation code (excluding prompts)?
> ___

**CrewAI:**
Approximately how many lines of implementation code (excluding prompts)?
> ___

### 4.2 Expressing the two-step workflow

**LangGraph:** Sequential calls with explicit variable passing.
How did AutoGen and CrewAI express the same dependency?

> **AutoGen:** _(describe how conversation 1's output was passed to conversation 2)_

> **CrewAI:** _(describe how task context=[write_spec] worked in practice)_

### 4.3 Prompt control

How much control did each framework give you over the exact system prompt
sent to the LLM?

> **LangGraph:** Full control — explicit system message in every call.

> **AutoGen:** _(describe — role, system_message, any injected framework text?)_

> **CrewAI:** _(describe — role + goal + backstory → what does the LLM actually see?)_

### 4.4 Token usage transparency

How easy was it to read token counts from each framework?

> **LangGraph:** `response.usage.prompt_tokens` directly on the API response object.

> **AutoGen:** _(describe your approach)_

> **CrewAI:** _(describe — result.token_usage or manual counting?)_

---

## 5. Framework Selection Criteria [REQUIRED]

Based on your experience with all three, complete this table with
"Best", "Acceptable", or "Poor" for each criterion.

| Criterion                          | LangGraph | AutoGen | CrewAI |
|------------------------------------|-----------|---------|--------|
| Sequential workflow clarity        |           |         |        |
| Prompt control                     |           |         |        |
| Token efficiency                   |           |         |        |
| Ease of token usage tracking       |           |         |        |
| Boilerplate / setup overhead       |           |         |        |
| Debugging experience               |           |         |        |

---

## 6. Recommendation [REQUIRED]

**For the PM + Coder pipeline you built in Lab 2.3, which framework
would you recommend, and why?**

> _(2–4 sentences. Justify using at least two specific observations from
> your measurements above. Reference token counts or code structure
> differences.)_

**Is there a scenario where you would choose one of the other frameworks
over your recommendation?**

> _(1–2 sentences — think about team size, task type, observability needs)_

---

## 7. Surprises and Open Questions [OPTIONAL]

> _(Anything that surprised you, behaved unexpectedly, or that you want
> to investigate further — framework-injected prompts, conversation
> termination quirks, token count discrepancies, etc.)_
