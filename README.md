# LAXM
## Local-First, Verifiable Financial AI

> **A personal financial intelligence system where AI can reason over sensitive financial state without becoming the authority over that state.**

---

## The project, rethought

This repository began as an early exploration of an autonomous financial AI.

The original plan combined:

- financial data ingestion
- machine learning
- reinforcement learning
- fuzzy logic
- voice interaction
- document processing
- privacy
- security
- banking integrations
- NPU / edge inference

That version was useful as a learning roadmap, but it mixed too many independent problems together and treated every interesting AI technology as something that needed to be part of the final system.

**LAXM is the second iteration of that idea.**

The new project is built around one systems thesis:

> **Financial truth should live in a deterministic, verifiable data layer. AI should sit around that layer as a reasoning, retrieval, and interaction system with controlled capabilities.**

This is deliberately not "another finance chatbot."

LAXM is an exploration of a **privacy-first financial agent architecture** combining:

- structured financial data
- machine learning
- deep learning
- time-series forecasting
- retrieval-augmented generation
- MCP tools and resources
- agent planning and verification
- local / edge inference
- CPU / GPU / NPU workload partitioning

---

# 1. Core Architecture

```text
                              USER
                                |
                                v
                       +----------------+
                       | Agent / Planner|
                       | LLM + Policy  |
                       +-------+--------+
                               |
                        Structured Intent
                               |
                               v
                     +--------------------+
                     |  MCP / Tool Layer  |
                     +----+-------+-------+
                          |       |
                +---------+       +----------+
                v                            v
        Financial Data Plane             RAG Layer
                |                            |
                v                            v
        Analytics / Forecasting       Evidence + Provenance
                |                            |
                +-------------+--------------+
                              |
                              v
                      Verification Layer
                              |
                              v
                       Grounded Response
                              |
                 +------------+------------+
                 |                         |
            Local / Edge              Optional Cloud
             Inference                  Escalation
```

## Central design rule

> **AI reasons. Software verifies.**

The model is allowed to interpret intent, plan a task, retrieve information, and explain results.

It is not allowed to become the unquestioned source of financial truth.

For example, if the user asks:

> "Can I afford to spend another ₹8,000 this month?"

the system should not have the LLM estimate the answer from conversational context.

Instead:

```text
User question
      |
      v
Agent interprets intent
      |
      v
get_cashflow_state()
      |
      v
Deterministic financial engine
      |
      +--> current balance
      +--> expected income
      +--> recurring obligations
      +--> forecast interval
      +--> savings target
      |
      v
Verified structured result
      |
      v
LLM explains the result
```

The LLM is therefore **not the financial source of truth**.

---

# 2. What LAXM Is Actually Trying to Solve

## Financial Data Intelligence

Import, normalize, classify, query, and analyze personal financial data.

Examples:

- transaction categorization
- merchant normalization
- recurring-payment detection
- spending analysis
- anomaly detection
- budget utilization
- cash-flow analysis
- net-worth calculations

---

## Time-Series Intelligence

Learn temporal patterns in income and expenses.

Examples:

- expense forecasting
- income forecasting
- recurring cash-flow modeling
- seasonality
- trend detection
- forecast intervals
- future budget pressure

---

## Retrieval-Augmented Generation

Use RAG for information that is not naturally represented as structured financial state.

Examples:

- bank statements
- uploaded financial documents
- tax/reference material
- financial-product documentation
- user-defined financial policies
- supporting evidence for generated answers

RAG exists to provide **evidence**, not to replace the transaction database.

---

## MCP-Native Agent Interaction

Expose financial capabilities through structured MCP tools and resources.

The agent interacts with the financial system through controlled interfaces rather than unrestricted database access.

---

## Local-First Intelligence

Sensitive financial state should remain local whenever practical.

Cloud models become an optional escalation path rather than the architectural default.

---

## Edge / NPU Research

Investigate how much of the agent can execute efficiently on:

- CPU
- GPU
- NPU
- hybrid local/cloud architectures

The objective is not "put AI on an NPU because NPUs are cool."

The objective is to measure the trade-offs.

---

# 3. Financial Data Plane

The data plane owns the canonical financial state.

Responsibilities:

- CSV / Excel / statement ingestion
- transaction normalization
- merchant normalization
- category labels
- duplicate detection
- recurring-transaction detection
- account and transaction IDs
- deterministic financial calculations
- provenance
- schema validation

Candidate technologies:

- Python
- Pandas / Polars
- SQLite
- DuckDB
- PostgreSQL for optional server deployments
- SQLAlchemy
- Pydantic

The database is **state**, not an LLM memory dump.

---

# 4. Machine Learning

ML is used where patterns actually need to be learned.

Initial targets:

- transaction categorization
- merchant classification
- anomaly detection
- recurring-payment detection
- user-specific spending patterns

Possible models:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- calibrated classifiers
- clustering for exploration

A complex model must earn its place against a simpler baseline.

The project will not use deep learning simply because deep learning is available.

---

# 5. Deep Learning

Deep learning becomes relevant where it provides measurable value.

Potential applications:

- learned transaction embeddings
- sequence representation
- document understanding
- semantic retrieval
- compact language models
- multimodal receipt / document processing

Candidate stack:

- PyTorch
- Transformers
- Sentence Transformers
- ONNX
- ExecuTorch

A classical or simpler baseline should exist before a more complex neural architecture is accepted.

---

# 6. Time-Series Forecasting

Financial behavior is temporal, so forecasting is a first-class subsystem.

The project will compare models rather than assuming one architecture is universally superior.

## Baselines

- Naive
- Seasonal Naive
- Moving Average
- Exponential Smoothing

## Statistical Models

- ARIMA
- SARIMA
- ETS

## Machine Learning

- lag features
- rolling statistics
- calendar features
- event features
- XGBoost
- LightGBM

## Deep Learning

- LSTM
- GRU
- Temporal CNN
- Transformer-based forecasting

Deep models remain experimental until they outperform simpler models under proper temporal validation.

## Evaluation

Forecasting experiments should use:

- walk-forward validation
- MAE
- RMSE
- sMAPE / MAPE where appropriate
- prediction interval coverage
- error by forecast horizon

Random train/test splits are not valid for time-series experiments.

---

# 7. RAG and Provenance

RAG is for knowledge that does not belong in the transaction ledger.

The intended pipeline:

```text
Document
   |
Parsing
   |
Chunking + Metadata
   |
Embedding
   |
Vector / Hybrid Index
   |
Retriever
   |
Reranker
   |
Evidence Set
   |
LLM
   |
Grounded Answer + Provenance
```

Every evidence item should retain metadata such as:

- document ID
- source
- page / section
- chunk ID
- timestamp
- retrieval metadata

The system should be able to answer:

> "Where did that claim come from?"

without fabricating the source.

RAG should therefore be evaluated not just by whether an answer sounds good, but by whether the answer is supported by the retrieved evidence.

---

# 8. MCP as the Capability Boundary

MCP is not a decorative integration.

It is part of the system boundary between the probabilistic model and the sensitive financial system.

The MCP server may expose:

- tools
- resources
- prompts

## Example read-oriented tools

```text
get_accounts()
get_transactions()
search_transactions()
summarize_spending()
get_budget_state()
forecast_cashflow()
detect_anomalies()
search_financial_documents()
explain_transaction()
generate_financial_report()
```

## Example resources

```text
financial://accounts
financial://transactions/{account_id}
financial://budgets/current
financial://forecasts/cashflow
financial://documents/{document_id}
financial://audit/{event_id}
```

## Future mutating capabilities

```text
create_budget_draft()
create_category_rule()
archive_transaction()
export_report()
```

Mutating operations should pass through authorization and, where appropriate, explicit user confirmation.

The model gets **capabilities**, not unrestricted authority.

---

# 9. Agent Control Plane

The agent should not simply be allowed to call arbitrary functions.

A control plane sits between the model and the real system.

```text
LLM
 |
 v
Intent
 |
 v
Tool selection
 |
 v
Schema validation
 |
 v
Authorization / Policy
 |
 +---- DENY -------> stop
 |
 +---- CONFIRM ----> user approval
 |
 +---- ALLOW ------> tool execution
 |
 v
Tool result
 |
 v
Verification
 |
 v
Response
```

Controls include:

- typed inputs
- typed outputs
- least privilege
- read-only vs mutating capabilities
- confirmation policies
- rate limits
- audit events
- provenance
- failure handling
- prompt-injection boundaries

Arbitrary model-generated SQL execution is intentionally excluded from the design.

---

# 10. Trust and Verification

LAXM treats trust as a systems problem.

## Financial correctness

Numbers should originate from:

- database state
- deterministic calculations
- validated model outputs
- explicit tool results

## Retrieval correctness

Evidence should retain provenance.

## Prediction uncertainty

Forecasts should be treated as estimates with measurable error and, where possible, uncertainty intervals.

## Abstention

When evidence is insufficient:

```text
Not enough evidence
        |
        v
Do not fabricate
        |
        v
Return uncertainty
or request more evidence
```

The system should prefer an explicit unknown over an invented answer.

SHAP/LIME can help explain some models, but explainability is not itself a general hallucination-prevention mechanism.

---

# 11. Security Model

The financial database is treated as a sensitive system boundary.

Planned controls:

- authentication
- authorization
- least-privilege tools
- parameterized SQL
- input validation
- secret isolation
- audit logging
- PII-aware logging
- secure configuration
- prompt-injection defenses
- document trust boundaries
- network egress controls
- separation of read and write capabilities

The governing principle is:

> **The model receives controlled capabilities, not raw authority.**

---

# 12. Privacy Model

The default architecture is local-first.

```text
Sensitive financial data
        |
        v
     Local device
        |
   +----+-----+
   |          |
Database    Local RAG
   |          |
   +----+-----+
        |
   Local models
```

Possible deployment modes:

### Local

Sensitive state and supported inference remain on-device.

### Hybrid

Structured financial state remains local while selected workloads are sent to a cloud model.

### Research / Cloud

Used for experimentation and controlled benchmarking.

Every deployment should document:

- what data stays local
- what data leaves the device
- why it leaves
- which model receives it
- whether the transfer is optional

---

# 13. Edge / NPU Research

The original NPU idea remains, but the research question is now much more precise:

> **Which parts of a personal financial agent should run on CPU, GPU, NPU, or cloud, and what measurable trade-off does each partition create?**

A possible split:

```text
CPU
 |
 |-- database
 |-- business logic
 |-- cryptography
 |-- orchestration
 +-- policy engine

NPU
 |
 |-- embeddings
 |-- classification
 |-- compact transformer inference
 +-- always-available local AI tasks

GPU
 |
 |-- training
 |-- heavier local experimentation
 +-- evaluation

Cloud
 |
 |-- optional large-model reasoning
 +-- expensive or non-local workloads
```

Candidate deployment technologies:

- ONNX Runtime
- ONNX Runtime QNN
- ExecuTorch
- Android on-device AI runtimes
- vendor-specific runtimes where necessary

The project will not claim accelerator execution unless hardware execution is actually verified.

---

# 14. Hardware-Aware Model Routing

Long-term, LAXM can use capability-aware routing.

```text
                 +----------------+
                 |  Agent Task    |
                 +-------+--------+
                         |
                         v
                 +----------------+
                 | Capability     |
                 | / Cost Router  |
                 +-------+--------+
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
        CPU/NPU          GPU          Cloud
       cheap/local    heavier local  optional
          |              |              |
          +--------------+--------------+
                         |
                         v
                   Unified Result
```

Routing decisions may consider:

- model availability
- latency
- privacy level
- model size
- device temperature
- power budget
- task complexity
- network availability
- accuracy requirements

This can eventually become one of the project's most interesting systems components.

---

# 15. India-Focused Direction

A future deployment target is India's financial ecosystem, including appropriate Account Aggregator-based data access where supported.

The architecture should remain:

- consent-driven
- purpose-limited
- auditable
- privacy-aware

Until real integrations are available, the project will use:

- synthetic data
- user-provided test data
- mock provider APIs

No bank credentials belong inside the agent.

---

# 16. What Is Deliberately NOT in the Core

Several technologies from the original project are intentionally removed from the foundation.

## No LLM from scratch

The interesting problem is the architecture around the model.

## No RL by default

Budgeting and financial planning are not automatically reinforcement-learning problems.

RL can return later if a clearly defined sequential decision problem justifies it.

## No fuzzy logic as a generic uncertainty layer

Uncertainty should be classified according to its actual source:

- data uncertainty
- model uncertainty
- forecast uncertainty
- retrieval uncertainty
- policy uncertainty

## No differential privacy just for decoration

Differential privacy is relevant when the system actually performs the type of shared-data learning or statistical release that calls for it.

## No unrestricted text-to-SQL

The agent interacts with typed financial capabilities rather than arbitrary SQL execution.

## No voice-first development

Voice is an interface layer.

It is not the foundation of the system.

## No bank-credential scraping

External data access must use supported and consent-based mechanisms.

---

# 17. Proposed Repository Structure

```text
Laxm_work-007-AI/
|
+-- apps/
|   +-- api/
|   +-- cli/
|   +-- android/
|
+-- core/
|   +-- domain/
|   +-- schemas/
|   +-- policies/
|   +-- config/
|
+-- data/
|   +-- ingestion/
|   +-- normalization/
|   +-- classification/
|   +-- storage/
|   +-- synthetic/
|
+-- analytics/
|   +-- spending/
|   +-- budgeting/
|   +-- anomalies/
|   +-- net_worth/
|
+-- forecasting/
|   +-- baselines/
|   +-- statistical/
|   +-- ml/
|   +-- deep/
|   +-- evaluation/
|
+-- rag/
|   +-- ingestion/
|   +-- chunking/
|   +-- embeddings/
|   +-- retrieval/
|   +-- reranking/
|   +-- provenance/
|
+-- agent/
|   +-- planner/
|   +-- state/
|   +-- memory/
|   +-- routing/
|   +-- verifier/
|
+-- mcp/
|   +-- server/
|   +-- tools/
|   +-- resources/
|   +-- prompts/
|   +-- policies/
|
+-- edge/
|   +-- export/
|   +-- quantization/
|   +-- onnx/
|   +-- executorch/
|   +-- benchmarks/
|
+-- security/
|   +-- auth/
|   +-- authorization/
|   +-- audit/
|   +-- threat_models/
|
+-- evals/
|   +-- financial/
|   +-- rag/
|   +-- agent/
|   +-- safety/
|   +-- edge/
|
+-- notebooks/
|   +-- ml/
|   +-- dl/
|   +-- time_series/
|   +-- rag/
|   +-- edge/
|
+-- docs/
|   +-- architecture/
|   +-- experiments/
|   +-- decisions/
|
+-- tests/
|
+-- Resources/
|
+-- pyproject.toml
+-- README.md
```

This is a **target architecture**, not a claim that all of these directories need to exist immediately.

---

# 18. Development Roadmap

## Phase 0 - Architecture

- define the financial domain model
- define schemas
- define trust boundaries
- write the threat model
- create synthetic financial data
- establish evaluation protocols
- define API and MCP contracts

**Exit condition:** the major contracts are stable enough to implement against.

---

## Phase 1 - Financial Data Plane

- statement ingestion
- transaction normalization
- local database
- deterministic calculations
- transaction categorization baseline
- anomaly detection baseline
- reproducible reports

**Exit condition:** useful financial questions can be answered without an LLM.

---

## Phase 2 - Forecasting

- naive baselines
- statistical forecasting
- feature-based ML forecasting
- walk-forward validation
- deep-model experiments
- uncertainty intervals

**Exit condition:** forecasting models are quantitatively comparable.

---

## Phase 3 - RAG

- document ingestion
- metadata
- embeddings
- vector / hybrid search
- reranking
- evidence packaging
- provenance

**Exit condition:** document questions are traceable to source evidence.

---

## Phase 4 - Agent + MCP

- MCP server
- read-only tools
- resources
- structured schemas
- tool routing
- authorization
- verification
- audit logging
- confirmation for state-changing actions

**Exit condition:** an agent can complete useful financial tasks through controlled capabilities.

---

## Phase 5 - Local Models

- local embeddings
- local classifiers
- compact local LLM
- model routing
- quantization
- offline operation

**Exit condition:** a meaningful subset of the system works without cloud inference.

---

## Phase 6 - Edge / NPU

- model export
- CPU benchmark
- GPU benchmark
- NPU benchmark
- quantization experiments
- fallback measurement
- Android / edge deployment
- hardware-aware routing

**Exit condition:** at least one real accelerator path is reproducibly benchmarked.

---

## Phase 7 - Integrated Financial Agent

Target architecture:

```text
                 USER
                   |
                   v
              Local Client
                   |
                   v
             Agent / Planner
                   |
           +-------+-------+
           |               |
           v               v
          RAG         MCP Server
           |               |
           |       +-------+-------+
           |       |               |
           |    Finance        Forecast
           |     Tools           Tools
           |       |               |
           +-------+---------------+
                   |
                   v
             Verification
                   |
                   v
            Grounded Answer
                   |
             +-----+-----+
             |           |
           Local       Cloud
           Model      Optional
```

---

# 19. Research Questions

The project is intentionally structured around questions that can be measured.

### RQ1
How much functionality can a personal financial agent execute locally without materially reducing usefulness?

### RQ2
Which workloads benefit most from NPU execution?

### RQ3
How much quality is lost through model quantization for:

- classification
- retrieval
- compact local reasoning

### RQ4
Does deterministic financial computation reduce numerical hallucination compared with direct LLM reasoning?

### RQ5
Can MCP act as a practical capability boundary between an LLM and sensitive financial operations?

### RQ6
How much do provenance and verification reduce unsupported financial claims?

### RQ7
What is the privacy / latency / quality trade-off between:

- local-only
- hybrid
- cloud-heavy

architectures?

### RQ8
What is the best workload partition across CPU, GPU, NPU, and cloud for a personal financial agent?

---

# 20. Evaluation

LAXM will not be evaluated purely through demos.

## Financial correctness

- arithmetic correctness
- reproducibility
- edge-case correctness
- reconciliation accuracy

## Transaction classification

- precision
- recall
- F1
- calibration

## Forecasting

- MAE
- RMSE
- sMAPE / MAPE where meaningful
- interval coverage
- performance by forecast horizon

## Retrieval

- Recall@k
- MRR
- nDCG
- citation correctness
- provenance completeness

## Agent

- tool-selection accuracy
- argument validity
- unnecessary tool calls
- invalid tool calls
- unsafe tool calls
- abstention behavior
- policy violations

## Edge inference

- latency
- throughput
- memory
- model size
- accelerator utilization
- CPU fallback frequency
- energy / power where measurable
- accuracy before and after quantization

> **The goal is benchmarks, not vibes.**

---

# 21. Experimental Methodology

Each major subsystem should follow the same loop:

```text
Problem
   |
   v
Simple baseline
   |
   v
Experiment
   |
   v
Measure
   |
   v
Compare
   |
   +---- Worse / unnecessary ---> remove
   |
   +---- Better                ---> keep
                              |
                              v
                         Document result
```

Every major architectural addition should answer:

1. What problem does it solve?
2. Why is a simpler method insufficient?
3. How will improvement be measured?
4. What are the failure modes?
5. What crosses the trust boundary?
6. Can the result be reproduced?

---

# 22. Success Criteria

LAXM is successful when it can demonstrate:

1. Realistic financial data can be imported into a local system.
2. Financial calculations remain deterministic and reproducible.
3. Forecasting is compared against proper temporal baselines.
4. RAG responses preserve source provenance.
5. An agent can use financial capabilities through MCP.
6. Unsafe or ambiguous operations are blocked or require confirmation.
7. The agent can abstain rather than fabricate evidence.
8. A useful subset of the workload runs locally.
9. At least one hardware-accelerated inference path is benchmarked.
10. Major claims about the system are backed by experiments.

---

# 23. Current Status

This repository is an **architecture reset of an old project idea**.

## Concept retained

- privacy-first financial AI
- trustworthy financial computation
- personal context
- local / edge inference
- NPU exploration

## Architectural reset

- [x] deterministic financial layer is the source of truth
- [x] forecasting is separated from LLM reasoning
- [x] RAG is a distinct evidence layer
- [x] MCP is a capability boundary
- [x] policy and verification are first-class components
- [x] edge inference is a measurable research track
- [x] evaluation is part of the system
- [x] RL is no longer mandatory
- [x] fuzzy logic is no longer a generic uncertainty layer
- [x] unrestricted text-to-SQL is excluded

## Implementation

- [ ] financial data plane
- [ ] analytics engine
- [ ] forecasting engine
- [ ] RAG pipeline
- [ ] MCP server
- [ ] agent planner
- [ ] policy / verification layer
- [ ] local inference runtime
- [ ] edge benchmark suite
- [ ] integrated client

---

# 24. Technology Direction

The exact stack will be selected experimentally.

| Layer | Direction |
|---|---|
| Data | Python, Pandas / Polars, SQLite / DuckDB, PostgreSQL |
| ML | scikit-learn, XGBoost / LightGBM |
| DL | PyTorch, Transformers |
| Time Series | statsmodels + ML/DL experiments |
| RAG | embeddings, hybrid retrieval, reranking, provenance |
| Agent | structured outputs, tool calling, routing, verification |
| MCP | tools, resources, prompts, controlled actions |
| Edge | ONNX Runtime, QNN, ExecuTorch, Android AI runtimes |
| Security | typed validation, authorization, secrets isolation, audit logging |

The core architecture should remain independent of a single model provider.

---

# 25. Why MCP Matters Here

The interesting question is not:

> "Can I connect an LLM to an MCP server?"

The interesting question is:

> **Can MCP serve as a practical capability boundary between a probabilistic agent and sensitive financial operations?**

Example:

```text
User:
"Has my food spending become unusually high?"

                |
                v

              LLM

                |
                v

compare_spending_to_baseline(
    category="food",
    period="current_month"
)

                |
                v

             MCP

                |
                v

     Deterministic financial engine

                |
                v

{
    current: ...,
    baseline: ...,
    delta: ...,
    interval: ...,
    provenance: [...]
}

                |
                v

           Verification

                |
                v

         LLM explains result
```

The model reasons over a structured result instead of inventing the result.

That separation gives the project something concrete to measure.

---

# 26. Why the NPU Part Still Matters

The NPU component remains because personal finance is a particularly sensitive domain for local inference.

Potential benefits of local inference include:

- reduced data exposure
- offline operation
- local latency
- reduced cloud dependence
- potentially lower inference cost
- private personal AI

The research challenge is to identify:

> **the smallest useful local models and the right workload partition**

rather than forcing a large model onto an accelerator simply because the hardware exists.

---

# 27. Long-Term Vision

The long-term system can evolve toward:

```text
                       PERSONAL AI
                            |
             +--------------+--------------+
             |              |              |
          Finance         Docs          Personal
           State          / RAG           Context
             |              |              |
             +--------------+--------------+
                            |
                            v
                      Agent Runtime
                            |
               +------------+------------+
               |            |            |
              CPU          NPU          GPU
               |            |            |
               +------------+------------+
                            |
                     Optional Cloud
                            |
                            v
                     Verified Output
```

The objective is not maximum autonomy.

The objective is **useful autonomy with explicit control boundaries**.

---

# 28. Project Philosophy

LAXM follows one rule:

> **Complexity must earn its place.**

A transformer, RAG pipeline, MCP server, agent loop, RL algorithm, NPU backend, or security mechanism is not a feature merely because it exists.

It belongs only when:

- it solves a real problem
- a simpler method is insufficient
- the improvement can be measured
- its failure modes are understood
- the security implications are understood
- the result can be reproduced

---

# 29. References

## Model Context Protocol

- MCP Architecture: https://modelcontextprotocol.io/docs/learn/architecture
- MCP Specification: https://modelcontextprotocol.io/specification
- MCP Tools: https://modelcontextprotocol.io/specification/2025-06-18/server/tools

## Edge Inference

- PyTorch ExecuTorch: https://pytorch.org/executorch/
- ExecuTorch Backends: https://docs.pytorch.org/executorch/stable/backends-section.html
- Qualcomm Backend: https://docs.pytorch.org/executorch/stable/backends-qualcomm.html
- ONNX Runtime Execution Providers: https://onnxruntime.ai/docs/execution-providers/
- ONNX Runtime QNN: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html

## Android / On-Device AI

- Android AI: https://developer.android.com/ai
- Google AI Edge: https://ai.google.dev/edge
- ExecuTorch Android: https://docs.pytorch.org/executorch/stable/large-models.html

## Financial Data

- Reserve Bank of India: https://www.rbi.org.in/
- Account Aggregator ecosystem information: https://www.rbi.org.in/

---

# 30. Final Principle

The old project asked:

> **"Can we build an autonomous financial AI?"**

The new project asks:

> **"Can we build a financial system in which AI can reason over sensitive personal state while computation, evidence, permissions, and hardware boundaries remain under explicit system control?"**

That is the problem LAXM is now built to investigate.

---

**Project:** LAXM  
**Repository:** https://github.com/AmanBanik/Laxm_work-007-AI  
**Status:** Architecture reset / implementation not started