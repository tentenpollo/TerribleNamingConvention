# PROJECT.md

> The what and why of this project.

---

## Problem Statement

Teams accumulate knowledge across meetings, documents, decisions, and conversations — and that knowledge evaporates. Six months into a project, nobody remembers why a critical decision was made, who owns what, or what was tried and abandoned. New members spend weeks piecing together context that veterans carry in their heads.

Existing tools fail at this in a specific way. Tools like Notion AI, Confluence search, and generic RAG chatbots treat documents as a flat index. They find text that looks similar to your query. They do not understand your project. Asking "why did we choose vendor X?" returns the document that mentions vendor X — not the reasoning trail, the people involved, the alternatives considered, or whether the assumptions made then still hold today.

This is the problem this platform solves.

---

## Core Mechanic

The platform maintains a **belief state** per project — a structured, living summary of what the project knows about itself. This is the CAG (Cache-Augmented Generation) layer. It is not a static system prompt or a summary document you write by hand. It is generated from the project's ingested content and updated automatically as new content comes in.

When a user queries a project, the query does not hit a flat vector index. It hits the belief state first — which orients the LLM on what this project is, what decisions have been made, what is currently open, and who the key people are. Then specific chunk retrieval (RAG) grounds the response in actual document content.

The result is answers that feel like they come from someone who knows the project — not a search engine.

---

## Who This Is For

**Primary users:**
- Project teams (engineering, product, design, ops) who work with documents, meeting notes, and accumulated decisions
- Internal knowledge managers at companies who want a self-hosted, private alternative to SaaS knowledge tools

**Secondary users:**
- Developers who want to extend the platform with custom ingestion adapters
- Organizations with compliance or data privacy requirements who cannot send internal data to a third-party SaaS

---

## User Roles

| Role | What They Can Do |
|---|---|
| **Member** | Query and ingest into assigned projects only. Cannot see or access any other project. |
| **Admin** | Query and ingest across all projects in their scope. Can run cross-project queries. Can manage team memberships and projects. |
| **Super Admin** | Full system access. Manages users, teams, projects, system config, and LLM settings. |

---

## Core Use Cases

**1. Decision retrieval**
"Why did we choose Qdrant over Pinecone?" → Returns the original reasoning, the alternatives considered, who made the call, and flags if any of the original assumptions have been superseded by more recent documents.

**2. Status orientation**
"What is the current status of the authentication feature?" → Returns a synthesis from the belief state (overall project context) grounded in specific recent meeting notes and docs.

**3. Onboarding acceleration**
A new team member can ask open-ended questions about the project — "what are the main open problems right now?", "who is responsible for the API layer?" — and get oriented answers rather than a wall of search results.

**4. Cross-project querying (Admin)**
"Have any of our projects dealt with GDPR compliance for user data?" → Admin queries across all accessible projects, surfaces relevant decisions and context.

**5. Knowledge health check**
Admins can view the belief state of any project to get a structured snapshot of what the system understands about it — useful for auditing knowledge quality.

---

## Non-Goals (v1)

These are explicitly out of scope for the first version:

- **Real-time connectors** — no live sync with Notion, Slack, Google Docs, GitHub. v1 is file upload only (markdown, txt, PDF).
- **Analytics and usage dashboards** — no query frequency metrics, no user behavior tracking.
- **SSO / OAuth** — v1 uses email/password with JWT. SSO is a v2 concern.
- **Multi-org support** — v1 is single-org. One deployment = one organization.
- **Fine-grained permissions below project level** — access is at the project level. Document-level ACL is v2.
- **GraphRAG entity relationship layer** — no explicit entity graph in v1. Cross-project querying is semantic, not graph-traversal.
- **Conflict or contradiction detection** — the system does not surface when two documents contradict each other. Future work.

---

## Success Criteria for v1

- A team can self-host the platform with `docker compose up` in under 10 minutes
- A member can upload meeting notes and query them with meaningful, grounded responses
- The belief state accurately reflects the project's key decisions and open items after 10+ documents
- Access control is provably enforced — a member cannot access another team's project at any layer
- The platform works with at least three LLM backends (OpenAI, Anthropic, Ollama)
- The codebase is legible enough that a new contributor can understand it and add an ingestion adapter

---

## Open Source Goals

This project is open source because:
- Organizations with privacy requirements need a self-hostable option — SaaS is not viable for them
- The hybrid CAG/RAG architecture should be available to the community to build on
- Open source is the right distribution model for internal tooling that needs to be trusted

The package should be clean enough to be used as a library, not just run as a service.
