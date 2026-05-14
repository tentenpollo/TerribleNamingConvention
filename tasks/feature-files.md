# Feature Files

> BDD-style feature descriptions broken down by phase. Use these to drive test writing and scope agreement.

---

## Phase 1 — Auth and Access Control

### Feature: User Registration

```gherkin
Feature: User Registration

  Scenario: Successful registration
    Given a valid email and password
    When I POST to /auth/register
    Then I receive a 201 response
    And a user record is created in the database
    And the password is stored hashed, never plaintext

  Scenario: Duplicate email
    Given an email that already exists
    When I POST to /auth/register with that email
    Then I receive a 400 response
    And no new user is created

  Scenario: Invalid email format
    Given a malformed email string
    When I POST to /auth/register
    Then I receive a 422 response
```

### Feature: User Login

```gherkin
Feature: User Login

  Scenario: Successful login
    Given a registered user
    When I POST to /auth/login with correct credentials
    Then I receive a 200 response
    And the response contains a JWT token
    And the token encodes the user's role

  Scenario: Wrong password
    Given a registered user
    When I POST to /auth/login with an incorrect password
    Then I receive a 401 response

  Scenario: Non-existent user
    Given an email that does not exist
    When I POST to /auth/login
    Then I receive a 401 response
```

### Feature: Role-Based Access Control

```gherkin
Feature: Role-Based Access Control

  Scenario: Member cannot access admin routes
    Given a user with role "member"
    When they request an admin-only endpoint
    Then they receive a 403 response

  Scenario: Admin can access admin routes
    Given a user with role "admin"
    When they request an admin-only endpoint
    Then they receive a 200 response

  Scenario: Unauthenticated request blocked
    Given no JWT token in the request
    When a protected endpoint is requested
    Then the response is 401
```

### Feature: Project Access Scoping

```gherkin
Feature: Project Access Scoping

  Scenario: Member sees only their team's projects
    Given a member belonging to Team A
    And Team A is assigned Project X
    And Team B is assigned Project Y
    When the member calls GET /projects
    Then only Project X is returned
    And Project Y is not visible

  Scenario: Admin sees all projects
    Given a user with role "admin"
    When they call GET /projects
    Then all projects in the system are returned

  Scenario: Member cannot access another team's project
    Given a member of Team A
    When they request Project Y (owned by Team B)
    Then they receive a 403 response
```

---

## Phase 2 — Ingestion Pipeline

### Feature: Document Upload

```gherkin
Feature: Document Upload

  Scenario: Successful markdown upload
    Given an authenticated member with access to Project X
    When they POST a markdown file to /projects/X/documents
    Then they receive a 202 response with a job_id
    And an ingestion job is created with status "pending"

  Scenario: Successful PDF upload
    Given an authenticated member with access to Project X
    When they POST a PDF file to /projects/X/documents
    Then they receive a 202 response with a job_id

  Scenario: Unsupported file type
    Given an authenticated member
    When they upload a .xlsx file
    Then they receive a 400 response
    And no job is created

  Scenario: Upload to inaccessible project
    Given a member with no access to Project Y
    When they attempt to upload to Project Y
    Then they receive a 403 response
```

### Feature: Ingestion Job Processing

```gherkin
Feature: Ingestion Job Processing

  Scenario: Job completes successfully
    Given a queued ingestion job
    When the ARQ worker processes it
    Then the job status becomes "complete"
    And chunks are present in the project's Qdrant collection
    And a document_summary row is created in Postgres
    And the raw document is stored in Postgres

  Scenario: Job fails gracefully
    Given a queued ingestion job with a corrupt file
    When the ARQ worker attempts processing
    Then the job status becomes "failed"
    And an error_message is stored on the job
    And no partial chunks are left in Qdrant

  Scenario: Collections are isolated between projects
    Given Project X and Project Y each have ingested documents
    When querying Qdrant for Project X's collection
    Then only Project X's chunks are returned
    And Project Y's chunks are not present
```

### Feature: Ingestion Config

```gherkin
Feature: Per-Project Ingestion Configuration

  Scenario: Contextual chunking uses configured context model
    Given a project configured with chunking_strategy: "contextual" and context_model: "gpt-4o-mini"
    When a document is ingested
    Then the context annotation LLM call uses "gpt-4o-mini"
    And not the query model

  Scenario: Naive chunking skips LLM annotation
    Given a project configured with chunking_strategy: "naive"
    When a document is ingested
    Then no LLM call is made during chunking
    And chunks are stored directly
```

---

## Phase 3 — CAG Layer

### Feature: Belief State Generation

```gherkin
Feature: Belief State Generation

  Scenario: Initial belief state created on first ingest
    Given a project with no prior ingestions
    When the first document is ingested and processed
    Then a belief_state row is created for the project
    And it has version 1
    And it contains a valid JSON structure

  Scenario: Belief state updates after threshold
    Given a project with cag_rebuild_threshold: 5
    When the 5th document is ingested
    Then a CAGUpdateJob is queued
    And after processing, a new belief_state row exists with version 2
    And the old version 1 row is preserved

  Scenario: Belief state reflects project content
    Given a project with meeting notes mentioning "we chose PostgreSQL"
    When the belief state is read
    Then the decisions array contains an entry related to the PostgreSQL choice
```

### Feature: Belief State Rebuild

```gherkin
Feature: Belief State Rebuild

  Scenario: Full rebuild regenerates from event log
    Given a project with 20 ingested documents
    When a full rebuild is triggered
    Then a new belief_state is created from scratch using all 20 document summaries
    And the result is a valid belief state JSON
    And all previous versions are preserved in the DB

  Scenario: Manual rebuild is admin-only
    Given a member user
    When they POST to /projects/{id}/cag/rebuild
    Then they receive a 403 response

  Scenario: Admin can trigger manual rebuild
    Given an admin user
    When they POST to /projects/{id}/cag/rebuild
    Then they receive a 202 response
    And a CAGRebuildJob is queued
```

---

## Phase 4 — Query Layer

### Feature: Project Query

```gherkin
Feature: Project Query

  Scenario: Successful query returns grounded answer
    Given a project with ingested meeting notes about "API rate limiting decisions"
    When a member queries "what did we decide about rate limiting?"
    Then the response contains a relevant answer
    And source_chunks are included in the response
    And each source chunk has a document reference

  Scenario: Belief state used as context
    Given a project with a belief state that includes a decision about vendor X
    When a member queries "why did we choose vendor X?"
    Then the belief state version is included in the response metadata
    And the answer reflects project-level context, not just chunk similarity

  Scenario: Member cannot query inaccessible project
    Given a member of Team A
    When they POST to /projects/Y/query (Team B's project)
    Then they receive a 403 response

  Scenario: Empty result handled gracefully
    Given a project with no relevant documents on a topic
    When a member queries about that topic
    Then the response indicates no relevant content was found
    And no hallucinated sources are returned
```

### Feature: Admin Cross-Project Query

```gherkin
Feature: Admin Cross-Project Query

  Scenario: Admin can query across all projects
    Given an admin with access to Project X and Project Y
    When they POST to /query with a broad query
    Then results from both Project X and Project Y are returned
    And results are ranked by relevance

  Scenario: Member cannot use cross-project query
    Given a member user
    When they POST to /query
    Then they receive a 403 response
```

---

## Phase 5 — Frontend

### Feature: Authentication Flow

```gherkin
Feature: Authentication Flow (E2E)

  Scenario: New user registers and logs in
    Given I am on the registration page
    When I enter a valid email and password and submit
    Then I am redirected to the project list page
    And my name appears in the navigation

  Scenario: Invalid credentials show error
    Given I am on the login page
    When I enter incorrect credentials
    Then an error message is displayed
    And I remain on the login page
```

### Feature: Document Upload and Query (E2E)

```gherkin
Feature: Document Upload and Query (E2E)

  Scenario: Member uploads a document and queries it
    Given I am logged in as a member with access to Project X
    When I upload a markdown file to Project X
    Then the upload shows as "processing"
    And eventually shows as "complete"
    When I type a query related to the document content
    And submit the query
    Then I receive a relevant answer
    And at least one source document is shown

  Scenario: Belief state viewer shows current state
    Given I am on the Project X page
    When I click the "Knowledge State" tab
    Then I see the current belief state
    And it shows decisions, open items, and key people
```

---

## Phase 6 — Hardening

### Feature: Self-Hosting

```gherkin
Feature: Self-Hosting

  Scenario: Fresh clone and startup
    Given a fresh clone of the repository
    When I copy .env.example to .env and fill in an LLM API key
    And run docker compose up
    Then all services start without error
    And the health check endpoint returns 200
    And the frontend is accessible at localhost:3000

  Scenario: Works with OpenAI backend
    Given the LLM configured as "gpt-4o-mini"
    When I ingest a document and query it
    Then I receive a valid response

  Scenario: Works with Ollama backend
    Given a local Ollama instance running llama3
    And the LLM configured as "ollama/llama3"
    When I ingest a document and query it
    Then I receive a valid response
```
