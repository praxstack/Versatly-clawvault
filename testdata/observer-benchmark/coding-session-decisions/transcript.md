user: We need to finalize the persistence layer for the observer benchmark harness.
assistant: Decision: use PostgreSQL for durable event storage and Redis for short-lived queue state.
user: Great, and we should keep search terms like "Compressor", "keyword preservation", and "Project Atlas" untouched.
assistant: TODO: add regression test that fails when PostgreSQL or Redis terms disappear from observations.
user: I will publish v3.4.0 after CI passes and update the changelog.
assistant: I talked with Maya and she approved shipping behind the benchmark feature flag.
