"""
AI layer — LLM provider interfaces and concrete adapters.

All LLM/AI calls must go through an abstract provider interface defined in
this layer so that:
  1. Tests can inject mock providers without live API keys.
  2. The underlying model/vendor can be swapped without touching domain logic.

No real API calls may be made unless an API key is present in the environment.
"""
