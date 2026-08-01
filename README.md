Udacity _AI-Powered Agentic Workflow for Project Management_ project: reusable agent library and multi-agent workflow that turns a product spec into user stories, features, and engineering tasks.

Setup with [uv](https://docs.astral.sh/uv/): `uv venv && source venv/bin/activate && uv pip install -r requirements.txt`

Add your OpenAI API key to a `.env` file: `OPENAI_API_KEY=your_key`

Then run Phase 1 agent tests from `phase_1/` (e.g. `python direct_prompt_agent.py`) and the Phase 2 workflow with `python phase_2/agentic_workflow.py`.
