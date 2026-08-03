# agentic_workflow.py

import os
from dotenv import load_dotenv

# Import agents from the shared Phase 1 library
from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent

# Loading the environment variables
load_dotenv()

# Loading the OpenAI key
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
product_spec = ""
filename = "./phase_2/Product-Spec-Email-Router.txt"
with open(filename, "r") as file:
    product_spec = file.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"{product_spec}"
)
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_product_manager, knowledge_product_manager)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
max_interactions_product_manager = 10
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_product_manager_eval,
    evaluation_criteria_product_manager,
    product_manager_knowledge_agent,
    max_interactions_product_manager,
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_program_manager, knowledge_program_manager)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
max_interactions_program_manager = 10
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    evaluation_criteria_program_manager,
    program_manager_knowledge_agent,
    max_interactions_program_manager,
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_dev_engineer, knowledge_dev_engineer)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_development_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
max_interactions_development_engineer = 10
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    evaluation_criteria_development_engineer,
    development_engineer_knowledge_agent,
    max_interactions_development_engineer,
)


# Routing Agent
# Keep prior good artifacts so later roles build on it
workflow_artifacts = []

def _is_meta_instructions(text):
    """Detect evaluator-style 'how to fix' text that should not enter the plan."""
    lower = (text or "").lower()
    return (
        "to fix the answer" in lower
        or "worker agent should" in lower
        or ("follow these steps" in lower and "organize the tasks according" in lower)
    )

def product_manager_support_function(query):
    response = product_manager_knowledge_agent.respond(query)
    result = product_manager_evaluation_agent.evaluate(query, response)
    return result["final_response"]

def program_manager_support_function(query):
    if workflow_artifacts:
        query = (
            f"{query}\n\n"
            f"Group these user stories into product features. "
            f"Output actual features using Feature Name, Description, Key Functionality, "
            f"and User Benefit for each feature. Do not output formatting instructions.\n\n"
            f"{workflow_artifacts[-1]}"
        )
    response = program_manager_knowledge_agent.respond(query)
    result = program_manager_evaluation_agent.evaluate(query, response)
    return result["final_response"]

def development_engineer_support_function(query):
    if workflow_artifacts:
        prior = "\n\n".join(workflow_artifacts)
        query = (
            f"{query}\n\n"
            f"Create engineering tasks for the Email Router based on this prior workflow output. "
            f"Output actual tasks using this exact structure for each task:\n"
            f"Task ID:\nTask Title:\nRelated User Story:\nDescription:\n"
            f"Acceptance Criteria:\nEstimated Effort:\nDependencies:\n"
            f"Do not output instructions about how to format the answer.\n\n"
            f"{prior}"
        )
    response = development_engineer_knowledge_agent.respond(query)
    result = development_engineer_evaluation_agent.evaluate(query, response)
    return result["final_response"]

# Define the agents for the routing agent
agents = [
    {
        "name": "Product Manager",
        "description": (
            "Responsible for defining product personas and user stories only. "
            "Does not define features or engineering tasks. Does not group stories."
        ),
        "func": product_manager_support_function,
    },
    {
        "name": "Program Manager",
        "description": (
            "Responsible for defining product features by grouping related user stories. "
            "Does not define user stories or engineering tasks."
        ),
        "func": program_manager_support_function,
    },
    {
        "name": "Development Engineer",
        "description": (
            "Responsible for defining detailed engineering development tasks for implementing user stories. "
            "Does not define user stories or product features. "
            "Does not compile or summarize an overall development plan."
        ),
        "func": development_engineer_support_function,
    },
]
routing_agent = RoutingAgent(openai_api_key, agents)


# Run the workflow
print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = (
    "Create a development plan for this product that includes user stories, "
    "product features, and engineering development tasks."
)
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")

# Extract steps from the 'workflow_prompt'
steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
# Keep only the three plan components (stories → features → tasks). Drop summary
# steps like "compile/create a development plan that includes all...".
def _is_summary_plan_step(step):
    lower = step.lower()
    mentions_all_parts = (
        "user stories" in lower
        and ("feature" in lower)
        and ("task" in lower or "development plan" in lower)
    )
    summary_verbs = (
        "compile",
        "includes all",
        "into a development plan",
        "create a development plan that includes",
        "combine all",
        "put together",
    )
    return mentions_all_parts and any(verb in lower for verb in summary_verbs)

steps = [step for step in steps if not _is_summary_plan_step(step)]
if len(steps) > 3:
    steps = steps[:3]
print(f"Steps extracted from the workflow prompt: {steps}")

# Initialize an empty list to store 'completed_steps'
completed_steps = []

# Loop through the steps and route them to one of the three agents we defined above
for step in steps:
    result = routing_agent.route(step)
    completed_steps.append(result)
    # Only feed real artifacts forward — never poison later steps with "to fix" text
    if not _is_meta_instructions(result):
        workflow_artifacts.append(result)
    print(f"\nStep:\n\n{step}\n\ncompleted with result:\n\n{result}")

# Rubric allows last item or a consolidated summary — print all artifacts
print("\n\n\n" + "-" * 100)
print("\n*** Final output of the workflow (consolidated) ***\n")
print("-" * 100)
for index, result in enumerate(completed_steps, start=1):
    print(f"\n--- Completed step {index} ---\n\n{result}")

print("\n" + "-" * 100)