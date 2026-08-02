# agentic_workflow.py

import os
from turtle import clear
from dotenv import load_dotenv
# import our agents from phase 1
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
# Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"{product_spec}"
)
# Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_product_manager, knowledge_product_manager)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_product_manager = "The answer should be stories that follow the following structure: " \
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
# Instatiate the product manager evaluation agent
max_interactions_product_manager = 3
product_manager_evaluation_agent = EvaluationAgent(openai_api_key, persona_product_manager_eval, evaluation_criteria_product_manager, product_manager_knowledge_agent, max_interactions_product_manager)


# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_program_manager, knowledge_program_manager)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_program_manager = "The answer should be product features that follow the following structure: " \
    "Feature Name: A clear, concise title that identifies the capability\n" \
    "Description: A brief explanation of what the feature does and its purpose\n" \
    "Key Functionality: The specific capabilities or actions the feature provides\n" \
    "User Benefit: How this feature creates value for the user"
# Instatiate the program manager evaluation agent
max_interactions_program_manager = 3
program_manager_evaluation_agent = EvaluationAgent(openai_api_key, persona_program_manager_eval, evaluation_criteria_program_manager, program_manager_knowledge_agent, max_interactions_program_manager)


# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_dev_engineer, knowledge_dev_engineer)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_development_engineer = "The answer should be tasks following this exact structure: " \
    "Task ID: A unique identifier for tracking purposes\n" \
    "Task Title: Brief description of the specific development work\n" \
    "Related User Story: Reference to the parent user story\n" \
    "Description: Detailed explanation of the technical work required\n" \
    "Acceptance Criteria: Specific requirements that must be met for completion\n" \
    "Estimated Effort: Time or complexity estimation\n" \
    "Dependencies: Any tasks that must be completed first"
# Instatiate the development engineer evaluation agent
max_interactions_development_engineer = 3
development_engineer_evaluation_agent = EvaluationAgent(openai_api_key, persona_dev_engineer_eval, evaluation_criteria_development_engineer, development_engineer_knowledge_agent, max_interactions_development_engineer)


# Routing Agent
# Define the support functions for the routes
def product_manager_support_function(query):
    product_manager_knowledge_agent.respond(query)
    result = product_manager_evaluation_agent.evaluate(query)
    return result["final_response"]

def program_manager_support_function(query):
    program_manager_knowledge_agent.respond(query)
    result = program_manager_evaluation_agent.evaluate(query)
    return result["final_response"]

def development_engineer_support_function(query):
    development_engineer_knowledge_agent.respond(query)
    result = development_engineer_evaluation_agent.evaluate(query)
    return result["final_response"]

# Define the agents for the routing agent
agents = [
    {
        "name": "Product Manager",
        "description": "Product Manager: responsible for personas/user stories only (not features/tasks)",
        "func": product_manager_support_function
    },
    {
        "name": "Program Manager",
        "description": "Program Manager: groups stories into features (not stories/tasks)",
        "func": program_manager_support_function
    },
    {
        "name": "Development Engineer",
        "description": "Development Engineer: defines engineering tasks (not stories/features)",
        "func": development_engineer_support_function
    }
]
routing_agent = RoutingAgent(openai_api_key, agents)


# Run the workflow
print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")

# Extract steps from the 'workflow_prompt'
steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print(f"Steps extracted from the workflow prompt: {steps}")

# Initialize an empty list to store 'completed_steps'
completed_steps = []

# Loop through the steps and route them to one of the three agents we defined above
for step in steps:
    # Route the step to the appropriate support function
    result = routing_agent.route(step)
    # Append the result to the list of completed steps
    completed_steps.append(result)
    # Print information about the step being executed and its result
    print(f"\nStep:\n\n{step}\n\ncompleted with result:\n\n{result}")

# The last completed step is the final output of the workflow, print it
print(f"Final output of the workflow: {completed_steps[-1]}")
