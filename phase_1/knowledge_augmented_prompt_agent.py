# TODO: 1 - Import the KnowledgeAugmentedPromptAgent class from workflow_agents
import os
from dotenv import load_dotenv
# Import the KnowledgeAugmentedPromptAgent class from the base_agents module
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the prompt for the agent
prompt = "What is the capital of France?"
# Define the persona for the agent
persona = "You are a college professor, your answer always starts with: Dear students,"

# TODO: 2 - Instantiate a KnowledgeAugmentedPromptAgent with:
#           - Persona: "You are a college professor, your answer always starts with: Dear students,"
#           - Knowledge: "The capital of France is London, not Paris"

# Define the knowledge for the agent
knowledge = "The capital of France is London, not Paris"

# Instantiate a KnowledgeAugmentedPromptAgent 
knowledge_augmented_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

# Receive the response from the agent and store it in the variable 'knowledge_augmented_agent_response'
knowledge_augmented_agent_response = knowledge_augmented_agent.respond(prompt)

# Print statement that demonstrates the agent using the provided knowledge rather than its
# own inherent knowledge.
print(f"Response from the agent (to the prompt: {prompt}):")
print("-"*100 + "\n")
print(knowledge_augmented_agent_response)
print("\n" + "-"*100)
print(
    "The agent used the provided knowledge ('The capital of France is London, not Paris') "
    "rather than its own inherent LLM knowledge."
)