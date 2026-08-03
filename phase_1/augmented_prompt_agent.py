import os
from dotenv import load_dotenv
# Import the AugmentedPromptAgent class from the base_agents module
from workflow_agents.base_agents import AugmentedPromptAgent

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Instantiation of an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# Receive the response from the agent and store it in the variable 'augmented_agent_response'
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(f"\nResponse from the agent (to the prompt: {prompt}):")
print("-"*100 + "\n")
print(augmented_agent_response)
print("\n" + "-"*100)

# The agent used the knowledge of OpenAI's LLM model gpt-3.5-turbo to answer the prompt.
# The system prompt specifying the persona affected the agent's response tone and style 
# (e.g. by adding the "Dear students," prefix as the college professor persona).
print(
    "The agent used the knowledge of OpenAI's LLM model gpt-3.5-turbo to answer the prompt. \
    The system prompt specifying the persona affected the agent's response tone and style \
    (e.g. by adding the 'Dear students,' prefix as the college professor persona)."
    )