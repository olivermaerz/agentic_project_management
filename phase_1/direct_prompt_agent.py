# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the OpenAI API key from the environment variables from the .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the prompt for the agent
prompt = "What is the Capital of France?"

# Instantiation of the DirectPromptAgent as direct_agent
direct_agent = DirectPromptAgent(openai_api_key)
# Using direct_agent to send the prompt defined above and store the response
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(f"\nResponse from the agent (to the prompt: {prompt}):")
print("-"*100 + "\n")
print(direct_agent_response)
print("\n" + "-"*100)

# Explanatory message describing the knowledge source used by the agent to generate the response
print(f"The agent used the following knowledge source LLM model gpt-3.5-turbo from OpenAI to generate the response.\n")
