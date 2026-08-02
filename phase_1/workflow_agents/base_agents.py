import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
from openai import OpenAI


# DirectPromptAgent class definition
class DirectPromptAgent:
    '''
    This is the DirectPromptAgent class definition.
    It directly relays a user's input (prompt) to the LLM and returns the LLM's response without incorporating 
    additional context, memory, or specialized tools. 
    It does not include a system prompt.
    '''
    def __init__(self, openai_api_key):
        '''
        Initialize the agent with the OpenAI API key.
        '''
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        '''
        Generate a response using the OpenAI API.
        '''
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # rubric requires gpt-3.5-turbo except for the routing agent
            # messages: The user's prompt. It does not include a system prompt.
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        # Return only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    '''     
    The Augmented Prompt Agent is a specialized agent designed to respond according to a predefined 
    persona. Unlike basic prompt-response interactions, this agent explicitly adopts a persona, 
    leading to more targeted and contextually relevant outputs.
    '''
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with the OpenAI API key and the persona."""
        # An attribute for the agent's persona
        self.persona = persona
        # Initialize the agent with the OpenAI API key.
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # Defining both the system prompt and the user prompt
                {"role": "system", "content": f"You are a {self.persona}. Forget all previous context."},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        # Return only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    '''
    The Knowledge Augmented Prompt Agent is designed to incorporate specific, provided knowledge alongside 
    a defined persona when responding to prompts, ensuring answers are based on that explicit information.
    '''
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with the OpenAI API key, the persona, and the knowledge."""
        self.persona = persona
        # Adding an attribute to store the agent's knowledge.
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # Defining both the an extended system prompt and the user prompt
                {
                    "role": "system",
                    "content": (
                        f"You are a {self.persona} knowledge-based assistant. "
                        "Forget all previous context. "
                        f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}. "
                        "Answer the prompt based on this knowledge, not your own."
                    ),
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0
        )
        # Return only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            # FAILURE: The original code resulted in an endless loop for me. Resulting in a memory error
            #          and the program crashing: "zsh: killed     python3 phase_1/rag_knowledge_prompt_agent.py"
            # FIX:     Added this condition to break the loop when end of the text is reached.
            if end == len(text):
                break

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content


# EvaluationAgent class definition
class EvaluationAgent:
    '''
    The Evaluation Agent is designed to assess responses from another agent (a "worker" agent) 
    against a given set of criteria, potentially refining the response through iterative feedback.
    '''
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        # Initialize the EvaluationAgent with given attributes.
        self.openai_api_key = openai_api_key
        self.persona = persona # persona is here the complete string "You are a ..." not just the name of the persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        """
        This method manages interactions between agents to achieve a solution.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            # Getting the response from the worker agent
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            # Getting the response from the evaluation agent
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages= [
                      {"role": "system", "content": self.persona},
                      {"role": "user", "content": instruction_prompt}
                    ],
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            # Dictionary containing the final response, evaluation, and number of iterations
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "number_of_iterations": i + 1 # +1 because it is 0-indexed
        }


# RoutingAgent class definition
class RoutingAgent():
    '''
    The Routing Agent is capable of directing user prompts to the most appropriate specialized agent from a collection, 
    based on semantic similarity between the prompt and descriptions of what each agent handles.
    '''
    def __init__(self, openai_api_key, agents):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        # Attribute to hold the agents, call it agents
        self.agents = agents

    def get_embedding(self, text):
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        # Calculate the embedding of the text using the text-embedding-3-large model
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        # Extract the embedding vector from the response
        embedding = response.data[0].embedding
        # Return the embedding vector
        return embedding 

    # Method to route user prompts to the appropriate agent
    def route(self, user_input):
        input_emb = self.get_embedding(user_input)

        # Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)
        best_agent = None,
        best_score = -1

        for agent in self.agents:
            # Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(f"Similarity score for {agent['name']} vs. user input: {similarity}")

            # Select the best agent based on the similarity score between the user prompt and the agent descriptions
            if similarity > best_score:
                # The new agent is the best agent so far
                best_score = similarity
                best_agent = agent

        # No agent was found to be the best fit
        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


# ActionPlanningAgent class definition
class ActionPlanningAgent:
    '''
    The Action Planning Agent is crucial for constructing agentic workflows. 
    This agent uses its provided knowledge to dynamically extract and list the steps 
    required to execute a task described in a user's prompt.
    '''
    pass
    # def __init__(self, openai_api_key, knowledge):
    #     # TODO: 1 - Initialize the agent attributes here

    # def extract_steps_from_prompt(self, prompt):

    #     # TODO: 2 - Instantiate the OpenAI client using the provided API key
    #     # TODO: 3 - Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
    #     # Provide the following system prompt along with the user's prompt:
    #     # "You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {pass the knowledge here}"

    #     response_text = ""  # TODO: 4 - Extract the response text from the OpenAI API response

    #     # TODO: 5 - Clean and format the extracted steps by removing empty lines and unwanted text
    #     steps = response_text.split("\n")

    #     return steps
