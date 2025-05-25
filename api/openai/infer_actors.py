# filepath: /Users/joeheapy/Documents/EvoSocialOne/api/openai/infer_actors.py
import os
from typing import List, Dict
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

# Define a dedicated Strategy model for better validation
class Strategy(BaseModel):
    id: str = Field(description="Strategy ID in the format '[actorID-index]' (e.g., 'CG-1')")
    description: str = Field(description="Brief description of the strategy")
    commitment_level: str = Field(description="Level of commitment: 'High', 'Medium', or 'Low'")

class ActorEntry(BaseModel):
    sector: str = Field(description="The sector the organisation belongs to.")
    role_in_alleviating_child_poverty: str = Field(description="Their specific role in alleviating the described problem.")
    actor_index: str = Field(description="Index in the format 'g=n' where n is a sequential number starting from 1.")
    actor_id: str = Field(description="A two-character ID inferred from the sector name (e.g., 'CG' for 'Central Government').")
    strategies: List[Strategy] = Field(description="Three evolutionary strategies for this actor, with varying levels of commitment.")

    @validator('strategies')
    def validate_strategies(cls, strategies):
        if len(strategies) != 3:
            raise ValueError("Each actor must have exactly 3 strategies")
        # Check strategy IDs match pattern and have correct commitment levels
        for i, strategy in enumerate(strategies):
            if not strategy.id.startswith(strategies[0].id.split('-')[0]):
                raise ValueError(f"Strategy ID prefix must match actor ID")
        return strategies

class ActorsTable(BaseModel):
    actors: List[ActorEntry] = Field(description="A list of key UK organisations and actors relevant to the problem.")

def infer_actors_from_problem(problem_description: str) -> ActorsTable | None:
    """
    Uses LangChain and OpenAI to infer actors from a problem description.
    Returns an ActorsTable object or None if an error occurs.
    """
    # Ensure API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    print(f"API key starts with: {api_key[:5]}... and is {len(api_key)} characters long")

    try:
        # Clear any proxy settings that might be interfering
        for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            if env_var in os.environ:
                del os.environ[env_var]
        
        # Use direct initialization with only the required parameters
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key
        )
        parser = PydanticOutputParser(pydantic_object=ActorsTable)

        prompt_template = """
        You are an expert in UK social issues, organisational structures, and Evolutionary Game Theory. Given the following problem description, identify AT LEAST 6 key UK organisations and actors that have a role in addressing this problem. Include a diverse range covering government, private sector, non-profit, and affected populations.

        Problem: "{problem_description}"

        For each actor, provide the following:
        1. The sector they belong to
        2. Their role in addressing the problem
        3. An index in the format 'g=n' where n is a sequential number starting from 1
        4. A two-character ID derived from the sector name (e.g., 'CG' for 'Central Government')
        5. Three GENERIC evolutionary strategies representing different approaches to the system, following this structure:

           a. Strategy 1: High commitment/investment approach
              - id: [actorID-1] (e.g., 'CG-1')
              - description: [general approach representing maximum engagement with the system]
              - commitment_level: "High"
           
           b. Strategy 2: Moderate commitment/investment approach
              - id: [actorID-2] (e.g., 'CG-2')
              - description: [general approach representing partial engagement with the system]
              - commitment_level: "Medium"
           
           c. Strategy 3: Minimal commitment/investment approach
              - id: [actorID-3] (e.g., 'CG-3')
              - description: [general approach representing minimal engagement with the system]
              - commitment_level: "Low"

        IMPORTANT STRATEGY GUIDELINES:
        - Strategies should represent GENERAL APPROACHES rather than specific implementations
        - Focus on cooperation levels, resource allocation patterns, or engagement styles
        - Each strategy should represent a fundamentally different behavioral pattern, not just different intensities of the same approach
        - Describe strategies in terms of how actors engage with the entire system rather than specific solutions

        Examples of generic strategies in evolutionary game theory:
        - Full cooperation vs. selective cooperation vs. non-cooperation
        - Proactive system-wide investment vs. reactive targeted intervention vs. minimal compliance
        - Information sharing vs. selective disclosure vs. information withholding
        - Long-term system focus vs. medium-term balanced approach vs. short-term self-interest

        IMPORTANT: 
        - You MUST identify AT LEAST 6 different actors/sectors
        - Each actor MUST have exactly three strategies
        - Each strategy MUST include an id, description, and commitment_level
        - Use UK English spellings. 

        Always include these key sectors IF relevant to the problem: Central Government, Local Authorities, Private Sector, Charities/NGOs, Healthcare Providers, Social Investors, Affected Populations.

        Please format your answer as a JSON object that strictly adheres to the following Pydantic schema: {format_instructions}
        """
        # Create the prompt template with the parser's format instructions
        prompt = PromptTemplate( 
            template=prompt_template, 
            input_variables=["problem_description"], 
            partial_variables={"format_instructions": parser.get_format_instructions()} 
        ) 
        chain = prompt | llm | parser

        print("\n--- CALLING LANGCHAIN/OPENAI FOR ACTOR INFERENCE ---") 
        print(f"Problem: {problem_description}")
        
        try:
            # Separate try block for the API call specifically
            actors_table_data = chain.invoke({"problem_description": problem_description})
            
            # Validate minimum number of actors
            if len(actors_table_data.actors) < 6:
                print(f"Error: Only {len(actors_table_data.actors)} actors were identified. At least 6 are required.")
                return None
            
            # Validate that all actors have exactly 3 strategies
            for actor in actors_table_data.actors:
                if len(actor.strategies) != 3:
                    print(f"Error: Actor {actor.sector} has {len(actor.strategies)} strategies instead of 3")
                    return None
            
            print("--- LANGCHAIN/OPENAI RESPONSE (PARSED) ---") 
            print(actors_table_data) 
            print("------------------------------------------\n")
            return actors_table_data
            
        except openai.RateLimitError as e:
            print(f"OpenAI API rate limit exceeded: {e}")
            return None
        except openai.AuthenticationError as e:
            print(f"OpenAI API authentication error - check your API key: {e}")
            return None
        except openai.APITimeoutError as e:
            print(f"OpenAI API request timed out: {e}")
            return None
        except openai.BadRequestError as e:
            print(f"Bad request to OpenAI API: {e}")
            return None
        except openai.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            return None
            
    except Exception as e: 
        print(f"Error during actor inference: {type(e).__name__}: {str(e)}") 
        import traceback
        traceback.print_exc()
        return None