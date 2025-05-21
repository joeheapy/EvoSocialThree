# filepath: /Users/joeheapy/Documents/EvoSocialOne/api/openai/infer-actors.py
import os
from typing import List
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Define the Pydantic models for the expected table structure
class ActorEntry(BaseModel):
    sector: str = Field(description="The sector the organisation belongs to (e.g., Central Government, Local Government, Charities/NGOs).")
    example_organisations_actors: str = Field(description="Specific examples of UK organisations or types of actors.")
    role_in_alleviating_child_poverty: str = Field(description="Their specific role in alleviating the described problem. If the problem is not child poverty, adapt the role description accordingly.")

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

        You are an expert in UK social issues and organisational structures. Given the following problem description, identify the key UK organisations including government, private and social sectors, and actors that have a role to play in alleviating this problem. Always include those most impacted by the problem as a sector, for example, 'Children and families'. Focus on UK-based entities.

        Problem: "{problem_description}"

        Please format your answer as a JSON object that strictly adheres to the following Pydantic schema: {format_instructions}

        Ensure the output contains a list of actors, where each actor has:

        "sector": The sector the organisation belongs to.
        "example_organisations_actors": Specific 'Examples' of UK organisations or types of actors.
        "role_in_alleviating_child_poverty": Their specific role in addressing the described problem. Adapt this field name if the problem is not child poverty, but ensure the content reflects their role for the given problem. """ 

        # The PydanticOutputParser will use the field names from ActorEntry for the JSON keys. 
        # The description for "role_in_alleviating_child_poverty" in ActorEntry guides the LLM. 
        # The column header in the HTML can remain as per the screenshot. 
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