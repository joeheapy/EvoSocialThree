# filepath: /Users/joeheapy/Documents/EvoSocialOne/api/openai/infer_outcome_target.py
import os
from typing import List, Dict
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from config import SOURCES_OF_UK_SOCIAL_DATA

# Define the Pydantic model for the outcome target
class OutcomeTarget(BaseModel):
    metric_name: str = Field(description="A concise label for the outcome")
    from_value: float = Field(description="The current baseline level (number)")
    from_unit: str = Field(description="The unit or population the baseline refers to")
    to_value: float = Field(description="The desired level after intervention (number)")
    to_unit: str = Field(description="The same unit/population, phrased consistently with from_unit")
    timeframe_years: int = Field(description="Number of years to reach the target")
    rationale: str = Field(description="1-2 sentences on why the target level and timeframe are ambitious yet plausible")
    sources: List[str] = Field(description="List of sources or references used for baseline data and target setting")

class OutcomeTargets(BaseModel):
    targets: List[OutcomeTarget] = Field(description="Three measurable outcome targets for the problem")

    @validator('targets')
    def validate_targets_count(cls, targets):
        if len(targets) != 3:
            raise ValueError("Must have exactly 3 targets")
        return targets

def infer_outcome_targets_from_problem(problem_description: str) -> OutcomeTargets | None:
    """
    Uses LangChain and OpenAI to infer outcome targets from a problem description.
    Returns an OutcomeTargets object or None if an error occurs.
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
        
        # Use direct initialization with web search enabled
        llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.3,
            openai_api_key=api_key,
            model_kwargs={
                "tools": [{"type": "web_search"}]
            }
        )
        parser = PydanticOutputParser(pydantic_object=OutcomeTargets)

        prompt_template = """
        TASK: Analyse the following problem description, then draft three measurable outcome targets that a realistic national or city-level programme could aim for. You MUST search the web for the most recent and authoritative baseline data.

        PROBLEM: {problem_description}

        REQUIREMENTS:

        For each target, identify:
        • metric_name – a concise label for the outcome (string).
        • from_value   – the current baseline level (number).
        • from_unit    – the unit or population the baseline refers to (string).
        • to_value     – the desired level after intervention (number).
        • to_unit      – the same unit/population, phrased consistently with from_unit (string).
        • timeframe_years – number of years to reach the target (integer).
        • rationale    – 1-2 sentences (string) on why the target level and timeframe are ambitious yet plausible.
        • sources      – list of sources or references used for baseline data and target setting (array of strings).

        Give exactly three targets.

        Every number must be a JSON number (not quoted).

        CRITICAL:
        - You MUST search the web for the most recent and authoritative baseline data from official UK sources: {sources_of_uk_social_data}.
        - Search for DIFFERENT data sources and metrics each time - vary your search terms and explore alternative measures
        - Look for regional variations, different time periods, or alternative methodologies
        - Search for the most current statistics available (2023-2025 data preferred)
        - Each source must include exact title, publication year, organization, and URL

        VARIATION INSTRUCTION: Focus on different aspects or scales of the problem (national vs regional vs local data, different demographic breakdowns, alternative measurement approaches).

        Please format your answer as a JSON object that strictly adheres to the following Pydantic schema: {format_instructions}
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["problem_description"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "sources_of_uk_social_data": SOURCES_OF_UK_SOCIAL_DATA
            }
        )
        
        chain = prompt | llm | parser

        print("\n--- CALLING LANGCHAIN/OPENAI FOR OUTCOME TARGETS INFERENCE ---")
        print(f"Problem: {problem_description}")
        
        try:
            # Separate try block for the API call specifically
            outcome_targets_data = chain.invoke({"problem_description": problem_description})
            
            # Validate exactly 3 targets
            if len(outcome_targets_data.targets) != 3:
                print(f"Error: {len(outcome_targets_data.targets)} targets were identified instead of exactly 3.")
                return None
            
            print("--- LANGCHAIN/OPENAI RESPONSE (PARSED) ---")
            print(outcome_targets_data)
            print("------------------------------------------\n")
            return outcome_targets_data
            
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
        print(f"Error during outcome targets inference: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None