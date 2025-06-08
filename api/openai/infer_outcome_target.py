# filepath: /Users/joeheapy/Documents/EvoSocialOne/api/openai/infer_outcome_target.py
import os
from typing import List, Dict, Union
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator, ValidationError
from config import SOURCES_OF_UK_SOCIAL_DATA
import json
import re

# Define the Pydantic model for the outcome target
class OutcomeTarget(BaseModel):
    metric_name: str = Field(description="A concise label for the outcome")
    from_value: Union[float, int] = Field(description="The current baseline level (number)")
    from_unit: str = Field(description="The unit or population the baseline refers to")
    to_value: Union[float, int] = Field(description="The desired level after intervention (number)")
    to_unit: str = Field(description="The same unit/population, phrased consistently with from_unit")
    timeframe_years: int = Field(description="Number of years to reach the target")
    rationale: str = Field(description="1-2 sentences on why the target level and timeframe are ambitious yet plausible")
    sources: List[str] = Field(description="List of sources or references used for baseline data and target setting")

    @validator('from_value', 'to_value')
    def validate_numeric_values(cls, v):
        """Ensure values are proper numbers"""
        if isinstance(v, str):
            try:
                # Remove common formatting characters
                clean_value = v.replace(',', '').replace('%', '').replace('£', '').replace('$', '')
                return float(clean_value)
            except ValueError:
                raise ValueError(f"Cannot convert '{v}' to a number")
        return float(v)

class OutcomeTargets(BaseModel):
    targets: List[OutcomeTarget] = Field(description="Three measurable outcome targets for the problem")

    @validator('targets')
    def validate_targets_count(cls, targets):
        if len(targets) != 3:
            raise ValueError(f"Must have exactly 3 targets, got {len(targets)}")
        return targets

class CustomJSONOutputParser:
    """Custom parser that can extract JSON from markdown code blocks"""
    
    def __init__(self, pydantic_object):
        self.pydantic_object = pydantic_object
    
    def parse(self, text: str):
        """Parse JSON from text, handling markdown code blocks"""
        try:
            # First, try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no code block, look for JSON-like content
                # Find content between first { and last }
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = text[start_idx:end_idx + 1]
                else:
                    raise ValueError("No JSON content found in response")
            
            # Parse the JSON
            parsed_json = json.loads(json_str)
            
            # Convert to Pydantic object
            return self.pydantic_object(**parsed_json)
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Attempted to parse: {json_str[:200]}...")
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            print(f"Parsing error: {e}")
            print(f"Raw response: {text[:500]}...")
            raise ValueError(f"Failed to parse response: {e}")
    
    def get_format_instructions(self):
        """Return format instructions for the prompt"""
        return """Return your response as a JSON object with the following structure:
{
  "targets": [
    {
      "metric_name": "string",
      "from_value": number,
      "from_unit": "string", 
      "to_value": number,
      "to_unit": "string",
      "timeframe_years": integer,
      "rationale": "string",
      "sources": ["string1", "string2", ...]
    }
  ]
}

CRITICAL: 
- Return ONLY the JSON object, no additional text
- All numeric values must be actual numbers, not strings
- from_value and to_value must NEVER BE THE SAME
- Include exactly 3 targets in the array"""

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

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            print(f"\n--- ATTEMPT {attempt + 1} OF {max_attempts} ---")
            
            # Clear any proxy settings that might be interfering
            for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
                if env_var in os.environ:
                    del os.environ[env_var]
            
            # Use ChatOpenAI with web search enabled
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.2,
                openai_api_key=api_key,
                max_retries=1
            )
            
            # Use our custom parser
            parser = CustomJSONOutputParser(pydantic_object=OutcomeTargets)

            prompt_template = """
            TASK: Analyse the following problem description, then draft three measurable outcome targets that a realistic national or city-level programme could aim for. Search the web for the most recent and authoritative baseline data.

            PROBLEM: {problem_description}

            REQUIREMENTS:
            1. Create exactly 3 different outcome targets.
            2. There must be a measureable difference between the from_value and to_value.
            3. Focus on different aspects or scales of the problem (national vs regional vs local data, different demographic breakdowns, alternative measurement approaches).
            4. Use actual numbers (not strings) for all numeric values.
            5. Search for DIFFERENT data sources and metrics each time - vary your search terms and explore alternative measures.
            6. Look for regional variations, different time periods, or alternative methodologies.
            7. Search for the most current statistics available (2023-2025 data preferred).
            8. Each source must include exact title, publication year, organization, and URL.

            For each target, provide EXACTLY the following fields:
            • metric_name: A concise label for the outcome
            • from_value: Current baseline level (as a number, not string)
            • from_unit: Unit or population description
            • to_value: Target level after intervention (as a number, not string)  
            • to_unit: Same unit phrasing as from_unit
            • timeframe_years: Number of years to reach target (integer 1-20)
            • rationale: 1-2 sentences explaining why target is ambitious yet achievable
            • sources: Array of source references with titles, organizations, and years

            SEARCH PRIORITY: Focus on official UK sources: {sources_of_uk_social_data}

            {format_instructions}
            """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["problem_description"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions(),
                    "sources_of_uk_social_data": SOURCES_OF_UK_SOCIAL_DATA
                }
            )
            
            print(f"Problem: {problem_description}")
            
            # Get response from LLM
            response = llm.invoke(prompt.format(problem_description=problem_description))
            
            # Extract content from response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            print(f"Raw response length: {len(response_text)} characters")
            print(f"Response preview: {response_text[:200]}...")
            
            # Parse the response
            outcome_targets_data = parser.parse(response_text)
            
            # Validate the result
            if not outcome_targets_data or not hasattr(outcome_targets_data, 'targets'):
                raise ValueError("Invalid response structure - no targets found")
            
            if len(outcome_targets_data.targets) != 3:
                raise ValueError(f"Expected 3 targets, got {len(outcome_targets_data.targets)}")
            
            # Additional validation for each target
            for i, target in enumerate(outcome_targets_data.targets):
                if not all([
                    hasattr(target, 'metric_name') and target.metric_name,
                    hasattr(target, 'from_value') and target.from_value is not None,
                    hasattr(target, 'to_value') and target.to_value is not None,
                    hasattr(target, 'timeframe_years') and target.timeframe_years,
                    hasattr(target, 'sources') and target.sources
                ]):
                    raise ValueError(f"Target {i+1} missing required fields")
            
            print("--- SUCCESSFUL PARSING ---")
            print(f"Successfully parsed {len(outcome_targets_data.targets)} targets")
            for i, target in enumerate(outcome_targets_data.targets):
                print(f"Target {i+1}: {target.metric_name}")
                print(f"  From: {target.from_value} {target.from_unit}")
                print(f"  To: {target.to_value} {target.to_unit}")
                print(f"  Timeframe: {target.timeframe_years} years")
                print(f"  Sources: {len(target.sources)} provided")
            print("---------------------------\n")
            
            return outcome_targets_data
            
        except ValidationError as e:
            print(f"Validation error on attempt {attempt + 1}: {e}")
            if attempt == max_attempts - 1:
                print("All validation attempts failed")
                return None
            continue
            
        except (openai.RateLimitError, openai.AuthenticationError, 
                openai.APITimeoutError, openai.BadRequestError, 
                openai.APIConnectionError) as e:
            print(f"OpenAI API error: {type(e).__name__}: {e}")
            return None
        
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {type(e).__name__}: {str(e)}")
            if attempt == max_attempts - 1:
                print("All attempts failed")
                import traceback
                traceback.print_exc()
                return None
            continue
    
    return None