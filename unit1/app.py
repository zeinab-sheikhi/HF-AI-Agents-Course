import datetime
import pytz
import yaml

from dotenv import load_dotenv
from Gradio_UI import GradioUI
from smolagents import CodeAgent, HfApiModel, tool, AgentImage
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebPageTool
from tools.web_search import DuckDuckGoSearchTool


@tool 
def my_custom_tool(arg1: str, arg2: str) -> str:
    """
    A tool that does nothing yet
    
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build?"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """
    A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g. 'America/NewYork')
    """

    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()
web_search = DuckDuckGoSearchTool()
visit_webpage = VisitWebPageTool()


load_dotenv()

model = HfApiModel(
    max_tokens=2096, 
    temperature=0.5,
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    custom_role_conversions=None,
)

with open("prompts.yaml") as stream:
    prompt_templates = yaml.safe_load(stream)


agent = CodeAgent(
    model=model, 
    tools=[final_answer, web_search, visit_webpage],
    max_steps=6, 
    verbosity_level=1, 
    grammar=None, 
    planning_interval=None, 
    name=None, 
    description=None, 
    prompt_templates=prompt_templates,
)

image_agent = AgentImage()


if __name__ == '__main__':
    GradioUI(agent).launch()
