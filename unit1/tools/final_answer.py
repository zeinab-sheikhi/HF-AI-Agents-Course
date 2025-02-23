from typing import Any
from smolagents.tools import Tool


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provide a final answer to a given problem"
    inputs = {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer
    
    def __init__(self, *args, **kwargs):
        self.is_initialized = False
