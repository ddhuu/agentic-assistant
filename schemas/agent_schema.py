from pydantic import BaseModel, Field
from typing import Literal

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    question: str = Field(
        description='Input from user'
    )
    content: str = Field(
        description='Content of the response'
    )

