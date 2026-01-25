from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    total_characters: int
    creation_date: str
    file_name: str
    label: str

    class Config:
        json_schema_extra = {
            "example": {
                "total_characters": 1234,
                "creation_date": "2023-10-01T12:00:00Z",
                "file_name": "example_document.txt",
                "label": "example_label"
            }
        }