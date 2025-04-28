import yaml
from pydantic import BaseModel

class IntentionDetection(BaseModel):
    should_query_vector_db: bool
    titles: list[str]
    query_to_vector_db: str
    answer_to_user: str

with open("static/prompts.yaml") as f:
    prompt_templates = yaml.safe_load(f)
