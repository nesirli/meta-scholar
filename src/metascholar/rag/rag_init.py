from openai import OpenAI

from metascholar.config import settings
from schemas import LLMCallRecord

client = OpenAI(api_key=settings.openai_api_key)

class RAG:

    llm_model = ''
    last_call: LLMCallRecord = None

    @staticmethod
    def calculate_cost(model, usage):
        cost = 0
        if "gpt-5.4-mini" in model:
            cost = (usage.input_tokens * 0.15 + usage.output_tokens * 0.60) / 1_000_000
        return cost

    def __init__(self):
        pass

    def embed(text: str, model: str = "text-embedding-3-small") -> list[float]:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

    def index(self):
        pass

    def search(self):
        pass

    def call_llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        return response

    def log_response(self, prompt, response, response_time):
        usage = response.usage
        cost = calculate_cost(self.model, usage)

        call_record = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=response.output_text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            response_time=response_time,
            cost=cost,
        )

        self.last_call = call_record