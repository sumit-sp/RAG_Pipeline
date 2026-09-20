"""Custom DeepEval judge model, since the default wants an OpenAI key we don't have.

Uses Groq's gpt-oss-120b (bigger than the gpt-oss-20b generator) as an independent
judge for faithfulness/relevancy/precision/recall scoring — see DECISIONS.md for
the trade-off (still same model family as the generator, so not a fully independent
judge, but better than self-grading with the identical model).
"""

from dotenv import load_dotenv
from groq import AsyncGroq, Groq

from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()

JUDGE_MODEL_NAME = "openai/gpt-oss-120b"


class GroqJudgeModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = JUDGE_MODEL_NAME):
        self.model_name = model_name
        self.client = Groq()
        self.async_client = AsyncGroq()
        super().__init__(model_name)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=4096,
            reasoning_effort="low",
        )
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=4096,
            reasoning_effort="low",
        )
        return response.choices[0].message.content or ""

    def get_model_name(self) -> str:
        return self.model_name
