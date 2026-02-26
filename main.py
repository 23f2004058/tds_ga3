from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Request model
class CommentRequest(BaseModel):
    comment: str

# Response model (also used as structured output schema)
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int  # 1-5

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. "
                        "Analyze the sentiment of the given comment and return:\n"
                        "- sentiment: 'positive', 'negative', or 'neutral'\n"
                        "- rating: integer 1-5 where 5=highly positive, 1=highly negative, 3=neutral"
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format=SentimentResponse,
        )

        result = response.choices[0].message.parsed
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")
