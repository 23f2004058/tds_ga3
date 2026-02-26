from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
import os
import json

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. "
                        "Analyze the sentiment of the given comment and return JSON with exactly two fields:\n"
                        "- sentiment: must be exactly 'positive', 'negative', or 'neutral'\n"
                        "- rating: integer 1-5 where 5=highly positive, 1=highly negative, 3=neutral\n"
                        "Return ONLY valid JSON, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        sentiment = data.get("sentiment", "neutral").lower()
        if sentiment not in ["positive", "negative", "neutral"]:
            sentiment = "neutral"

        rating = int(data.get("rating", 3))
        rating = max(1, min(5, rating))

        return SentimentResponse(sentiment=sentiment, rating=rating)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")
