import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# Define schema for structured output


class Review(BaseModel):

    key_themes: list[str] = Field(
        description="List of key themes mentioned in the review"
    )
    summary: str = Field(description="Summary of the review")
    sentiment: str = Field(
        description="Sentiment of the review either positive, negative or neutral"
    )
    pros: Optional[list[str]] = Field(
        default=None, description="List of pros mentioned in the review"
    )
    cons: Optional[list[str]] = Field(
        default=None, description="List of cons mentioned in the review"
    )
    rating: float = Field(
        gt=0.0,
        le=10.0,
        description="Rating of the product between 0.0 and 10.0",
    )


structured_model = model.with_structured_output(Review)

response = structured_model.invoke(
    """ The Samsung Galaxy S25 Ultra is a premium smartphone that excels in various aspects. It features a stunning 6.8-inch Dynamic AMOLED display with a 120Hz refresh rate and a peak brightness of 1750 nits, providing vibrant colors and smooth scrolling. Powered by the latest Snapdragon 8 Gen 3 processor, it offers exceptional performance for gaming and multitasking. The camera system is impressive, with a 200MP main sensor that captures stunning detail, along with ultra-wide and telephoto lenses for versatile photography. The battery life is robust, lasting over a day with heavy usage, thanks to its 5000mAh battery, and supports fast charging and wireless charging. The design is sleek and premium, with an IP68 rating for water and dust resistance. 

Pros:
- Exceptional camera quality
- High-performance Snapdragon 8 Gen 3 processor
- Stunning display with 1750 nits brightness
- Long battery life with 5000mAh capacity
- Premium build quality

Cons:
- High price point
- Bulky design
- Limited software updates compared to competitors."""
)

print(response)
