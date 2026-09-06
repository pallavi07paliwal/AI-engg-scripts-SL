import os

from openai import OpenAI
from dotenv import load_dotenv
from scraper import fetch_website_contents

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MAX_WEBSITE_CHARS = 24000

system_prompt = """You analyze the contents of a website and
give a short, friendly summary. Ignore navigation menus.
Respond in markdown."""

def summarize(url):
    website = fetch_website_contents(url)
    website = website[:MAX_WEBSITE_CHARS]
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Summarize this website:\n\n{website}"},
        ],
        max_tokens=800,
    )
    return response.choices[0].message.content
