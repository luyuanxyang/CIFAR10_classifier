A Image Classification Using Vision Language Model using the ai.sooners.us API.

## Requirements

- Python 3.8 or higher
- API key from ai.sooners.us

## Setup Instructions

1. **Get your API key:**
   - Visit https://ai.sooners.us
   - Sign up with your OU email
   - Go to Settings Account API Keys
   - Create a new API key and copy it
2. **Create environment file:**
   
   Create a file at `~/.soonerai.env` with:

   SOONERAI_API_KEY=your_key_here
   SOONERAI_BASE_URL=https://ai.sooners.us
   SOONERAI_MODEL=gemma3:4b
   
   Replace `your_key_here` with your actual API key.

## How to Run
pip install -r requirements.txt
python cifar10_classify.py

## SELECT YOUR EXPERIMENT in cifar10_classify.py file:
SYSTEM_PROMPT = SYSTEM_PROMPT_V1

## Project Structure

- `cifar10_classify.py` - Main classification script
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `.gitignore` - Excludes env files from Git
