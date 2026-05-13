# Garmin R10 HelpBot

This project is for Study.com CS 311 Project #2. It builds a simple FAQ chatbot for the Garmin Approach R10 portable golf launch monitor.

The chatbot uses:

- Python
- LangChain
- FAISS vector search
- OpenAI API when an `OPENAI_API_KEY` is provided
- A local fallback mode when no API key is available

## Files

- `garmin_r10_helpbot.py`: main Python application
- `garmin_r10_faq_data.csv`: FAQ dataset used by the chatbot
- `requirements.txt`: Python libraries needed to run the project
- `.env.example`: example environment file for the OpenAI API key
- `Garmin_R10_HelpBot_Report.docx`: project report
- `sample_chatbot_results.png`: sample terminal-style results image

## How to Run

1. Install Python 3.
2. Open a terminal in this project folder.
3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Optional but recommended: create a `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

5. Run the chatbot:

```bash
python garmin_r10_helpbot.py
```

6. Ask a Garmin Approach R10 question, such as:

```text
How far behind the ball should I place the R10?
How do I pair it with my phone?
How long does the battery last?
What does the solid blue light mean?
```

7. Type `quit` to exit.

## Notes

The full assignment version is designed to use LangChain with an LLM API. If no API key is found, the program still runs in a local fallback mode so the retrieval workflow can be tested.
