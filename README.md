# Study Coach

A local Streamlit app that answers questions from your uploaded notes and helps break assignments into study plans.

## What it does

- Upload PDF, text, markdown, and CSV files into `./data/notes`
- Chat with a model that answers strictly from those files
- Track tasks and deadlines in the sidebar
- Responses are structured for study questions and planning

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set an API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
   or
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Upload your notes in the sidebar and start asking questions.

## Notes

- Everything runs locally except calls to your chosen LLM provider.
- Default models: `gpt-4o-mini` for OpenAI, `claude-3-5-haiku-20241022` for Anthropic. Change these in `utils.py` if needed.
