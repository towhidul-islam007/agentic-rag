# Vertex AI Chatbot

A Streamlit chatbot application that connects to Google Vertex AI for conversational AI capabilities.

## Setup

### Prerequisites
- Python 3.10+
- Poetry (install from https://python-poetry.org/docs/#installation)
- Google Cloud account with Vertex AI API enabled

### Installation

1. **Clone and setup the project:**
   ```bash
   poetry install
   ```

2. **Set up Google Cloud:**
   - Create a Google Cloud project
   - Enable the Vertex AI API
   - Create a service account with Vertex AI User permissions
   - Download the service account key JSON file

3. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your actual values:
   - `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID
   - `GOOGLE_APPLICATION_CREDENTIALS`: If this is not provided ADC will be used
   - `VERTEX_AI_LOCATION`: Region (default: global)

### Running the App

```bash
poetry run streamlit run app.py
```

Or use the configured script:
```bash
poetry run start
```

### Development

Enter the Poetry shell:
```bash
poetry shell
```
