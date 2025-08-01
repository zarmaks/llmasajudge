# LLM-as-a-Judge

An automated evaluation system that uses Large Language Models to judge the quality of question-answer pairs in a RAG (Retrieval-Augmented Generation) system.

## Features

- **Multi-label Classification**: Classifies answers as "Correct", "Incorrect", or "Dangerous"
- **Safety Gate**: Deterministic safety checks before LLM evaluation
- **Flexible Provider**: Uses Mistral AI for LLM inference
- **Evaluation Metrics**: Comprehensive precision, recall, F1, and accuracy metrics
- **Report Generation**: Automated Markdown reports with statistics

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables by creating a `.env` file:
   ```bash
   MISTRAL_API_KEY=your_mistral_api_key_here
   TEMPERATURE=0.0
   SEED=0
   ```

## Usage

### Command Line Interface

```bash
python -m src.cli --in data/my.csv [--model mistral-large-latest] [--out out.csv] [--temperature 0.0] [--seed 0]
```

**Arguments:**
- `--in`: Input CSV file path (required)
- `--out`: Optional output CSV path (defaults to input_file.judged.csv)
- `--model`: Mistral model name (default: mistral-large-latest)
- `--temperature`: Sampling temperature (default: 0.0)
- `--seed`: Random seed (default: 0)

### Input CSV Format

Your CSV should contain these columns:
- `Current User Question` or `question`: The user's question
- `Assistant Answer` or `answer`: The assistant's response
- `Fragment Texts` or `fragments`: Supporting text fragments
- `Label` (optional): Ground truth labels for evaluation

### Example

```bash
python -m src.cli --in data/rag_evaluation_07_2025.csv --model mistral-large-latest --temperature 0.0 --seed 0
```

## Output

The system generates:
1. **Judged CSV**: Original data with added `Predicted_Label` and `Predicted_CoT` columns
2. **Markdown Report**: Summary statistics and metrics in `reports/` directory

## Architecture

- **Judge**: Core evaluation logic with safety gate + LLM reasoning
- **Safety**: Deterministic pattern matching for dangerous content
- **Evaluation**: Dependency-free metrics calculation
- **IO**: CSV handling utilities
- **OpenAI Client**: Mistral AI wrapper (maintains backward compatibility)

## Testing

Run tests with:
```bash
pytest tests/
```

## Configuration

The system supports various Mistral models:
- `mistral-large-latest` (default, most accurate)
- `mistral-medium-latest`
- `mistral-small-latest`

## License

[Add your license here]
