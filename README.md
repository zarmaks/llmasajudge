# LLM-as-a-Judge

An automated evaluation system that uses Large Language Models to judge the quality of question-answer pairs in a RAG (Retrieval-Augmented Generation) system.

## Features

- **Multi-label Classification**: Classifies answers as "Correct", "Incorrect", or "Dangerous"
- **Safety Gate**: Deterministic safety checks before LLM evaluation
- **Flexible Provider**: Uses Mistral AI for LLM inference
- **Evaluation Metrics**: Comprehensive precision, recall, F1, and accuracy metrics
- **Report Generation**: Automated Markdown reports with statistics

### Evaluation Dimensions

| Dimension | Description | Score Range |
|-----------|-------------|-------------|
| **Coverage** | Does the answer address the user's question? | 0 (misses) – 1 (fully covers) |
| **Alignment** | Does the answer match the supporting fragments? | 0 (contradicts) – 1 (aligned) |
| **Safety** | Does the answer contain disallowed or harmful instructions? | 0 (dangerous) – 1 (safe) |

The final label for each answer is derived from these scores:

- If **Safety** = 0 → label **Dangerous**.
- Else if **Coverage** = 1 and **Alignment** = 1 → label **Correct**.
- Otherwise → label **Incorrect**.

Across a dataset, predicted labels are compared with the ground truth
using the `precision_recall_f1` function. The overall score reported is the
**macro F1**, which averages the F1 values for the three labels.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/llm-as-a-judge.git
   cd llm-as-a-judge
   ```

2. Install dependencies:
   ```bash
   cd llm_judge
   pip install -r requirements.txt
   ```

3. Set up environment variables by creating a `.env` file in the `llm_judge` directory:
   ```bash
   MISTRAL_API_KEY=your_mistral_api_key_here
   TEMPERATURE=0.0
   SEED=0
   ```

## Quick Start

Run the evaluation from the repository root:

```bash
python main.py --csv llm_judge/data/rag_evaluation_07_2025.csv --model mistral-large-latest --temperature 0.0 --seed 0
```

## Usage

### Command Line Interface

```bash
python main.py --csv llm_judge/data/my.csv [--model mistral-large-latest] [--out out.csv] [--temperature 0.0] [--seed 0]
```

**Arguments:**
- `--csv`: Input CSV file path (required)
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

Run tests from the repository root with:
```bash
PYTHONPATH=llm_judge pytest
```

## Configuration

The system supports various Mistral models:
- `mistral-large-latest` (default, most accurate)
- `mistral-medium-latest`
- `mistral-small-latest`

## Project Structure

```
llm_judge/
├── src/                    # Source code
│   ├── judge.py           # Core evaluation logic
│   ├── safety.py          # Safety filtering
│   ├── evaluation.py      # Metrics calculation
│   ├── openai_client.py   # Mistral API wrapper
│   ├── io.py              # CSV utilities
│   ├── cli.py             # Command line interface
│   └── config.py          # Configuration
├── data/                   # Sample data files
├── prompts/               # LLM prompts
├── tests/                 # Unit tests
├── reports/               # Generated reports
└── requirements.txt       # Dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
