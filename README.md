# Function Calling Auto-Evaluation Pipeline

An automated testing pipeline for evaluating function calling capabilities of language models. This tool measures accuracy across function selection, query understanding, and response generation.

## Overview

This pipeline evaluates three key aspects:
- Function selection accuracy
- Query understanding accuracy 
- Response generation accuracy

The evaluation uses Gemini API to assess semantic similarity between predicted and expected outputs.

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 
- Transformers
- Google Cloud API access

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fc_autoeval_pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration Files

1. Create `credentials.py` with your Google API key:
```python
# filepath: credentials.py
api_key = "YOUR_GOOGLE_API_KEY"
```

2. Create `prompt.py` with your system prompt:
```python
# filepath: prompt.py
system_prompt = """Your system prompt here..."""
```

3. Create `tools.py` with your function definitions:
```python
# filepath: tools.py
tools = '''[
    {
        "name": "get_recipe_details",
        "description": "Lấy chi tiết công thức nấu ăn",
        "parameters": {
            "type": "object",
            "properties": {
                "recipe_name": {
                    "type": "string",
                    "description": "Tên công thức nấu ăn"
                }
            },
            "required": ["recipe_name"]
        }
    },
    // Add more tools as needed
]'''
```

## Usage

Run the evaluation script with:

```bash
python test_fc.py <model_path> <test_data_path>
```

Arguments:
- `model_path`: Path to the pretrained model
- `test_data_path`: Path to test data Excel file 

Example:
```bash
python test_fc.py /path/to/model /path/to/test_data.xlsx
```

## Test Data Format

The test data Excel file should contain the following columns:
- `title`: The input message/command
- `custom_nlp_sample`: Expected query of function calling
- `custom_nlp_expected_dialog`: Expected response
- `custom_nlp_expected_intent`: Expected function name

## Output

The script outputs:
- Function selection accuracy
- Query understanding accuracy
- Response generation accuracy
- Detailed results in `answer.jsonl`

Example output:
```
Function accuracy: 0.85
Query accuracy: 0.78
Response accuracy: 0.73
```

## Features

- Automated evaluation of function calling capabilities
- Semantic similarity assessment using Gemini API
- Support for multiple model architectures
- Detailed logging and results analysis
- Configurable system prompts and tools

## Technical Details

- Uses PyTorch for model inference
- Supports quantization configurations
- Implements custom tokenization
- Handles both text and audio inputs
- Supports batched processing

## License

[Your License]

## Contact

[Your Contact Information]