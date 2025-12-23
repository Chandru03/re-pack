# Re-pack Optimisation

A packing optimisation tool that uses Google OR-Tools to optimise carton packing into boxes and pallet stacking.

## Features

- Optimise carton packing into shipping boxes
- Optimise box stacking on pallets
- Generate PDF reports with visualisations
- Interactive Streamlit web interface
https://i.pinimg.com/1200x/05/cc/34/05cc349d20a99372db0ea32462137d5c.jpg


## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit Web App

```bash
streamlit run multipack_poc/main.py
```

### CLI Script

```bash
python multipack_poc/test_run.py
```

## Project Structure

```
multipack_poc/
├── main.py              # Streamlit entrypoint
├── test_run.py         # CLI script
├── core/               # Solver logic
├── models/             # Data models
├── report/             # PDF generation
├── visualization/      # Plotting utilities
└── config/             # Configuration files
```

test webhook
test webhook 2
