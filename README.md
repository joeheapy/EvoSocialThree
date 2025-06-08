# EvoSocialThree

A computational policymaking simulator that uses evolutionary game theory to model and analyse social problems in the UK. The web application helps policymakers understand complex social systems by identifying key actors, calculating strategic payoffs, and predicting behavioral outcomes.

## What It Does

EvoSocialThree transforms complex social problems into game-theoretic models to reveal:

- Which actors have the most influence on outcomes
- What strategies are economically attractive to different players
- How actors are likely to behave given current incentives
- Where policy interventions might be most effective

## How It Works

The application follows an evolutionary game theory workflow:

### 1. Problem Definition

Users input a description of a UK social problem (default examples include child poverty, prison overcrowding, homelessness, housing quality, and NHS strain from aging population).

### 2. Actor Identification (10-15 seconds)

AI analyzes the problem and identifies at least 6 key actors across sectors:

- Central Government
- Local Authorities
- Private Sector
- Social Investors
- Healthcare Providers
- Charities/NGOs
- Affected Populations

Each actor gets three evolutionary strategies representing different commitment levels (High/Medium/Low).

### 3. Outcome Targets (8-10 seconds)

The system proposes three measurable outcome targets that can track progress toward solving the problem. Users select one target to drive the simulation.

### 4. Payoff Calculation (40-50 seconds)

For each actor-strategy combination, the system calculates:

- **Δ-effect**: How much the strategy changes the target metric
- **Private cost**: Financial/political cost to the actor
- **Weight**: How much the actor genuinely cares about the outcome
- **Payoff**: Using the formula `weight × (-delta) - private_cost`
- **Behavior shares**: Proportion of actor's effort allocated to each strategy

### 5. Strategic Analysis (10-15 seconds)

Analyzes three representative strategies to show their economic attractiveness and likely adoption patterns.

## Technical Architecture

### Backend

- **Flask** web framework
- **OpenAI GPT-4** for AI inference via LangChain
- **Pydantic** for data validation and parsing
- **Pandas/NumPy** for mathematical calculations
- Modular API structure for different analysis steps

### Key Components

- `main.py`: Flask application with routing and state management
- `api/openai/`: AI inference modules for each analysis step
- `maths/`: Mathematical algorithms for payoff calculations
- `templates/`: HTML templates with component-based structure
- `config.py`: Configuration and default problem definitions

### Data Models

The application uses structured Pydantic models:

- `ActorEntry`: Represents organizations with strategies
- `Strategy`: Individual strategic options with payoff data
- `OutcomeTarget`: Measurable system objectives
- Analysis response models for each step

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd EvoSocial3

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Running the Application

```bash
python main.py
```

The application will automatically open in your browser at `http://localhost:5001`.

## Use Cases

This tool is designed for:

- **Policy analysts** modeling complex social interventions
- **Government departments** evaluating strategy effectiveness
- **Researchers** studying multi-actor social systems
- **NGOs** understanding ecosystem dynamics
- **Academic** evolutionary game theory applications

## Example Analysis Flow

1. **Input**: "UK's ageing population is straining the NHS..."
2. **Actors**: Central Government, NHS Trusts, Private Healthcare, Local Authorities, etc.
3. **Target**: "Reduce NHS waiting times by 15%"
4. **Strategies**: Each actor gets High/Medium/Low commitment options
5. **Results**: Payoff matrix showing which strategies are economically attractive

## Mathematical Foundation

The system implements evolutionary game theory principles:

- **Replicator dynamics**: Strategies with higher payoffs gain adoption share
- **Multi-actor modeling**: Complex systems with diverse stakeholder incentives
- **Payoff calculations**: Balancing social benefits against private costs
- **Weight-based preferences**: Actors value outcomes differently

## Project Status

Current version focuses on initial system modeling and payoff analysis. Future development may include:

- Full evolutionary simulation over multiple epochs
- Incentive design and policy intervention modeling
- Sensitivity analysis and robustness testing
- Enhanced visualization and reporting
