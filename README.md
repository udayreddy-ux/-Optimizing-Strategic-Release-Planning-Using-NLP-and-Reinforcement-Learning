# Optimizing Software Release Planning with Sentiment Analysis and Q-Learning

This project integrates Natural Language Processing (NLP) and Reinforcement Learning (RL) to optimize the strategic release planning process in software development. Utilizing sentiment analysis through the VADER tool, software requirements are categorized based on sentiment, and a Q-learning model is then trained to select an ideal mix of requirements for various release plans, ensuring a balance between positive and negative sentiment features.

## Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations](#limitations)
- [Contributing](#contributing)

## About The Project

Strategic release planning is critical to software development, ensuring the right features are delivered at the right time. This project aims to enhance this process by applying sentiment analysis to requirement descriptions, thereby capturing stakeholder sentiment. Reinforcement learning, through a Q-learning model, is then used to strategically select requirements that meet the sentiment balance criteria for each release plan.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- Python 3.6 or higher
- pip

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your-username/strategic-release-planning.git
   ```

### Usage
1. To run the program:
   ```sh
   python reinforcementLearningSrpWithNlp.py
   ```

### Dataset
The dataset consists of requirement descriptions. Each has been analyzed using sentiment analysis to provide a sentiment score, which guides the release planning process.

### Model Training
The Q-learning model is trained over 1000 episodes. During each episode, the model updates the Q-table based on the reward structure, which encourages the selection of requirements to achieve a balanced sentiment profile for each release plan.

### Evaluation Metrics
The model is evaluated using the following metrics:

Balance of sentiments across selected requirements
Diversity of requirements within each release plan
Adherence to the predefined number of requirements for each release plan
Reward convergence across episodes, indicating the model's learning stability

### Limitations
The model's effectiveness is contingent upon the accuracy of sentiment analysis and the dataset's representation of the complexity of requirements. The computational demands and the model's generalizability also present challenges, particularly in adapting to client-imposed prioritization constraints.

### Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.
