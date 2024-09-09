# README
## Project Overview
This project aims to train and evaluate a model for detecting anti-elitism. Various approaches to data augmentation and diversity measurement are employed to explore the model's performance. This documentation provides an overview of the project structure and explains the functionality of the included scripts.
## Important Requirement
**Note:** For this project to run correctly, the [PopBERT model](https://github.com/luerhard/PopBERT) by Erhard et al.,2023, must be cloned and located in a directory parallel to the `scripts` directory. The project structure should look like this:
```
project_root/
├── PopBERT/                  # Cloned PopBERT repository
└── scripts/
    ├── anti_elitism_model.py
    ├── common_methods.py
    ├── evaluate.py
    ├── approaches/           # Contains the different approaches for generating additional training data
    │   ├── multi_step_prompting.py
    │   ├── few_shot.py
    │   ├── role_playing_basic.py
    │   ├── role_playing_diverse.py
    │   └── topic.py
    ├── diversity/            # Contains classes for calculating the similarity scores
    │   ├── chamfer_remote_evaluation.py
    │   └── evaluate_diversity.py
    ├── results/              # Directory for storing evaluation results including the comparison of the predicted labels and the gold labels
    └── generated_data/       # Directory for storing generated training and analysis data
        ├── csv_training_data/    
        └── qualitative_analysis/ 
```



## Directory Structure

### `scripts/`
- **`anti_elitism_model.py`**: 
  - This script is responsible for training and evaluating the anti-elitism model. It loads the data, preprocesses it, trains the model, and evaluates its performance on the test dataset. Training and test data are provided by Erhard et al.,2023.
  
- **`common_methods.py`**: 
  - Contains shared methods and helper functions used by other scripts. This file includes the specified Majority Vote Label Strategy for data annotated by coders and various functions to support operations in the scripts.
  
- **`evaluate.py`**: 
  - Evaluates the trained models on the test dataset, generating detailed performance reports and showing an overview of the comparison of actual labels with predicted labels.

### `scripts/approaches/`
- This directory contains scripts for generating additional training data. These approaches are designed to enhance the anti-elitism model by increasing the diversity of the training data.
  - **`multi_step_prompting.py`**: Implements a multi-step prompting strategy to generate training examples.
  - **`few_shot.py`**: Implements a few-shot learning technique for data augmentation.
  - **'few_shot_balanced.py'**: Implements the few-shot learning technique with generating additional examples according a proportional label distribution
  - **`role_playing_basic.py`**: Provides a basic role-playing scenario for data generation.
  - **`role_playing_diverse.py`**: Implements diverse role-playing scenarios to target data variety.
  - **`topic.py`**: Contains methods for generating data based on different topics.

### `scripts/diversity/`
- This directory includes scripts for calculating diversity scores and evaluating the diversity of datasets.
  - **`chamfer_remote_evaluation.py`**: Implements methods for Chamfer Distance and the Remote Clique Score to measure dataset diversity.
  - **`evaluate_diversity.py`**: Provides functions for evaluating the diversity of datasets.

### `scripts/results/`
- Stores the saved model and a comparison of the gold labels vs the predicted labels

### `scripts/generated_data/`
- Contains generated data used for training and analysis.
  - **`csv_training_data/`**: Stores CSV files with generated training data serving as input for evaluation and training.
  - **`qualitative_analysis/`**: Contains files for qualitative analysis related to the model's performance to enable the manual inspection with the assessment template.

## Usage

1. **Cloning PopBERT**: Before running any scripts, ensure that the PopBERT repository by Erhard et al., 2023, is cloned into a directory parallel to the `scripts/` directory.

   ```bash
   git clone https://github.com/luerhard/PopBERT
   ```

2. **Data Augmentation**: Explore the `approaches/` directory for different methods to generate additional training data. If you execute these approaches, the data is loaded to be evaluated.

3. **Model Training and Evaluation**: After the augmentation, run the script `evaluate.py` to evaluate the model on a separate test dataset. The results will be stored in the `results/` directory.

   ```bash
   python scripts/evaluate.py
   ```

4. **Diversity Measurement**: Use the classes in the `diversity/` directory to analyze the diversity of the datasets.


