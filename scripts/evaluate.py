from anti_elitism_model import AntiElitismTrainer
from common_methods import load_training_data

import pandas as pd

def main():
    
    print("Baseline: Training data provided by Erhard et al.,2023 without additional training examples:")
    trainer_base = AntiElitismTrainer(train_data=load_training_data())
    trainer_base.train_model()
    trainer_base.evaluate_model("_baseline")
    trainer_base.save_model()

    print("\n" + "="*50 + "\n")

    print("Few-Shot approach:")
    few_shot_data = pd.read_csv("generated_data/csv_training_data/expanded_train_data_few_shot.csv")
    trainer_few_shot = AntiElitismTrainer(few_shot_data)
    trainer_few_shot.train_model()
    trainer_few_shot.evaluate_model("_few_shot2")

    print("\n" + "="*50 + "\n")

    print("Chain-Of-thought:")
    cot_data = pd.read_csv("generated_data/csv_training_data/expanded_train_data_chain.csv")
    trainer_cot = AntiElitismTrainer(cot_data)
    trainer_cot.train_model()
    trainer_cot.evaluate_model("_chain_of_thought")

    print("\n" + "="*50 + "\n")

    print("Topic Approach:")
    role_topic_data = pd.read_csv("generated_data/csv_training_data/expanded_train_data_role_topic.csv")
    trainer_role_topic = AntiElitismTrainer(role_topic_data)
    trainer_role_topic.train_model()
    trainer_role_topic.evaluate_model("_role_topic")

    print("\n" + "="*50 + "\n")

    print("Role Playing: Basic")
    role_basic_data = pd.read_csv("generated_data/csv_training_data/expanded_train_data_role_basic.csv")
    trainer_role_basic = AntiElitismTrainer(role_basic_data)
    trainer_role_basic.train_model()
    trainer_role_basic.evaluate_model("_role_basic")

    print("\n" + "="*50 + "\n")

    print("Role Playing: more diverse roles")
    diverse_roles_data = pd.read_csv("generated_data/csv_training_data/expanded_train_data_role_diverse.csv")
    trainer_role_diverse = AntiElitismTrainer(diverse_roles_data)
    trainer_role_diverse.train_model()
    trainer_role_diverse.evaluate_model("_diverse_roles")


if __name__ == "__main__":
    main()