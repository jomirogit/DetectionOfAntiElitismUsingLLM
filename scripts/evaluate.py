from anti_elitism_model import AntiElitismTrainer
from common_methods import load_training_data

import pandas as pd

def main():
    
    print("Overall dataset information")
    train_data = load_training_data()
    print("Distribution of the labels in the test set:")
    print(train_data["elite"].value_counts())
    
    train_data["text_length"] = train_data["text"].apply(len)
    average_length = train_data["text_length"].mean()
    print(f"Durchschnittliche Textlaenge: {average_length:.2f} Zeichen")

    print("\n" + "="*50 + "\n")

    print("Baseline: Training data provided by Erhard et al.,2023 without additional training examples:")
    trainer_base = AntiElitismTrainer(train_data=load_training_data())
    trainer_base.train_model()
    trainer_base.evaluate_model("_baseline")
    trainer_base.save_model()

    print("\n" + "="*50 + "\n")

    print("Few-Shot approach:")
    print("Augmented only with data labeled as true for anti-elitism and maximum token length 1000")
    few_shot_data = pd.read_csv("generated_data/csv_training_data/expanded_train_data_few_shot_onlyAE_l1000.csv")
    trainer_few_shot = AntiElitismTrainer(few_shot_data)
    trainer_few_shot.train_model()
    trainer_few_shot.evaluate_model("_few_shot")
    print("Augmented with data labeled as true for anti-elitism and labeled as false for anti-elitsm")
    few_shot_data_balanced = pd.read_csv("generated_data/csv_training_data/expanded_train_data_few_shot_balanced.csv")
    trainer_few_shot_balanced = AntiElitismTrainer(few_shot_data_balanced)
    trainer_few_shot_balanced.train_model()
    trainer_few_shot_balanced.evaluate_model("_few_shot_balanced")




    print("\n" + "="*50 + "\n")

    print("Multi-Step Prompting:")
    print("Augmented only with data labeled as true for anti-elitism and maximum token length 1000")
    msp_data = pd.read_csv("generated_data/csv_training_data/expanded_train_data_msp_onlyAE_l1000.csv")
    trainer_msp = AntiElitismTrainer(msp_data)
    trainer_msp.train_model()
    trainer_msp.evaluate_model("_multi_step_prompting")

    print("\n" + "="*50 + "\n")

    print("Topic Approach:")
    print("Augmented only with data labeled as true for anti-elitism and maximum token length 1000")
    role_topic_data = pd.read_csv("generated_data/csv_training_data/expanded_train_data_role_topic_onlyAE_l1000.csv")
    trainer_role_topic = AntiElitismTrainer(role_topic_data)
    trainer_role_topic.train_model()
    trainer_role_topic.evaluate_model("_role_topic")

    print("\n" + "="*50 + "\n")

    print("Role Playing: Basic")
    print("Augmented only with data labeled as true for anti-elitism and maximum token length 1000")
    role_basic_data = pd.read_csv("generated_data/csv_training_data/expanded_train_data_role_basic_onlyAE_l1000.csv")
    trainer_role_basic = AntiElitismTrainer(role_basic_data)
    trainer_role_basic.train_model()
    trainer_role_basic.evaluate_model("_role_basic")

    print("\n" + "="*50 + "\n")

    print("Role Playing: more diverse roles")
    print("Augmented only with data labeled as true for anti-elitism and maximum token length 1000")
    diverse_roles_data = pd.read_csv("generated_data/csv_training_data/expanded_train_data_role_diverse_onlyAE_l1000.csv")
    trainer_role_diverse = AntiElitismTrainer(diverse_roles_data)
    trainer_role_diverse.train_model()
    trainer_role_diverse.evaluate_model("_diverse_roles")
    print("Augmented with data labeled as true for anti-elitism and labeled as false for anti-eltisim")


if __name__ == "__main__":
    main()