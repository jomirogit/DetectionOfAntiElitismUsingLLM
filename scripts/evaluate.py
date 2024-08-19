from anti_elitism_model import AntiElitismTrainer
from common_methods import load_training_data

def main():
    
    print("Baseline: Training data provided by Erhard et al.,2023 without additional training examples:")
    trainer_base = AntiElitismTrainer(train_data=load_training_data())
    trainer_base.train_model()
    trainer_base.evaluate_model()
    trainer_base.save_model()

    print("Few-Shot approach:")

    print("Chain-Of-thought:")

    print("Topic Approach:")

    print("Role Playing: Basic")

    print("Role Playing: more diverse roles")


if __name__ == "__main__":
    main()