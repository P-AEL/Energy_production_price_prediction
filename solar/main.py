# file: main.py
import os 
import argparse
from trainer.trainer_manager import TrainerManager
dirname = os.path.dirname(__file__)

def main():
    # ArgumentParser initialisieren, um CLI-Argumente zu verarbeiten
    parser = argparse.ArgumentParser(description="Model Training with Configurable Iteration")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number for the current run")
    parser.add_argument("--config", type=str, default=dirname+"/config/trainer_config.yaml", help="Path to the central config YAML file")
    args = parser.parse_args()
    
    # TrainerManager initialisieren und die aktuelle Iteration sowie die Konfigurationspfade übergeben
    trainer = TrainerManager(config_path=args.config, iteration=args.iteration)
    
    # Trainings- und Evaluationsprozess starten
    trainer.train_models()
    trainer.execute_eval_notebook()

if __name__ == "__main__":
    main()
