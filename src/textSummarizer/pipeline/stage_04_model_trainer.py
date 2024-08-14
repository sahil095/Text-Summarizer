from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.logging import logger



class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer__config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer__config)
        model_trainer.train()