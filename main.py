import cv2
import numpy as np
import networks.script_identifier.models.inference as inference

from networks.script_identifier.models.training import TrainPipeline


def run_script_identifier_training():
    data_path = "./dataset/lang-detect-dataset/"
    train_pipeline = TrainPipeline(data_path)
    history = train_pipeline.train(epochs=10, batch_size=32)
    history = train_pipeline.test()

if __name__ == '__main__':
    """Main entry point of training for all kinds
    of model architectures.  
    """
    run_script_identifier_training() # Start training for the Script Identifier Network
    # inference.predict_script() # For prediction using pre-trained model
    