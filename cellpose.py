from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.endpoint import EndpointThreshold
from kartezio.dataset import read_dataset
from kartezio.training import train_model
from kartezio.preprocessing import SelectChannels
import sys

DATASET = "../datasets/cellpose"
MODELS = "./models"
CHANNELS = [1, 2]
preprocessing = SelectChannels(CHANNELS)
run = sys.argv[1] 

if __name__ == "__main__":
    generations = 20000
    _lambda = 5
    frequency = 5
    indices = [12, 26, 76, 59, 58, 37, 11, 79, 34, 35, 36, 81, 67, 17, 13]
    model = create_instance_segmentation_model(
        generations, _lambda, inputs=2, outputs=2,
    )
    dataset = read_dataset(DATASET, indices=indices)
    
    file = MODELS + "/raw_test_data_" + run + ".txt"
    _, elite = train_model(model, dataset, MODELS, file, run, preprocessing=preprocessing, callback_frequency=frequency)