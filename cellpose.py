import pkg_resources
pkg_resources.require("Kartezio==1.0.0a1")

from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.endpoint import EndpointThreshold
from kartezio.dataset import read_dataset
from kartezio.training import train_model
from kartezio.preprocessing import SelectChannels
import sys

DATASET = "/tmpdir/lavinas/cellpose"
MODELS = "/tmpdir/lavinas/results_cellpose_cgp"
CHANNELS = [1, 2]
preprocessing = SelectChannels(CHANNELS)
run = sys.argv[1] 

if __name__ == "__main__":
    generations = 20000
    _lambda = 5
    frequency = 10
    indices = [2, 52, 29, 1, 40, 28, 19, 79]
    model = create_instance_segmentation_model(
        generations, _lambda, inputs=2, outputs=2,
    )
    dataset = read_dataset(DATASET, indices=indices)
    
    file = MODELS + "/raw_test_data_" + run + ".txt"
    _, elite = train_model(model, dataset, MODELS, file, run, preprocessing=preprocessing, callback_frequency=frequency)