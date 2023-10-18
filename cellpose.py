import pkg_resources
pkg_resources.require("Kartezio==1.0.0a1")

from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.endpoint import EndpointThreshold
from kartezio.dataset import read_dataset
from kartezio.training import train_model
from kartezio.preprocessing import SelectChannels
import sys
from kartezio.plot import save_prediction
from kartezio.utils.viewer import KartezioViewer
import os

DATASET = "/tmpdir/lavinas/cellpose"
MODELS = "/tmpdir/lavinas/results_cellpose_cgp"
CHANNELS = [1, 2]
preprocessing = SelectChannels(CHANNELS)
run = sys.argv[1] 

try:
    os.makedirs(RESULTS)
except:
    print('folder already exists, continuing...')


if __name__ == "__main__":
    generations = 20000
    _lambda = 5
    frequency = 1
    indices = None
    model = create_instance_segmentation_model(
        generations, _lambda, inputs=2, outputs=2,
    )
    dataset = read_dataset(DATASET, indices=indices)
    
    # file = MODELS + "/raw_test_data_" + run + ".txt"
    file = f"{MODELS}/raw_test_data_{run}.txt"
    _, elite = train_model(model, dataset, MODELS, file, run, preprocessing=preprocessing, callback_frequency=frequency)
    
    dataset = read_dataset(DATASET, indices=None)
    test_x, test_y, test_v = dataset.test_xyv
    test_x = preprocessing.call(test_x)
    
    model.save_elite(MODELS + "/elite.json", dataset)  
    y_hat, _ = model.predict(test_x)
    # imgs_name = MODELS + "/gen_model_run_" + str(run) + "_.png"
    imgs_name = f"{MODELS}/gen_model_run_{run}_.png"
    save_prediction(imgs_name, test_v[0], y_hat[0]["mask"])

    # insight = KartezioInsight(model.parser)
    # insight.create_node_images(al.elites[i], test_x[0], prefix="./results/cgp/images/model_" + str(i) + "_cgp")
    
    viewer = KartezioViewer(
        model.parser.shape, model.parser.function_bundle, model.parser.endpoint
    )
    model_graph = viewer.get_graph(
        elite, inputs=["In_1","In_2","In_3"], outputs=["Mask"]
    )
    path = f"{MODELS}/graph_model_run_{run}_.png"
    model_graph.draw(path=path)