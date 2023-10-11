import pkg_resources
pkg_resources.require("Kartezio==1.0.0a1")

from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.dataset import read_dataset
from kartezio.training import train_model
import sys
from kartezio.plot import save_prediction
from kartezio.utils.viewer import KartezioViewer

DATASET = "/tmpdir/lavinas/ssi/"
MODELS = "/tmpdir/lavinas/results_ssi_cgp"
filename = f"dataset.csv"
DATASET = "../datasets/ssi/"
MODELS = "../results_ssi_cgp"
filename = f"dataset.csv"
meta_filename = "META_rgb.json"

run = sys.argv[1] 

if __name__ == "__main__":
    generations = 20000
    _lambda = 5
    frequency = 10000
    indices = None
    model = create_instance_segmentation_model(
        generations, _lambda, inputs=3, outputs=2,
    )
    dataset = read_dataset(DATASET, indices=indices, filename=filename, meta_filename=meta_filename)
    
    file = MODELS + "/raw_test_data_" + run + ".txt"
    _, elite = train_model(model, dataset, MODELS, file, run, callback_frequency=frequency)
    
    dataset = read_dataset(DATASET, indices=None)
    test_x, test_y, test_v = dataset.test_xyv

    model.save_elite(MODELS + "/elite.json", dataset)  
    y_hat, _ = model.predict(test_x)
    imgs_name = MODELS+"/gen_model_run_" + str(run) + "_.png"
    save_prediction(imgs_name, test_v[0], y_hat[0]["mask"])

    viewer = KartezioViewer(
        model.parser.shape, model.parser.function_bundle, model.parser.endpoint
    )
    model_graph = viewer.get_graph(
        elite, inputs=["In_1","In_2","In_3"], outputs=["Mask"]
    )
    path = MODELS+"/graph_model_run_" + str(run) + "_.png"
    model_graph.draw(path=path)