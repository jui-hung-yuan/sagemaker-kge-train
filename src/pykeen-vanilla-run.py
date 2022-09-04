"""
Modified to run on Amazon SageMaker. The original version is here: 
https://github.com/optuna/optuna/blob/master/examples/pytorch_simple.py
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
import logging
import sys
import time

# from neo4j import GraphDatabase
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline_from_config
from pykeen.hpo import hpo_pipeline_from_config
import torch

OUTPUT_METRICS_FNAME = "metrics_df.csv"
OUTPUT_ENTITY_FNAME = "id_to_entity.tsv"
OUTPUT_RELATION_FNAME = "id_to_relation.tsv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # The following environment variables are used by SageMaker instance to allow
    # data passage between s3 and the instance container

    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

#     parser.add_argument('--data-dir', type=str)
#     parser.add_argument('--output-dir', type=str)
   
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--training-env', type=str, default=json.loads(os.environ['SM_TRAINING_ENV']))
    
    parser.add_argument('--data-version', type=str)
    parser.add_argument('--config-fname', type=str)
    parser.add_argument('--random-seed', type=int, default=666)

    args, _ = parser.parse_known_args()    

    # load the config file, it can be either for training or HPO
    config_fpath = os.path.join(args.data_dir, args.config_fname )
    if os.path.isfile(config_fpath):
        with open(config_fpath) as file:
            pykeen_configs = json.load(file)
    else:
        raise FileNotFoundError

    # if the argument data_version is not None, we provide our own triples to Pykeen, e.g. all_triples_20220218-1506.txt
    if args.data_version is not None:
        # load the triples to pd.DataFrame
        triple_fpath = os.path.join(args.data_dir, "all_triples_{}.txt".format(args.data_version) )
        if os.path.isfile(triple_fpath):
            labeled_triples = pd.read_csv(triple_fpath, sep='\t', header=None).values
        else:
            raise FileNotFoundError

        # initialize the trainer with the triples
        tf = TriplesFactory.from_labeled_triples(labeled_triples)
        train_data, valid_data, test_data = tf.split([0.9, 0.05, 0.05], random_state=args.random_seed)

        pykeen_configs['pipeline']['training'] = train_data
        pykeen_configs['pipeline']['testing'] = test_data
        pykeen_configs['pipeline']['validation'] = valid_data

        entity_df = pd.DataFrame.from_dict(tf.entity_id_to_label, orient='index', columns=['entity'])
        relation_df = pd.DataFrame.from_dict(tf.relation_id_to_label, orient='index', columns=['relation'])
        entity_df.to_csv( os.path.join(args.output_dir, OUTPUT_ENTITY_FNAME), sep='\t' )
        relation_df.to_csv( os.path.join(args.output_dir, OUTPUT_RELATION_FNAME), sep='\t' )
    
    # if the argument data_version is None, we use one of Pykeen's available dataset.
    # it requires the 'dataset' key in the pykeen_configs['pipeline'] to be specified, e.g. "BioKG".
    else:
        if 'dataset' not in pykeen_configs['pipeline']:
            raise ValueError("No dataset is specified")

    fit_start = time.time()
    # depending on whether the config file has key "metadata" - training, or "optuna" - HPO
    if 'optuna' in pykeen_configs:
        # start HPO pipeline
        pipeline_result = hpo_pipeline_from_config(pykeen_configs)
    elif 'metadata' in pykeen_configs:
        # start training pipeline
        pykeen_configs['pipeline']['random_seed'] = args.random_seed
        pipeline_result = pipeline_from_config(pykeen_configs)
        metrics_df = pipeline_result.metric_results.to_df()
        metrics_df.to_csv( os.path.join(args.output_dir, OUTPUT_METRICS_FNAME) , index=False)
        trained_model = pipeline_result.model.to('cpu')
        trained_model_fpath = args.output_dir + "/trained_model_cpu.pkl"
        torch.save(trained_model, trained_model_fpath, pickle_protocol=4)
    
    pipeline_result.save_to_directory(directory=args.output_dir)
    total_fit_time = time.time() - fit_start
    print("total run time: {:.1f} min".format(total_fit_time/60))



    
    