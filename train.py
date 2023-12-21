from train_hlm import *
from train_llm import *
from configs import *
import traceback
import datetime
import logging
import time
import json


logging.getLogger('PIL').setLevel(logging.WARNING)


def exp1():

    # Setting the run directory
    if exp_config['run_type'] == 'll_model_train':
        run_number = len(os.listdir(exp_config['results_path_llm']))
        curr_result_dir = os.path.join(exp_config['results_path_llm'], f'Run{run_number:04}')
    elif exp_config['run_type'] == 'hl_model_train':
        run_number = len(os.listdir(exp_config['results_path_hlm']))
        curr_result_dir = os.path.join(exp_config['results_path_hlm'], f'Run{run_number:04}')
    if exp_config['resume_training']:
        run_number = int(exp_config['resume_path'].split('/')[2][3:])
        curr_result_dir = exp_config['resume_path'].split('Train')[0]

    exp_config['results_dir'] = curr_result_dir
    if not os.path.exists(curr_result_dir):
        os.mkdir(curr_result_dir)

    # Setting the log files to easily access the train and test results. Also saving the config file used to run the
    # experiment.
    details_path= os.path.join(exp_config['results_dir'], 'details.txt')
    with open(details_path, 'w'):
        logging.basicConfig(filename= details_path, filemode='a', level=logging.DEBUG, format='')
    config_details_path= os.path.join(exp_config['results_dir'], 'config_details.json')
    json_object = json.dumps(exp_config, indent= 4)
    with open(config_details_path, "w") as outfile:
        outfile.write(json_object)
    logging.info(exp_config['run_type'])
    logging.info(f'Run{run_number:04}')
    
    # Setting train and test configurations
    train_datasets = []
    for dataset in exp_config['datasets'].keys():
        if exp_config['datasets'][dataset]['train']:
            train_datasets.append(dataset)
    test_domains = ['CLIVE']
    
    # Training the chosen model
    if exp_config['run_type'] == 'll_model_train':
        model = TrainQCLLLM(exp_config, train_datasets, test_domains)
        model.learn()
    elif exp_config['run_type'] == 'hl_model_train':
        model = TrainGCLHLM(exp_config, train_datasets)
        model.learn()
    return


def main():
    exp1()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
