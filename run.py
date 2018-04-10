#!/usr/bin/env python

#############################
# ChaLearn AutoML2 challenge #
#############################

# Usage: python program_dir/run.py input_dir output_dir program_dir

# program_dir is the directory of this program

#
# The input directory input_dir contains 5 subdirectories named by dataset,
# including:
# 	dataname/dataname_feat.type          -- the feature type "Numerical", "Binary", or "Categorical" (Note: if this file is abscent, get the feature type from the dataname.info file)
# 	dataname/dataname_public.info        -- parameters of the data and task, including metric and time_budget
# 	dataname/dataname_test.data          -- training, validation and test data (solutions/target values are given for training data only)
# 	dataname/dataname_train.data
# 	dataname/dataname_train.solution
# 	dataname/dataname_valid.data
#
# The output directory will receive the predicted values (no subdirectories):
# 	dataname_valid.predict           
# 	dataname_test.predict
# We have 2 test sets named "valid" and "test", please provide predictions for both.
# 
# We implemented 2 classes:
#
# 1) DATA LOADING:
#    ------------
# Use/modify 
#                  D = DataManager(basename, input_dir, ...) 
# to load and preprocess data.
#     Missing values --
#       Our default method for replacing missing values is trivial: they are replaced by 0.
#       We also add extra indicator features where missing values occurred. This doubles the number of features.
#     Categorical variables --
#       The location of potential Categorical variable is indicated in D.feat_type.
#       NOTHING special is done about them in this sample code. 
#     Feature selection --
#       We only implemented an ad hoc feature selection filter efficient for the 
#       dorothea dataset to show that performance improves significantly 
#       with that filter. It takes effect only for binary classification problems with sparse
#       matrices as input and unbalanced classes.
#
# 2) LEARNING MACHINE:
#    ----------------
# Use/modify 
#                 M = MyAutoML(D.info, ...) 
# to create a model.
#     Number of base estimators --
#       Our models are ensembles. Adding more estimators may improve their accuracy.
#       Use M.model.n_estimators = num
#     Training --
#       M.fit(D.data['X_train'], D.data['Y_train'])
#       Fit the parameters and hyper-parameters (all inclusive!)
#       What we implemented hard-codes hyper-parameters, you probably want to
#       optimize them. Also, we made a somewhat arbitrary choice of models in
#       for the various types of data, just to give some baseline results.
#       You probably want to do better model selection and/or add your own models.
#     Testing --
#       Y_valid = M.predict(D.data['X_valid'])
#       Y_test = M.predict(D.data['X_test']) 
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 
#
# Main contributors: Isabelle Guyon and Arthur Pesah, March-October 2014
# Lukasz Romaszko April 2015
# Originally inspired by code code: Ben Hamner, Kaggle, March 2013
# Modified by Ivan Judson and Christophe Poulain, Microsoft, December 2013
# Last modifications Isabelle Guyon, November 2017

# =========================== BEGIN USER OPTIONS ==============================

TIME_RESIDUAL = 50.0

# Verbose mode: 
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
############## 
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there are 5 datasets). 
# The code should keep track of time spent and NOT exceed the time limit 
# in the dataset "info" file, stored in D.info['time_budget'], see code below.
# If debug >=1, you can decrease the maximum time (in sec) with this variable:
max_time = 1200 

# Maximum number of cycles, number of samples, and estimators
#############################################################
# Your training algorithm may be fast, so you may want to limit anyways the 
# number of points on your learning curve (this is on a log scale, so each 
# point uses twice as many time than the previous one.)
# The original code was modified to do only a small "time probing" followed
# by one single cycle. We can now also give a maximum number of estimators 
# (base learners).
max_cycle = 0
max_estimators = 100
max_samples = float('Inf')

# I/O defaults
##############
# If true, the previous output directory is not overwritten, it changes name
save_previous_results = False
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = "../"
default_input_dir = root_dir + "sample_data"
default_output_dir = root_dir + "AutoML2_sample_result_submission"
#default_program_dir = root_dir + "AutoML2_sample_code_program"
default_program_dir = root_dir + "automl-challenge-2018"

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# Version of the sample code
version = 5 

# General purpose functions
import time
import numpy as np
overall_start = time.time()
import os
from sys import argv, path
import datetime
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

import sklearn
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.metrics
try:
    import sklearn.model_selection as sklm
except:
    import sklearn.cross_validation as sklm

import lib.bayeso.bayeso.bo as bayesobo

def _get_random_hyps(space_hyps):
    list_hyps = []
    for elem_hyps in space_hyps:
        cur_hyp = np.random.uniform(elem_hyps[3], elem_hyps[4])
        list_hyps.append(cur_hyp)
    return list_hyps

def _convert_to_hyp(space_hyps, result_hyp=None, base_hyps=[]):
    scale = 100.0
    if result_hyp is not None:
        list_hyp = base_hyps
        for idx_elem, elem_hyps in enumerate(space_hyps):
            if elem_hyps[5]:
                cur_elem = result_hyp[idx_elem] / scale
            else:
                cur_elem = result_hyp[idx_elem]
            if elem_hyps[0]:
                cur_elem = 10**cur_elem
            if elem_hyps[1]:
                cur_elem = int(cur_elem)
            list_hyp.append(cur_elem)
    else:
        list_hyp = base_hyps
        for elem_hyps in space_hyps:
            if elem_hyps[5]:
                cur_elem = elem_hyps[2] / scale
            else:
                cur_elem = elem_hyps[2]
            if elem_hyps[0]:
                cur_elem = 10**cur_elem
            if elem_hyps[1]:
                cur_elem = int(cur_elem)
            list_hyp.append(cur_elem)
    return list_hyp

def _check_same(cur_hyp, list_hyps_all, space_hyps):
    cur_hyp = np.array(cur_hyp)
    for elem_hyps in list_hyps_all:
        is_same = True
        for cur_elem, elem_elem_hyps in zip(cur_hyp, elem_hyps):
            if np.abs(float(cur_elem) - float(elem_elem_hyps)) > 0.01:
                is_same = False
        if is_same:
            return True
    return False

if __name__=="__main__" and debug_mode < 4:
    try:
        print('scikit-learn version: ' + sklearn.__version__)
        print('numpy version: ' + np.__version__)
    except:
        pass

    #### Check whether everything went well (no time exceeded)
    execution_success = True
    
    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir = default_program_dir
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])
        
    if verbose: 
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)
        
	# Our libraries
    path.append(program_dir + "/lib/")
    path.append(input_dir)
    import data_io # general purpose input/output functions
    from data_io import vprint # print only in verbose mode
    from data_manager import DataManager # load/save data and get info about them
    from models import MyAutoML # example model

    if debug_mode >= 4: # Show library version and directory structure
        data_io.show_dir(".")
        
    # Move old results and create a new output directory (useful if you run locally)
    if save_previous_results:
        data_io.mvdir(output_dir, output_dir + '_' + the_date) 
    data_io.mkdir(output_dir) 
    
    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    # Overwrite the "natural" order
    
    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode >= 3:
        data_io.show_version()
        data_io.show_io(input_dir, output_dir)
        print('\n****** Ingestion program version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')
        data_io.write_list(datanames)
        datanames = [] # Do not proceed with learning and testing
        
    #### MAIN LOOP OVER DATASETS: 
    overall_time_budget = 0
    time_left_over = 0
    for basename in datanames: # Loop over datasets
#        vprint(verbose, "\n========== Ingestion program version " + str(version) + " ==========\n") 
#        vprint(verbose, "************************************************")
        vprint(verbose, "******** Processing dataset " + basename.capitalize() + " ********")
#        vprint(verbose, "************************************************")
        
        # ======== Learning on a time budget:
        # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
        start = time.time()
        
        # ======== Creating a data object with data, informations about it
        vprint(verbose, "========= Reading and converting data ==========")
        D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, max_samples=max_samples, verbose=verbose)

#        n_components = int(D.info['feat_num'])
        try:
            num_train = int(D.info['train_num'])
        except:
            num_train = D.data['X_train'].shape[0]
            D.info['train_num'] = num_train
        try:
            num_valid = int(D.info['valid_num'])
        except:
            num_valid = D.data['X_valid'].shape[0]
            D.info['valid_num'] = num_valid
        try:
            num_test = int(D.info['test_num'])
        except:
            num_test = D.data['X_test'].shape[0]
            D.info['test_num'] = num_test
        dim_data = D.data['X_train'].shape[1]

        is_valid = True
        is_chronological_order = int(D.info['is_chronological_order'])
        if is_valid:
            train_size = 0.6
            if is_chronological_order == 0:
                vprint(verbose, '[+] Split with non-chronological order')
                X_train_train, X_train_valid, Y_train_train, Y_train_valid = sklm.train_test_split(D.data['X_train'], D.data['Y_train'], train_size=train_size, stratify=D.data['Y_train'])
            else:
                vprint(verbose, '[+] Split with chronological order')
                num_train_train = int(train_size * num_train)
                X_train_train = D.data['X_train'][0:num_train_train]
                X_train_valid = D.data['X_train'][num_train_train:]
                Y_train_train = D.data['Y_train'][0:num_train_train]
                Y_train_valid = D.data['Y_train'][num_train_train:]
                if np.unique(Y_train_train).shape[0] != np.unique(Y_train_valid).shape[0]:
                    vprint(verbose, '[-] Fail to split with chronological order')
                    X_train_train, X_train_valid, Y_train_train, Y_train_valid = sklm.train_test_split(D.data['X_train'], D.data['Y_train'], train_size=train_size, stratify=D.data['Y_train'])

        if debug_mode < 1:
            time_budget = D.info['time_budget']
        else:
            time_budget = max_time
        time_budget = float(time_budget)
        overall_time_budget = overall_time_budget + time_budget
        vprint(verbose, "[+] Cumulated time budget (all tasks so far)  %5.2f sec" % (overall_time_budget))
        # We do not add the time left over form previous dataset: time_budget += time_left_over
        vprint(verbose, "[+] Time budget for this task %5.2f sec" % time_budget)
        time_spent = time.time() - start
        vprint(verbose, "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
        if time_spent >= time_budget:
            vprint(verbose, "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue
        
        # ========= Creating a model, knowing its assigned task from D.info['task'].
        # The model can also select its hyper-parameters based on other elements of info.
        
        # ========= Iterating over learning cycles and keeping track of time
        time_spent = time.time() - start
        vprint(verbose, "[+] Remaining time after building model %5.2f sec" % (time_budget-time_spent))        
        if time_spent >= time_budget:
            vprint(verbose, "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue

        time_budget = time_budget - time_spent
        start = time.time()
        time_spent = 0.0
        cycle = 0
        time_one_loop = 0.0

        is_bo = True
        num_all = num_train + num_valid + num_test
        if num_all > 10000 and dim_data > 1000:
            is_bo = False
            init_n_estimators_1 = 90
            init_n_estimators_2 = 90
            init_n_estimators_3 = 90
            rate_n_estimators = 1.0 / 2.0
            init_max_depth = 300 / 100.0
        elif num_all < 2000 and dim_data > 5000:
            init_n_estimators_1 = 300
            init_n_estimators_2 = 300
            init_n_estimators_3 = 300
            rate_n_estimators = 1.0 / 2.0
            init_max_depth = 750 / 100.0
        elif is_chronological_order == 1:
            init_n_estimators_1 = 400
            init_n_estimators_2 = 400
            init_n_estimators_3 = 400
            rate_n_estimators = 1.0 / 2.0
            init_max_depth = 750 / 100.0
        else:
            init_n_estimators_1 = 800
            init_n_estimators_2 = 800
            init_n_estimators_3 = 800
            rate_n_estimators = 1.0 / 2.0
            init_max_depth = 750 / 100.0

        # list of [is log10, is integer, initial value, start, end, is scaled 100]
        space_hyps = [
            [True, False, 0.0, -0.60, 0.70, False], # second / first weight for votingclassifier
            [True, False, 0.0, -0.60, 0.70, False], # third / first weight for votingclassifier
            [False, True, init_n_estimators_1, int(init_n_estimators_1 * rate_n_estimators), init_n_estimators_1, False], # n_estimators_1
#            [False, True, init_n_estimators_2, int(init_n_estimators_2 / 2), init_n_estimators_2, False], # n_estimators_2
#            [False, True, init_n_estimators_3, int(init_n_estimators_3 / 2), init_n_estimators_3, False], # n_estimators_3
            [False, True, init_max_depth, 250 / 100.0, init_max_depth, False], # max_depth
#            [True, False, -100.0, -200.0, 0.0, True], # learning_rate
#            [False, False, 100.0, 50.0, 100.0, True], # subsample
        ]
        D.info['hyps'] = _convert_to_hyp(space_hyps, base_hyps=[is_bo])

        arr_range = np.array([[elem_hyps[3], elem_hyps[4]] for elem_hyps in space_hyps])
        model_bo = bayesobo.BO(arr_range, str_acq='ucb')
        cur_hyps = [elem_hyps[2] for elem_hyps in space_hyps]
        list_hyps_all = []
        list_measures_all = []
        
        while time_spent <= time_budget - (2 * time_one_loop) - TIME_RESIDUAL and is_bo and True:
            vprint(verbose, "=========== " + basename.capitalize() + " Training cycle " + str(cycle) + " ================") 
            M = MyAutoML(D.info, verbose=False, debug_mode=debug_mode)

            M.fit(X_train_train, Y_train_train)

            vprint(verbose, "[+] Fitting success, time spent so far %5.2f sec" % (time.time() - start))
            # Make predictions
            # -----------------
            pred_train_valid = M.predict(X_train_valid)
            if 'classification' in D.info['task']:
                performance = sklearn.metrics.roc_auc_score(Y_train_valid, pred_train_valid)
                performance = 2 * performance - 1
                vprint(verbose, "[+] AUC for X_train_valid, %5.4f" % (performance))
            else:
                preformance = 0.0
                vprint(verbose, "[-] Performance cannot be measured")

            list_hyps_all.append(cur_hyps)
            list_measures_all.append([(1.0 - performance) * 10.0])
            cur_hyps, _, _, _ = model_bo.optimize(np.array(list_hyps_all), np.array(list_measures_all), is_grid_optimized=False, verbose=False)
            if _check_same(cur_hyps, list_hyps_all, space_hyps):
                vprint(verbose, "[-] Random selection")
                cur_hyps = _get_random_hyps(space_hyps)
            D.info['hyps'] = _convert_to_hyp(space_hyps, cur_hyps, base_hyps=[is_bo])

            Y_valid = M.predict(D.data['X_valid'])
            Y_test = M.predict(D.data['X_test'])
            vprint(verbose, "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
            # Write results
            # -------------
            filename_valid = basename + '_valid.predict'
            filename_test = basename + '_test.predict'
            data_io.write(os.path.join(output_dir,filename_valid), Y_valid)
            data_io.write(os.path.join(output_dir,filename_test), Y_test)
            vprint(verbose, "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
            time_spent = time.time() - start
            if cycle == 0:
                time_one_loop = time_spent
            vprint(verbose, "[+] Time for the first loop, %5.2f sec" % (time_one_loop))
            time_left_over = time_budget - time_spent
            vprint(verbose, "[+] End cycle, time left %5.2f sec" % time_left_over)
            if time_left_over <= 0:
                break
            cycle += 1

        if True:
#        if cycle > 1:
            vprint(verbose, "======== Creating last model ==========")
            if is_bo and list_measures_all is not [] and list_hyps_all is not []:
                sorted_hyps = sorted(zip(list_measures_all, list_hyps_all), key=lambda elem_: elem_[0])
                print(sorted_hyps)
                cur_hyps = sorted_hyps[0][1]
                D.info['hyps'] = _convert_to_hyp(space_hyps, cur_hyps, base_hyps=[is_bo])

            else:
                D.info['hyps'] = _convert_to_hyp(space_hyps, base_hyps=[is_bo])
            M = MyAutoML(D.info, verbose=False, debug_mode=debug_mode)
            print(M)
            M.fit(D.data['X_train'], D.data['Y_train'])

            vprint(verbose, "[+] Fitting success, time spent so far %5.2f sec" % (time.time() - start))

            Y_valid = M.predict(D.data['X_valid'])
            Y_test = M.predict(D.data['X_test'])
            vprint(verbose, "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
            # Write results
            # -------------
            filename_valid = basename + '_valid.predict'
            filename_test = basename + '_test.predict'
            data_io.write(os.path.join(output_dir, filename_valid), Y_valid)
            data_io.write(os.path.join(output_dir, filename_test), Y_test)
            vprint(verbose, "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
            vprint(verbose, "[+] Time for the first loop, %5.2f sec" % (time_one_loop))
            time_spent = time.time() - start
            time_left_over = time_budget - time_spent
            vprint(verbose, "[+] End cycle, time left %5.2f sec" % time_left_over)
               
    overall_time_spent = time.time() - overall_start
    if execution_success:
        vprint(verbose, "[+] Done")
        vprint(verbose, "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget %5.2f sec" % overall_time_budget)
    else:
        vprint(verbose, "[-] Done, but some tasks aborted because time limit exceeded")
        vprint(verbose, "[-] Overall time spent %5.2f sec " % overall_time_spent + " > Overall time budget %5.2f sec" % overall_time_budget)

