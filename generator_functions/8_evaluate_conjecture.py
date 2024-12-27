"""
evaluate_conjecture.py
"""
import sys
import json
import importlib.util
from typing import List, Dict, Any
import os
import re
from datetime import datetime
import types
from types import MappingProxyType, GetSetDescriptorType
import logging

evaluate_conjecture_module_logger = None
PRUNED_SCORE_PLACEHOLDER = 'N8cT89C044lqvJ'
DEFINED_OBJECTS_PLACEHOLDER = "vcM/juON9%1gv!lLp"

"""
Identify the functions, classes, instances associated with 'PCO False'. Select targets for elimination/improvement, then evaluate improvement using PCO_false.
"""
def separate_dicts_by_bool(ewm_observation_data):
    # Collect all predictions observed as 'False'
    pco_false_dicts = []
    pco_true_dicts = []
    pco_none_dicts = []
    prediction_error_dicts = []

    for observation_dict in ewm_observation_data:
        if 'pco_bool' in observation_dict and observation_dict['pco_bool'] == False:
            pco_false_dicts.append(observation_dict)
        elif 'pco_bool' in observation_dict and observation_dict['pco_bool'] == True:
            pco_true_dicts.append(observation_dict)
        elif 'pco_bool' in observation_dict and observation_dict['pco_bool'] == None:
            pco_none_dicts.append(observation_dict)
        elif 'computation_error' in observation_dict:
            prediction_error_dicts.append(observation_dict)
    # of the objects that interact with false observations, which of these objects has the highest proportion of false interactions
    return pco_false_dicts, pco_true_dicts, pco_none_dicts, prediction_error_dicts


# def parse_tuple_string(tuple_string):
#     """
#     Parses a tuple string and return the actual tuple.

#     Args:
#         tuple_string [str]: sting representing tuple.

#     Returns:
#         Tuple[str, str]: tuple with the variable path and class name.
#     """
#     pattern = r"\('([^']*)', <class '([^']*)'>\)"

#     match = re.match(pattern, tuple_string.strip())
#     if match:
#         variable_path = match.group(1)
#         class_name = match.group(2)
#         tuple_result=(variable_path, class_name)
#     else:
#         # Handle the case where the string doesn't match the expected format
#         raise ValueError(f"String '{tuple_string}' does not match expected format.")

#     return tuple_result

# def recursive_get_object_names_from_dict(tuple_string_dict):
#     obj_name_dict = {}
#     for key, value in tuple_string_dict.items():
#             # Apply parse_tuple_string to the key
#             name_type_tuple = parse_tuple_string(key)
#             name = name_type_tuple[0]
            
#             # If the value is a dictionary, recursively process it
#             if isinstance(value, dict):
#                 # Recursively apply the changes to the inner dictionary
#                 inner_dict = recursive_get_object_names_from_dict(value)
#                 obj_name_dict[name] = inner_dict
#             else:
#                 # Otherwise, just insert the value
#                 obj_name_dict[name] = value
#     return obj_name_dict


# def generate_observation_graph_dict_list(ewm_observation_data):
#     """
#     Generate a list of dictionaries representing directed graph of dependencies for each 
#     observation. e.g. Sub-Functions and Inputs -> Function -> LLM -> PCO bool
#     """
#     # Make a deep copy of ewm_observation_data and initialize the graph dictionary
#     ewm_observation_data_copy = copy.deepcopy(ewm_observation_data)

#     # Initialize observation_dict_list to store the graph dictionaries of each observation
#         # Each observation dictionary will be a hyperedge
#     observation_dict_list = []

#     successful_observation_keys = {'pco_bool', 
#                      'method_or_function', 
#                      'function_name', 
#                      'instance_name(s)', 
#                      'accessed_objects', 
#                      'result', 
#                      'observer_software'}
#     computation_error_keys = {'computation_error', 
#                      'method_or_function', 
#                      'function_name', 
#                      'instance_name(s)'}
    
#     for obs_dict in ewm_observation_data_copy:
#         if successful_observation_keys.issubset(obs_dict):
#             observation_graph_dict = {}
#             # Create a variable that stores sections of observation_graph_dict
#             obs_graph_dict_navigator = observation_graph_dict

#             # Add the 'pco_bool' as highest level key
#             obs_graph_dict_navigator = obs_graph_dict_navigator.setdefault(obs_dict['pco_bool'], {})
#             # Add the 'observer_software' below that
#             obs_graph_dict_navigator = obs_graph_dict_navigator.setdefault(obs_dict['observer_software'], {})
#             # Add 'result' below that
#             result_string = json.dumps(obs_dict['result'], default=str)  # Convert 'result' to a JSON string
#             obs_graph_dict_navigator = obs_graph_dict_navigator.setdefault(result_string, {})

#             # Add 'function' below that
#             function_object_type_tuple_string = get_function_name(obs_dict)
#             tuple_function_item = parse_tuple_string(function_object_type_tuple_string)
#             computation_function_name = tuple_function_item[0]
#             obs_graph_dict_navigator = obs_graph_dict_navigator.setdefault(computation_function_name, {})

#             # Add nested 'accessed_objects' below the function
#             # Note that the function name should always be the first key in the 'accessed_objects' dictionary, so it should be skipped
#             accessed_objects_tuple_string_dict = obs_dict['accessed_objects']
#             accessed_objects_name_dict = recursive_get_object_names_from_dict(accessed_objects_tuple_string_dict)

#             # Check if accessed_objects_name_dict is not empty
#             if accessed_objects_name_dict:
#                 # Get the function name (the first key)
#                 accessed_function_name = next(iter(accessed_objects_name_dict))
#                 if accessed_function_name == computation_function_name:
#                     # Get the dictionary under the function name
#                     function_contents_dict = accessed_objects_name_dict[accessed_function_name]
#                     # Update obs_graph_dict_navigator with the contents under the function name
#                     obs_graph_dict_navigator.update(function_contents_dict)
#                 else:
#                     # Update obs_graph_dict_navigator with the entire accessed_objects dict (replace the specified function name)
#                     obs_graph_dict_navigator.update(accessed_objects_name_dict)
#                     evaluate_conjecture_module_logger.warning(f"Function name in accessed objects '{accessed_function_name}' does not match funciton name of computation {computation_function_name}")


#             # Add 'inputs' below the highest level item (the function name) in accessed_objects
#             # Create a list of nested keys to the 'function' key
#             keys_to_function = [obs_dict['pco_bool'], obs_dict['observer_software'], result_string, computation_function_name]
#             # Reset obs_graph_dict_navigator to the full dictionary
#             obs_graph_dict_navigator = observation_graph_dict

#             for key in keys_to_function:
#                 if isinstance(obs_graph_dict_navigator, dict) and key in obs_graph_dict_navigator:
#                     obs_graph_dict_navigator = obs_graph_dict_navigator[key]
#                 else:
#                     raise KeyError(f"Function key '{key}' not found at this level in the dictionary.")

#             # Now add all instance names under the function key
#             inputs_tuple_string_list = get_inputs(obs_dict)
#             for input_name_type_tuple_string in inputs_tuple_string_list:
#                 input_name_type_tuple = parse_tuple_string(input_name_type_tuple_string)
#                 input_name = input_name_type_tuple[0]
#                 obs_graph_dict_navigator.setdefault(input_name, {})

#             # Append the observation graph dictionary to the list
#             observation_dict_list.append(observation_graph_dict)

#         elif computation_error_keys.issubset(obs_dict):
#             observation_graph_dict = {}
#             # Create a variable that stores sections of observation_graph_dict
#             obs_graph_dict_navigator = observation_graph_dict

#             # Add 'False' boolean as highest level key
#             obs_graph_dict_navigator = obs_graph_dict_navigator.setdefault(False, {})
#             # Add 'computation_error' below that
#             computation_error_str = str(obs_dict['computation_error'])
#             obs_graph_dict_navigator = obs_graph_dict_navigator.setdefault(computation_error_str, {})
#             # Add 'function' below that
#             function_object_type_tuple_string = get_function_name(obs_dict)
#             tuple_function_item = parse_tuple_string(function_object_type_tuple_string)
#             computation_function_name = tuple_function_item[0]
#             obs_graph_dict_navigator = obs_graph_dict_navigator.setdefault(computation_function_name, {})
#             # Add 'inputs' below that
#             inputs_tuple_string_list = get_inputs(obs_dict)
#             for input_name_type_tuple_string in inputs_tuple_string_list:
#                 input_name_type_tuple = parse_tuple_string(input_name_type_tuple_string)
#                 input_name = input_name_type_tuple[0]
#                 obs_graph_dict_navigator.setdefault(input_name, {})

#             # Append the observation graph dictionary to the list
#             observation_dict_list.append(observation_graph_dict)

#     print(f"observation_dict_list length = {len(observation_dict_list)}\newm_observation_data length = {len(ewm_observation_data)}\n")

#     return observation_dict_list

# def compare_obj_pco_dictionaries(old_obj_pcos_dict, new_obj_pcos_dict):
#     """
#     Compares two PCO value dictionaries and returns two lists with lost object names and higher inconsistency scores.
#         Input lists have format:  {'object_name':{'true': 0, 'false': 0, 'none': 0, 
#              'false_to_true': 0}, ...}

#     Args:
#         dict1 (dict): The first dictionary to compare.
#         dict2 (dict): The second dictionary to compare.

#     Returns:
#         tuple: A tuple containing two lists:
#             - list_pco_true_missing_or_zero: Names from dict1 with 'true' >= 1 that are either
#               missing in dict2 or have 'true' < 1 in dict2.
#             - list_new_pco_true: Names from dict2 with 'true' >=1 that are either missing in 
#               dict1 or have 'true < 1 in dict1
#             - list_inconsistency_higher: Names from dict1 that are either missing in dict2 or have
#               a higher 'false_and_none_to_true' value than in dict2.
#             - list_inconsistency_lower: Names from dict2 that are either missing in dict1 or have
#               a lower 'false_and_none_to_true' value than in dict1.

#     """
#     # List to store names of objects that have become unobservable
#     set_pco_true_missing_or_zero = set()
#     # List to store names of newly observed objects
#     set_new_pco_true = set()
#     # List to store names of objects with increased "False":"True" observations
#     set_consistency_lower = set()
#     # List to store names of objects with decreased "False":"True" observations
#     set_consistency_higher = set()

#     for name, values in old_obj_pcos_dict.items():
#         if values.get('true', 0) >= 1:
#             # Find all objects in olf PCOs dict with at least one "True" observation
#             if name not in new_obj_pcos_dict:
#                 # If the object is not found in new PCOs dict, add it to set_pco_true_missing_or_zero
#                 set_pco_true_missing_or_zero.add(name)
#             if new_obj_pcos_dict.get(name, {}).get('true', 0) < 1:
#                 # If the 'true' count for the object in the new PCOs dict is 0, add it to the list.
#                 set_pco_true_missing_or_zero.add(name)
            
#             if name not in new_obj_pcos_dict:
#                 # If the object is not found in the new PCOs dict, also add it to list_consistency_lower
#                 # Note: Objects that only appear in old PCOs dict will be listed in both 'set_pco_true_missing_or_zero' and 'list_consistency_lower'
#                 set_consistency_lower.add(name)

#             else:
#                 new_false_none_and_error_to_true = values.get('false_none_and_error_to_true', 0)
#                 old_false_none_and_error_to_true = old_obj_pcos_dict[name].get('false_none_and_error_to_true', 0)
#                 if old_false_none_and_error_to_true is None:
#                     # If object is not observable in old EWM (and is observable in old EWM), skip next checks
#                     pass
#                 elif new_false_none_and_error_to_true is None:
#                     # If object is not observable in new EWM, add it to list.
#                     set_consistency_lower.add(name)
#                 elif old_false_none_and_error_to_true < new_false_none_and_error_to_true:
#                     # If increased ratio of "False":"True" observations, add it to list
#                     set_consistency_lower.add(name)
    
#     for name, values in new_obj_pcos_dict.items():
#         if values.get('true', 0) >= 1:
#             # Find all objects in new PCOs dict with at least one "True" observation
#             if name not in old_obj_pcos_dict:
#                 # If the object is not found in old PCOs dict, acc it to list_new_pco_true
#                 set_new_pco_true.add(name)
#             if old_obj_pcos_dict.get(name, {}).get('true', 0) < 1:
#                 # If the 'true' count for the object in the old PCOs dict is 0, add it to the list.
#                 set_new_pco_true.add(name)
            
#             if name not in old_obj_pcos_dict:
#                 # If the object is not found in the old PCOs dict, also add it to list_consistency_higher
#                 # Note: Objects that only appear in new PCOs dict will be listed in both 'list_new_pco_true' and 'list_consistency_higher'
#                 set_consistency_higher.add(name)

#             else:
#                 new_false_none_and_error_to_true = values.get('false_none_and_error_to_true', 0)
#                 old_false_none_and_error_to_true = old_obj_pcos_dict[name].get('false_none_and_error_to_true', 0)
#                 if new_false_none_and_error_to_true is None:
#                     # If object is not observable in new EWM (and is observable in old EWM), skip next checks
#                     pass
#                 elif old_false_none_and_error_to_true is None:
#                     # If object is not observable in old EWM, add it to list.
#                     set_consistency_higher.add(name)
#                 elif old_false_none_and_error_to_true > new_false_none_and_error_to_true:
#                     # If decreased ratio of "False":"True" observations, add it to list
#                     set_consistency_higher.add(name)

#     return set_pco_true_missing_or_zero, set_new_pco_true, set_consistency_lower, set_consistency_higher


def count_true_obs(obs_dicts): 
    """Counts number of 'True' (physically consistent) observations in the observation JSON of an EWM"""
    true_obs_count = 0   
    for obs_dict in obs_dicts:
        if 'pco_bool' in obs_dict and obs_dict['pco_bool'] == True:
            true_obs_count += 1
    return true_obs_count

def score_ewm_organization(ann_obj_dicts):
    """Adds scores of all definitions in an EWM, plus the scores of code outside the found definitions (pruned score)"""
    # Initialize variable to sum definition lengths
    total_adjusted_definition_length = 0

    for fq_name, obj_info in ann_obj_dicts.items():
        if fq_name == PRUNED_SCORE_PLACEHOLDER:
            # Add the scores of the additional nodes outside the scored definitions (e.g. 'print()' statements)
            total_adjusted_definition_length += ann_obj_dicts[PRUNED_SCORE_PLACEHOLDER]

        # Note: newly added definitions are also scored. 
            # This should not be a problem because selection for consistency overrides selection for organization.
            # I.e. Total organization score may worsen when learning new things but that won't prevent learning new things.
            # Leaving 'added' definitions in scoring helps punish new definitions that fail to increase consistency. 
        
        elif fq_name == DEFINED_OBJECTS_PLACEHOLDER:
            continue
        elif obj_info['type'] == 'attribute':
            #  Don't count attribute scores in total because they are already included in instance static definitions (they're duplicated)
            continue
        elif 'static_definition' in obj_info:
            # Score all remaining static definitions recorded in the annotated object dict
            for statement in obj_info['static_definition']:
                total_adjusted_definition_length += statement['adjusted_length']

    return total_adjusted_definition_length

"""
HELPER FUNCTIONS
"""
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MappingProxyType):
            # Convert mappingproxy to a regular dictionary
            return dict(obj)
        elif isinstance(obj, GetSetDescriptorType):
            # Return a string representation or skip it
            return str(obj)
        elif hasattr(obj, '__dict__'):
            class_name = obj.__class__.__name__
            attributes = {}
            for key, value in obj.__dict__.items():
                # Skip methods and descriptors
                if callable(value) or isinstance(value, (
                    types.FunctionType, types.MethodType, types.BuiltinFunctionType,
                    types.BuiltinMethodType, types.GetSetDescriptorType, types.MemberDescriptorType
                )):
                    continue
                try:
                    attributes[key] = self.default(value)
                except TypeError:
                    # Optionally, include a string representation
                    attributes[key] = str(value)
            return {class_name: attributes}
        elif isinstance(obj, (list, tuple, set)):
            return [self.default(item) for item in obj]
        elif isinstance(obj, dict):
            return {self.default(key): self.default(value) for key, value in obj.items()}
        else:
            # Fallback to the superclass default method
            try:
                return super().default(obj)
            except TypeError:
                # Return string representation if not serializable
                return str(obj)

def save_results(json_path, data):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    # Write the results to the file, overwriting if it already exists
    with open(json_path, 'w') as f:
        json.dump(
            data,
            f,
            indent=2,
            cls=CustomJSONEncoder
        )

def load_json(file_path: str) -> List[Dict[Any, Any]]:
    with open(file_path, 'r') as file:
        return json.load(file)
    
def import_module_from_path(module_name, module_path):
    # Ensure the module path exists
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"No file found at {module_path}")

    # If module_path is a directory, construct the full path to the .py file
    if os.path.isdir(module_path):
        module_path = os.path.join(module_path, f"{module_name}.py")

    # Create the spec
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module '{module_name}' at {module_path}")

    # Create the module
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Execute the module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Error during import of module '{module_name}': {e}")

    return module

def set_logging(verbose_bool, log_dir, module_name):
    """Creates and configures a logger for a specific module with its own log file"""
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_filename = f'{timestamp}_{module_name}.log'
    log_filepath = os.path.join(log_dir, log_filename)

    # Get a named logger for this module
    logger = logging.getLogger(module_name)
    
    # Only configure if not already configured
    if not logger.handlers:
        file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        if verbose_bool:
            logger.setLevel(logging.DEBUG)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            console_handler.setLevel(logging.DEBUG)
            logger.addHandler(console_handler)
        else:
            logger.setLevel(logging.INFO)
            
    return logger

"""
MAIN FUNCTION
"""
def main_function(
    last_ewm_name, 
    next_ewm_name,                 
    ann_obj_dir,
    obs_dir,
    conj_dir,
    log_dir,
    verbose_bool):
                
    # Generate file paths
    last_annotated_obj_json_path = f"{ann_obj_dir}/{last_ewm_name}_ann_obj.json"
    next_annotated_obj_json_path = f"{ann_obj_dir}/{next_ewm_name}_ann_obj.json"
    last_obs_json_path = f"{ann_obj_dir}/{last_ewm_name}_obs.json"
    next_obs_json_path = f"{ann_obj_dir}/{next_ewm_name}_obs.json"

    # Set default values for evaluation
    consistency_decreased = None
    consistency_increased = None
    organization_improved = None
    accept_update = None

    # Configure logging
    global evaluate_conjecture_module_logger
    evaluate_conjecture_module_logger = set_logging(verbose_bool, log_dir, 'evaluate_conjecture_module')

    # Ensure observation JSON and annotated object JSONs exist; if not, end:
    if not os.path.exists(last_annotated_obj_json_path):
        evaluate_conjecture_module_logger.error(f"last_annotated_obj_json not at path {last_annotated_obj_json_path}.")
        print(f"Evaluate Conjecture Module Error: last_annotated_obj_json not at path {last_annotated_obj_json_path}.")
        sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.
    if not os.path.exists(next_annotated_obj_json_path):
        evaluate_conjecture_module_logger.error(f"next_annotated_obj_json not at path {next_annotated_obj_json_path}.")
        print(f"Evaluate Conjecture Module Error: next_annotated_obj_json not at path {next_annotated_obj_json_path}.")
        sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.
    if not os.path.exists(last_obs_json_path):
        evaluate_conjecture_module_logger.error(f"last_obs_json_path not at path {last_obs_json_path}.")
        print(f"Evaluate Conjecture Module Error: last_obs_json_path not at path {last_obs_json_path}.")
        sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.
    if not os.path.exists(next_obs_json_path):
        evaluate_conjecture_module_logger.error(f"next_obs_json_path not at path {next_obs_json_path}.")
        print(f"Evaluate Conjecture Module Error: next_obs_json_path not at path {next_obs_json_path}.")
        sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.

    # Load data
    try:
        old_annotated_obj_dict= load_json(last_annotated_obj_json_path)
        new_annotated_obj_dict= load_json(next_annotated_obj_json_path)
        old_obs_dicts= load_json(last_obs_json_path)
        new_obs_dicts= load_json(next_obs_json_path)

    except () as e:
        evaluate_conjecture_module_logger.error(f"Error: {e}")
        print(f"Evaluate Conjecture Module Error: unable to load obs or ann_obj data.")
        sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.

    """
    Compare the true and false observation data of the old and new EWMs
    """
    # Count number of 'True' observations in old obs data and new obs data
    old_ewm_consistent_obs = count_true_obs(old_obs_dicts)
    evaluate_conjecture_module_logger.info(f"Number of consistent obs in old EWM: {old_ewm_consistent_obs}")
    new_ewm_consistent_obs = count_true_obs(new_obs_dicts)
    evaluate_conjecture_module_logger.info(f"Number of consistent obs in new EWM: {new_ewm_consistent_obs}")

    # Update consistency evaluation bools:
    if new_ewm_consistent_obs>old_ewm_consistent_obs:
        consistency_decreased = False
        consistency_increased = True
    elif new_ewm_consistent_obs<old_ewm_consistent_obs:
        consistency_decreased = True
        consistency_increased = False
    elif new_ewm_consistent_obs==old_ewm_consistent_obs:
        consistency_decreased = False
        consistency_increased = False

    # Check if organization improved
        # Initialize variables to sum 'adjusted_definition_length' total scores
    old_total_adjusted_definition_length = score_ewm_organization(old_annotated_obj_dict)
    evaluate_conjecture_module_logger.info(f"Total adjusted definition length of old EWM: {old_total_adjusted_definition_length}")
    new_total_adjusted_definition_length = score_ewm_organization(new_annotated_obj_dict)
    evaluate_conjecture_module_logger.info(f"Total adjusted definition length of new EWM: {new_total_adjusted_definition_length}")

    if old_total_adjusted_definition_length > new_total_adjusted_definition_length:
        organization_improved = True
    else:
        organization_improved = False

    """
    Decide to accept or reject update
    """
    # Generate 'accept_update' bool
    conjecture_evaluation_summary = ""
    if not consistency_decreased:
        if consistency_increased or organization_improved:
            accept_update = True
        else:
            accept_update = False
    else:
        accept_update = False  # Always reject update if consistency decreases

    # Generate evaluation summary text
    if accept_update:
        conjecture_evaluation_summary += f"CONJECTURE ACCEPTED:\n"
    else:
        conjecture_evaluation_summary +=f"CONJECTURE REJECTED:\n"

    conjecture_evaluation_summary +=f"    consistency_decreased:      {consistency_decreased}\n\
    consistency_increased:      {consistency_increased}\n\
    organization_improved:      {organization_improved}\n"
    
    print(conjecture_evaluation_summary)

    # Update conjecture dictionary
    conjecture_evaluation_summary_dict = {'consistency_decreased':consistency_decreased,
                                          'consistency_increased':consistency_increased,
                                          'organization_increased':organization_improved}
    conjecture_evaluation_data_dict = {'last_ewm_num_pco_true':old_ewm_consistent_obs,
                                       'next_ewm_num_pco_true':new_ewm_consistent_obs,
                                       'last_ewm_organization_score':old_total_adjusted_definition_length,
                                       'next_ewm_organization_score':new_total_adjusted_definition_length}
    
    """
    Update the conj_dicts of 'last_ewm'
    """
    conj_dicts_path = os.path.join(conj_dir, f"{last_ewm_name}_conj_dicts.json")
    evaluate_conjecture_module_logger.info(f"Updating conj_dicts of 'last_ewm' ('{last_ewm_name}') at: {conj_dicts_path}")

    # Load conj_dicts data
    try:
        conj_dicts= load_json(conj_dicts_path)  # conj_dicts should already be assigned
        evaluate_conjecture_module_logger.info(f"Loaded conj_dicts: \n{json.dumps(conj_dicts,indent=2)}")
    except () as e:
        evaluate_conjecture_module_logger.error(f"Unable to find conj_dicts of last ewm at: {conj_dicts_path}\nUnable to update it.")
        evaluate_conjecture_module_logger.error(f"Error: {e}") 
        sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.

    try: 
        # Get the ids of the latest question series and latest update attempt for last_ewm
        latest_q_series_id = max(series['id'] for series in conj_dicts['question_series'])
        evaluate_conjecture_module_logger.info(f"Found latest_q_series_id: {latest_q_series_id}")
        for question_series in conj_dicts['question_series']:
            if question_series['id'] == latest_q_series_id:
                latest_q_series_dict = question_series
        latest_update_id = max(update['id'] for update in latest_q_series_dict['update_attempts'])
        evaluate_conjecture_module_logger.info(f"Found latest_update_id: {latest_update_id}")
        for update_dict in latest_q_series_dict['update_attempts']:
            if update_dict['id'] == latest_update_id:
                latest_update_dict = update_dict    
        latest_update_dict['conj_eval_result']={}
        latest_update_dict['conj_eval_result']['accepted_bool']=accept_update
        latest_update_dict['conj_eval_result']['summary']=conjecture_evaluation_summary_dict
        latest_update_dict['conj_eval_result']['eval_data']=conjecture_evaluation_data_dict
                
    except () as e: 
        evaluate_conjecture_module_logger.error(f"Missing 'question_series' data in 'conj_dicts' of next EWM: {last_ewm_name}. A conjecture EWM has been created: {next_ewm_name}.")
        evaluate_conjecture_module_logger.error(f"Error: {e}")

    # Save updated conj_dicts data
    save_results(conj_dicts_path, conj_dicts)
    evaluate_conjecture_module_logger.info(f"Updated conjecture dictionary of 'question_series' 'id' {latest_q_series_dict}, 'update_attempts' 'id' {latest_update_id} with 'conj_evaluation_result'.\n Saved at: {conj_dicts_path}")

    return accept_update, conjecture_evaluation_summary_dict

"""
    Conjecture dictionary format, for reference:
    conj_dicts = {'conj_origin':{
                    'last_EWM':None, 
                    'seed_prompt':seed_prompt, 
                    'conjecturer':conj_llm_model},
                'question_series':{
                    'id':1,
                    'question_path':None,
                    'problem_obj':None,
                    'prompt_series':None,
                    'tokens':{
                        'prompt_tokens':[],
                        'completion_tokens':[],
                        'total_tokens':[]},
                    'update_attempts':{
                        'id':int,
                        'update_dict':{},
                        'timestamp':None,
                        'compile_check':{
                            'result':bool, 
                            'message':str}}
                        'conj_eval_result':{
                            'accepted_bool':bool, 
                            'summary': {'consistency_decreased': bool,
                                        'consistency_increased': bool,
                                        'organization_increased': bool},
                            'eval_data: {'last_ewm_num_pco_true': int,
                                            'next_ewm_num_pco_true': int,
                                            'last_ewm_organization_score': float,
                                            'next_ewm_organization_score': float}
                            }
                        }

                }   
""" 

if __name__ == "__main__":
    # Previous EWM data:
    test_old_ewm_name = "test_prediction_environment_09_test_02"
    test_old_ewm_path = f".../graph_visualization/test_ewms/{test_old_ewm_name}.py"
    test_old_ewm_obs_json = f".../generator_functions/predictions_and_observations/{test_old_ewm_name}.json"

    # New EWM data:
    test_new_ewm_name ="test_prediction_environment_10_test_02"
    test_new_ewm_path =f".../graph_visualization/test_ewms/{test_new_ewm_name}.py"
    test_new_ewm_obs_json = f".../generator_functions/predictions_and_observations/{test_new_ewm_name}.json"

    # Conjecture assessment JSON paths:
    old_annotated_obj_json_path = f".../generator_functions/predictions_and_observations/{test_old_ewm_name} - ann_obj_dict.json"
    new_annotated_obj_json_path = f".../generator_functions/predictions_and_observations/{test_new_ewm_name} - ann_obj_dict.json"

    # Observation graph dictionary paths:
    old_obs_graph_dict_path = f".../generator_functions/predictions_and_observations/{test_old_ewm_name} - graph_dict.json"
    new_obs_graph_dict_path = f".../generator_functions/predictions_and_observations/{test_new_ewm_name} - graph_dict.json"

    main_function(last_ewm_name=test_old_ewm_name, 
                  old_ewm_path=test_old_ewm_path, 
                  old_obs_json_path=test_old_ewm_obs_json, 
                  old_annotated_obj_json_path=old_annotated_obj_json_path,
                  old_obs_graph_dict_path=old_obs_graph_dict_path,
                  next_ewm_name=test_new_ewm_name, 
                  new_ewm_path=test_new_ewm_path, 
                  new_obs_json_path=test_new_ewm_obs_json, 
                  new_annotated_obj_json_path=new_annotated_obj_json_path, 
                  new_obs_graph_dict_path=new_obs_graph_dict_path, 
                  verbose_bool=False)


"""
Additional Resources:
    - Graph Neural Networks
        - Stanford lecture series: https://www.youtube.com/watch?v=lRCDpfJoMiE&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=34
        - https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
        - https://www.youtube.com/watch?v=4dVwlE9jYxY&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=5
        - GraphGym (to test out what analysis method suites your data)
            - https://pytorch-geometric.readthedocs.io/en/latest/advanced/graphgym.html
            - https://github.com/snap-stanford/GraphGym
        - 'Survey on hypergraph neural networks'(2024): https://arxiv.org/pdf/2404.01039
        - 'Hypergraph Neural Networks' (2019): https://arxiv.org/pdf/1809.09401 

"""