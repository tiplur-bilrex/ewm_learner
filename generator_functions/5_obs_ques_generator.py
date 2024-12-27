"""
obs_ques_generator.py

Notes:
- convert observation data to natural language because the NL patterns may help the LLM match questions to knowledge of the physical world (which it learned via NL).

Possible alternate techniques:
- can generate subtrees of EWM and observe those (DreamCoder paper Fig2 Sleep: Abstracton - https://arxiv.org/pdf/2006.08381)
- can generate multiple statements for each question and select from statements that are most similar (most likely correct) - mentioned somewhere in AlphaCode paper (https://www.science.org/doi/10.1126/science.abq1158) 
"""
import json
from datetime import datetime
import logging
import os

obs_question_generator_module_logger = None

DEFAULT_VALUE_PLACEHOLDER = 'l8jNWV52p34HjK'
# natural_language_prompt_context = "Below is the output of a python expression converted to natural language:\n\n\""
natural_language_prompt_request = "\"\n\nAssuming that the above hypothetical statement is referring to real things or concepts, expressed using Python syntax, does it seem correct or is there a factual error in the statement? Answer 'True' when you think output is correct, 'None' when you do not know if the output is correct, and 'False' when the output is incorrect.\nANSWER ONLY: 'True' , 'None', or 'False'."


def configure_logging_by_call(verbose, log_dir):
    global obs_question_generator_module_logger
    obs_question_generator_module_logger = set_logging(verbose, log_dir, module_name="obs_question_generator_module")

def process_json(ewm_name, obs_json_dir, ann_obj_dir):
    
    # Generate the obs_json_path and ann_obj_json_path
    obs_filename = f"{ewm_name}_obs.json"
    obs_json_path = f"{obs_json_dir}/{obs_filename}"
    ann_obj_filename = f"{ewm_name}_ann_obj.json"
    ann_obj_json_path = f"{ann_obj_dir}/{ann_obj_filename}"

    # Load data from JSONs
    with open(obs_json_path, 'r') as f1, open(ann_obj_json_path, 'r') as f2:
        obs_json_data = json.load(f1)
        ann_obj_json_data = json.load(f2)

    # Count how many dictionaries have the 'computation_error' key
    num_failed_computations_in_json = sum('computation_error' in d for d in obs_json_data)
    num_computations_in_json = sum('result' in d for d in obs_json_data)
    num_computations_with_prompt = sum('prompt' in d for d in obs_json_data)

    # List of observation keys
    observation_keys = ["prompt", "observer_context", "physically_consistent_observation", "pco_bool"]

    # Convert each prediction dictionary to a natural language statement, generate prompt with NL statement, then merge prompt with the original data
    for i, d in enumerate(obs_json_data):
        if 'prompt' in d:
            continue
        elif any(obs_key in d for obs_key in observation_keys):
            print(f"Warning: Some but not all observation keys present. Something went wrong. Dictionary: \n{d}")
        elif 'result' in d:
            statement = dict_to_statement(d, ann_obj_json_data)
            if statement is not None:
                obs_json_data[i]['prompt'] = statement + natural_language_prompt_request
            else: print(f"Warning: problem generating prompt statement from computation dictionary.")
    
    num_prompts_added = (sum('prompt' in d for d in obs_json_data)) - num_computations_with_prompt

    # Save the updated data as a new JSON file
    with open(obs_json_path, 'w') as f:
        json.dump(obs_json_data, f, indent=2)

    print(f"{num_computations_with_prompt} existing prompts found. {num_prompts_added} new prompts added. Total of {num_computations_in_json} successful computations and {num_failed_computations_in_json} failed computations. JSON file has been saved at: {obs_json_path}\n")
          
    return obs_json_data

def convert_string_to_tuple(string_rep):
    try:
        # Remove outer parentheses and split by comma
        # The string format is like "('venus_planet', 'Planet')"
        clean_string = string_rep.strip('()').split(', ')
        
        # Extract name and class, removing quotes
        name = clean_string[0].strip("'")
        class_name = clean_string[1].strip("'")
        
        return (name, class_name)
        
    except Exception as e:
        print(f"Error converting string to tuple: {e}")
        return None

## Below function processes string tuples with full class names, including module name. 
# def convert_string_to_tuple(string_rep):
#     try:
#         # Extract the first part ('earth_planet.radius') and the class part
#         # This assumes that the string is formatted like the example provided
#         arg_name_part, class_part = string_rep.split(', ', 1)
        
#         # Remove the outer parentheses and the surrounding quotes
#         arg_name_part = arg_name_part.strip("('")
#         class_part = class_part.strip(")")

#         # Extract module and class from the class part
#         if class_part.startswith("<class '") and class_part.endswith("'>"):
#             class_path = class_part[8:-2]  # Strip the <class ''> parts

#             # Split the string at the first period (to remove the file name)
#             class_path_parts = class_path.split('.', 1)
#             class_name = class_path_parts[1] if len(class_path_parts) > 1 else class_path
            
#             # Return the tuple ('earth_planet.radius', ClassObject)
#             return (arg_name_part, class_name)
        
#         # If the string doesn't match the expected pattern
#         raise ValueError("Unexpected format")
    
#     except (ValueError, SyntaxError, ImportError, AttributeError) as e:
#         print(f"Error converting string to tuple: {e}")
#         return None

def dict_to_statement(d, ann_obj_json_data):

    # Return None when there is no result (i.e. the computation failed)
    if not 'result' in d:
        return None

    # Initialize statement string
    statement = ''

    string_tuple_args = d['instance_name(s)']['args']
    string_tuple_kwargs = d['instance_name(s)']['kwargs']

    arg_name_class_tuples = []
    arg_initialized_defs = []
    if string_tuple_args:
        for string_tuple_arg in string_tuple_args:
            # Remove any default value placeholders stored in the instances when forming question.
            if DEFAULT_VALUE_PLACEHOLDER in string_tuple_arg:
                continue
            arg_tuple = convert_string_to_tuple(string_tuple_arg)
            arg_name_class_tuples.append(arg_tuple)
            if arg_tuple[0] in ann_obj_json_data:
                if 'initialized_definition' in ann_obj_json_data[arg_tuple[0]]:
                    if ann_obj_json_data[arg_tuple[0]]['initialized_definition']:
                        arg_initialized_defs.append(ann_obj_json_data[arg_tuple[0]]['initialized_definition'])
                    else:
                        arg_initialized_defs.append(None)
                else:
                    arg_initialized_defs.append(None)
            else:
                arg_initialized_defs.append(None)

    kwarg_key_name_class = {}
    if string_tuple_kwargs:
        for key, value in string_tuple_kwargs.items():
            kwarg_key_name_class[key]=[]
            for string_tuple_arg in value:
                # Remove any default value placeholders stored in the instances when forming question.
                if DEFAULT_VALUE_PLACEHOLDER in string_tuple_arg:
                    continue
                arg_tuple = convert_string_to_tuple(string_tuple_arg)
                kwarg_key_name_class[key].append(arg_tuple)

    if d["class_if_method"]:
        function_name=f"{d['class_if_method']}.{d['function_name']}"
        statement += f"**Method:** `{function_name}`\n**Input Parameters:**\n  **Args:**\n"
    else:
        function_name=f"{d['function_name']}"
        statement += f"**Function:** `{function_name}`\n**Input Parameters:**\n  **Args:**\n"

    # else:
    #     obs_question_generator_module_logger.error(f"Unexpected behavior. Observation funciton is neither method or function: {json.dumps(d)}")
    #     print(f"Unexpected behavior. Observation funciton is neither method or function: {json.dumps(d)}")
    #     return None

    if arg_name_class_tuples:
        for i, (arg_name, arg_class) in enumerate(arg_name_class_tuples):
            statement += f"    {i+1}. `{arg_name}` of class `{arg_class}`\n"
            if arg_name in ann_obj_json_data:
                if 'initialized_definition' in ann_obj_json_data[arg_name]:
                    if ann_obj_json_data[arg_name]['initialized_definition']:
                        statement += f"    Initialised Definition:{ann_obj_json_data[arg_name]['initialized_definition']}\n"
    else:
        statement += "  None"

    if kwarg_key_name_class:
        statement += "  **Kwargs:**\n"
        for key, values in kwarg_key_name_class.items():
            statement += f"    Key: `{key}`\n"
            for i, (arg_name, arg_class) in enumerate(values):
                statement += f"      {i+1}. `{arg_name}` of class `{arg_class}`\n"
                if arg_name in ann_obj_json_data:
                    if 'initialized_definition' in ann_obj_json_data[arg_name]:
                        if ann_obj_json_data[arg_name]['initialized_definition']:
                            statement += f"      Initialised Definition:{ann_obj_json_data[arg_name]['initialized_definition']}\n"
  
    statement += f"**Result:**\n  {d['result']}"
    
    return statement

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

# def set_logging(verbose_bool, log_dir):
#     # when 'verbose_bool' is True, debug logging is collected, printed, and saved. When 'False', only info logging is saved, no logging is printed.
#     """
#     For reference, here are the logging message types:
#         logging.debug('Debug message: Useful for diagnostic purposes')
#         logging.info('Info message: Confirmation that things are working as expected')
#         logging.warning('Warning message: Something unexpected happened')
#         logging.error('Error message: A serious problem occurred')
#         logging.critical('Critical message: A very serious issue occurred')
#     """
#     # Create timestamp for unique filename
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
#     # Construct the log filename with timestamp
#     log_filename = f'{timestamp}_o_question_generator.log'
#     log_filepath = os.path.join(log_dir, log_filename)

#     # Create a file handler with the timestamped filename
#     logging_file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
#     logging_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

#     # Get the root logger and remove any existing handlers
#     logger = logging.getLogger()
#     logger.handlers = []  # Clear existing handlers to avoid duplicate logging

#     # Add the file handler (this will always log to the file)
#     logger.addHandler(logging_file_handler)

#     # Check if verbose mode is enabled
#     if verbose_bool:
#         # Set the logging level to DEBUG for verbose mode
#         logger.setLevel(logging.DEBUG)
        
#         # Create a console handler for printing logs to the console
#         console_handler = logging.StreamHandler()
#         console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
#         # Add the console handler (for printing logs to the console)
#         logger.addHandler(console_handler)

#         # Log debug message indicating verbose mode is enabled
#         logging.debug("Verbose mode is enabled. Debugging info will be shown.")
#     else:
#         # Set the logging level to INFO (only log to file)
#         logger.setLevel(logging.INFO)


if __name__ == "__main__":
    # # This testing block will only run if the script is executed directly
    # test_json_path = '.../generator_functions/predictions_and_observations/test_prediction_environment_09_test_02.json'
    # result = process_json(test_json_path)
    # print(result)

    # Generate the obs_json_path and ann_obj_json_path
    ann_obj_json_path = ".../generator_functions/obs_ann_conj_files/test_prediction_environment_10 (copy)_ann_obj.json"

    test_dict_1 = {
    "method_or_function": "function",
    "function_name": "calculate_gravitational_force_in_newtons",
    "class_if_method": None,
    "instance_name(s)": {
      "args": [
        "('pool_ball.radius', <class 'test_prediction_environment_10 (copy).Radius'>)",
        "('venus_planet.mass', <class 'test_prediction_environment_10 (copy).Mass'>)"
      ],
      "kwargs": {}
    },
    "result": {
      "number": 1.000426008002462e+17,
      "unit": {
        "name": {
          "name": "Newton"
        },
        "symbols": [
          {
            "symbol": "m\u00b3 kg\u207b\u00b9 s\u207b\u00b2"
          },
          {
            "symbol": "N"
          }
        ]
      }
    }}

    test_dict_2 = {
    "method_or_function": "method",
    "function_name": "calculate_density",
    "class_if_method": "PhysicalObject",
    "instance_name(s)": {
      "args": [
        "('mars_planet', <class 'test_prediction_environment_10 (copy).Planet'>)"
      ],
      "kwargs": {"keyword_a": [
        "('gravitational_constant', <class 'test_prediction_environment_10 (copy).PhysicalConstant'>)",
        "('foot_unit', <class 'test_prediction_environment_10 (copy).Unit'>)"], "keyword_b": ["('gravitational_constant', <class 'test_prediction_environment_10 (copy).PhysicalConstant'>)"]}
    },
    "result": {
      "number": 3915.7370568499055,
      "unit": {
        "name": {
          "name": "kilogram per cubic meter"
        },
        "symbols": [
          {
            "symbol": "kg/m\u00b3"
          }
        ]
      }
    }}

    # Load data from JSONs
    with open(ann_obj_json_path, 'r') as f:
        ann_obj_json_data = json.load(f)

    statement_1 = dict_to_statement(test_dict_1, ann_obj_json_data)
    statement_2 = dict_to_statement(test_dict_2, ann_obj_json_data)

    print(statement_1+natural_language_prompt_request+"\n\n\n")
    print(statement_2+natural_language_prompt_request)