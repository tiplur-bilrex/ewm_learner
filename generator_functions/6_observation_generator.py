"""
observation_generator.py

- Observe all predictions by giving their observable prediction statements to an observer (LLM).
    - Ask observer to provide a boolean observation ("is the prediction consistent or inconsistent with your knowledge?"
"""
import json
from huggingface_hub import InferenceClient
from huggingface_hub.utils._errors import HfHubHTTPError
from datetime import datetime
import logging
import os
import types
from types import MappingProxyType, GetSetDescriptorType

observation_generator_module_logger = None

def generate_pco_value(physically_consistent_observation):
    """
    Generate "physically consistent observation" (PCO) bool values from LLM observations.
    - read the first word in each 'physically_consistent_observation' and convert "True", "False", or "None" into bools. 
    """
    # Remove any surrounding quotation marks and whitespace, then convert to lowercase
    pco = physically_consistent_observation.strip().strip("'\"").lower()

    # Split by newline and take the first non-empty line
    pco_lines = [line.strip() for line in pco.split('\n') if line.strip()]
    if pco_lines:
        pco = pco_lines[0]
        
    # Check if the string starts with 'true', 'false', or 'none'
    if pco.startswith('true'):
        return True
    elif pco.startswith('false'):
        return False
    # Note that this software does not check for 'pco.startswith('none')'. 
    # It assumes that any failure to state 'true' or 'false' is equivalent to not knowing the answer.
    else:
        return None


"""
List predictions according to observation bool value
"""
# Note: This function is not used by main_function.py
def list_predictions_by_pco_bool(json_data):
    def categorize_dictionaries(json_data):
        true_list = []
        false_list = []
        none_list = []
        no_key_list = []

        for item in json_data:
            if 'pco_bool' in item:
                if item['pco_bool'] is True:
                    true_list.append(item)
                elif item['pco_bool'] is False:
                    false_list.append(item)
                elif item['pco_bool'] is None:
                    none_list.append(item)
            else:
                no_key_list.append(item)

        return true_list, false_list, none_list, no_key_list


    # Call the function
    true_list, false_list, none_list, no_key_list = categorize_dictionaries(json_data)

    # Print the result summary
    print(f"Number of dictionaries with pco_bool = True: {len(true_list)}")
    print(f"Number of dictionaries with pco_bool = False: {len(false_list)}")
    print(f"Number of dictionaries with pco_bool = None: {len(none_list)}")
    print(f"Number of dictionaries without pco_bool key: {len(no_key_list)}\n")
    # Log the result summary
    observation_generator_module_logger.info(f"Number of dictionaries with pco_bool = True: {len(true_list)}")
    observation_generator_module_logger.info(f"Number of dictionaries with pco_bool = False: {len(false_list)}")
    observation_generator_module_logger.info(f"Number of dictionaries with pco_bool = None: {len(none_list)}")
    observation_generator_module_logger.info(f"Number of dictionaries without pco_bool key: {len(no_key_list)}")

"""
Helper functions
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

"""
MAIN
"""
def main_function(
    obs_api_key, 
    ewm_name, 
    obs_json_dir, 
    log_dir,
    obs_llm_name="mistralai/Mistral-7B-Instruct-v0.3",
    verbose=False):
    
    # Configure logging
    global observation_generator_module_logger
    observation_generator_module_logger = set_logging(verbose, log_dir, 'observation_generator_module')

    # Generate the obs_json_path
    obs_filename = f"{ewm_name}_obs.json"
    obs_json_path = f"{obs_json_dir}/{obs_filename}"

    # Read the JSON file
    with open(obs_json_path, 'r') as file:
        ewm_obs_dicts = json.load(file)

    # Initialize inference client
    client = InferenceClient(api_key= obs_api_key)

    # Process only a subset of prompts from the JSON file
    observation_generator_module_logger.info(f"Beginning to generate observations using: {obs_json_path}")
    for item in ewm_obs_dicts:
        # Check if 'prompt' key exists in the dictionary
        if 'prompt' not in item:
            observation_generator_module_logger.info(f"Skipping observation of item with no prompt key: {item}")
            continue

        elif 'pco_bool' in item:
            observation_generator_module_logger.info(f"Prediction already observed as {item['pco_bool']}. \nPrompt was: {item['prompt']}\n")
            continue
        
        prompt = item.get('prompt')
        
        # Convert the prompt to the required message format
        messages = [{"role": "user", "content": prompt}]
        
        observation_generator_module_logger.info(f"Requesting observation for: {prompt}")

        try: 
            # Make the API call
            completion = client.chat.completions.create(
                model= obs_llm_name, 
                messages=messages, 
                temperature=0.5,
                max_tokens=2048,
                top_p=0.7)
        
            # Get text of the response
            response=completion.choices[0].message.content

            # Update the JSON data item with the API's response and observation context
            item['physically_consistent_observation'] = response.strip()
            observation_generator_module_logger.info(f"{obs_llm_name} response: '{item['physically_consistent_observation']}'")
            print(f"{obs_llm_name} response: '{item['physically_consistent_observation']}'")

            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            item['observer_software'] = obs_llm_name
            item['observer_host'] = "huggingface_hub, InferenceClient"
            item['observation_time'] = formatted_time

            pco_bool = generate_pco_value(item['physically_consistent_observation'])
            item['pco_bool'] = pco_bool

        except HfHubHTTPError as e:
            observation_generator_module_logger.warning(f"Error occurred while processing prompt: {prompt}")
            observation_generator_module_logger.warning(f"Error details: {str(e)}")
            # item['error'] = str(e)
            break
        except Exception as e:
            observation_generator_module_logger.warning(f"Unexpected error occurred while processing prompt: {prompt}")
            observation_generator_module_logger.warning(f"Error details: {str(e)}")
            # item['error'] = str(e)
            break

        # Save the JSON file after each item is processed
        with open(obs_json_path, 'w') as file:
            json.dump(ewm_obs_dicts, file, indent=2)

    print(f"Finished generating observations. Updated the JSON file: {obs_json_path}\n")

    # Print summary of observations (now with observation data)
    list_predictions_by_pco_bool(ewm_obs_dicts) 

    # Save 'ewm_obs_dicts' with 'all_objects' info added
    save_results(obs_json_path, ewm_obs_dicts)
    observation_generator_module_logger.info(f"ewm_obs_dicts (updated with 'all_objects' info) saved at: {obs_json_path}")

    return ewm_obs_dicts

if __name__ == "__main__":
    # obs_llm_name="mistralai/Mistral-7B-Instruct-v0.3"
    obs_llm_name= 'meta-llama/Llama-3.1-8B-Instruct'
    obs_api_key=''
    log_dir = '.../generator_functions/log_files'
    # This testing block will only run if the script is executed directly
    test_ewm_name = "p_generator_test"
    obs_json_dir = '.../generator_functions/obs_ann_conj_files'
    
    # Initialize inference client
    client = InferenceClient(api_key=obs_api_key)
    messages = [{"role": "user", "content": "Write a sentence about woodworking."}]

    try: 
        completion = client.chat.completions.create(
            model= obs_llm_name, 
            messages=messages, 
            temperature=0.5,
            max_tokens=2048,
            top_p=0.7)

    except HfHubHTTPError as e:
        observation_generator_module_logger.warning(f"Error occurred while processing prompt")
        observation_generator_module_logger.warning(f"Error details: {str(e)}")
        # item['error'] = str(e)
        
    except Exception as e:
        observation_generator_module_logger.warning(f"Unexpected error occurred while processing prompt")
        observation_generator_module_logger.warning(f"Error details: {str(e)}")
        # item['error'] = str(e)
        
    print(completion.choices[0].message.content)