"""
conjecture_generator.py
- Selects a question path to direct updating of EWM (e.g. targeting a problem object, improving organization, adding new definitions)
- Generates question prompts and communicates with LLM via API
- Updates EWM with new conjecture and saves as new EWM

Notes about object scoring using observation data:
- find objects, functions, classes with highest false:true ratios
    - for classes, take a weighted (by total false) average of all the instances belonging to the class
- use objects with highest ratios or highest num False... May prioritize classes over instances (classes will always have less extreme false_true ratios than instances)

Try these three:
    - Weighted product:product of frequency and amplitude (ratio); can weight products by raising each to a different exponent
        * Multiplicative methods emphasize items that are high in both frequency and amplitude.
            p, q = 1, 1  # Equal influence
            scores = [f**p * a**q for f, a in zip(frequencies, amplitudes)]

    - Z-Score Standardization and Summation: 
            import statistics

            mean_f, std_f = statistics.mean(frequencies), statistics.stdev(frequencies)
            mean_a, std_a = statistics.mean(amplitudes), statistics.stdev(amplitudes)

            z_frequencies = [(f - mean_f) / std_f for f in frequencies]
            z_amplitudes = [(a - mean_a) / std_a for a in amplitudes]

            scores = [zf + za for zf, za in zip(z_frequencies, z_amplitudes)]
    
    - Sum of ranks:
            from scipy.stats import rankdata

            # Rank frequencies and amplitudes (rank 1 is highest value)
            rank_frequencies = rankdata([-f for f in frequencies], method='average')
            rank_amplitudes = rankdata([-a for a in amplitudes], method='average')

            combined_ranks = [rf + ra for rf, ra in zip(rank_frequencies, rank_amplitudes)]


    - Other options: weighted sum, geometric mean (lower impact of extreme values), 

Select the objects with the greatest number of dependent False observations.

"""

import ast
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from together import Together
import importlib.util
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import re
from scipy.stats import rankdata
import statistics
import sys
import tokenize
import types
import traceback
from types import MappingProxyType, GetSetDescriptorType, ModuleType
from typing import List, Dict, Any, Set, Tuple, Optional

conjecture_generator_module_logger = None
PRUNED_SCORE_PLACEHOLDER = 'N8cT89C044lqvJ'
DEFINED_OBJECTS_PLACEHOLDER = "vcM/juON9%1gv!lLp"
ANY_TYPE_PLACEHOLDER = "kHNMWb9b^1!gzcx"

"""
Score observation data and select problem object
    - Review previous conjecture attempts
    - Select next highest False scored object
"""
def score_obj_observations(annotated_object_dict):
    """
    Compute Weighted Product, Z-Score Standardization and Summation, and Sum of Ranks for objects in annotated_object_dict.
    Store these values in annotated_object_dict.

    """
    fq_names = []
    no_prediction_objs = []  # A list of defined objects with no predictions
    no_true_observation_objs = []  # A list of defined objects with predictions but no true observations
    false_none_error_totals = []
    false_none_and_error_to_true_ratios = []

    # Create lists of values for rank calculations
    for fq_name, obj_info in annotated_object_dict.items():
        if fq_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
            continue
        elif obj_info['pco_val_counts']['total'] == 0:
            obj_info['no_predictions'] = True
            no_prediction_objs.append([fq_name, obj_info['type'], 1]) # The '1' is used for priority scoring later
        elif obj_info['pco_val_counts']['false_none_and_error_to_true'] == float('inf'):
            obj_info['no_observations'] = True
            no_true_observation_objs.append([fq_name, obj_info['type'], 1])
        else:
            fq_names.append(fq_name)
            false_none_error_total = (obj_info['pco_val_counts']['false'] + 
                                      obj_info['pco_val_counts']['none'] + 
                                      obj_info['pco_val_counts']['error'])
            false_none_error_totals.append(false_none_error_total)
            obj_info['pco_val_counts']['false_none_error_total'] = false_none_error_total
            false_none_and_error_to_true_ratios.append(obj_info['pco_val_counts']['false_none_and_error_to_true'])

    mean_total, std_total = statistics.mean(false_none_error_totals), statistics.stdev(false_none_error_totals)
    mean_ratio, std_ratio = statistics.mean(false_none_and_error_to_true_ratios), statistics.stdev(false_none_and_error_to_true_ratios)

    # Sum of ranks:
    # Rank frequencies and amplitudes (rank 1 is highest value)
    rank_totals = rankdata([-f for f in false_none_error_totals], method='average')
    rank_ratios = rankdata([-a for a in false_none_and_error_to_true_ratios], method='average')
    combined_ranks = [rt + rr for rt, rr in zip(rank_totals, rank_ratios)]

    # Calculate rank for each fq_name
    for fq_name, false_none_error_total, false_none_and_error_to_true_ratio, combined_rank in zip(fq_names, false_none_error_totals, false_none_and_error_to_true_ratios, combined_ranks):
        # Weighted product: the product of frequency and amplitude (ratio); can weight products by raising each to a different exponent
        # Multiplicative methods emphasize items that are high in both frequency and amplitude.
        p, q = 1, 1  # Equal influence
        weighted_product = false_none_error_total**p * false_none_and_error_to_true_ratio**q

        # Z-Score Standardization and Summation: 
        z_frequency = (false_none_error_total - mean_total) / std_total
        z_amplitude = (false_none_and_error_to_true_ratio - mean_ratio) / std_ratio
        std_sum_score = z_frequency + z_amplitude
    
        # Update annotated_object_dict with scores 
        annotated_object_dict[fq_name]['pco_val_counts']['weighted_product']=weighted_product
        annotated_object_dict[fq_name]['pco_val_counts']['std_sum_score']=std_sum_score
        annotated_object_dict[fq_name]['pco_val_counts']['combined_rank']=combined_rank
    no_prediction_objs = []  # A list of defined objects with no predictions
    conjecture_generator_module_logger.info(f"annotated_object_dict updated with PCO value scores: 'weighted_product', 'std_sum_score', 'combined_rank'.")
    if no_prediction_objs:
        conjecture_generator_module_logger.info(f"'score_obj_observations' function returning 'no_prediction_objs':")
        for obj_name_type_tuple in no_prediction_objs:
            conjecture_generator_module_logger.info(f"   - {obj_name_type_tuple[0]} (type {obj_name_type_tuple[1]})")
    else:
        conjecture_generator_module_logger.info(f"'score_obj_observations' function returning empty 'no_prediction_objs' list.")
    if no_true_observation_objs:
        conjecture_generator_module_logger.info(f"score_obj_observations function returning 'no_true_observation_objs':")
        for obj_name_type_tuple in no_true_observation_objs:
            conjecture_generator_module_logger.info(f"   - {obj_name_type_tuple[0]} (type {obj_name_type_tuple[1]})")     
    else:
        conjecture_generator_module_logger.info(f"'score_obj_observations' function returning empty 'no_true_observation_objs' list.") 

    return (no_prediction_objs, no_true_observation_objs)

"""
LLM API functions
"""
class TogetherLLM:
    def __init__(self, conj_llm_name, api_key, sys_prompt=None):
        self.model = conj_llm_name
        self.client = Together(api_key=api_key)
        self.conversation = []

        # Initialize conversation with properly formatted first prompt if provided
        if sys_prompt:
            self.conversation = [{"role": "system", "content": sys_prompt}]

        # Use the client instance instead of the class method
        model_list = self.client.models.list()
        conjecture_generator_module_logger.info(f"Models available: {len(model_list)}\n")

    def reset_conversation(self, new_system_prompt=None):
        self.conversation = []
        if new_system_prompt:
            self.conversation = [{"role": "system", "content": new_system_prompt}]
            conjecture_generator_module_logger.info(f"Reset conversation with new system prompt: {new_system_prompt}")
        else:
            conjecture_generator_module_logger.info("Reset conversation (empty system prompt)")

    def get_response(self, prompt):
        self.conversation.append({"role": "user", "content": prompt})

        print(f"Passing conjecture question prompt to '{self.model}':\n{prompt}\n")
        print("Awaiting response...\n")
        conjecture_generator_module_logger.info(f"Passing conjecture question prompt to LLLM: {prompt}\n")
        conjecture_generator_module_logger.info("Awaiting response...\n")
        conjecture_generator_module_logger.info(f"Conversation history:\n{self.conversation}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            max_tokens=8000,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=None,  # Let the model finish naturally
            stream=False  # Streaming text is not needed
        )  
        # For more parameter info: https://docs.together.ai/reference/chat-completions

        message = response.choices[0].message.content

        print(f"Response received: {json.dumps(message)}\n")
        conjecture_generator_module_logger.info(f"Response received: {json.dumps(message)}\n")

        # Append assistant's response to conversation
        self.conversation.append({"role": "assistant", "content": message})
        
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        conjecture_generator_module_logger.info(f"Response generated (completion_tokens={completion_tokens}, total_tokens={total_tokens}):\n{message}")
        return message, prompt_tokens, completion_tokens, total_tokens

class TerminalLLM:
   def __init__(self, conj_llm_name=None, api_key=None, sys_prompt=None):
       self.model = conj_llm_name  # Kept for compatibility but unused
       self.conversation = []

       # Initialize conversation with system prompt if provided
       if sys_prompt:
           self.conversation = [{"role": "system", "content": sys_prompt}]
           
       conjecture_generator_module_logger.info("Terminal LLM initialized")

   def reset_conversation(self, new_system_prompt=None):
       self.conversation = []
       if new_system_prompt:
           self.conversation = [{"role": "system", "content": new_system_prompt}]
           conjecture_generator_module_logger.info(f"Reset conversation with new system prompt: {new_system_prompt}")
       else:
           conjecture_generator_module_logger.info("Reset conversation (empty system prompt)")

   def get_response(self, prompt):
       self.conversation.append({"role": "user", "content": prompt})
       
       # Print conversation history
       print("\nConversation history:")
       for msg in self.conversation:
           if msg["role"] == "system":
               print(f"System: \n{msg['content']}\n")
           elif msg["role"] == "user":
               print(f"User: \n{msg['content']}\n")
           else:
               print(f"\nAssistant: {msg['content']}\n")
       
       print("\nPlease enter your response (press Enter twice to finish):")
       
       # Collect multi-line input
       message_lines = []
       while True:
           line = input()
           if line == "":
               break
           message_lines.append(line)
       message = "\n".join(message_lines)

       # Append assistant's response to conversation
       self.conversation.append({"role": "assistant", "content": message})
       
       # Calculate character counts instead of tokens
       prompt_chars = len(prompt)
       completion_chars = len(message)
       total_chars = prompt_chars + completion_chars
       
       conjecture_generator_module_logger.info(f"Response received (completion_chars={completion_chars}, total_chars={total_chars}):\n{message}")
       return (message, prompt_chars, completion_chars, total_chars)
"""
Functions to generate first conjecture prompt 
"""
def get_last_line(module_path):
    with open(module_path, 'r') as file:
        tree = ast.parse(file.read())
        
    # Find the highest line number in the AST
    max_lineno = 0
    for node in ast.walk(tree):
        if hasattr(node, 'lineno'):
            max_lineno = max(max_lineno, node.lineno)
    
    return max_lineno

def count_types_in_ann_obj_dicts(ann_obj_dicts):
    """
    Count occurrences of different types in 'ann_obj_dicts'.
    
    Args:
        ann_obj_dicts (dict): Dictionary of values with 'type' keys
        
    Returns:
        dict: Counts of each type (class, function, instance, attribute)
    """
    # Initialize counters
    type_counts = {
        'class': 0,
        'function': 0,
        'instance': 0,
        'attribute': 0
    }
    
    # Iterate through all items in dictionary
    for fq_name, obj_info in ann_obj_dicts.items():
        if fq_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
            continue
        # Check if value is a dictionary and has 'type' key
        if isinstance(obj_info, dict) and 'type' in obj_info:
            item_type = obj_info['type']
            # Increment counter if type matches one we're looking for
            if item_type in type_counts:
                type_counts[item_type] += 1
    
    return type_counts

"""
Functions to extract info from observation dictionaries
"""
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

## Below function processes tuple strings with complete class info, including module name (no longer included in these tuples)
# def convert_string_to_tuple(tuple_string):
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

def get_obs_inputs(obs_dict):
    """
    Extracts inputs from 'instance_name(s)', including 'args' and 'kwargs' values.
    """
    input_tuples = []
    input_names = []
    args_and_kwargs_dict = obs_dict.get('instance_name(s)', {})
    # Get positional arguments
    args = args_and_kwargs_dict.get('args', [])
    input_tuples.extend(args)
    # Get keyword argument values
    kwargs = args_and_kwargs_dict.get('kwargs', {})
    input_tuples.extend(kwargs.values())
    for name_type_tuple_string in input_tuples:
        name_type_tuple = convert_string_to_tuple(name_type_tuple_string)
        object_name = name_type_tuple[0]
        input_names.append(object_name)
    return input_names

def get_obs_func_name(obs_dict):
    """
    Returns the function name, whether it is a function or a method.
    """
    if obs_dict['method_or_function'] == 'function':
        function = obs_dict['function_name']
    elif obs_dict['method_or_function'] == 'method':
        function = f"{obs_dict['class_if_method']}.{obs_dict['function_name']}"
    else:
        function = None  # Handle unexpected cases if necessary
        print(f"Warning: function or method name could not be located in an observation dictionary.")

    return function

def get_other_accessed_objects(obs_dict):
    """
    Retrieves all the accessed object names from the 'accessed_objects' dictionary.
    """
    other_accessed_object_tuples = []
    other_accessed_object_names = []

    accessed_objects = obs_dict.get('accessed_objects', {})
    for function_name, objects in accessed_objects.items():
        if isinstance(objects, dict):
            other_accessed_object_tuples.extend(objects.keys())
    for name_type_tuple_string in other_accessed_object_tuples:
        name_type_tuple = convert_string_to_tuple(name_type_tuple_string)
        object_name = name_type_tuple[0]
        other_accessed_object_names.append(object_name)

    return other_accessed_object_names

"""
Functions to update and validate EWM
"""
def save_code_to_file(code_text: str, directory: str) -> str:
    """
    Used for creating a seed EWM:

    Save code text to a Python file in the specified directory with a timestamp-based name.

    Args:
        code_text: The Python code to save
        directory: Directory path where file should be saved
        
    Returns:
        Path to the created file
    """
    # Create directory if it doesn't exist
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{timestamp}.py"

    # Create full file path
    file_path = os.path.join(directory, filename)

    # Write code to file
    with open(file_path, 'w') as f:
        f.write(code_text)
        
    conjecture_generator_module_logger.info(f"Seed EWM saved at: {file_path}")
    return file_path, timestamp

def update_module_content(module_path: str, update_dict: Dict[Tuple[int, int], Optional[str]]):
    """
    Creates a new version of a Python module with specified line modifications.
    For each update range (start, end):
    - Keeps the start and end lines
    - Removes all lines between them
    - Inserts the new text (if any) between start and end lines
    - If end = start + 1, no lines are removed and text is inserted between them
    - If text is None, nothing is inserted between the lines
    
    Args:
        module_path (str): Path to the original Python module
        update_dict (dict): Dictionary with tuple keys (start_line, end_line) and string values
                           representing the text to insert. Value can be None.
    
    Returns:
        str: Path to the newly created module file
        list: List of any errors that occured while updating   
            ModuleUpdateError: For various validation errors in the input parameters
            FileNotFoundError: If the module_path doesn't exist
            PermissionError: If there are permission issues with reading/writing files
        * If the file cannot be updated, (None, None) is returned. 
        * If the file can be updated, the new file path is returned.
        * If the file can be partially updated (minor errors reading the update dict), the new file path is returned along with error messages. 
    """
   
    # Validate module path
    if not os.path.exists(module_path):
        conjecture_generator_module_logger.error(f"Module path does not exist: {module_path}")
        return None, None, None
    
    # Read the original file
    try:
        with open(module_path, 'r') as file:
            lines = file.readlines()
    except Exception as e:
        conjecture_generator_module_logger.error(f"Unexpected error reading file: {str(e)}")
        return None, None, None
    
    # Validate update dictionary
    validated_updates = {}
    file_length = len(lines)
    error_messages = []
    
    for (start, end), text in update_dict.items():
        # Validate start and end are integers
        if not (isinstance(start, int) and isinstance(end, int)):
            conjecture_generator_module_logger.warning(f"Line numbers must be integers: ({start}, {end})")
            error_messages.append(f"Line numbers must be integers: ({start}, {end})")
        
        # Validate start <= end
        if start > end:
            conjecture_generator_module_logger.warning(f"Start line ({start}) cannot be greater than end line ({end})")
            error_messages.append(f"Start line ({start}) cannot be greater than end line ({end})")
        
        # Validate line numbers are within file bounds
        if start < 0 or end > file_length+1:
            conjecture_generator_module_logger.warning(f"Line numbers ({start}, {end}) out of range for file with {file_length} lines")
            error_messages.append(f"Line numbers ({start}, {end}) out of range for file with {file_length} lines")
        
        # Validate text is string or None
        if text is not None and not isinstance(text, str):
            conjecture_generator_module_logger.warning(f"Update text must be string or None, got {type(text)}")
            error_messages.append(f"Update text must be string or None, got {type(text)}")
        
        # Log valid update
        conjecture_generator_module_logger.info(f"Validated update for lines {start}-{end}" + 
                   (f" with {len(text)} characters" if text else " with no insertion"))
        
        validated_updates[(start, end)] = text
    
    # Convert 1-based line numbers to 0-based indices
    updates = {(start-1, end-1): text for (start, end), text in validated_updates.items()}
    
    # Sort updates by start index to process in order
    sorted_updates = sorted(updates.items(), key=lambda x: x[0][0])
    conjecture_generator_module_logger.info(f"Processing {len(sorted_updates)} updates in order")
    
    # Find all lines that need to be deleted
    lines_to_delete: Set[int] = set()
    for (start_idx, end_idx), _ in sorted_updates:
        if end_idx > start_idx + 1:  # Only delete lines if there's a gap
            lines_to_delete.update(range(start_idx + 1, end_idx))
    
    conjecture_generator_module_logger.info(f"Will delete {len(lines_to_delete)} lines")
    
    # Create list of updates with their positions
    processed_updates: List[Tuple[int, str]] = []
    for (start_idx, end_idx), new_text in sorted_updates:
        if new_text is not None:
            # Insert after start line
            # Ensure text ends with newline
            new_text = new_text if new_text.endswith('\n') else new_text + '\n'
            processed_updates.append((start_idx + 1, new_text))
    
    # Create new content
    new_lines = []
    current_line = 0
    
    # Sort processed updates by position
    processed_updates.sort(key=lambda x: x[0])
    
    # Process all lines and updates
    while current_line < len(lines) or processed_updates:
        # If we have updates and the current position is where we need to insert
        if processed_updates and current_line == processed_updates[0][0]:
            new_lines.append(processed_updates[0][1])
            processed_updates.pop(0)
        
        # If we still have original lines to process
        if current_line < len(lines):
            # Only keep lines that weren't marked for deletion
            if current_line not in lines_to_delete:
                new_lines.append(lines[current_line])
            current_line += 1
    
    # Generate new filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    original_path = Path(module_path)
    new_filename = f"{timestamp}{original_path.suffix}"
    new_path = original_path.parent / new_filename
    
    # Write the modified content to the new file
    try:
        with open(new_path, 'w') as file:
            file.writelines(new_lines)
        print(f"Updates added to:\n {original_path}\nNew version saved at:\n{new_path}\n")
        conjecture_generator_module_logger.info(f"Successfully wrote updated module to: {new_path}")
        return  new_path, timestamp, error_messages
    except Exception as e:
        conjecture_generator_module_logger.error(f"Unexpected error writing file: {str(e)}")
        return None, None, None    

def validate_python_module(module_path: str) -> Tuple[bool, List[str]]:
    """
    Validates a Python module for syntax errors and other basic issues.
    
    Args:
        module_path (str): Path to the Python module file
    
    Returns:
        Tuple[bool, List[str]]: A tuple containing:
            - bool: True if the module is valid, False if it contains errors
            - List[str]: List of error messages (empty if no errors)
    """
    path = Path(module_path)
    errors = []
    
    # Check if file exists
    if not path.exists():
        errors.append(f"File does not exist: {module_path}")
        return False, errors
    
    # Check if it's a Python file
    if path.suffix != '.py':
        errors.append(f"Not a Python file: {module_path}")
        return False, errors
        
    try:
        # Check for syntax errors by parsing the code
        with open(path, 'r', encoding='utf-8') as file:
            source = file.read()
            
        try:
            ast.parse(source)
        except SyntaxError as e:
            error_line_number = e.lineno
            problem_range_start = max(error_line_number-4, 1)
            problem_range_end = (error_line_number+4)
            problem_line_range = [(problem_range_start, problem_range_end)]
            line_text = find_lines_in_module(module_path, problem_line_range)
            errors.append(f"Syntax error at line {e.lineno}, offset {e.offset}: {e.msg}\nYour suggested update would appear as:\n{line_text}")
            return False, errors
        except IndentationError as e:
            error_line_number = e.lineno
            problem_range_start = max(error_line_number-4, 1)
            problem_range_end = (error_line_number+4)
            problem_line_range = [(problem_range_start, problem_range_end)]
            line_text = find_lines_in_module(module_path, problem_line_range)
            errors.append(f"Indentation error at line {e.lineno}: {e.msg}\nYour suggested update would appear as:\n{line_text}")
            return False, errors
            
        # Attempt to import the module
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                errors.append("Failed to create module specification")
                return False, errors
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
        except ImportError as e:
            tb = sys.exc_info()[2] # Returns the traceback object of the current exception being handled
            error_line_number = traceback.extract_tb(tb)[-1].lineno # Returns the line number of the last frame of the traceback stack
            problem_range_start = max(error_line_number-4, 1)
            problem_range_end = (error_line_number+4)
            problem_line_range = [(problem_range_start, problem_range_end)]
            line_text = find_lines_in_module(module_path, problem_line_range)
            errors.append(f"Import error: {str(e)}\nYour suggested update would appear as:\n{line_text}")
            return False, errors
        except AttributeError as e:
            tb = sys.exc_info()[2] # Returns the traceback object of the current exception being handled
            error_line_number = traceback.extract_tb(tb)[-1].lineno # Returns the line number of the last frame of the traceback stack
            problem_range_start = max(error_line_number-4, 1)
            problem_range_end = (error_line_number+4)
            problem_line_range = [(problem_range_start, problem_range_end)]
            line_text = find_lines_in_module(module_path, problem_line_range)
            errors.append(f"Attribute error: {str(e)}\nYour suggested update would appear as:\n{line_text}")
            return False, errors
        except NameError as e:
            tb = sys.exc_info()[2] # Returns the traceback object of the current exception being handled
            error_line_number = traceback.extract_tb(tb)[-1].lineno # Returns the line number of the last frame of the traceback stack
            problem_range_start = max(error_line_number-4, 1)
            problem_range_end = (error_line_number+4)
            problem_line_range = [(problem_range_start, problem_range_end)]
            line_text = find_lines_in_module(module_path, problem_line_range)
            errors.append(f"Name error: {str(e)}\nYour suggested update would appear as:\n{line_text}")
            return False, errors
        except TypeError as e:
            tb = sys.exc_info()[2] # Returns the traceback object of the current exception being handled
            error_line_number = traceback.extract_tb(tb)[-1].lineno # Returns the line number of the last frame of the traceback stack
            problem_range_start = max(error_line_number-4, 1)
            problem_range_end = (error_line_number+4)
            problem_line_range = [(problem_range_start, problem_range_end)]
            line_text = find_lines_in_module(module_path, problem_line_range)
            errors.append(f"Type error: {str(e)}\nYour suggested update would appear as:\n{line_text}")
            return False, errors
        except ValueError as e:
            tb = sys.exc_info()[2] # Returns the traceback object of the current exception being handled
            error_line_number = traceback.extract_tb(tb)[-1].lineno # Returns the line number of the last frame of the traceback stack
            problem_range_start = max(error_line_number-4, 1)
            problem_range_end = (error_line_number+4)
            problem_line_range = [(problem_range_start, problem_range_end)]
            line_text = find_lines_in_module(module_path, problem_line_range)
            errors.append(f"Value error: {str(e)}\nYour suggested update would appear as:\n{line_text}")
            return False, errors
        except Exception as e:
            tb = sys.exc_info()[2] # Returns the traceback object of the current exception being handled
            error_line_number = traceback.extract_tb(tb)[-1].lineno # Returns the line number of the last frame of the traceback stack
            problem_range_start = max(error_line_number-4, 1)
            problem_range_end = (error_line_number+4)
            problem_line_range = [(problem_range_start, problem_range_end)]
            line_text = find_lines_in_module(module_path, problem_line_range)
            errors.append(f"Unexpected error: {str(e)}\nYour suggested update would appear as:\n{line_text}")
            return False, errors
            
        return True, []
        
    except Exception as e:
        tb = sys.exc_info()[2] # Returns the traceback object of the current exception being handled
        error_line_number = traceback.extract_tb(tb)[-1].lineno # Returns the line number of the last frame of the traceback stack
        problem_range_start = max(error_line_number-4, 1)
        problem_range_end = (error_line_number+4)
        problem_line_range = [(problem_range_start, problem_range_end)]
        line_text = find_lines_in_module(module_path, problem_line_range)
        errors.append(f"Failed to read file: {str(e)}\nYour suggested update would appear as:\n{line_text}")
        return False, errors

def find_lines_in_module(module_path, lines_tuple_list):
    """
    Validates a list of range tuples and creates a RangeSet for efficient querying.
    Each tuple must contain two positive integers where the second is >= the first.
    Find the lines within the ranges in the python module. 
    Return requested lines.
    
    Args:
        lines_tuple_list (list): List of tuples of line ranges to return.
    
    Returns:
        str: Formatted string containing search results
    """
    collected_lines = []
    acceptable_ranges = []
    return_text = ''

    # Validate each tuple
    for lines_tuple in lines_tuple_list:
        start, end = lines_tuple
        error = False
        # Check if both values are integers
        if not isinstance(start, int) or not isinstance(end, int):
            conjecture_generator_module_logger.warning(ValueError(f"Range {(start, end)} contains non-integer values"))
            error=True
            
        # Check if both values are positive
        if start <= 0 or end <= 0:
            conjecture_generator_module_logger.warning(ValueError(f"Range {(start, end)} contains non-positive values"))
            error=True
            
        # Check if end is >= start
        if end < start:
            conjecture_generator_module_logger.warning(ValueError(f"Range {(start, end)} has end value less than start value"))
            error=True

        if not error:
            acceptable_ranges.append((start, end))

    # Sort ranges by start value for easier processing
    sorted_ranges = sorted(acceptable_ranges)
    
    # Find overall min and max
    min_value = min(start for start, _ in sorted_ranges)
    max_value = max(end for _, end in sorted_ranges)

    # Initialize results list
    try:
        # Read all lines from the file
        with open(module_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for i, line in enumerate(lines, start=1):
            # Quick bounds check
            if i < min_value or i > max_value:
                continue

            # Check each range
            line_found = False
            for start, end in sorted_ranges:
                if line_found:
                    continue
                if start <= i <= end: 
                    collected_lines.append((i, line))
                    line_found = True
        
        # Format the results
        for line_no, line in collected_lines: 
            line_no_chars = len(str(abs(line_no)))
            no_char_to_fill = 4-line_no_chars
            if no_char_to_fill > 0:
                line_no_str = "0"*no_char_to_fill+str(line_no)
            else:
                line_no_str = str(line_no)
            
            return_text+=f"[{line_no_str}]{line}"
            
        return return_text
    
    except FileNotFoundError:
        conjecture_generator_module_logger.error(f"'find_lines_in_module' function could not find file at {module_path}")
    except Exception as e:
        conjecture_generator_module_logger.error(f"Error occurred while processing file: {str(e)}")
        
@dataclass
class Definition:
    name: str
    line_no: int
    col_offset: int
    end_line_no: int
    end_col_offset: int
    type: str
    scope: Tuple[str, ...]
    text: str
    parent_node: ast.AST
    control_flow_depth: int  # Added to track nesting level of control flow

def get_source_segment(source: str, node: ast.AST) -> str:
    """Extract the full lines of source code for a given AST node."""
    def format_num(line_no):
        # Format line numbers to add leading zeros if it is less than 4 digits
        line_no_chars = len(str(abs(line_no)))
        no_char_to_fill = 4-line_no_chars
        if no_char_to_fill > 0:
            line_no_str = "0"*no_char_to_fill+str(line_no)
        else:
            line_no_str = str(line_no)    
        return line_no_str

    lines = source.splitlines()
    # Single line, no end_lineno:
    if node.end_lineno is None:
        return f"[{format_num(node.lineno)}]{lines[node.lineno - 1]}"
    # Single line, with start/end columns:
    if node.lineno == node.end_lineno:
        return f"[{format_num(node.lineno)}]{lines[node.lineno - 1]}"
    # Multi-line:
    result = [f"[{format_num(node.lineno)}]{lines[node.lineno - 1]}"]

    for line_num in range(node.lineno, node.end_lineno - 1):
        result.append(f"[{format_num(line_num + 1)}]{lines[line_num]}")
    result.append(f"[{format_num(node.end_lineno)}]{lines[node.end_lineno - 1]}")
    return '\n'.join(result)


class DefinitionVisitor(ast.NodeVisitor):
    def __init__(self, source: str):
        self.definitions: List[Definition] = []
        self.source = source
        self.scope_stack: List[str] = []
        self.current_parent: ast.AST = None
        self.control_flow_depth: int = 0  # Track depth of control flow structures
        
    def visit(self, node: ast.AST):
        """Override visit to track parent nodes."""
        previous_parent = self.current_parent
        self.current_parent = node
        
        # Increment control flow depth for control flow structures
        is_control_flow = isinstance(node, (ast.If, ast.While, ast.For, ast.Try, 
                                          ast.With, ast.Match, ast.AsyncWith))
        if is_control_flow:
            self.control_flow_depth += 1
            
        super().visit(node)
        
        # Decrement control flow depth when exiting control flow structures
        if is_control_flow:
            self.control_flow_depth -= 1
            
        self.current_parent = previous_parent
        
    def visit_ClassDef(self, node: ast.ClassDef):
        self.add_definition(node, "class")
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.add_definition(node, "function")
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.add_definition(node, "async_function")
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()
        
    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.add_definition(target, "variable")
        self.generic_visit(node)
        
    def add_definition(self, node: ast.AST, def_type: str):
        name = node.name if hasattr(node, 'name') else node.id
        definition = Definition(
            name=name,
            line_no=node.lineno,
            col_offset=node.col_offset,
            end_line_no=getattr(node, 'end_lineno', node.lineno),
            end_col_offset=getattr(node, 'end_col_offset', node.col_offset + len(name)),
            type=def_type,
            scope=tuple(self.scope_stack),
            text=get_source_segment(self.source, node),
            parent_node=self.current_parent,
            control_flow_depth=self.control_flow_depth
        )
        self.definitions.append(definition)

def are_definitions_in_same_control_flow(def1: Definition, def2: Definition) -> bool:
    """Check if two definitions are in the same control flow block."""
    if def1.control_flow_depth == 0 and def2.control_flow_depth == 0:
        return True
    return False

def is_within_duplicate_scope(def_: Definition, duplicate_scopes: Set[Tuple[str, ...]]) -> bool:
    """Check if a definition is within any of the duplicate scopes."""
    def_scope = def_.scope
    return any(
        len(def_scope) > len(dup_scope) and def_scope[:len(dup_scope)] == dup_scope
        for dup_scope in duplicate_scopes
    )

def analyze_duplicates(file_path: str) -> Optional[str]:
    """
    Analyze a Python file for redundant duplicate definitions.
    
    Args:
        file_path: Path to the Python file to analyze
        
    Returns:
        A string containing warnings about redundant duplicate definitions,
        or None if no redundant duplicates are found.
    """
    try:
        with open(file_path, 'r') as file:
            source = file.read()
            
        # Parse the AST
        tree = ast.parse(source)
        visitor = DefinitionVisitor(source)
        visitor.visit(tree)
        
        # Group definitions by name and scope
        definitions_by_scope: Dict[Tuple[str, Tuple[str, ...]], List[Definition]] = defaultdict(list)
        for definition in visitor.definitions:
            definitions_by_scope[(definition.name, definition.scope)].append(definition)
            
        # Track scopes that contain duplicates to exclude their nested duplicates
        duplicate_scopes: Set[Tuple[str, ...]] = set()
        
        # Check for redundant duplicates
        warnings = []
        # Sort by scope length to process outer scopes first
        sorted_keys = sorted(definitions_by_scope.keys(), key=lambda x: len(x[1]))
        
        for key in sorted_keys:
            name, scope = key
            defs = definitions_by_scope[key]
            
            # Skip if this definition is within a scope that already has duplicates
            if is_within_duplicate_scope(defs[0], duplicate_scopes):
                continue
                
            if len(defs) > 1:
                # Sort by line number to identify which definition overrides which
                defs.sort(key=lambda x: x.line_no)
                
                # Group definitions by their control flow context
                base_level_defs = [d for d in defs if d.control_flow_depth == 0]
                
                # Only warn if there are multiple definitions at the base level (not in control flow)
                if len(base_level_defs) > 1:
                    scope_str = ".".join(scope) if scope else "global"
                    warning = f"Here are some possible redundant duplicate definitions of '{name}' found in scope '{scope_str}':\n"
                    for idx, def_ in enumerate(base_level_defs):
                        warning += (f"[Definition {idx + 1} at lines {def_.line_no}-{def_.end_line_no}]\n"
                                  f"{def_.text}\n\n")
                    if base_level_defs[-1].type == base_level_defs[0].type:  # Only warn if they're the same type
                        warnings.append(warning)
                        # Add this scope to the set of duplicate scopes
                        duplicate_scopes.add(scope)
                
        return "\n".join(warnings) if warnings else None
        
    except FileNotFoundError:
        conjecture_generator_module_logger.warning(f"Error: File '{file_path}' not found.")
        return None
    except SyntaxError as e:
        conjecture_generator_module_logger.warning(f"Error: Invalid Python syntax in file: {str(e)}")
        return None
    except Exception as e:
        conjecture_generator_module_logger.warning(f"Error analyzing file: {str(e)}")
        return None
    
"""
Class to walk the question paths, prompt LLM, answer context requests, and apply update to EWM
"""
class QuestionPathWalker:
    """
    Generate first prompt.
    - collect all context info
    - build prompt with context info
    - if last conjecture was rejected; include additional info about prior attempts
    - if improve organization path, generate prompt for this.

    ----
    Collect response.
    Check response for final answer or context request.

    If not final answer, respond with context.
        - Parse context request.
        - Retrieve requested informaion.
        - Construct response prompt.
        - Add next response to message list and submit to llm

    If final answer.
        - Parse answer. 

    If final answer can be integrated with EWM, pass duplicate check, and pass compile check:
        - update conjecture dicts

    If num_attempts is 0, update conjecture dicts and exit.

    Update conj_dicts:
        - Initialize new question series dictionary (id += 1)
        - Prompt/response series
        - 

    Save conj_dicts. 
    """  
    def __init__(
        self, 
        ewm_name,
        ewm_path,
        prob_obj_name,
        prob_obj_obs_type,
        question_path,
        # LLM API info 
        conj_api_key,
        conj_llm_name,
        # Data
        obs_dicts,
        ann_obj_dicts,
        conj_dicts,
        # Paths to update conj_dicts
        conj_dicts_path,
        conj_dir,
        ):

        self.ewm_name=ewm_name
        self.ewm_path=ewm_path
        self.prob_obj_name=prob_obj_name
        self.prob_obj_obs_type=prob_obj_obs_type # will be set to one of ('no_pred', 'no_true_obs', 'true_obs')
        self.question_path=question_path

        self.conj_api_key=conj_api_key
        self.conj_llm_name=conj_llm_name

        self.obs_dicts=obs_dicts
        self.ann_obj_dicts=ann_obj_dicts
        self.conj_dicts=conj_dicts

        self.conj_dicts_path = conj_dicts_path
        self.conj_dir=conj_dir

        self.num_update_attempts = 8
        self.num_context_requests = 16
        
        # Computed self attributes
        self.prob_obj_type = None
        if self.prob_obj_name and self.ann_obj_dicts:
            self.prob_obj_type = self.ann_obj_dicts[self.prob_obj_name]['type']
        
        self.total_lines = None
        if self.ewm_path:
            self.total_lines = get_last_line(self.ewm_path)

        if self.ann_obj_dicts:
            type_counts = count_types_in_ann_obj_dicts(self.ann_obj_dicts)
            conjecture_generator_module_logger.info(f"'type_counts' from 'ann_obj_dicts': {type_counts}")
        
        self.total_num_classes = None
        self.total_num_instances = None
        self.total_attributes = None
        self.total_functions_and_methods = None
        if type_counts:
            self.total_num_classes = type_counts['class']
            self.total_num_instances = type_counts['instance']
            self.total_attributes = type_counts['attribute']
            self.total_functions_and_methods = type_counts['function']

        self.shared_obs_ids = []
        self.llm = None

    def walk(self):
        prob_obj_sys_prompt = (
            f"You will receive information about an object defined inside a Python module. The module is {self.total_lines} lines, contains {self.total_num_classes} classes, {self.total_functions_and_methods} functions and methods, and {self.total_num_instances} instances with {self.total_attributes} attributes. All of the objects defined in the module are intended to represent real things or concepts.\n"
            "The user evaluates how realistic object defintions are by inputting instances and attributes into functions or methods (following type hints), and computing results. When computations with an object produce errors, unexpected, or unrealistic results the user will ask you to help improve the defintion of that potential problem object. Your final answer will supply locations and content for new lines of code to update the module.\n"
            "Your PRIMARY GOAL is to improve the module such that computations produce more results that are consistent with reality and fewer unexpected or incorrect results. Your SECONDARY GOALS are a) to improve organization of the code, decreasing redundancy by using references to existing definitions whenever possible, b) introduce new definitions when you think they will enhance the module’s overall representation of real things or concepts.\n\n"
            "#Steps\n\n"
            "1. **Choose your response type**: You may either request more context or submit a code update. You may submit the final code update(s) only once. Reviewing context information may improve your answer.\n"
            f"2. **Request more context**: If you think more information about the module or about computation errors will improve your final answer, submit a context request. The user will return the information you requested. You can make context requests multiple times before you submit the final code update. The simplist request is to view the entire module: @<<(lines:1,{self.total_lines})>>@. If the module is very long, you may request specific details.\n"
            "Context request formats include:\n"
            f"  • Text from any line range in the module (between 1 and {self.total_lines}): “lines:[insert start line],[insert end line]”, e.g. (lines:12,18).\n"
            "  • List of all class names: (all_classes)\n"
            "  • List of all subclasses of a class: (subclasses:[insert a class name]), e.g. (subclasses:Planet) \n"
            "  • List of all instances of a class: (instances:[insert a class name]), e.g. (instances:Elements)\n"
            "  • List of all statically defined attributes of a type: (attributes:[insert a type name]), e.g. (attributes:Name)\n"
            "  • List of all function and method names: (all_functions)\n"
            "  • List of all instance names (organized by their classes): (all_instances)\n"
            "  • List of all functions that accept arguments of a given type: (domain:[insert type name]), e.g. (domain:Distance)\n"
            "  • The initialized definition of any instance or attribute: (i_def:[insert name]), e.g. (i_def:Planet).\n"
            "  • The complete static definition of any class, function, or instance: (s_def:[insert name]), e.g. (s_def:Planet).\n"
            "  • A sample of potential unexpected or incorrect computations from a class, instance, or function: (comps:[insert object name]), e.g. (comps:earth_planet)”\n"
            "  • The code lines containing some input string (including one line above and below): (find:[insert name]), e.g. (find:print).\n\n"
            "3. **Submit your final code update**: When you submit, you must specify the line(s) where each block of your generated code will be inserted. Only the lines _between_ the two integer boundary numbers will be deleted, and the boundary number lines are not deleted. Your code will insert below the first line you specified. E.g. line references '@>>(122,122)...' and '@>>(122,123)...' will delete no lines and will insert code below line 122. The response format '@>>(11,14)...' will replace code on lines 12 and 13. The line reference 0, '@>>(0,...' can be used to insert or delete at line 1. To delete the last line, use line reference number '[last line #] + 1'.\n\n"
            "#Output Formats\n\n"
            "Your responses should include either a request for more context information or a final answer. Requests for more information must be formatted as: @<<(str),(str),(str),...>>@. Final answers must be formatted as: @>>(int,int):```str```,...<<@. Note that you can submit multiple context requests or code updates in a single response. You may include the context request or final answer anywhere in your response. Additional text is acceptable but it will not be read.\n"
            "#Examples\n\n"
            "[Example 1: context request]\n"
            "@<<(lines:1,18),(s_def:Planet),(all_instances),(i_def:earth_planet)>>@\n"
            "[Example 1 end]\n"
            "[Example 2: final code improvement answer]\n"
            "@>>(167,189):```class Planet(PhysicalObject, Sphere):\n    def __init__(self: 'Planet', name: Name, mass: Mass, radius: Radius, volume: Volume = None, atmosphere: Atmosphere = None):\n        PhysicalObject.__init__(self, name=name, mass=mass)\n        Sphere.__init__(self, radius=radius, volume=volume)\n        self.atmosphere = atmosphere\n\n    def calculate_surface_gravity(self: 'Planet') -> Value:\n        \"\"\"Calculate the surface gravity of the planet in m/s^2.\"\"\"\n        if self.mass.value.unit == kilogram_unit and self.radius.value.unit == meter_unit:\n            G = gravitational_constant.get_value(newton_unit)\n            gravity = G * self.mass.value.number / (self.radius.value.number ** 2)\n            return Value(number=gravity, unit=Unit(name=Name(\"meters per second squared\"), symbols=[Symbol(\"m/s\u00b2\")]))\n        else:\n            raise AttributeError(\"Mass must be in kilograms and radius in meters to calculate surface gravity\")```<<@\n"
            "[Example 2 end]\n\n"
            """[Example 3: improper use of line numbers in final answer]
# Code being updated:
[0070]class PhysicalConstant:
[0071]    def __init__(self: 'PhysicalConstant', name: Name, symbol: Symbol, values: list[Value]):
[0072]        self.name: Name = name
[0073]        self.symbol: Symbol = symbol
[0074]        self.values: list[Value] = values
[0075]    
[0076]    def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:
[0077]        for value in self.values:
[0078]            if value.unit == unit:
[0079]                return value.number
[0080]        raise ValueError(f"No value found for unit {unit.name.name}")

# Final answer submitted (boundary line numbers 76 and 80 are not deleted, leaving duplicate lines):
@>>(76,80):```def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:\n    for value in self.values:\n        if value.unit == unit:\n            return value.number\n    # Try to convert units if no direct match is found\n    for value in self.values:\n        if isinstance(value.number, (int, float)) and value.unit.symbols:\n            # Implement unit conversion logic here\n            # For now, just raise an error\n            raise NotImplementedError(\"Unit conversion not implemented\")\n    raise ValueError(f\"No value found for unit {unit.name.name}\")```<<@

# Updated code (producing syntax errors at lines 77 and 88):
[0070]class PhysicalConstant:
[0071]    def __init__(self: 'PhysicalConstant', name: Name, symbol: Symbol, values: list[Value]):
[0072]        self.name: Name = name
[0073]        self.symbol: Symbol = symbol
[0074]        self.values: list[Value] = values
[0075]    
[0076]    def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:
[0077]def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:
[0078]    for value in self.values:
[0079]        if value.unit == unit:
[0080]            return value.number
[0081]    # Try to convert units if no direct match is found
[0082]    for value in self.values:
[0083]        if isinstance(value.number, (int, float)) and value.unit.symbols:
[0084]            # Implement unit conversion logic here
[0085]            # For now, just raise an error
[0086]            raise NotImplementedError("Unit conversion not implemented")
[0087]    raise ValueError(f"No value found for unit {unit.name.name}")
[0088]        raise ValueError(f"No value found for unit {unit.name.name}")

# The final answer would have been correct if it used boundary numbers 75 and 81, the numbers above and below the lines that should have been deleted.
[Example 3 end]

"""
            "#Notes\n\n"
            "Any changes to the code are permitted as long as the change improves the code’s overall organization or consistency with reality. E.g. You may introduce a new definition; modify the content of any definition; change the name of any instance, function, or method; or reassign an instance or method to a new class.\n"
            "Remember to use type hints in your function and method definitions. In some cases, the best way to improve a function or method definition is by modifying its type hints. Likewise, you can sometimes improve an instance by reassigning it to a new class; or improve a class by ensuring it inherits from the correct parent class(s).\n"
            "If you intend to update any object you should check if any references to that object need to be updated as well. Recall, you can find references to a name in the code by adding (find:[name of object]) in a context request."
            )

        variety_prompt = (
            f"You will receive information about a Python module. The module is {self.total_lines} lines, contains {self.total_num_classes} classes, {self.total_functions_and_methods} functions and methods, and {self.total_num_instances} instances with {self.total_attributes} attributes. All of the objects defined in the module are intended to represent real things or concepts.\n"
            "The user evaluates how realistic object defintions are by inputting instances and attributes into functions or methods (following type hints), and computing results. The user will ask you to help improve the module by suggesting new definitions to add. Your final answer will supply locations and content for new lines of code to update the module.\n"
            "Your PRIMARY GOAL is to improve the module by introducing one or more definitions that can be combined with existing objects in the module to compute results that are consistent with reality. Your SECONDARY GOAL is to improve organization of the code, decreasing redundancy by using references to existing definitions whenever possible.\n\n"
            "#Steps\n\n"
            "1. **Choose your response type**: You may either request more context or submit a code update. You may submit the final code update(s) only once. Reviewing context information may improve your answer.\n"
            f"2. **Request more context**: If you think more information about the module will improve your final answer, submit a context request. The user will return the information you requested. You can make context requests multiple times before you submit the final code update. The simplist request is to view the entire module: @<<(lines:1,{self.total_lines})>>@. If the module is very long, you may request specific details.\n"
            "Context request formats include:\n"
            f"  • Text from any line range in the module (between 1 and {self.total_lines}): “lines:[insert start line],[insert end line]”, e.g. (lines:12,18).\n"
            "  • List of all class names: (all_classes)\n"
            "  • List of all subclasses of a class: (subclasses:[insert a class name]), e.g. (subclasses:Planet) \n"
            "  • List of all instances of a class: (instances:[insert a class name]), e.g. (instances:Elements)\n"
            "  • List of all statically defined attributes of a type: (attributes:[insert a type name]), e.g. (attributes:Name)\n"
            "  • List of all function and method names: (all_functions)\n"
            "  • List of all instance names (organized by their classes): (all_instances)\n"
            "  • List of all functions that accept arguments of a given type: (domain:[insert type name]), e.g. (domain:Distance)\n"
            "  • The initialized definition of any instance or attribute: (i_def:[insert name]), e.g. (i_def:Planet).\n"
            "  • The complete static definition of any class, function, or instance: (s_def:[insert name]), e.g. (s_def:Planet).\n"
            "  • The code lines containing some input string (including one line above and below): (find:[insert name]), e.g. (find:print).\n\n"
            "3. **Submit your final code update**: When you submit, you must specify the line(s) where each block of your generated code will be inserted. Only the lines _between_ the two integer boundary numbers will be deleted, and the boundary number lines are not deleted. Your code will insert below the first line you specified. E.g. line references '@>>(122,122)...' and '@>>(122,123)...' will delete no lines and will insert code below line 122. The response format '@>>(11,14)...' will replace code on lines 12 and 13. The line boundary number 0, '@>>(0,...' can be used to insert or delete at line 1. To delete the last line, use line boundary number '[last line #] + 1'.\n\n"
            "#Output Formats\n\n"
            "Your responses should include either a request for more context information or a final answer. Requests for more information must be formatted as: @<<(str),(str),(str),...>>@. Final answers must be formatted as: @>>(int,int):```str```,...<<@. Note that you can submit multiple context requests or code updates in a single response. You may include the context request or final answer anywhere in your response. Additional text is acceptable but it will not be read.\n"
            "#Examples\n\n"
            "[Example 1: context request]\n"
            "@<<(lines:1,18),(s_def:Planet),(all_instances),(i_def:earth_planet)>>@\n"
            "[Example 1 end]\n"
            "[Example 2: final code improvement answer]\n"
            "@>>(167,189):```class Planet(PhysicalObject, Sphere):\n    def __init__(self: 'Planet', name: Name, mass: Mass, radius: Radius, volume: Volume = None, atmosphere: Atmosphere = None):\n        PhysicalObject.__init__(self, name=name, mass=mass)\n        Sphere.__init__(self, radius=radius, volume=volume)\n        self.atmosphere = atmosphere\n\n    def calculate_surface_gravity(self: 'Planet') -> Value:\n        \"\"\"Calculate the surface gravity of the planet in m/s^2.\"\"\"\n        if self.mass.value.unit == kilogram_unit and self.radius.value.unit == meter_unit:\n            G = gravitational_constant.get_value(newton_unit)\n            gravity = G * self.mass.value.number / (self.radius.value.number ** 2)\n            return Value(number=gravity, unit=Unit(name=Name(\"meters per second squared\"), symbols=[Symbol(\"m/s\u00b2\")]))\n        else:\n            raise AttributeError(\"Mass must be in kilograms and radius in meters to calculate surface gravity\")```<<@\n"
            "[Example 2 end]\n\n"
            """[Example 3: improper use of line numbers in final answer]
# Code being updated:
[0070]class PhysicalConstant:
[0071]    def __init__(self: 'PhysicalConstant', name: Name, symbol: Symbol, values: list[Value]):
[0072]        self.name: Name = name
[0073]        self.symbol: Symbol = symbol
[0074]        self.values: list[Value] = values
[0075]    
[0076]    def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:
[0077]        for value in self.values:
[0078]            if value.unit == unit:
[0079]                return value.number
[0080]        raise ValueError(f"No value found for unit {unit.name.name}")

# Final answer submitted (boundary line numbers 76 and 80 are not deleted, leaving duplicate lines):
@>>(76,80):```def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:\n    for value in self.values:\n        if value.unit == unit:\n            return value.number\n    # Try to convert units if no direct match is found\n    for value in self.values:\n        if isinstance(value.number, (int, float)) and value.unit.symbols:\n            # Implement unit conversion logic here\n            # For now, just raise an error\n            raise NotImplementedError(\"Unit conversion not implemented\")\n    raise ValueError(f\"No value found for unit {unit.name.name}\")```<<@

# Updated code (producing syntax errors at lines 77 and 88):
[0070]class PhysicalConstant:
[0071]    def __init__(self: 'PhysicalConstant', name: Name, symbol: Symbol, values: list[Value]):
[0072]        self.name: Name = name
[0073]        self.symbol: Symbol = symbol
[0074]        self.values: list[Value] = values
[0075]    
[0076]    def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:
[0077]def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:
[0078]    for value in self.values:
[0079]        if value.unit == unit:
[0080]            return value.number
[0081]    # Try to convert units if no direct match is found
[0082]    for value in self.values:
[0083]        if isinstance(value.number, (int, float)) and value.unit.symbols:
[0084]            # Implement unit conversion logic here
[0085]            # For now, just raise an error
[0086]            raise NotImplementedError("Unit conversion not implemented")
[0087]    raise ValueError(f"No value found for unit {unit.name.name}")
[0088]        raise ValueError(f"No value found for unit {unit.name.name}")

# The final answer would have been correct if it used boundary numbers 75 and 81, the numbers above and below the lines that should have been deleted.
[Example 3 end]

"""
            "#Notes\n\n"
            "Any changes to the code are permitted as long as the change improves the code’s overall organization or consistency with reality. E.g. You may introduce a new definition; modify the content of any definition; change the name of any instance, function, or method; or reassign an instance or method to a new class.\n"
            "Remember to use type hints in your function and method definitions.\n"
            "If you intend to update any object you should check if any references to that object need to be updated as well. Recall, you can find references to a name in the code by adding (find:[name of object]) in a context request."
            )

        org_sys_prompt = (
            f"You will receive some examples of objects defined inside a Python module. The module is {self.total_lines} lines, contains {self.total_num_classes} classes, {self.total_functions_and_methods} functions and methods, and {self.total_num_instances} instances with {self.total_attributes} attributes. All of the objects defined in the module are intended to represent real things or concepts.\n"
            "The purpose of the code is to compute realistic results when instances and attributes are input into functions or methods (following type hints). This module simply stores objects that another module uses to compute results with.\n"
            "Your final answer will supply locations and content for new lines of code to update the module."
            "Your PRIMARY GOAL is to the improve organization of the module. This means:\n  a) decreasing redundancy by using references to existing definitions rather than re-writing code available elsewhere in the module. \n  b) referencing definitions within the module rather than definitions outside the module, when possible. c) removing unnecessary print statements or other unnecessary code in the module.\n\n"
            "Your SECONDARY GOAL is to introduce new definitions when you think they will enhance the module’s overall representation of reality.\n"
            "#Steps\n\n"
            "1. **Choose your response type**: You may either request more context or submit a code update. You may submit the final code update(s) only once. Reviewing context information may improve your answer.\n"
            f"2. **Request more context**: If you think more information about the module will improve your final answer, submit a context request. The user will return the information you requested. You can make context requests multiple times before you submit the final code update. The simplist request is to view the entire module: @<<(lines:1,{self.total_lines})>>@. If the module is very long, you may instead request specific details.\n"
            "Context request formats include:\n"
            f"  • Text from any line range in the module (between 1 and {self.total_lines}): “lines:[insert start line],[insert end line]”, e.g. (lines:12,18).\n"
            "  • List of all class names: (all_classes)\n"
            "  • List of all subclasses of a class: (subclasses:[insert a class name]), e.g. (subclasses:Planet) \n"
            "  • List of all instances of a class: (instances:[insert a class name]), e.g. (instances:Elements)\n"
            "  • List of all statically defined attributes of a type: (attributes:[insert a type name]), e.g. (attributes:Name)\n"
            "  • List of all function and method names: (all_functions)\n"
            "  • List of all instance names (organized by their classes): (all_instances)\n"
            "  • List of all functions that accept arguments of a given type: (domain:[insert type name]), e.g. (domain:Distance)\n"
            "  • The initialized definition of any instance or attribute: (i_def:[insert name]), e.g. (i_def:Planet).\n"
            "  • The complete static definition of any class, function, or instance: (s_def:[insert name]), e.g. (s_def:Planet).\n"
            "  • The code lines containing some input string (including one line above and below): (find:[insert name]), e.g. (find:print).\n\n"
            "3. **Submit your final code update**: When you submit, you must specify the line(s) where each block of your generated code will be inserted. Only the lines _between_ the two integer boundary numbers will be deleted, and the boundary number lines are not deleted. Your code will insert below the first line you specified. E.g. line references '@>>(122,122)...' and '@>>(122,123)...' will delete no lines and will insert code below line 122. The response format '@>>(11,14)...' will replace code on lines 12 and 13. The line reference 0, '@>>(0,...' can be used to insert or delete at line 1. To delete the last line, use line reference number '[last line #] + 1'.\n\n"
            "#Output Formats\n\n"
            "Your responses should include either a request for more context information or a final answer. Requests for more information must be formatted as: @<<(str),(str),(str),...>>@. Final answers must be formatted as: @>>(int,int):```str```,...<<@. Note that you can submit multiple context requests or code updates in a single response. You may include the context request or final answer anywhere in your response. Additional text is acceptable but it will not be read.\n\n"
            "#Examples\n\n"
            "[Example 1: context request]\n"
            "@<<(lines:1,18),(s_def:Planet),(all_instances),(i_def:earth_planet)>>@\n"
            "[Example 1 end]\n"
            "[Example 2: final code improvement answer]\n"
            "@>>(167,189):```class Planet(PhysicalObject, Sphere):\n    def __init__(self: 'Planet', name: Name, mass: Mass, radius: Radius, volume: Volume = None, atmosphere: Atmosphere = None):\n        PhysicalObject.__init__(self, name=name, mass=mass)\n        Sphere.__init__(self, radius=radius, volume=volume)\n        self.atmosphere = atmosphere\n\n    def calculate_surface_gravity(self: 'Planet') -> Value:\n        \"\"\"Calculate the surface gravity of the planet in m/s^2.\"\"\"\n        if self.mass.value.unit == kilogram_unit and self.radius.value.unit == meter_unit:\n            G = gravitational_constant.get_value(newton_unit)\n            gravity = G * self.mass.value.number / (self.radius.value.number ** 2)\n            return Value(number=gravity, unit=Unit(name=Name(\"meters per second squared\"), symbols=[Symbol(\"m/s\u00b2\")]))\n        else:\n            raise AttributeError(\"Mass must be in kilograms and radius in meters to calculate surface gravity\")```<<@\n"
            "[Example 2 end]\n\n"
            """[Example 3: improper use of line numbers in final answer]
# Code being updated:
[0070]class PhysicalConstant:
[0071]    def __init__(self: 'PhysicalConstant', name: Name, symbol: Symbol, values: list[Value]):
[0072]        self.name: Name = name
[0073]        self.symbol: Symbol = symbol
[0074]        self.values: list[Value] = values
[0075]    
[0076]    def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:
[0077]        for value in self.values:
[0078]            if value.unit == unit:
[0079]                return value.number
[0080]        raise ValueError(f"No value found for unit {unit.name.name}")

# Final answer submitted (boundary line numbers 76 and 80 are not deleted, leaving duplicate lines):
@>>(76,80):```def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:\n    for value in self.values:\n        if value.unit == unit:\n            return value.number\n    # Try to convert units if no direct match is found\n    for value in self.values:\n        if isinstance(value.number, (int, float)) and value.unit.symbols:\n            # Implement unit conversion logic here\n            # For now, just raise an error\n            raise NotImplementedError(\"Unit conversion not implemented\")\n    raise ValueError(f\"No value found for unit {unit.name.name}\")```<<@

# Updated code (producing syntax errors at lines 77 and 88):
[0070]class PhysicalConstant:
[0071]    def __init__(self: 'PhysicalConstant', name: Name, symbol: Symbol, values: list[Value]):
[0072]        self.name: Name = name
[0073]        self.symbol: Symbol = symbol
[0074]        self.values: list[Value] = values
[0075]    
[0076]    def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:
[0077]def get_value(self: 'PhysicalConstant', unit: Unit) -> Any:
[0078]    for value in self.values:
[0079]        if value.unit == unit:
[0080]            return value.number
[0081]    # Try to convert units if no direct match is found
[0082]    for value in self.values:
[0083]        if isinstance(value.number, (int, float)) and value.unit.symbols:
[0084]            # Implement unit conversion logic here
[0085]            # For now, just raise an error
[0086]            raise NotImplementedError("Unit conversion not implemented")
[0087]    raise ValueError(f"No value found for unit {unit.name.name}")
[0088]        raise ValueError(f"No value found for unit {unit.name.name}")

# The final answer would have been correct if it used boundary numbers 75 and 81, the numbers above and below the lines that should have been deleted.
[Example 3 end]

"""
            "#Notes\n\n"
            "Any changes to the code are permitted as long as the change improves the code’s overall organization or consistency with reality. E.g. You may introduce a new definition; modify the content of any definition; change the name of any instance, function, or method; or reassign an instance or method to a new class.\n"
            "Remember to use type hints in your function and method definitions. Sometimes the best way to improve a function or method definition is by modifying its type hints. Likewise, you can sometimes improve an instance by reassigning it to a new class; or improve a class by ensuring it inherits from the correct parent class(s).\n"
            "If you intend to update any object you should check if any references to that object need to be updated as well. Recall, you can find references to a name in the code by adding (find:[name of object]) in a context request."
            )
        
        conjecture_generator_module_logger.info(f"Generating first prompt for question_path: '{self.question_path}', prob_obj_name '{self.prob_obj_name}'.")
        
        if self.question_path == 'default_problem_obj_conjecture':
            sys_prompt = prob_obj_sys_prompt
            first_user_prompt = self.generate_prob_obj_prompt()
        elif self.question_path == 'rejected_problem_obj_conjecture':
            sys_prompt = prob_obj_sys_prompt
            first_user_prompt = self.generate_rejected_problem_obj_prompt()
        elif self.question_path == 'improve_organization':
            sys_prompt = org_sys_prompt
            first_user_prompt = self.generate_organization_prompt()
        elif self.question_path == 'improve_variety':
            sys_prompt = variety_prompt
            first_user_prompt = self.generate_variety_prompt()

        conjecture_generator_module_logger.info(f"sys_prompt: {sys_prompt}")
        conjecture_generator_module_logger.info(f"first_user_prompt: {first_user_prompt}")
        conjecture_generator_module_logger.info(f"Passing first prompt to '{self.conj_llm_name}'")

        # Initialize LLM
        conjecture_generator_module_logger.info(f"Generating conjecture for EWM: {self.ewm_name}")
        print(f"Generating conjecture for EWM: {self.ewm_name}\n")

        self.llm = TogetherLLM(
            conj_llm_name=self.conj_llm_name, 
            api_key=self.conj_api_key,
            sys_prompt=sys_prompt)
        
        # # TerminalLLM mimics LLM API but user submits responses (conjecture updates and context requests)
        # self.llm = TerminalLLM(
        #     conj_llm_name=self.conj_llm_name, 
        #     api_key=self.conj_api_key,
        #     sys_prompt=sys_prompt)
        
        # Pass first prompt to LLM
        message, prompt_tokens, completion_tokens, total_tokens = self.llm.get_response(first_user_prompt)

        self.num_context_requests -= 1

        if not message:
            conjecture_generator_module_logger.error("No message received after passing first prompt to LLM.")
            print("Error: No message received after passing first prompt to LLM.")
            sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.
        
        conjecture_generator_module_logger.info(f"'self.conj_dicts':\n{json.dumps(self.conj_dicts, indent=2)}")
        # If no prior question series for the EWM, initialize dict for current quesiton path
        if not 'question_series' in self.conj_dicts:
            current_q_series_id = 1
            self.conj_dicts['question_series']=[{
                'id':1,
                'question_path':self.question_path,
                'problem_obj':self.prob_obj_name,
                'conversation':self.llm.conversation,
                'tokens':{
                    'prompt_tokens':[prompt_tokens],
                    'completion_tokens':[completion_tokens],
                    'total_tokens':[total_tokens]},
                'update_attempts':[]}]
            save_results(self.conj_dicts_path, self.conj_dicts)

        # If prior question series for the EWM, add a new dict for the current question path   
        else:
            last_q_series_id = max(series['id'] for series in self.conj_dicts['question_series'])
            current_q_series_id=last_q_series_id+1
            self.conj_dicts['question_series'].append({
                'id':current_q_series_id,
                'question_path':self.question_path,
                'problem_obj':self.prob_obj_name,
                'conversation':self.llm.conversation,
                'tokens':{
                    'prompt_tokens':[prompt_tokens],
                    'completion_tokens':[completion_tokens],
                    'total_tokens':[total_tokens]},
                'update_attempts':[]})
            
            save_results(self.conj_dicts_path, self.conj_dicts)
        
        conjecture_generator_module_logger.info(f"current_q_series_id:\n{current_q_series_id}")
        conjecture_generator_module_logger.info(f"'self.conj_dicts':\n{json.dumps(self.conj_dicts, indent=2)}")

        # Assign the current_q_series_dict to a variable
        for question_series in self.conj_dicts['question_series']:
            if question_series['id'] == current_q_series_id:
                current_q_series_dict = question_series
                break  # Exit 'for' loop once found ('while' loop is not exited)
            
        new_ewm_path=None # Initialize new_ewm_path variable

        while self.num_context_requests > 0 and self.num_update_attempts > 0:
            """
            break: Exits the entire loop immediately
            continue: Skips the rest of the current iteration and moves to the next one
            """
            response_errors = []
            ctx_msg_dict = None
            next_prompt = ''

            # Determine the response type
            response_type = self.detect_response_type(message)
            conjecture_generator_module_logger.info(f"Response type: {response_type}")
            print(f"Response type: {response_type}\n")
            if response_type == "both":
                response_errors.append(f"Response error: unable to process because both context request and final answer detected in response.")
                conjecture_generator_module_logger.info(f"Response error: unable to process because both context request and final answer detected in response: '{message}'")

            elif response_type == "context_request":
                ctx_msg_dict, error_msg_lst = self.process_context_messages(message)
                conjecture_generator_module_logger.info(f"ctx_msg_dict: {json.dumps(ctx_msg_dict,indent=2, cls=CustomJSONEncoder)}")
                conjecture_generator_module_logger.info(f"error_msg_lst: {error_msg_lst}")
                response_errors.extend(error_msg_lst)

            elif response_type == "final_answer":
                final_msg_dict, error_msg_lst = self.process_final_answer(message)
                final_msg_dict_str_key = {str(key): value for key, value in final_msg_dict.items()}
                conjecture_generator_module_logger.info(f"final_msg_dict: {final_msg_dict}")
                conjecture_generator_module_logger.info(f"error_msg_lst: {error_msg_lst}")
                response_errors.extend(error_msg_lst)

                # Subtract an update attempt (default is 8) and reset num context requests (default is 16)
                self.num_update_attempts -= 1
                self.num_context_requests = 16
                                
                if not response_errors:
                    new_ewm_path, timestamp, update_errors_list = update_module_content(self.ewm_path, final_msg_dict)
                    if new_ewm_path:
                        if update_errors_list:
                            response_errors.append(update_errors_list)
                        else: 
                            # Initialize first update dictionary if none exist
                            if not current_q_series_dict['update_attempts']:
                                current_update_dict = {'id':1, 
                                                    'update_dict':{},
                                                    'timestamp':None, 
                                                    'compile_check':{'result':None, 'message':''},
                                                    'duplicate_check':{'result':None, 'message':''}}
                            # Initialize next update dictionary if prior attempts exist
                            else:
                                last_update_attempt = max(update['id'] for update in current_q_series_dict['update_attempts'])
                                current_update_dict = {'id':last_update_attempt+1, 
                                                    'update_dict':{},
                                                    'timestamp':None, 
                                                    'compile_check':{'result':None, 'message':''},
                                                    'duplicate_check':{'result':None, 'message':''}}
                            
                            current_update_dict['update_dict'] = final_msg_dict_str_key
                            current_update_dict['timestamp'] = timestamp

                            # Check if the new EWM code compiles
                            validated_bool, errors = validate_python_module(new_ewm_path)
                            conjecture_generator_module_logger.info(f"Updated module validation result: {validated_bool}\n  Validation errors: {errors}")
                            current_update_dict['compile_check'] = {'result':validated_bool, 'message':errors}
                            
                            # Check if the new EWM contains duplicate definitions 
                              # (presence of duplicate definitions generates errors during ann_obj_dict generation)
                            duplicate_defs = analyze_duplicates(new_ewm_path) # Returns None if no duplicate definitions found, otherwise error message
                            if duplicate_defs:
                                conjecture_generator_module_logger.info("Duplicate definition result: Duplicate definitions found.")
                                errors.append(duplicate_defs)
                                validated_bool = False
                                current_update_dict['duplicate_check'] = {'result':False, 'message':duplicate_defs}
                            else:
                                current_update_dict['duplicate_check'] = {'result':True, 'message':None}

                            current_q_series_dict['update_attempts'].append(current_update_dict)

                            print(f"Updated module validation result: {validated_bool}\n  Validation errors: {errors}\n")
                            if validated_bool and not errors:
                                break
                            else:
                                response_errors.extend(errors)
                                rename_files([new_ewm_path], prefix="COMPILE_FAIL_")

            elif response_type == "none":
                response_errors.append(f"Response error: No context request or final answer detected in response.\n")
                conjecture_generator_module_logger.info(f"Response error: No context request or final answer detected in response: '{message}'\n")
            
            else:
                print(f"Error determining response type for response.\n")
                conjecture_generator_module_logger.warning(f"Error determining response type for response: '{message}'")
                response_errors.append(f"Response error: {response_type}")
            
            if ctx_msg_dict:
                context_prompt = self.retrieve_requested_context(ctx_msg_dict)
                next_prompt +=  "\n\n________________\n\n"+ context_prompt

            if response_errors:
                next_prompt +='\n'.join(response_errors) 
                next_prompt += f"\nTry submitting your answer again. Remember:\n  - Follow the output format of in the system prompt.\n  - Ensure definitions are ordered correctly in the code (e.g. place class definitions above their instance definitions).\n  - Only use triple backticks if they are _within_ a formatted final answer response (e.g. within this part '@>>(173,175):```...```<<@') \n  - When submitting a final answer, boundary numbers are not deleted. Only the lines _between_ the two integer boundary numbers will be deleted. E.g. to replace code on lines 2 and 8, the response format must be '@>>(1,9):```[code here]```<<@'."

            if next_prompt:
                # 'message' var is reassigned here, and will be used in the next loop
                message, prompt_tokens, completion_tokens, total_tokens = self.llm.get_response(next_prompt)
                # Update self.conj_dicts
                current_q_series_dict['conversation']=self.llm.conversation
                current_q_series_dict['tokens']['prompt_tokens'].append(prompt_tokens)
                current_q_series_dict['tokens']['completion_tokens'].append(completion_tokens)
                current_q_series_dict['tokens']['total_tokens'].append(total_tokens)

                save_results(self.conj_dicts_path, self.conj_dicts)

            else:
                conjecture_generator_module_logger.error(f"Something went wrong. No errors found but EWM could not update")
                sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.

            self.num_context_requests -= 1

        # Update the conj_dicts at conj_dicts_path of the base EWM
        try:
            conjecture_generator_module_logger.info(self.conj_dicts)
            save_results(self.conj_dicts_path, self.conj_dicts)
            conjecture_generator_module_logger.info(f"Successfully saved conjecture dictionaries at: {self.conj_dicts_path}")
        except Exception as e:
            conjecture_generator_module_logger.error(f"Unexpected error writing file at {self.conj_dicts_path}: {str(e)}")
            raise

        if not validated_bool:
            conjecture_generator_module_logger.warning(f"Reached maximum update attempts for question path. Try again.")
            print(f"Reached maximum update attempts for question path. Try again.\n")
            return None
               
        elif new_ewm_path:
            # Create a conj_dicts JSON for the new EWM
            new_conj_dicts = {'conj_origin':{'last_EWM':self.ewm_name, 
                                'seed_prompt':None, 
                                'conjecturer':self.conj_llm_name}}
            try:
                new_conj_dicts_path = os.path.join(self.conj_dir, f"{timestamp}_conj_dicts.json")
                save_results(new_conj_dicts_path, new_conj_dicts)
                conjecture_generator_module_logger.info(f"Successfully saved new conjecture dictionaries at: {new_conj_dicts_path}")
            except Exception as e:
                conjecture_generator_module_logger.error(f"Unexpected error writing file at {new_conj_dicts_path}: {str(e)}")
                raise
            return timestamp
        
        else:
            sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.
        
    """
    Functions to retrieve EWM info:
    """
    def retrieve_requested_context(self, ctx_msg_dict):
        """
        Receives a 'ctx_msg_dict' of format:
            {'lines': set(),  # Will store tuples of integers
            'subclasses': set(),
            'instances': set(),
            'attributes': set(),
            'all_functions': False,
            'all_instances': False,
            'all_classes': False,
            'domain': set(),
            'i_def': set(),
            's_def': set(),
            'find': set(),
            'comps': set()}        

        Returns a string with requested info. 
        """
        # Collect context requests if any:
        retrieved_context = ''
    
        if ctx_msg_dict['lines']:
            lines_tuple_list = ctx_msg_dict['lines']
            line_text = find_lines_in_module(self.ewm_path, lines_tuple_list)
            retrieved_context += "---------------\nRETRIEVED LINES\n---------------\n"
            if line_text:
                retrieved_context += "[LINE]\n"
                retrieved_context += line_text
            else:
                retrieved_context += "Requested lines could not be retrieved."
            
        if ctx_msg_dict['subclasses']:
            retrieved_context += "--------------------\nRETRIEVED SUBCLASSES\n--------------------\n"
            found_subclasses = {}
            for search_class in ctx_msg_dict['subclasses']:
                try:
                    if search_class in self.ann_obj_dicts:
                        if 'class_connections' in self.ann_obj_dicts[search_class]:
                            subclasses = self.ann_obj_dicts[search_class]['class_connections']['subclasses']
                            if subclasses:
                                found_subclasses[search_class]=subclasses
                            else:
                                retrieved_context += f"No subclasses found for class `{search_class}`.\n"
                        else:
                            retrieved_context += f"`{search_class}` in module but it is type {self.ann_obj_dicts[search_class]['type']}.\n"
                    else:
                        retrieved_context += f"Class `{search_class}` not found in module.\n"    
                except:
                    conjecture_generator_module_logger.exception("An error occurred")
            for search_class, subclass_list in found_subclasses.items():
                retrieved_context += f"Subclasses of `{search_class}`:"
                retrieved_context += f" {', '.join(subclass_list)}\n"

        if ctx_msg_dict['instances']:
            retrieved_context += "-------------------\nRETRIEVED INSTANCES\n-------------------\n"
            found_instances = {}
            for search_class in ctx_msg_dict['instances']:
                try:
                    if search_class in self.ann_obj_dicts:
                        if 'class_connections' in self.ann_obj_dicts[search_class]:
                            instances = self.ann_obj_dicts[search_class]['class_connections']['instances']
                            if instances:
                                found_instances[search_class]=instances
                            else:
                                retrieved_context += f"No instances found for class `{search_class}`\n"
                        else:
                            retrieved_context += f"`{search_class}` in module but it is type {self.ann_obj_dicts[search_class]['type']}.\n"
                    else:
                        retrieved_context += f"Class `{search_class}` not found in module.\n"      
                except:
                    conjecture_generator_module_logger.exception("An error occurred")
            for search_class, instance_list in found_instances.items():
                retrieved_context += f"Instances of class `{search_class}`:"
                retrieved_context += f" {', '.join(instance_list)}\n"

        if ctx_msg_dict['attributes']:
            retrieved_context += "--------------------\nRETRIEVED ATTRIBUTES\n--------------------\n"
            found_attributes = {}
            for search_type in ctx_msg_dict['attributes']:
                found_attributes[search_type]=[]
                for obj_name, obj_info in self.ann_obj_dicts.items():
                    if obj_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                        continue
                    try:
                        if 'type' in obj_info:
                            if obj_info['type'] == 'attribute':
                                if obj_info['attribute_type'] == search_type:
                                    found_attributes[search_type].append(obj_name)      
                    except:
                        conjecture_generator_module_logger.exception("An error occurred")
            for search_type, attr_list in found_attributes.items():
                if attr_list:
                    retrieved_context += f"Attribute type `{search_type}`:"
                    retrieved_context += f" {', '.join(attr_list)}\n"
                else:
                    retrieved_context += f"No attributes of type `{search_type}` found.\n"

        if ctx_msg_dict['all_functions']:
            retrieved_context += "-------------\nALL FUNCTIONS\n-------------\n"
            func_list = self.ann_obj_dicts[DEFINED_OBJECTS_PLACEHOLDER]['in_module']['function_fq_names']
            if func_list:
                retrieved_context += f"{(', '.join(func_list))}\n"
            else:
                retrieved_context += "No funcitons found in module.\n"
            
        if ctx_msg_dict['all_instances']:
            retrieved_context += "-------------\nALL INSTANCES\n-------------\n"
            found_instances = {}
            for obj_name, obj_info in self.ann_obj_dicts.items():
                if obj_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                    continue
                try:
                    if 'type' in obj_info:
                        if obj_info['type'] == 'class':
                            found_instances[obj_name]=obj_info['class_connections']['instances']     
                except:
                    conjecture_generator_module_logger.exception("An error occurred")
            retrieved_context += "Dictionary with class keys and instance values:\n"
            retrieved_context += json.dumps(found_instances, indent=2)
            retrieved_context += "\n"

        if ctx_msg_dict['all_classes']:
            retrieved_context += "-----------\nALL CLASSES\n-----------\n"
            class_list = self.ann_obj_dicts[DEFINED_OBJECTS_PLACEHOLDER]['in_module']['class_fq_names']                   
            if class_list:
                retrieved_context += f"{(', '.join(class_list))}\n"
            else:
                retrieved_context += "No classes found in module.\n"

        if ctx_msg_dict['domain']:
            retrieved_context += "-----------------\nRETRIEVED DOMAINS\n-----------------\n"
            compatible_functions={}
            for arg_type in ctx_msg_dict['domain']:
                compatible_functions[arg_type]={'accept_class':[], 'accept_any':[]}
                # Find compatible funcitons
                for func_name, func_info in self.ann_obj_dicts.items():
                    if func_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                        continue
                    elif func_info['type'] == 'function':
                        if arg_type in func_info['arg_types']: 
                            compatible_functions[arg_type]['accept_class'].append(func_name)
                        elif ANY_TYPE_PLACEHOLDER in func_info['arg_types']:
                            compatible_functions[arg_type]['accept_any'].append(func_name)
            # Generate prompt
            for arg_type, found_dict in compatible_functions.items():
                found_funcs = False
                if found_dict['accept_class']:
                    found_funcs = True
                    retrieved_context +=f"Functions accepting `{arg_type}`: {(', '.join(found_dict['accept_class']))}\n"
                if found_dict['accept_any']:
                    found_funcs = True
                    rand_sample_len = min(16, len(found_dict['accept_any']))
                    rand_func_sample = random.sample(found_dict['accept_any'],rand_sample_len) 
                    retrieved_context +=f"Functions accepting any type (random sample): {(', '.join(rand_func_sample))}\n"
                if not found_funcs:
                    retrieved_context +=f"No functions accepting '{arg_type}' found.\n" 

        if ctx_msg_dict['i_def']:
            retrieved_context += "-----------------------\nINITIALIZED DEFINITIONS\n-----------------------\n"
            for fq_name in ctx_msg_dict['i_def']:
                initialized_def_text = self.generate_initialized_def_text(fq_name)
                retrieved_context += initialized_def_text+"\n"

        if ctx_msg_dict['s_def']:
            retrieved_context += "------------------\nSTATIC DEFINITIONS\n------------------\n"
            for fq_name in ctx_msg_dict['s_def']:
                static_def_text = self.generate_static_def_text(fq_name)
                retrieved_context += static_def_text
        
        if ctx_msg_dict['comps']:
            retrieved_context += "--------------------\nEXAMPLE COMPUTATIONS\n--------------------\n"
            for search_obj in ctx_msg_dict['comps']:
                false_examples_text = self.generate_false_examples(search_obj, sample_size=4)
                retrieved_context += (false_examples_text + "\n")

        if ctx_msg_dict['find']:
            retrieved_context += "-----------\nFIND STRING\n-----------\n"
            list_of_strings_to_find = ctx_msg_dict['find']
            found_strings_text = self.find_strings_in_module(list_of_strings_to_find)
            retrieved_context += found_strings_text

        return retrieved_context

    def generate_static_def_text(self, obj_name):
        """
        Generate static definition prompt section
        """
        static_def_text = ''
        if obj_name in self.ann_obj_dicts.keys():
            static_def_text += (f"Definition of `{obj_name}`:" "\n")
            if 'type' in self.ann_obj_dicts[obj_name]:
                obj_type = self.ann_obj_dicts[obj_name]['type']
                        # If the object is not an attribute type, update static_def with data from fq_name's dict
                if obj_type != 'attribute':
                    if "static_definition" in self.ann_obj_dicts[obj_name]:
                        # Add line #'s and text form each static def dict to the static_def string
                        for def_dict in self.ann_obj_dicts[obj_name]['static_definition']:
                            st_def_lines = [(def_dict['location']['start'][0],def_dict['location']['end'][0])]
                            static_def_text += find_lines_in_module(self.ewm_path, st_def_lines)+"\n"
                    else:
                        static_def_text += f"`{obj_name}` found but no static definition available.\n"
                
                # If the object is an attribute type, update static_def and initialized_def with data from the instance dict               
                if obj_type == 'attribute':
                    for instance_name, instance_info in self.ann_obj_dicts.items():
                        if instance_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                            continue
                        if instance_info['type'] == 'instance':
                            if 'instance_attributes' in instance_info:
                                if obj_name in instance_info['instance_attributes']:
                                    if 'static_definition' in self.ann_obj_dicts[instance_name]:
                                        # Add line #'s and text form each static def dict to the static_def string
                                        for def_dict in self.ann_obj_dicts[instance_name]['static_definition']:
                                            st_def_lines = [(def_dict['location']['start'][0],def_dict['location']['end'][0])]
                                            conjecture_generator_module_logger.info(f"instance_name: {instance_name}")
                                            conjecture_generator_module_logger.info(f"st_def_lines: {st_def_lines}")
                                            static_def_text += find_lines_in_module(self.ewm_path, st_def_lines)+"\n"
                                    else:
                                        static_def_text += f"`{obj_name}` found but no static definition available.\n"
                conjecture_generator_module_logger.info(f"Static definition prompt part generated:\n{static_def_text}")
            else:
                static_def_text += f"`{obj_name}` found but no static definition available.\n"

        else: 
            static_def_text += f"`{obj_name}` not found in module."
        conjecture_generator_module_logger.info(f"Static definition prompt part generated:\n{static_def_text}")

        return static_def_text
    
    def generate_initialized_def_text(self, obj_name):
        """
        Generate initialized definition prompt section
        """
        initialized_def_text=''
        # If there is an intitalized definition, store it as a string 
        if obj_name in self.ann_obj_dicts.keys():
            if 'initialized_definition' in self.ann_obj_dicts[obj_name]:
                initialized_def_text += f'Initialized definition of `{obj_name}`:\n'
                initialized_def_text += json.dumps(self.ann_obj_dicts[obj_name]['initialized_definition'], indent=2)
                initialized_def_text += "\n"
            else:
                try:
                    conjecture_generator_module_logger.info(f"`{obj_name}` found but no initialized definition available. Object is type `{self.ann_obj_dicts[obj_name]['type']}`.")
                except Exception as e:
                    conjecture_generator_module_logger.warning(f"An error occurred: {str(e)}")
        else: 
            conjecture_generator_module_logger.error(f"`{obj_name}` not found in module.\n")
        conjecture_generator_module_logger.info(f"Initialized definition prompt part generated:\n{initialized_def_text}")
        return initialized_def_text

    def generate_class_context(self, fq_name):
        """ 
        Generate context info for the class
        """
        class_instances = self.ann_obj_dicts[fq_name]['class_connections']['instances']
        class_subclasses = self.ann_obj_dicts[fq_name]['class_connections']['subclasses']
        # Get sample of compatible functions for class (the domain of the class)
        compatible_functions = {'accept_class':[], 'accept_any':[]} 
        # Lists of functions that specfically accept the class and functions that accept any type
        for func_name, func_info in self.ann_obj_dicts.items():
            # conjecture_generator_module_logger.info(f"func_name : {func_name}")
            # conjecture_generator_module_logger.info(f"func_info : {func_info}")
            if fq_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                continue
            elif isinstance(func_info, dict) and 'type' in func_info: # This line should be redundant with the 'if' statement above 
                                        # but is was added because, for an unknown reason PRUNED_SCORE_PLACEHOLDER and DEFINED_OBJECTS_PLACEHOLDER keys were not being screened out
                if func_info['type'] == 'function':
                    if fq_name in func_info['arg_types']: 
                        compatible_functions['accept_class'].append(func_name)
                    elif ANY_TYPE_PLACEHOLDER in func_info['arg_types']:
                        compatible_functions['accept_any'].append(func_name)
        class_instances_text = ''
        if class_instances:
            class_instances_text +=f"Instances of class `{fq_name}`: {(', '.join(class_instances))}."
        class_subclasses_text = ''
        if class_subclasses:
            class_subclasses_text +=f"Subclasses of class `{fq_name}`: {(', '.join(class_subclasses))}."
        compatible_functions_text = ''
        if compatible_functions['accept_class']:
            compatible_functions_text +=f"Functions that recieve arguments of type `{fq_name}`: {(', '.join(compatible_functions))}."
        if compatible_functions['accept_any']:
            compatible_functions_text +=f"Functions that recieve arguments of any type: {(', '.join(compatible_functions))}."

        return class_instances_text, class_subclasses_text, compatible_functions_text

    def generate_function_context(self, fq_name):
        # Get sample of compatible instances
        compatible_instances_text = ''
        arg_types = set(self.ann_obj_dicts[fq_name]['arg_types'])
        if not arg_types: 
            compatible_instances_text += 'None\n'
        for arg_type in arg_types:
            if arg_type == ANY_TYPE_PLACEHOLDER:
                rand_sample_len = min(16, len(self.ann_obj_dicts[DEFINED_OBJECTS_PLACEHOLDER]['in_module']['instance_fq_names']))
                rand_inst_sample = random.sample(self.ann_obj_dicts[DEFINED_OBJECTS_PLACEHOLDER]['in_module']['instance_fq_names'], rand_sample_len)
                if rand_inst_sample:
                    compatible_instances_text += f"  - Instances of any type (random sample): "
                    compatible_instances_text += f"{(', '.join(rand_inst_sample))}.\n"
                else:
                    conjecture_generator_module_logger.warning("Unexpected empty 'rand_inst_sample'.")
            elif arg_type in self.ann_obj_dicts:
                compatible_instances_text += f"  - Instances of class `{arg_type}`: "
                type_instances = self.ann_obj_dicts[arg_type]['class_connections']['instances']
                compatible_instances_text += f"{(', '.join(type_instances))}.\n"
            else:
                compatible_instances_text += f"  - Instances of class `{arg_type}`: "
                compatible_instances_text += "No instances found.\n"
        return compatible_instances_text

    def generate_instance_context(self, fq_name):
        # Get some compatible functions and other instances of the same class
          # Possible upgrade: get some other instances with the same name component
        parent_class = None
        sibling_instances = []
        # Lists of functions that specfically accept the class and functions or accept any type
        compatible_functions = {'accept_class':[], 'accept_any':[]} 
        # Find the name of the parent class
        for class_name, class_info in self.ann_obj_dicts.items():
            if class_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                continue
            elif class_info['type'] == 'class':
                if fq_name in class_info['class_connections']['instances']:
                    parent_class = class_name
        # Get sibling instances
        if parent_class:
            sibling_instances = self.ann_obj_dicts[parent_class]['class_connections']['instances']
            # Remove the problem instance name from the sibling instance list
            if fq_name in sibling_instances:
                sibling_instances.remove(fq_name)

            # Find compatible funcitons
            for func_name, func_info in self.ann_obj_dicts.items():
                if func_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                    continue
                elif func_info['type'] == 'function':
                    if parent_class in func_info['arg_types']: 
                        compatible_functions['accept_class'].append(func_name)
                    elif ANY_TYPE_PLACEHOLDER in func_info['arg_types']:
                        compatible_functions['accept_any'].append(func_name)

        sibling_instances_text = ''
        if sibling_instances:
            sibling_instances_text +=f"Other instances of type `{parent_class}`: {(', '.join(sibling_instances))}."
        else:
            sibling_instances_text += f"No other instances of class '{parent_class}'."
        
        compatible_functions_text = ''
        if compatible_functions['accept_class']:
            compatible_functions_text +=f"Functions that can receive `{fq_name}` as an argument: {(', '.join(compatible_functions['accept_class']))}."
        elif compatible_functions['accept_any']:
            compatible_functions_text +=f"Functions that can receive any type of argument: {(', '.join(compatible_functions['accept_any']))}."
        else:
            compatible_functions_text +=f"No compatible functions."  
        
        return sibling_instances_text, compatible_functions_text

    def generate_attribute_context(self, fq_name):
       # Get some compatible functions and other examples of that attribute
        parent_instance = None
        parent_class = None
        sibling_instances = []
        sibling_attributes = []
        # Lists of functions that specfically accept the class and functions that accept any type
        compatible_functions = {'accept_class':[], 'accept_any':[]} 

        # Find the name of the parent instance
        for inst_name, inst_info in self.ann_obj_dicts.items():
            if inst_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                continue
            elif inst_info['type'] == 'instance':
                if fq_name in inst_info['instance_attributes']:
                    parent_instance = inst_name
        if parent_instance:
            # Find the name of the parent class
            for class_name, class_info in self.ann_obj_dicts.items():
                if class_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                    continue
                elif class_info['type'] == 'class':
                    if parent_instance in class_info['class_connections']['instances']:
                        parent_class = class_name
        else: 
            conjecture_generator_module_logger.warning(f"No parent instance found for attribute '{fq_name}")
        # Get sibling instances
        if parent_class:
            sibling_instances = self.ann_obj_dicts[parent_class]['class_connections']['instances']
            # Remove the problem instance name from the sibling instance list
            if fq_name in sibling_instances:
                sibling_instances.remove(fq_name)
        else: 
            conjecture_generator_module_logger.warning(f"No parent class found for parent instance of attribute '{fq_name}")
        if sibling_instances:
            # Get sibling attributes
            common_inst_path_len = len(parent_instance.split(".")) # Path length of sibling_instances
            problem_attr_type_path = ".".join(fq_name.split(".")[common_inst_path_len:]) # The path to the problem attribute with parent instance path removed
            for instance in sibling_instances:
                inst_attributes = self.ann_obj_dicts[instance]['instance_attributes']
                if inst_attributes:
                    for inst_attr in inst_attributes:
                        attr_type_path = ".".join(inst_attr.split(".")[common_inst_path_len:]) # The path to the sibling attribute with parent instance path removed
                        if attr_type_path == problem_attr_type_path:
                            sibling_attributes.append(inst_attr)
        else: conjecture_generator_module_logger.info(f"No sibling instances class found for parent instance of attribute '{fq_name}")

        attr_type=self.ann_obj_dicts[fq_name]['attribute_type'] # This key should have custom type or default type (e.g. int) info

        # Find compatible funcitons
        for func_name, func_info in self.ann_obj_dicts.items():
            if fq_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                continue
            elif func_info['type'] == 'function':
                if attr_type in func_info['arg_types']: 
                    compatible_functions['accept_class'].append(func_name)
                elif ANY_TYPE_PLACEHOLDER in func_info['arg_types']:
                    compatible_functions['accept_any'].append(func_name)

        sibling_attributes_text = ''
        if sibling_attributes:
            sibling_attributes_text +=f"Other attributes of type `{attr_type}`: {(', '.join(sibling_attributes))}."
        else: conjecture_generator_module_logger.info(f"No sibling attributes found for attribute '{fq_name}")
        
        compatible_functions_text = ''
        if compatible_functions['accept_class']:
            compatible_functions_text +=f"Functions that can receive `{fq_name}` as an argument: {(', '.join(compatible_functions['accept_class']))}."
        elif compatible_functions['accept_any']:
            rand_sample_len = min(16, len(compatible_functions['accept_any']))
            rand_func_sample = random.sample(compatible_functions['accept_any'],rand_sample_len) 
            compatible_functions_text +=f"Functions that can receive any type of argument (random sample): {(', '.join(rand_func_sample))}."
        else:
            compatible_functions_text +=f"No compatible functions."  
        
        return sibling_attributes_text, compatible_functions_text

    def generate_false_examples(self, fq_name, sample_size=4):
        """
        Generate the 'false_examples' prompt section
        """
        false_examples_text = ''
        conjecture_generator_module_logger.info(f"self.shared_obs_ids: {self.shared_obs_ids}")

        # if not fq_name in self.ann_obj_dicts:
        #     return f"Unable to find `{fq_name}` defined in module.\n"
        
        def collect_false_predictions(obj_name):
            inconsistent_obs_dicts = {}
            # Collect all predictions observed as 'False'
            for obs_dict in self.obs_dicts:
                if obs_dict['prediction_id'] in self.shared_obs_ids:
                    continue # Skip observations that have been shared already
                if obj_name in obs_dict['all_objects']:
                    if 'pco_bool' in obs_dict and obs_dict['pco_bool'] == False:
                        conjecture_generator_module_logger.info(f"Found 'False' observation of '{obj_name}': {obs_dict}\n")
                        if 'false' in inconsistent_obs_dicts:
                            inconsistent_obs_dicts['false'].append(obs_dict)
                        else:
                            inconsistent_obs_dicts['false']=[obs_dict]
                    elif 'pco_bool' in obs_dict and obs_dict['pco_bool'] == None:
                        conjecture_generator_module_logger.info(f"Found 'None' observation of '{obj_name}': {obs_dict}\n")
                        if 'none' in inconsistent_obs_dicts:
                            inconsistent_obs_dicts['none'].append(obs_dict)
                        else:
                            inconsistent_obs_dicts['none']=[obs_dict]
                    elif 'computation_error' in obs_dict:
                        conjecture_generator_module_logger.info(f"Found 'computation_error' observation of '{obj_name}': {obs_dict}\n")
                        if 'error' in inconsistent_obs_dicts:
                            inconsistent_obs_dicts['error'].append(obs_dict)
                        else:
                            inconsistent_obs_dicts['error']=[obs_dict]
            return inconsistent_obs_dicts

        """
        Evenly sample items from lists in a dictionary, redistributing remaining samples
        when lists are emptied.
        """
        def sample_inconsistent_obs(inconsistent_obs_dicts, false_examples_text, sample_size):
            # Create working copy and result dictionary
            remaining_items = {k: list(v) for k, v in inconsistent_obs_dicts.items()}
            obs_sample_dict = {k: [] for k in inconsistent_obs_dicts}
            
            while sample_size > 0 and remaining_items:
                # Calculate samples per list for this round
                active_lists = len(remaining_items)
                samples_per_list = math.ceil(sample_size / active_lists)
                
                # Process each remaining list
                empty_keys = []
                for key in remaining_items:
                    # Calculate actual samples for this list
                    actual_samples = min(samples_per_list, len(remaining_items[key]), sample_size)
                    
                    # Take samples
                    key_sample_items = random.sample(remaining_items[key], actual_samples)
                    for item in key_sample_items:
                        remaining_items[key].remove(item)
                        obs_sample_dict[key].append(item)
                        sample_size -= 1
                                                    
                    # Track empty lists
                    if not remaining_items[key]:
                        empty_keys.append(key)
                
                # Remove empty lists
                for key in empty_keys:
                    del remaining_items[key]

            obs_sample_list = [item for sublist in obs_sample_dict.values() for item in sublist]

            # Optionally shuffle the final list of inconsistent observations
            random.shuffle(obs_sample_list)

            return obs_sample_list, false_examples_text, sample_size
        
        inconsistent_obs_dicts = collect_false_predictions(fq_name)
        num_inconsistent_obs_available = sum(len(obs_list) for obs_list in inconsistent_obs_dicts.values())
        if num_inconsistent_obs_available == 1:
            false_examples_text += f"There is {num_inconsistent_obs_available} unexpected or incorrect computation that included `{fq_name}`. "
        else:
            false_examples_text += f"There are {num_inconsistent_obs_available} unexpected or incorrect computations that included `{fq_name}`. "

        obs_sample_list, false_examples_text, sample_size = sample_inconsistent_obs(inconsistent_obs_dicts, false_examples_text, sample_size)

        if not obs_sample_list:
            inconsistent_obs_dicts = {}
            # If no obs_samples are found, check if there are obs_samples for sub-objects (instances if class, attributes if instance)
            if self.ann_obj_dicts[fq_name]['type'] == "instance":
                observed_attrs = [] # Track attribute names that have false observations
                screened_attrs = [] # Track attribute names that have been screened (because ann_obj_dicts only stores attributes with static defintions - a subset initialized and accessed attributes)
                for attribute in self.ann_obj_dicts[fq_name]['instance_attributes']:
                    screened_attrs.append(attribute)
                    attr_obs_dicts = collect_false_predictions(attribute)
                    for obs_type, obs_dict_list in attr_obs_dicts.items():
                        if obs_dict_list:
                            observed_attrs.append(attribute)
                            # Merge the attr_obs_dicts with inconsistent_obs_dicts
                            inconsistent_obs_dicts[obs_type] = inconsistent_obs_dicts.get(obs_type, []) + obs_dict_list
                obs_sample_list, false_examples_text, sample_size = sample_inconsistent_obs(inconsistent_obs_dicts, false_examples_text, sample_size)
                num_inconsistent_obs_available = sum(len(obs_list) for obs_list in inconsistent_obs_dicts.values())
                if num_inconsistent_obs_available == 1:
                    false_examples_text += f"  There is {num_inconsistent_obs_available} unexpected or incorrect computation that included attributes of `{fq_name}`. "
                else:
                    false_examples_text += f"  There are {num_inconsistent_obs_available} unexpected or incorrect computations that included attributes of `{fq_name}`. "
                if screened_attrs:
                    false_examples_text += (f"Searched attributes included: "+', '.join(screened_attrs)+". More specific attributes, if they exist, were not included in search.")
                if observed_attrs:
                    false_examples_text += (f"Computations were found for: "+', '.join(screened_attrs)+".")
                false_examples_text += "\n"
            if self.ann_obj_dicts[fq_name]['type'] == "class":              
                for instance in self.ann_obj_dicts[fq_name]['class_connections']['instances']:
                    inst_obs_dicts = collect_false_predictions(instance)
                    for obs_type, obs_dict_list in inst_obs_dicts.items():
                        if obs_dict_list:
                            inconsistent_obs_dicts[obs_type] = inconsistent_obs_dicts.get(obs_type, []) + obs_dict_list
                for method in self.ann_obj_dicts[fq_name]['class_connections']['methods']:
                    meth_obs_dicts = collect_false_predictions(method)
                    for obs_type, obs_dict_list in meth_obs_dicts.items():
                        if obs_dict_list:
                            inconsistent_obs_dicts[obs_type] = inconsistent_obs_dicts.get(obs_type, []) + obs_dict_list            
                obs_sample_list, false_examples_text, sample_size = sample_inconsistent_obs(inconsistent_obs_dicts, false_examples_text, sample_size)
                num_inconsistent_obs_available = sum(len(obs_list) for obs_list in inconsistent_obs_dicts.values())
                if num_inconsistent_obs_available == 1:
                    false_examples_text += f"  There is {num_inconsistent_obs_available} unexpected or incorrect computation that included instances or methods of `{fq_name}`. "
                else:
                    false_examples_text += f"  There are {num_inconsistent_obs_available} unexpected or incorrect computations that included instances or methods of `{fq_name}`. "

        # Use 'inconsistent_obs_sample' to generate the prompt section
        if num_inconsistent_obs_available != 0:
            false_examples_text += f"Here are some examples (which have not been reviewed for accuracy): \n"
        for obs_dict in obs_sample_list:
            self.shared_obs_ids.append(obs_dict['prediction_id']) # Update the list of shared computations
            inputs = get_obs_inputs(obs_dict)
            function = get_obs_func_name(obs_dict)
            prompt_part = "  Argument input(s): "+', '.join(inputs)+f"\n  Function name: {function}\n"
            if fq_name not in (inputs, function):
                other_accessed_objects = []
                if 'accessed_objects' in obs_dict:
                    other_accessed_objects = get_other_accessed_objects(obs_dict)
                    if fq_name in other_accessed_objects:
                        prompt_part += f"  * '{fq_name}' was accessed by '{function}' during the computation.\n"
                        # Optionally share the access graph 'obs_dict['accessed_objects']
                    else:
                        conjecture_generator_module_logger.warning(f"Unable to find problem object '{fq_name}' in inputs, function, or accessed objects of observation in 'inconsistent_obs_sample'.\n{json.dumps(obs_dict, indent=2)}")
                else: 
                    conjecture_generator_module_logger.warning(f"Unable to find problem object '{fq_name}' in inputs or function, and no 'accessed objects' key for observation in 'inconsistent_obs_sample'.\n{json.dumps(obs_dict, indent=2)}")
            if 'result' in obs_dict:
                prompt_part += f"  Result: {json.dumps(obs_dict['result'], indent=2)}\n"
            elif 'computation_error' in obs_dict:
                prompt_part += "  Result: "+', '.join(obs_dict['computation_error'])+"\n"
            else:
                conjecture_generator_module_logger.warning(f"Unable to find result or error message for observation of problem object '{fq_name}': \n{json.dumps(obs_dict, indent=2)}")            
            false_examples_text += prompt_part
            false_examples_text += "\n"

        return false_examples_text
        
    def find_strings_in_module(self, search_strings):
        """
        Search for strings in a Python module and return context around each occurrence.
        
        Args:
            search_strings (list): List of strings to search for
        
        Returns:
            str: Formatted string containing search results
        """
        # Initialize results dictionary
        results = {s: set() for s in search_strings}
        
        try:
            # Read all lines from the file
            with open(self.ewm_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            # For each search string
            for search_str in search_strings:
                # Check each line for the search string
                lines_found = []
                
                for i, line in enumerate(lines, start=1):
                    if search_str in line:
                        context = ""
                        lines_found.append(i)
                        # Get previous, current, and next lines
                        prev_line = lines[i-2] if i > 1 else None
                        next_line = lines[i] if i < len(lines) else None
                        if not i-2 in lines_found and not i-1 in lines_found:
                            # Only add previous line if it was not already found
                            prev_line_no = i-1
                            prev_line_no_chars = len(str(abs(prev_line_no)))
                            prev_line_no_char_to_fill = 4-prev_line_no_chars
                            if prev_line_no_char_to_fill > 0:
                                prev_line_no_str = "0"*prev_line_no_char_to_fill+str(prev_line_no)
                            else:
                                prev_line_no_str = str(prev_line_no)
                            if prev_line:
                                context += f"[{prev_line_no_str}]{prev_line}"
                        # Add the found line
                        if not  i-1 in lines_found:
                            line_no_chars = len(str(abs(i)))
                            line_no_char_to_fill = 4-line_no_chars
                            if line_no_char_to_fill > 0:
                                line_no_str = "0"*line_no_char_to_fill+str(i)
                            else:
                                line_no_str = str(i)
                            context += f"[{line_no_str}]{line}"
                        # Add the next line
                        next_line_no = i+1
                        next_line_no_chars = len(str(abs(next_line_no)))
                        next_line_no_char_to_fill = 4-next_line_no_chars
                        if next_line_no_char_to_fill > 0:
                            next_line_no_str = "0"*next_line_no_char_to_fill+str(next_line_no)
                        else:
                            next_line_no_str = str(next_line_no)
                        if next_line:
                            context += f"[{next_line_no_str}]{next_line}"      

                        # Add the tuple to the set
                        results[search_str].add((i, context))
            
            # Format the results
            output = []
            for search_str, findings in results.items(): 
                example_counter = 32
                occurences_found = len(findings)   
                num_sharable = min(example_counter, occurences_found)
                if not findings:
                    output.append(f"`{search_str}` not found in module.\n\n")
                else:
                    # Create sorted list of line numbers for summary
                    found_lines = sorted([line_num for line_num, _ in findings])
                    if occurences_found == 1:
                        output.append(f"`{search_str}` found on 1 line: {', '.join(map(str, found_lines))}\n")
                    else:
                        if occurences_found>example_counter:
                            output.append(f"`{search_str}` found on {occurences_found} lines: {', '.join(map(str, found_lines))}\nLines around first {num_sharable} examples:\n")
                        else:
                            output.append(f"`{search_str}` found on {occurences_found} lines: {', '.join(map(str, found_lines))}\nLines around each found string:\n")
                    # Add each context block
                    output += "[LINE]\n"
                    last_line=None
                    for line_num, context in sorted(findings):
                        if num_sharable > 0:
                            # Calculate the line range for context
                            if last_line is not None and last_line != line_num + 1:
                                output.append("[LINE BREAK]\n")
                            output.append(context)
                            num_sharable -= 1
                            last_line = line_num
                    output.append("\n")  # Empty line between different search results
            
            return "".join(output)
        
        except FileNotFoundError:
            conjecture_generator_module_logger.error(f"'find_strings_in_module' function could not find file at {self.ewm_path}")
        except Exception as e:
            conjecture_generator_module_logger.error(f"Error occurred while processing file: {str(e)}")
    
    def collect_max_score_defs(self,num_defs=4):
        """
        Manages a list of dictionaries, keeping the 3 dictionaries with highest key values.
        
        Args:
            dict_list: List of dictionaries to maintain (target length of 3)
            dict_dicts: Dictionary of dictionaries to process
        """
        max_score_defs_text = ''
        max_score_objs_set = set()

        for fq_name, obj_info in self.ann_obj_dicts.items():
            total_score = 0
            if fq_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                continue
            
            if 'static_definition' in obj_info:
                for static_def in obj_info['static_definition']:
                    if 'adjusted_length' in static_def:
                        total_score += static_def['adjusted_length']
        
            if len(max_score_objs_set) < num_defs:
                # If list not full, add the current dictionary
                  # Dicts are added as two item lists, where total_score is the first item
                current_dict = {fq_name:obj_info}
                max_score_objs_set.add((total_score, fq_name))
            else:
                # Find tuple with lowest key value in dict_list
                lowest_tuple = min(max_score_objs_set, key=lambda x: x[0])
                min_value = lowest_tuple[0]
                # If current dictionary has higher key value, replace the minimum
                if total_score > min_value:
                    max_score_objs_set.remove(lowest_tuple)
                    max_score_objs_set.add((total_score,fq_name))
        
        for item in max_score_objs_set:
            fq_name = item[1]
            static_def_text = self.generate_static_def_text(fq_name)
            max_score_defs_text += f"{static_def_text}"

        return max_score_defs_text

    def find_uncovered_lines(self, module_path):
        """
        Find lines in a Python module that aren't covered by any of the ranges in the static definitions.
        
        Args:
            module_path (str): Path to the Python module file
            ranges (list): List of dictionaries containing 'start' and 'end' positions
                        where each position is [line_no, col_offset]
        
        Returns:
            list: List of tuples (line_number, line_content) for uncovered lines
        """
        # Read all lines from the file
        with open(module_path, 'r') as file:
            module_lines = file.readlines()
        
        # Create a set of all line numbers that are covered by ranges
        covered_lines = set()
        for fq_name, obj_info in self.ann_obj_dicts.items():
            if fq_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                continue
            elif 'static_definition' in obj_info:
                for static_def in obj_info['static_definition']:
                    start_line = static_def['location']['start'][0]
                    end_line = static_def['location']['end'][0]
                    # Add all line numbers in this range to covered_lines
                    for line_num in range(start_line, end_line + 1):
                        covered_lines.add(line_num)
        
        # Find uncovered lines that aren't empty
        uncovered_lines = []
        for line_num, line_content in enumerate(module_lines, start=1):
            if line_num not in covered_lines and line_content.strip():
                uncovered_lines.append((line_num, line_content.strip()))
        
        return uncovered_lines
    
    """
    Functions to generate first prompt
    """
    def generate_prob_obj_prompt(self):
        initialized_def_text = self.generate_initialized_def_text(self.prob_obj_name)
        static_def_text = self.generate_static_def_text(self.prob_obj_name)
        obj_type_context = ""  # Depending on object type, list context info: instances, subclasses, compatible functions, etc."
        false_examples_text = self.generate_false_examples(self.prob_obj_name)  # Randomly sampled (without replacement) examples or false/error results

        if self.prob_obj_type == 'class':
            class_instances_text, class_subclasses_text, compatible_functions_text = self.generate_class_context(self.prob_obj_name)
            if class_instances_text:
                obj_type_context += f"  - {class_instances_text}\n"
            if class_subclasses_text:
                obj_type_context += f"  - {class_subclasses_text}\n"
            if compatible_functions_text:
                obj_type_context += f"  - {compatible_functions_text}\n"

        if self.prob_obj_type == "function":
            compatible_instances_text = self.generate_function_context(self.prob_obj_name)
            if compatible_instances_text:
                obj_type_context += compatible_instances_text
        
        if self.prob_obj_type == "instance":
            sibling_instances_text, compatible_functions_text = self.generate_instance_context(self.prob_obj_name)
            if sibling_instances_text:
                obj_type_context += f"  - {sibling_instances_text}\n"
            if compatible_functions_text:
                obj_type_context += f"  - {compatible_functions_text}\n"
        
        if self.prob_obj_type == "attribute":
            sibling_attributes_text, compatible_functions_text = self.generate_attribute_context(self.prob_obj_name)
            if sibling_attributes_text:
                obj_type_context += f"  - {sibling_attributes_text}\n"
            if compatible_functions_text:
                obj_type_context += f"  - {compatible_functions_text}\n"

        comp_prob_text = ''
        if self.prob_obj_obs_type == 'no_pred':
            comp_prob_text += f"I could not successfully generate computations with `{self.prob_obj_name}`. "
        elif self.prob_obj_obs_type == 'no_true_obs':
            comp_prob_text += f"None of the predictions generated with `{self.prob_obj_name}` make sense. "
        else: # if self.prob_obj_obs_type = 'true_obs'
            comp_prob_text += f"Some of the predictions generated with `{self.prob_obj_name}` do not make sense. "
        
        # Complete the prob_obj_prompt.
        prob_obj_user_prompt = (
            f"Help me improve my Python module. {comp_prob_text}"
            f"Here is more information about the object:\n"
            f"Static definition (original definition and references): \n{static_def_text}"
            f"{initialized_def_text if initialized_def_text else ''}\n"
            f"Other objects related to `{self.prob_obj_name}` include:\n" 
            f"{obj_type_context}"
            "\n"
            f"{'Sometimes computations from this module produce unrealistic, unexpected, or incorrect results. ' + false_examples_text}\n"
            f"For reference, below is the complete code:\n[LINE]\n{find_lines_in_module(self.ewm_path, [(1,self.total_lines)])}"
            )

        return prob_obj_user_prompt

    def generate_rejected_problem_obj_prompt(self):
        """
        Just add something like: 
        "You have attempted this _ times:
            Attempt _ was to update with "_" 
            Attempt _ failed because "[evaluation result dict]...
            ...
        """

        prompt = ''
        base_prompt = self.generate_prob_obj_prompt()
        hist_prompt = '\n---------------------\nPRIOR UPDATE ATTEMPTS\n---------------------\n'

        num_prev_ques = len(self.conj_dicts['question_series'])
        hist_prompt += f"You have previously attempted to update the code {num_prev_ques} time(s). Updates included:\n"

        for i, question_dict in enumerate(self.conj_dicts['question_series'], start=1):
            update_rejection_reason = ''
            last_update_id = max(update_attempt['id'] for update_attempt in question_dict['update_attempts'])
            conjecture_generator_module_logger.info(f"last_update_id: {last_update_id}")
            for update_attempt in question_dict['update_attempts']:
                if update_attempt['id'] == last_update_id:
                    last_update_attempt = update_attempt
                    conjecture_generator_module_logger.info(f"last_update_attempt: {json.dumps(last_update_attempt, indent=2)}")
            if 'conj_eval_result' in last_update_attempt:
                try:
                    accept_update = last_update_attempt['conj_eval_result']['accepted_bool']
                    consistency_decreased = last_update_attempt['conj_eval_result']['summary']['consistency_decreased']
                    consistency_increased = last_update_attempt['conj_eval_result']['summary']['consistency_increased']
                    organization_increased = last_update_attempt['conj_eval_result']['summary']['organization_increased']
                    if accept_update:
                        conjecture_generator_module_logger.error(f"Unexpected behavior. Prior update was accepted but generate_rejected_problem_obj_prompt was called for this EWM: {self.ewm_name}")
                        sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.
                    if consistency_decreased:
                        update_rejection_reason += f"  - The last update resulted in one or more defined objects producing a higher proportion of unexpected or unrealistic computation results.\n"
                    elif not consistency_increased:
                        update_rejection_reason += f"  - The last update did not increase the number of realistic computation results.\n"
                        if not organization_increased:
                            update_rejection_reason += f"  - The last update resulted in no improvement in organization of the code.\n" 
                except Exception as e:
                    conjecture_generator_module_logger.error(f"Error: {e}")
                    conjecture_generator_module_logger.error(f"Error retrieving last update in generate_rejected_problem_obj_prompt. Exiting.")
                    sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.
                if not update_rejection_reason:
                    conjecture_generator_module_logger.error(f"Error parsing update rejection reason for last update to EWM '{self.ewm_name}' in generate_rejected_problem_obj_prompt. \nquestion_series{i}\nlast_update_id{last_update_id}\naccept_update{accept_update}\nconsistency_decreased{consistency_decreased}\nall(dimensionality_increased, consistency_increased, organization_improved){all(consistency_increased, organization_increased)}")
                    sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.
                hist_prompt += (f"Attempt {i}:\n"
                    f"{json.dumps(last_update_attempt['update_dict'], indent=2)}\n"
                    f"{update_rejection_reason}\n\n")
                continue

        prompt = base_prompt + hist_prompt
        return prompt

    def generate_organization_prompt(self):
        """
        Generate a prompt with information the the LLM can use to improve organization of the EWM.
        """
        def sample_list(list, max_size = 32):
            if not list:  # Handle empty list
                return []
        
            # Take min of max_size and list length to avoid ValueError
            sample_size = min(len(list), max_size)
            return random.sample(list, sample_size)
        
        max_score_defs_text = self.collect_max_score_defs(num_defs=4)
        
        # Collect lines from code outside all static defs, which are not empty
        unscored_lines = self.find_uncovered_lines(self.ewm_path)
        unscored_lines_text = ''

        if unscored_lines:
            unscored_lines_text += "Here are some _possibly_ superfluous lines from the module (these have not been reviewed for accuracy):\n"
            unscored_lines_text += "[LINE]\n"
            conjecture_generator_module_logger.info(f"unscored_lines:\n{unscored_lines}")
            for line_no, content in unscored_lines:
                line_no_chars = len(str(abs(line_no)))
                no_char_to_fill = 4-line_no_chars
                if no_char_to_fill > 0:
                    line_no_str = "0"*no_char_to_fill+str(line_no)
                else:
                    line_no_str = str(line_no)
                unscored_lines_text += f"[{line_no_str}]{content}\n"

        func_name_list = self.ann_obj_dicts[DEFINED_OBJECTS_PLACEHOLDER]['in_module']['function_fq_names']
        outside_obj_name_list = self.ann_obj_dicts[DEFINED_OBJECTS_PLACEHOLDER]['outside_module']
        func_name_sample = sample_list(func_name_list)
        outside_obj_name_sample = sample_list(outside_obj_name_list)
        
        func_name_sample_text = ", ".join(func_name_sample)
        outside_obj_name_sample_text = ", ".join(outside_obj_name_sample)

        org_user_prompt = (
            f"Here are a few static definitions (line numbers bracketed on left) of specific objects that may benefit from better organization:\n{max_score_defs_text}"
            f"{unscored_lines_text}\n\n"
            f"Here is a list of some objects used in the module but defined outside it: {outside_obj_name_sample_text}.\n\n\n"
            f"Here is a random selection of function and method names from the module: {func_name_sample_text}.\n\n\n"
            f"You may update any part of the code, you do not need to update the specific definitions provided above.\n"
            f"For reference, below is the complete code:\n[LINE]\n{find_lines_in_module(self.ewm_path, [(1,self.total_lines)])}"
            )
        
        return org_user_prompt

    def generate_variety_prompt(self):
        """
        Generate a prompt with information the the LLM can use to improve variety of the EWM.
        """        
        # Generate text to introduce the instances and classes
        all_instances_text = "Below are names of classes and instances.\n"
        found_instances = {}
        for obj_name, obj_info in self.ann_obj_dicts.items():
            if obj_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                continue
            try:
                if 'type' in obj_info:
                    if obj_info['type'] == 'class':
                        found_instances[obj_name]=obj_info['class_connections']['instances']     
            except:
                conjecture_generator_module_logger.exception("An error occurred")
        all_instances_text += json.dumps(found_instances, indent=2)
        all_instances_text += "\n\n"

        # Generate text to introduce the functions
        all_funcs_text = "Below are the names of all functions and methods defined in the module:\n"
        func_list = self.ann_obj_dicts[DEFINED_OBJECTS_PLACEHOLDER]['in_module']['function_fq_names']
        if func_list:
            all_funcs_text += f"{(', '.join(func_list))}\n\n"
        else:
            all_funcs_text += "  No funcitons found in module.\n\n"

        variety_user_prompt = (
            f"Help improve my Python module by adding one or more new definitions.\n"
            f"{all_instances_text}"
            f"{all_funcs_text}"
            f"You may update any part of the code.\n"
            f"For reference, below is the complete code:\n[LINE]\n{find_lines_in_module(self.ewm_path, [(1,self.total_lines)])}"
            )
        
        return variety_user_prompt

    """
    Read response from LLM
    """
    def detect_response_type(self, input_string):
        """
        Analyzes a string for specific patterns while ignoring content within triple backticks.
        Looks for context_request ('<<' followed by '>>') and final_answer ('>>' followed by '<<') patterns.
        
        Args:
            input_string (str): The input string to analyze
            
        Returns:
            str: 'context_request', 'final_answer', 'both', or 'none' based on patterns found
                Or error message if patterns are incomplete
        """        
        # Initialize variables
        i = 0
        found_context_request = False
        found_final_answer = False
        
        while i < len(input_string):
            if i + 1 < len(input_string):
                current = input_string[i:i+2]
                
                if current == '<<':
                    # Look for matching >>
                    j = i + 2
                    found_close = False
                    while j + 1 < len(input_string):
                        if input_string[j:j+2] == '>>':
                            found_context_request = True
                            found_close = True
                            i = j + 2
                            break
                        j += 1
                    if not found_close:
                        conjecture_generator_module_logger.info("No message found. Missing '>>@' closure.")
                        return "No message found. Missing '>>@' closure."
                    continue
                    
                elif current == '>>':
                    # Look for matching <<
                    j = i + 2
                    found_close = False
                    while j + 1 < len(input_string):
                        if input_string[j:j+2] == '<<':
                            found_final_answer = True
                            found_close = True
                            i = j + 2
                            break
                        j += 1
                    if not found_close:
                        conjecture_generator_module_logger.info("No message found. Missing '@<<' closure.")
                        return "No message found. Missing '@<<' closure."
                    continue
                    
            i += 1
        
        # Determine final result
        if found_context_request and found_final_answer:
            return "both"
        elif found_context_request:
            return "context_request"
        elif found_final_answer:
            return "final_answer"
        else:
            return "none"

    def process_context_messages(self, input_string):
        """
        Parse message sequences of format <<(msg1),(msg2),...>> from input string
        and populate a dictionary with extracted information.
        
        Args:
            input_string (str): Input string containing message sequences
            
        Returns:
            tuple: (dict: Dictionary containing parsed message information with sets and tuples,
                list: List of error messages)
        """
        # Initialize messages dictionary with default values - now using sets for strings
        messages_dict = {
            'lines': set(),  # Will store tuples of integers
            'subclasses': set(),
            'instances': set(),
            'attributes': set(),
            'all_functions': False,
            'all_instances': False,
            'all_classes': False,
            'domain': set(),
            'i_def': set(),
            's_def': set(),
            'find': set(),
            'comps': set()
        }
        
        # Initialize error messages list
        error_messages = []
            
        # Find all message sequences, handling multiline input
        sequences = re.finditer(r'<<(.*?)>>', input_string, re.DOTALL)
        sequence_found = False
        
        for sequence in sequences:
            sequence_found = True
            # Extract messages from sequence
            sequence_content = sequence.group(1)
            # Split by comma, handling potential spaces and newlines
            messages = re.finditer(r'\(\s*(.*?)\s*\)', sequence_content, re.DOTALL)
            
            for message in messages:
                # Store original message for error reporting
                original_message = message.group(1).strip()
                # Remove quotes but keep spaces for now (needed for lines message)
                msg_content = original_message.replace('"', '').replace("'", '')
                
                # Parse lines message (lines:int,int)
                if msg_content.lower().strip().startswith('lines:'):
                    try:
                        # Extract numbers part and split, stripping spaces from each number
                        numbers_part = msg_content[msg_content.find(':')+1:]
                        nums = [num.strip() for num in numbers_part.split(',')]
                        
                        if len(nums) != 2:
                            error_messages.append(f"Error in '({original_message})': Expected two numbers.\n")
                            continue
                        
                        try:
                            num1, num2 = int(nums[0]), int(nums[1])
                        except ValueError:
                            error_messages.append(f"Error in '({original_message})': Invalid number format.\n")
                            continue
                        
                        if num1 < 0 or num2 < 0:
                            error_messages.append(f"Error in '({original_message})': Negative line numbers not allowed.\n")
                            continue
                            
                        if num1 > num2:
                            error_messages.append(f"Error in '({original_message})': First number should be less than or equal to second number.\n")
                            continue
                            
                        messages_dict['lines'].add((num1, num2))
                    except ValueError:
                        error_messages.append(f"Error in ({original_message}): Invalid number format.\n")
                        continue
                        
                # Parse simple string messages
                elif ':' in msg_content:
                    # Now remove all spaces for other message types
                    msg_content = msg_content.replace(' ', '')
                    try:
                        msg_type, value = msg_content.split(':')
                        if msg_type in ['subclasses', 'instances', 'attributes', 
                                    'domain', 's_def', 'i_def', 'find', 'comps']:
                            messages_dict[msg_type].add(value)
                        else:
                            error_messages.append(f"Unknown message type: ({original_message}).\n")
                    except ValueError:
                        error_messages.append(f"Invalid message format: ({original_message}).\n")
                        
                # Parse boolean flags
                elif msg_content.replace(' ', '') == 'all_functions':
                    messages_dict['all_functions'] = True
                elif msg_content.replace(' ', '') == 'all_instances':
                    messages_dict['all_instances'] = True
                elif msg_content.replace(' ', '') == 'all_classes':
                    messages_dict['all_classes'] = True
                else:
                    error_messages.append(f"Unrecognized message format: ({original_message}).\n")
        
        if not sequence_found:
            error_messages.append("No valid message sequences found in input string.\n")
        
        return messages_dict, error_messages


    def process_final_answer(self, input_string):
        """
        Process a message sequence string and find start/end positions of message content.
        
        Args:
            input_string (str): Input string containing message sequences in format:
                ">>(int,int):```str```, (int,int):```str```,...<<"
        
        Returns:
            tuple: (processed_copy, position_pairs)
                - processed_copy: String with content between ``` replaced with _
                - position_pairs: List of (start, end) positions between >> and <<
        """
        error_messages = []

        # Make a copy of the input string
        processed_copy = input_string
        
        # Find and replace content between triple backticks with underscores
        start = 0
        end = 0
        while True:
            start = processed_copy.find('```', end)
            if start == -1:
                break
                
            end = processed_copy.find('```', start + 3)
            if end == -1:
                break
                
            # Replace characters between backticks with underscores
            content_length = end - (start + 3)
            processed_copy = (
                processed_copy[:start + 3] + 
                '_' * content_length + 
                processed_copy[end:]
            )
        
        # Find start and end positions between >> and <<
        position_pairs = []
        current_pos = 0
        
        while True:
            # Find next >> marker
            start_marker = processed_copy.find('>>', current_pos)
            if start_marker == -1:
                break
                
            # Find corresponding << marker
            end_marker = processed_copy.find('<<', start_marker + 3)
            if end_marker == -1:
                break
                
            # Store positions (after >> and before <<)
            position_pairs.append((start_marker + 2, end_marker))
            
            # Update current position for next iteration
            current_pos = end_marker + 3

        """
        Parse messages from original string using position pairs and create a dictionary.
        
        Args:
            original_string (str): Original input string containing message sequences
            position_pairs (list): List of (start, end) positions to process
            
        Returns:
            dict: Dictionary with (int, int) tuples as keys and message content as values
        """
        messages_dict = {}
        
        for start_pos, end_pos in position_pairs:
            substring = input_string[start_pos:end_pos]
            current_pos = 0

            section_error_messages = []
            section_messages = {}
            
            while current_pos < len(substring):
                try:
                    # Find opening parenthesis for integers
                    open_paren = substring.find('(', current_pos)
                    if open_paren == -1:
                        break
                        
                    # Find closing parenthesis
                    close_paren = substring.find(')', open_paren)
                    if close_paren == -1:
                        break
                    
                    # Make sure we actually move forward
                    if close_paren <= current_pos:
                        current_pos += 1
                        continue
                        
                    # Extract and process the integer pair
                    int_pair_str = substring[open_paren + 1:close_paren]
                    
                    # Add validation for integer pair format
                    try:
                        int_pair = [int(x.strip()) for x in int_pair_str.split(',')]
                        if len(int_pair) != 2:
                            section_error_messages.append(f"Line error: did not fine two integers in '({int_pair_str})' part of update message.")
                            current_pos = close_paren + 1
                            continue
                    except ValueError:
                        current_pos = close_paren + 1
                        continue
                        
                    # Find the triple backticks
                    start_backticks = substring.find('```', close_paren)
                    if start_backticks == -1:
                        break
                        
                    end_backticks = substring.find('```', start_backticks + 3)
                    if end_backticks == -1:
                        break
                    
                    # Make sure we're moving forward
                    if end_backticks <= current_pos:
                        current_pos += 1
                        continue
                        
                    message_content = substring[start_backticks + 3:end_backticks]
                    section_messages[(int_pair[0], int_pair[1])] = message_content
                    
                    # Ensure we move forward
                    current_pos = end_backticks + 3
                    
                except Exception as e:
                    # If anything goes wrong, move forward one position
                    current_pos += 1
                    continue
            
            error_messages.extend(section_error_messages)
            messages_dict.update(section_messages)
            if not section_error_messages and not section_messages:
                error_messages.append(f"Update error: invalid message format in: {substring}")

        keys_to_remove = []  # Stores keys in messages_dict that have problems with their start and end positions. 
        for key_tuple, value in messages_dict.items():
            start_int, end_int = key_tuple
            if start_int > end_int:
                error_messages.append(f"Line error: start line greater than end line in '{key_tuple}: ```{value}```'")
                keys_to_remove.append(key_tuple)
            if start_int < 0 or end_int < 0:
                error_messages.append(f"Line error: out of range line numbers in '{key_tuple}: ```{value}```'")
                keys_to_remove.append(key_tuple)    
            if start_int > self.total_lines or end_int > self.total_lines+1:
                error_messages.append(f"Line error: line numbers exceed acceptable range (0 to {self.total_lines+1}) '{key_tuple}: ```{value}```'")
                keys_to_remove.append(key_tuple) 

        for key in keys_to_remove:
            messages_dict.pop(key, None)   # Won't raise KeyError if key is missing
        
        # If there are no update messages, add this as an error message
        if not messages_dict:
            error_messages.append(f"No valid updates found in response.")
        
        return messages_dict, error_messages


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

# Helper function to get the base object of an attribute
def get_base_name(node):
    if isinstance(node, ast.Attribute):
        return get_base_name(node.value)  # Recur to find the base
    elif isinstance(node, ast.Name):
        return node.id  # Return the base object name
    return None

def rename_files(file_paths, prefix="REJECTED_"):
    """
    Rename files by prepending a custom prefix to their filenames.
    
    Args:
        file_paths (list): List of file paths to rename
        prefix (str): String to prepend to filenames (default: "REJECTED__")
        
    """
    for path in file_paths:
        try:
            # Get the directory and filename
            directory = os.path.dirname(path)
            filename = os.path.basename(path)
            
            # Create new filename with custom prefix
            new_filename = f"{prefix}{filename}"
            new_path = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(path, new_path)
            
            conjecture_generator_module_logger.info(f"Renamed '{path}' to '{new_path}'")
            
        except FileNotFoundError:
            conjecture_generator_module_logger.warning(f"Could not rename '{path}'. File not found.")

        except PermissionError:
            conjecture_generator_module_logger.warning(f"Could not rename '{path}'. Permission denied.")

        except Exception as e:
            conjecture_generator_module_logger.warning(f"Could not rename '{path}'. {str(e)}")
    
               
"""
MAIN FUNCTION
"""
def main_function(
    # LLM API info 
    conj_api_key,
    conj_llm_name,
    # Directories
    ewm_dir,
    obs_json_dir,
    ann_obj_dir,
    conj_dir,
    log_dir,
    # Seed knowledge
    seed_prompt=None,
    last_ewm_name=None,
    next_ewm_name=None,
    # Logging bool
    verbose=False):

    # Initialize logging
    global conjecture_generator_module_logger
    conjecture_generator_module_logger = set_logging(verbose, log_dir, 'conjecture_generator_module')

    # Set a default value for 'next_ewm_accepted' 
      # Updated to True or False if 'conj_eval_summary_dict' available
    next_ewm_accepted = None

    # Ensure one seed prompt or ewm path is input (never both)
    if seed_prompt and any([last_ewm_name, next_ewm_name]):
        conjecture_generator_module_logger.error(f"c_generator can take either seed prompt or ewm_path, not both:\n  - seed_prompt: {seed_prompt}\n  - ewm_name: {last_ewm_name}")
        return

    """
    Generate code from a seed prompt.
    """
    # If a seed prompt exists, only supply prompt question and generate new EWM. 
    if seed_prompt:
        # Number of attempts allowed to generate seed EWM
        num_attempts_allowed = 8
        seed_ewm_name=''

        conjecture_generator_module_logger.info(f"Passing seed prompt to '{conj_llm_name}': \n{seed_prompt}")
        print(f"Passing seed prompt to '{conj_llm_name}': \n{seed_prompt}\n")

        # llm = TerminalLLM(
        #     conj_llm_name=conj_llm_name, 
        #     api_key=conj_api_key)
        llm = TogetherLLM(
            conj_llm_name=conj_llm_name, 
            api_key=conj_api_key)
        message = llm.get_response(seed_prompt)[0]  # prompt_tokens, completion_tokens, total_tokens are not needed for seed EWM
        num_attempts_allowed -= 1  # Decrement number of future attempts allowed 

        # Look for code between ``` and ```
        pattern = r"```(.*?)```"
        match = re.search(pattern, message, re.DOTALL)  # re.DOTALL allows matching across lines
        
        if match:
            seed_ewm_text = match.group(1)
            # group(1) refers to what's in the parentheses in the pattern r"```(.*?)```"
            # group(0) would be the entire match including the backticks
            seed_ewm_text.strip()
            seed_ewm_path, seed_ewm_name = save_code_to_file(seed_ewm_text, ewm_dir)
            module_validated_bool, error_msg_lst = validate_python_module(seed_ewm_path)
            if error_msg_lst:
                conjecture_generator_module_logger.warning("Error messages from new seed ewm: {error_msg_lst}")
            if not module_validated_bool:
                # Try to generate again if the module could not be validated
                match = None

        # If a match was not found, try to generate code again.
        while num_attempts_allowed and not match:
            conjecture_generator_module_logger.info(f"No code block found in the message:\n{message}")
            conjecture_generator_module_logger.info(f"{num_attempts_allowed} attempts remaining. Attempting to generate code again.")
            print(f"No code block found in the message:\n{message}")
            print(f"{num_attempts_allowed} attempts remaining. Attempting to generate code again.")
            
            follow_up_prompt = f'I searched for code in your response using: `pattern = r"```(.*?)```"` and `match = re.search(pattern, message, re.DOTALL)`. No code was found. Try to generate code again.'
            new_message= llm.get_response(follow_up_prompt)[0]
            num_attempts_allowed -= 1
            # Check for a pattern match
            match = re.search(pattern, new_message, re.DOTALL)
            if match:
                seed_ewm_text = match.group(1)
                # group(1) refers to what's in the parentheses in the pattern r"```(.*?)```"
                # group(0) would be the entire match including the backticks
                seed_ewm_text.strip()
                seed_ewm_path, seed_ewm_name = save_code_to_file(seed_ewm_text, ewm_dir)
                module_validated_bool, error_msg_lst = validate_python_module(seed_ewm_path)
                if error_msg_lst:
                    conjecture_generator_module_logger.warning("Error messages from new seed ewm: {error_msg_lst}")
                if not module_validated_bool:
                    # Try to generate again if the module could not be validated
                    match = None

        if match == None:
            conjecture_generator_module_logger.error(f"Reached maximum number of attempts to generate code form seed prompt. Seed EWM could not be created.")
            print(f"Reached maximum number of attempts to generate code form seed prompt. Seed EWM could not be created.\n")
            return
        
        else:
            # If a seed EWM was successfully generated, save its conj_dict JSON and return it to main module
            seed_ewm_conj_dicts_path = os.path.join(conj_dir, f"{seed_ewm_name}_conj_dicts.json")
            seed_ewm_conj_dicts = {'conj_origin':{'last_EWM':None, 
                                'seed_prompt':seed_prompt, 
                                'conjecturer':conj_llm_name}}
            save_results(seed_ewm_conj_dicts_path, seed_ewm_conj_dicts)
            conjecture_generator_module_logger.info(f"Successfully saved seed EWM conjecture dictionaries at: {seed_ewm_conj_dicts_path}")
            return seed_ewm_name
    """
    Find or generate conj_dicts for last_ewm
    """
    # If there is no last_ewm_name (i.e. only a seed EWM is provided in next_ewm_name)
    if not last_ewm_name:
        next_ewm_accepted = None
        conjecture_generator_module_logger.info(f"Processing seed EWM: {next_ewm_name}.\nlast_ewm_name: {last_ewm_name}")

        seed_ewm_conj_dicts_path = os.path.join(conj_dir, f"{next_ewm_name}_conj_dicts.json")
        if not os.path.exists(seed_ewm_conj_dicts_path):
            try:
                # Generate a conj_dicts JSON for seed EWM
                seed_ewm_conj_dicts = {'conj_origin':{'last_EWM':None, 
                                    'seed_prompt':seed_prompt, 
                                    'conjecturer':conj_llm_name}}
                save_results(seed_ewm_conj_dicts_path, seed_ewm_conj_dicts)
                conjecture_generator_module_logger.info(f"Successfully saved seed EWM conjecture dictionaries at: {seed_ewm_conj_dicts_path}")
            except Exception as e:
                conjecture_generator_module_logger.error(f"Unexpected error writing file at {seed_ewm_conj_dicts_path}: {str(e)}")
                raise

        """
        Go ahead to generating next conjecture using ann_obj and obs data of seed EWM ('next_ewm'). 
        """

    # If there is a 'last_ewm_name' (an earlier EWM that 'next_ewm_name' branched from), find or generate its conj_dicts data to determine if 'next_ewm_name' was accepted or rejected.
    else:
        conj_dicts_path = os.path.join(conj_dir, f"{last_ewm_name}_conj_dicts.json")

        # Load conj_dicts data
        try:
            conj_dicts= load_json(conj_dicts_path)  # If a seed EWM was input, conj_dicts may not be assigned
        except () as e:
            conjecture_generator_module_logger.error(f"Error: {e}")   
            conjecture_generator_module_logger.error(f"Unexpected behavior:\n'last_ewm_name' given: {last_ewm_name}. However, unable to load conj_dicts from '{conj_dicts_path}'.")
            print(f"Unexpected behavior:\n'last_ewm_name' given: {last_ewm_name}. However, unable to load conj_dicts from '{conj_dicts_path}'.")
            sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.

        """
        Process 'next_ewm_name' according to whether it was rejected or accepted.
        """
        # Load conj_dicts data (it should exist)
        conj_dicts_path = os.path.join(conj_dir, f"{last_ewm_name}_conj_dicts.json")
        try:
            conj_dicts= load_json(conj_dicts_path)
        except () as e:
            conjecture_generator_module_logger.error(f"Error: {e}")
        
        try: 
            # Get 'accepted_bool' of 'next_ewm_name':
            for question_series in conj_dicts['question_series']:
                for update_attempt in question_series['update_attempts']:
                    if update_attempt['timestamp'] == next_ewm_name:
                        next_ewm_accepted = update_attempt['conj_eval_result']['accepted_bool']
                    
        except () as e: 
            conjecture_generator_module_logger.error(f"Could not access 'accepted_bool' of next EWM '{next_ewm_name}' in last EWM '{last_ewm_name}'. Exiting.")
            conjecture_generator_module_logger.error(f"Error: {e}")
            sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.

    """
    Load data to generate next conjecture:

    If a seed EWM was received:
        - Generate next conjecture using 'ann_obj_dicts', 'obs_dicts', and 'conj_dicts' data from 'next_ewm'
    If a new conjecture EWM was received, and it was REJECTED:
        - Generate next conjecture using 'ann_obj_dicts', 'obs_dicts', and 'conj_dicts' data from 'last_ewm'
        - Use 'question_series' data to provide context and generate a unique conjecture
    If a new conjecture EWM was received, and it was ACCEPTED:
        - Generate next conjecture using 'ann_obj_dicts', 'obs_dicts', and 'conj_dicts' data from 'next_ewm'
    """

    # Define the EWM and data that will be updated with next conjecture
    # If conjecture accepted or only a seed EWM was received
    if next_ewm_accepted in (True, None):
        # Ensure files for next_ewm_name: observation JSON, annotated object JSON, and conj_dicts JSON exist
            # Note that conj_dicts data for the next_ewm must be loaded
        ewm_path = os.path.join(ewm_dir, f"{next_ewm_name}.py")
        if not os.path.exists(ewm_path):
            conjecture_generator_module_logger.error(f"next_ewm_name input '{next_ewm_name}' but '{ewm_path}' file not found in '{ewm_dir}")
            sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.
        obs_dicts_path = os.path.join(obs_json_dir, f"{next_ewm_name}_obs.json")
        if not os.path.exists(obs_dicts_path):
            conjecture_generator_module_logger.error(f"next_ewm_name input '{next_ewm_name}' but '{obs_dicts_path}' file not found in '{obs_json_dir}")
            sys.exit(1)
        ann_obj_dicts_path = os.path.join(ann_obj_dir, f"{next_ewm_name}_ann_obj.json")
        if not os.path.exists(ann_obj_dicts_path):
            conjecture_generator_module_logger.error(f"next_ewm_name input '{next_ewm_name}' but '{ann_obj_dicts_path}' file not found in '{ann_obj_dir}")
            sys.exit(1)
        conj_dicts_path = os.path.join(conj_dir, f"{next_ewm_name}_conj_dicts.json")
        if not os.path.exists(conj_dicts_path):
            conjecture_generator_module_logger.error(f"next_ewm_name input '{next_ewm_name}' but '{conj_dicts_path}' file not found in '{conj_dir}")
            sys.exit(1)
        
    # If 'next_ewm_accepted' is False (i.e. the conjecture was rejected)
    elif next_ewm_accepted is False:                  
        # If the next EWM version was not accepted, prepend that EWM's file names with "REJECTED_"
        next_ewm_path = os.path.join(ewm_dir, f"{next_ewm_name}.py")
        next_obs_dicts_path = os.path.join(obs_json_dir, f"{next_ewm_name}_obs.json")
        next_ann_obj_dicts_path = os.path.join(ann_obj_dir, f"{next_ewm_name}_ann_obj.json")
        next_conj_dicts_path = os.path.join(conj_dir, f"{next_ewm_name}_conj_dicts.json")
        files_to_rename = [next_ewm_path, next_obs_dicts_path, next_ann_obj_dicts_path, next_conj_dicts_path]
        rename_files(files_to_rename, prefix="REJECTED_")

        # Ensure files for last_ewm_name: observation JSON, annotated object JSON, and conj_dicts JSON exist
        ewm_path = os.path.join(ewm_dir, f"{last_ewm_name}.py")
        if not os.path.exists(ewm_path):
            conjecture_generator_module_logger.error(f"EWM name given '{last_ewm_name}' but '{ewm_path}' file not found in '{ewm_dir}")
            sys.exit(1) # Exit module. This kind of error is unexpected and not handled in module.
        obs_dicts_path = os.path.join(obs_json_dir, f"{last_ewm_name}_obs.json")
        if not os.path.exists(obs_dicts_path):
            conjecture_generator_module_logger.error(f"EWM name given '{last_ewm_name}' but '{obs_dicts_path}' file not found in '{obs_json_dir}")
            sys.exit(1)
        ann_obj_dicts_path = os.path.join(ann_obj_dir, f"{last_ewm_name}_ann_obj.json")
        if not os.path.exists(ann_obj_dicts_path):
            conjecture_generator_module_logger.error(f"EWM name given '{last_ewm_name}' but '{ann_obj_dicts_path}' file not found in '{ann_obj_dir}")
            sys.exit(1)
        conj_dicts_path = os.path.join(conj_dir, f"{last_ewm_name}_conj_dicts.json")
        if not os.path.exists(conj_dicts_path):
            conjecture_generator_module_logger.error(f"last_ewm_name input '{last_ewm_name}' but '{conj_dicts_path}' file not found in '{conj_dir}")
            sys.exit(1)

    else:
        conjecture_generator_module_logger.error(f"Something went wrong with evaluating 'next_ewm_accepted': {next_ewm_accepted}")
        sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.

    # Load obs_dicts, ann_obj_dicts, and conj_dicts data
        # Use the conj_dicts data that has already been loaded for last_ewm_name
    try:
        obs_dicts= load_json(obs_dicts_path)
        ann_obj_dicts= load_json(ann_obj_dicts_path)
        conj_dicts = load_json(conj_dicts_path)
    except () as e:
        conjecture_generator_module_logger.error(f"Error: {e}")
        sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.

    """
    Define the 'question_path' (the style of prompting that will be used)
    - Each EWM condition has a set of 'question_path' options
    - Each set of 'question_path' options has probability values for each path
    This allows focused learning for both correcting inconsistent conjectures and on improving knowledge organization.
    """
    ewm_to_question = None # The EWM to pass to question_path_walker
    # If seed EWM:
    if next_ewm_accepted is None:
        path_options=[('default_problem_obj_conjecture', 0.5),
                      ('improve_variety', 0.5)]
        ewm_to_question = next_ewm_name
    # If conjecture accepted
    elif next_ewm_accepted is True:
        path_options=[('default_problem_obj_conjecture', 0.5),
                      ('improve_variety', 0.4),
                      ('improve_organization', 0.1)]
        ewm_to_question = next_ewm_name
    # If conjecture rejected
    elif next_ewm_accepted is False: 
        path_options=[('rejected_problem_obj_conjecture', 0.5),
                      ('improve_variety', 0.4),
                      ('improve_organization', 0.1)]
        ewm_to_question = last_ewm_name

    ewm_path = os.path.join(ewm_dir, f"{ewm_to_question}.py")

    # Split into separate lists for items and weights
    options = [item[0] for item in path_options]
    weights = [item[1] for item in path_options]
    # Make weighted selection of a question path
    question_path = random.choices(options, weights=weights, k=1)[0]

    """
    Select a problem object if necessary:
      - Used for problem question paths
      - Can select an object with high False:True weighted product score, 
        or an unscorable object (with no predictions or not True observations)
      - If prior conjecture data in 'question_series', adjust scores before selection
        (lower scores of objects already used for conjecture generation with current EWM)

    """
    selected_obj_name = None  # Default value
    obj_obs_type = None  # Default value
    if not question_path in ['improve_organization', 'improve_variety']:
        # Update annotated_object_dict with object observation scores
            # Note that re-scoreing is needed if EWM was rejected because new observations are created for last_ewm during comparison against next_ewm
                # and if next_ewm was accepted, it will not have been scored yet
            # Returns lists of tuples containing object fq_name and type info
        no_prediction_objs, no_true_observation_objs= score_obj_observations(ann_obj_dicts)

        # Create a dictionary with non-computable scores filtered out
        no_pred_no_obs_filt_dict = {}
        for fq_name, obj_info in ann_obj_dicts.items():
            if fq_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
                continue
            elif 'no_predictions' in obj_info:
                continue
            elif 'no_observations' in obj_info:
                continue
            else:
                no_pred_no_obs_filt_dict[fq_name] = obj_info

        # Adjust scores for objects already queried (by 1/2 for each prior query)
        if 'question_series' in conj_dicts:
            queried_fq_names = []
            # Collect names of previously queried objects
            for conj_dict in conj_dicts['question_series']:
                if conj_dict['question_path']=='improve_organization':
                    continue
                else:
                    queried_fq_names.append(conj_dict['problem_obj']) # 'problem_obj' value should be a FQ name string

            if queried_fq_names:
                for queried in queried_fq_names:
                    query_count = queried_fq_names.count(queried)
                    score_adjustment = 1/(2**query_count)
                    # Adjust scores in no_pred_no_obs_filt_dict
                    if queried in no_pred_no_obs_filt_dict:
                        no_pred_no_obs_filt_dict[queried]['pco_val_counts']['weighted_product'] *= score_adjustment
                        conjecture_generator_module_logger.info(f"Weighted product score adjusted for observed object: {queried}. Adjustment: {score_adjustment}")
                    # Adjust scores in 'no_prediction_objs' and 'no_true_observation_objs' lists
                    elif no_prediction_objs:
                        for sublist in no_prediction_objs:
                            if sublist[0] == queried:
                                sublist[2] *= score_adjustment
                                conjecture_generator_module_logger.info(f"Weighted product score adjusted for 'no prediction' object: {queried}. Adjustment: {score_adjustment}")
                    elif no_true_observation_objs:
                        for sublist in no_true_observation_objs:
                            if sublist[0] == queried:
                                sublist[2] *= score_adjustment
                                conjecture_generator_module_logger.info(f"Weighted product score adjusted for 'no true observation' object: {queried}. Adjustment: {score_adjustment}")

        # Use scores (adjusted by prior queries) to select the next problem object to query
          # Choose whether to query an observed object or not (i.e. an object where 'weighted_product' could not be computed due to no predictions or no 'True' observations)
        choose_observed_object = True
        # If unobserved objects were found
        if no_prediction_objs or no_true_observation_objs:
            # Randomly decide whether to query an unobserved object with (probability 0.7 of choosing to query an observed object)
            choose_observed_object = random.choices([True, False], weights=[0.7, 0.3])[0] 

        # Store the type of observation (e.g. no prediction or no True observation) to customize prompt
        obj_obs_type = '' # will be set to one of ('no_pred', 'no_true_obs', 'true_obs')

        if not choose_observed_object:
            conjecture_generator_module_logger.info("Querying an observed object.")
            unobserved_objs = no_prediction_objs + no_true_observation_objs
            conjecture_generator_module_logger.info(f"unobserved_objs: {unobserved_objs}")
            # Extract weights from position 2 of each sublist
            weights = [item[2] for item in unobserved_objs]
            
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            normalized_weights = [w/total_weight for w in weights]
            conjecture_generator_module_logger.info(f"normalized_weights: {normalized_weights}")
            
            # Use random.choices() with weights for selection
              # I use k=1 to get one item, and [0] to get it out of the list
            selected_obj = random.choices(unobserved_objs, weights=normalized_weights, k=1)[0]
            selected_obj_name = selected_obj[0]
            conjecture_generator_module_logger.info(f"Selected problem object: {selected_obj_name}")
            if selected_obj_name in no_prediction_objs:
                obj_obs_type = 'no_pred'
            else:
                obj_obs_type = 'no_true_obs'

        else:  # If an observed object must be selected
            obj_obs_type = 'true_obs'

            # Sort the dictionary of scored objects by the adjusted weighted_product values in descending order
            conjecture_generator_module_logger.info("Querying an unobserved object.")
            weighted_product_sorted = sorted(no_pred_no_obs_filt_dict.items(), 
                                key=lambda item: (item[1]['pco_val_counts']['weighted_product']), 
                                reverse=True)
            conjecture_generator_module_logger.info(f"Filtered and adjusted weighted product sorted ann_obj_dicts: {json.dumps(weighted_product_sorted, indent=2)[:500]}")

            # Log the top weighted_product values
            top_n = 16
            conjecture_generator_module_logger.info(f"\nTop {top_n} weighted_product values:")
            for key, value in weighted_product_sorted[:top_n]:
                conjecture_generator_module_logger.info(f"{key}: {value['pco_val_counts']['weighted_product']}")

            # Select the object with the top score
            selected_obj_name = weighted_product_sorted[0][0]
            conjecture_generator_module_logger.info(f"selected_obj_name: {selected_obj_name}")

    # Begin prompting LLM to generate new conjecture code
    path_walker = QuestionPathWalker(
        ewm_name = ewm_to_question,
        ewm_path = ewm_path,
        prob_obj_name = selected_obj_name,  # Defaults to None when no object is selected (e.g. 'improve_organization' path)
        prob_obj_obs_type = obj_obs_type, # will be set to one of ('no_pred', 'no_true_obs', 'true_obs')
        question_path = question_path,
        conj_api_key=conj_api_key, 
        conj_llm_name=conj_llm_name,
        obs_dicts=obs_dicts,
        ann_obj_dicts=ann_obj_dicts,
        conj_dicts=conj_dicts,
        conj_dicts_path=conj_dicts_path,
        conj_dir=conj_dir
        )
    
    new_ewm_timestamp = path_walker.walk()

    conjecture_generator_module_logger.info(f"Returning: {new_ewm_timestamp}")
    return new_ewm_timestamp # Return the name of the new EWM


if __name__ == "__main__":
    # #EWM data:
    # test_ewm_name ="test_prediction_environment_10 (copy)"
    # test_ewm_path =".../generator_functions/ewm_files/test_prediction_environment_10 (copy).py"
    # obs_json_path = ".../generator_functions/obs_ann_conj_files/test_prediction_environment_10 (copy)_obs04.json"
    # ann_obj_json_path = ""
    # log_dir = ".../generator_functions/log_files"
    
    # set_logging(verbose_bool=False, log_dir=log_dir)

    # try:
    #     obs_dicts= load_json(obs_json_path)  # If a seed EWM was input, conj_dicts may not be assigned
    #     ann_obj_dicts= load_json(ann_obj_json_path)
    # except () as e:
    #     conjecture_generator_module_logger.info(f"Error: {e}")   

    # walker = QuestionPathWalker(
    #     ewm_name=test_ewm_name,
    #     ewm_path=test_ewm_path,
    #     prob_obj_name=None,
    #     question_path=None,
    #     # LLM API info 
    #     conj_api_key=None,
    #     conj_llm_name=None,
    #     # Data
    #     obs_dicts=obs_dicts,
    #     ann_obj_dicts=ann_obj_dicts,
    #     conj_dicts=None)
    
    # # Test context message
    # message_a = ("Good question. let me think... OK I want context. "
    #            +"<<"
    #            +"(lines:17,19), " # Should return def for 'kilogram_unit'
    #            +"(all_classes), "
    #            +"(subclasses:Planet),   " # No subclasses
    #            +"(subclasses: PhysicalObject), "
    #            +"(instances: ) " # No comma
    #            +"(instances:Car), " # Not in module
    #            +"(instances:Planet), "
    #            +"(attributes:Name), "
    #            +"(attributes:bool), "
    #            +"(attributes:Mass), "
    #            +"(attributes:Car), "
    #            +"(all_functions), "
    #            +"(all_instances), "
    #            +"(domain:Distance), "
    #            +"(domain:int), "
    #            +"(domain:Mass), "
    #            +"(domain:Car), "
    #            +">>"
    #            +"<<"
    #            +"(i_def:Planet), "
    #            +"(i_def:earth_planet), "
    #            +"(i_def:Car), "
    #            +"(s_def:Planet), "
    #            +"(s_def: earth_planet),"
    #            +"(s_def: Car)"
    #            +"(find:venus_planet), "
    #            +"(find:gravity), "
    #            +"(find:Car), "
    #            +"(comps:mars_planet), "
    #            +"(comps:earth_planet.atmosphere.pressure)"
    #            +"(comps:gravitational_constant)"   
    #            +">>"
    #            +" How's that?"
    #         )
    
    # # Test update message
    # message_b = ("Good question. let me think... OK I want context. "
    #            +">>"
    #            +"(167,189):```class Planet(PhysicalObject, Sphere):\n    def __init__(self: 'Planet', name: Name, mass: Mass, radius: Radius, volume: Volume = None, atmosphere: Atmosphere = None):\n        PhysicalObject.__init__(self, name=name, mass=mass)\n        Sphere.__init__(self, radius=radius, volume=volume)\n        self.atmosphere = atmosphere\n\n    def calculate_surface_gravity(self: 'Planet') -> Value:\n        \"\"\"Calculate the surface gravity of the planet in m/s^2.\"\"\"\n        if self.mass.value.unit == kilogram_unit and self.radius.value.unit == meter_unit:\n            G = gravitational_constant.get_value(newton_unit)\n            gravity = G * self.mass.value.number / (self.radius.value.number ** 2)\n            return Value(number=gravity, unit=Unit(name=Name(\"meters per second squared\"), symbols=[Symbol(\"m/s\u00b2\")]))\n        else:\n            raise AttributeError(\"Mass must be in kilograms and radius in meters to calculate surface gravity\")```}"
    #            +"(0,1):```Yes Hello```,"
    #            +"(-1,1):```Yes Hello```,"
    #            +"(0,0):```Yes Hello```,"
    #            +"(0,500):```Yes Hello```,"
    #            +"(500,5555):```Yes Hello```,"
    #            +"(55,22):```Yes Hello```,"
    #            +"<<"
    #            +">>"
    #            +"(i_def:Planet), "  
    #            +"<<"
    #            +" How's that?"
    #         )
    
    # response_errors = []
    # ctx_msg_dict = None
    # # Determine the response type
    # final_msg_dict, error_msg_lst = walker.process_final_answer(message_b)

    # print(f"error_msg_lst:\n")
    # for error in error_msg_lst:
    #     print(f"{error}\n")
    # print("\n\n")
    # print(f"final_msg_dict:\n")
    # for lines, message_dict in final_msg_dict.items():
    #     print(lines)
    #     print(f"{message_dict}\n")

    ewm_path = '.../generator_functions/ewm_files/20241223_153122_270575.py'
    duplicate_results = analyze_duplicates(ewm_path)
    print(duplicate_results)
    
    """
    Conjecture dictionary format, for reference:
    conj_dicts = {'conj_origin':{
                    'last_EWM':None, 
                    'seed_prompt':seed_prompt, 
                    'conjecturer':conj_llm_model}
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
    
"""
Printing alternate sort results below:
"""
    # # Sort the dictionary by its false:true values in descending order, sort by number of false when ties
    # false_none_error_to_true_sorted = sorted(no_pred_no_obs_filt_dict.items(), key=lambda item: (item[1]['pco_val_counts']['false_none_and_error_to_true'], item[1]['pco_val_counts']['false_none_error_total']), reverse=True)
    # # Print the top false_none_and_error_to_true ratios
    # print(f"\nTop {top_n} ratios (false+none+error:true, false+none+error PCOs):")
    # for key, value in false_none_error_to_true_sorted[:top_n]:
    #     print(f"{key}: ({value['pco_val_counts']['false_none_and_error_to_true']}, {value['pco_val_counts']['false_none_error_total']})")

    # # Sort the dictionary by its std_sum_score values in descending order.
    # false_to_true_sorted = sorted(no_pred_no_obs_filt_dict.items(), 
    #                     key=lambda item: (item[1]['pco_val_counts']['std_sum_score']), 
    #                     reverse=True)
    # # Print the top false_none_and_error_to_true ratios
    # print(f"\nTop {top_n} std_sum_score values:")
    # for key, value in false_to_true_sorted[:top_n]:
    #     print(f"{key}: {value['pco_val_counts']['std_sum_score']}")
        
    # # Sort the dictionary by its combined_rank values in ascending order.
    # false_to_true_sorted = sorted(no_pred_no_obs_filt_dict.items(), 
    #                     key=lambda item: (item[1]['pco_val_counts']['combined_rank']), 
    #                     reverse=False)
    # # Print the top combined_rank ratios
    # print(f"\nTop {top_n} combined_rank values:")
    # for key, value in false_to_true_sorted[:top_n]:
    #     print(f"{key}: {value['pco_val_counts']['combined_rank']}")