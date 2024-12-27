"""
x_compute_obs.py

- Recieve prediction-observation JSONs for two last EWMs
- Check if any predictions are unique, try to cross-compute unique predictions
- Transfer observations for duplicate predictions
- Print comparison between new and old observation sets

"""
import inspect
from inspect import Parameter
import json
from typing import get_origin, get_args, get_type_hints, Union, Callable, Tuple, Generator, Optional, Type, Dict, Set, Any, List
import types
from types import MappingProxyType, GetSetDescriptorType
import functools
import sys
import os
import importlib.util
from tabulate import tabulate
import traceback
from collections import defaultdict
import logging
from datetime import datetime

cross_compute_module_logger = None
DEFAULT_VALUE_PLACEHOLDER = 'l8jNWV52p34HjK'

"""
CONSISTENCY CHECK
"""
# Create keys for unique combinations of function and instance data
def create_func_and_inst_key(item: Dict[Any, Any]) -> str:
    return (str(item['function_name']) + 
        str(item['class_if_method']) + 
        json.dumps(item['instance_name(s)']))

# Create keys for unique combinations of function, instance, and prediction data
def create_prediction_key(item: Dict[Any, Any]) -> str:
    return (str(item['function_name']) + 
        str(item['class_if_method']) + 
        json.dumps(item['instance_name(s)']) + 
        str(item['result']))   

def compare_json_files(prev_json_data: list, new_json_data: list) -> tuple:
    prev_json_key_dict = {create_func_and_inst_key(item): item for item in prev_json_data}
    cross_compute_module_logger.info("\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n")
    cross_compute_module_logger.info(f"prev_json_key_dict: {json.dumps(prev_json_key_dict, indent=2)}")
    new_json_key_dict = {create_func_and_inst_key(item): item for item in new_json_data}
    cross_compute_module_logger.info("\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n")
    cross_compute_module_logger.info(f"new_json_key_dict: {json.dumps(new_json_key_dict, indent=2)}")

    only_in_prev_json = []
    only_in_new_json = []
    different_results = []
    in_both = []

    for key, item1 in prev_json_key_dict.items():
        if key not in new_json_key_dict:
            only_in_prev_json.append(item1)
            cross_compute_module_logger.info(f"only_in_prev_json.append: {key}")
        else:
            item2 = new_json_key_dict[key]
            if 'computation_error' in item1 and 'computation_error' in item2:
                in_both.append([item1,item2])
                cross_compute_module_logger.info(f"in_both.append: {key}")
            elif 'computation_error' in item1 or 'computation_error' in item2:
                different_results.append([item1,item2])
                cross_compute_module_logger.info(f"different_results.append: {key}")
            elif item1['result'] != item2['result']:
                different_results.append([item1,item2])
                cross_compute_module_logger.info(f"different_results.append: {key}")
            else:
                in_both.append([item1,item2])
                cross_compute_module_logger.info(f"in_both.append: {key}")

    for key, item2 in new_json_key_dict.items():
        if key not in prev_json_key_dict:
            only_in_new_json.append(item2)
            cross_compute_module_logger.info(f"only_in_new_json.append: {key}")

    cross_compute_module_logger.info("\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n")
    cross_compute_module_logger.info(f"only_in_prev_json: \n{json.dumps(only_in_prev_json, indent=2)}\n")
    cross_compute_module_logger.info("\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n")
    cross_compute_module_logger.info(f"only_in_new_json: \n{json.dumps(only_in_new_json, indent=2)}\n")
    cross_compute_module_logger.info("\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n")
    cross_compute_module_logger.info(f"different_results: \n{json.dumps(different_results, indent=2)}\n")
    cross_compute_module_logger.info("\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n~~~\n")
    cross_compute_module_logger.info(f"in_both: \n{json.dumps(in_both, indent=2)}\n")

    return only_in_prev_json, only_in_new_json, different_results, in_both

"""
Reproduce Computations
"""
def compute_across_modules(module, module_name, computation_dict, func_tracer_module, prediction_counter):

    def create_method_caller(cls, method_name):
        try:
            method = getattr(cls, method_name)
            
            @functools.wraps(method)
            def method_caller(*args, **kwargs):
                return method(*args, **kwargs)
            return method_caller
        except AttributeError:
            cross_compute_module_logger.warning(f"Warning: Class '{cls.__name__}' has no method '{method_name}'")
            return None

    def get_instance(module, instance_name):
        cross_compute_module_logger.info(f"get_instance inputs:\n  module: {module}, instance_name: {instance_name}")
        try:
            parts = instance_name.split('.')
            cross_compute_module_logger.info(f"parts: {parts}")
            obj = module
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    cross_compute_module_logger.info(f"{obj} does not have attribute '{part}'")
                    return None
            return obj
        except AttributeError:
            cross_compute_module_logger.warning(f"Warning: Instance object '{instance_name}' could not be found in '{module}'")
            return None

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
    #     cross_compute_module_logger.info(f"tuple_string: {tuple_string}")
    #     try:
    #         # Extract the first part ('earth_planet.radius') and the class part
    #         # This assumes that the string is formatted like the example provided
    #         arg_name_part, class_part = tuple_string.split(', ', 1)
            
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
    #         cross_compute_module_logger.warning(f"Error converting string to tuple: {e}")
    #         return None
    
    def generate_args_and_kwargs_from_names(module, func, arg_name_type_tuple_strings, kwarg_name_type_tuple_strings):
        args = []
        arg_names = []
        kwargs = {}
        kwarg_names = {}
        input_names = set() # Store FQ names of all arg and kwarg inputs excluding default values
                        # Recall, default values are only used in prediction generation if no compatible defined objects exist; therefore, default values used are not defined objects and do not need to be recorded in 'all_objects'
        
        # Get the function's signature
        sig = inspect.signature(func)
        """
        For reference, the signature dictionary looks like this:
            # Example function
            def example(a: int, b: str = "hello", *args: tuple, kw: float, **kwargs: dict) -> bool:
                pass

            # Parameters dictionary :
            {
                'a': Parameter(name='a', kind=<_ParameterKind.POSITIONAL_OR_KEYWORD: 1>, default=<class 'inspect._empty'>, annotation=<class 'int'>),
                'b': Parameter(name='b', kind=<_ParameterKind.POSITIONAL_OR_KEYWORD: 1>, default='hello', annotation=<class 'str'>),
                'args': Parameter(name='args', kind=<_ParameterKind.VAR_POSITIONAL: 2>, default=<class 'inspect._empty'>, annotation=<class 'tuple'>),
                'kw': Parameter(name='kw', kind=<_ParameterKind.KEYWORD_ONLY: 3>, default=<class 'inspect._empty'>, annotation=<class 'float'>),
                'kwargs': Parameter(name='kwargs', kind=<_ParameterKind.VAR_KEYWORD: 4>, default=<class 'inspect._empty'>, annotation=<class 'dict'>)
            }
        """
        parameters = list(sig.parameters.values())

        """
        For reference, get_type_hints() from the typing module produces a dictionary mapping parameter names to their type annotations.
        E.g. 
            def example(a: int, b: str = "hello", *args: tuple, kw: float, 
                items: List[str], mapping: Dict[str, int], **kwargs: dict) -> bool:
                pass

            # get_type_hints(example) would produce:
            {
                'a': int,
                'b': str,
                'args': tuple,
                'kw': float,
                'items': List[str],
                'mapping': Dict[str, int],
                'kwargs': dict,
                'return': bool
            }
            It includes a 'return' key for the return type annotation
            It evaluates string annotations and forward references
            It doesn't include any information about parameter kinds or defaults
            It only includes parameters that have type annotations
        """

        # Generate args
        for i, string_arg_name_tuple in enumerate(arg_name_type_tuple_strings):
            if string_arg_name_tuple == DEFAULT_VALUE_PLACEHOLDER:
                arg_names.append(DEFAULT_VALUE_PLACEHOLDER)
                if i < len(parameters) and parameters[i].default != inspect.Parameter.empty:
                    args.append(parameters[i].default)
                else:
                    args.append(None)
            else:
                cross_compute_module_logger.info(f"string_arg_name_tuple: {string_arg_name_tuple}")
                arg_name_tuple = convert_string_to_tuple(string_arg_name_tuple)
                cross_compute_module_logger.info(f"arg_name_tuple: {arg_name_tuple}")
                arg_name = arg_name_tuple[0]
                next_arg = get_instance(module, arg_name)
                if next_arg:  # If the argument can be found in the module, add it to the arguments passed to function tracer module
                    args.append(next_arg)
                    cross_compute_module_logger.info(f"next_arg: {next_arg}")
                    arg_names.append(arg_name)
                    input_names.add(arg_name)
                else:
                    cross_compute_module_logger.info(f"next_arg: {next_arg}")
                    return None 
                        # 'get_instance' returns None when it cannot find the object.
                        # Cannot re-compute a prediction if an argument object is not available in the module.
        
        # Generate kwargs
        for kwarg_name, value_string_name_tuple in kwarg_name_type_tuple_strings.items():
            param = sig.parameters.get(kwarg_name)
            if value_string_name_tuple == DEFAULT_VALUE_PLACEHOLDER:
                if param and param.default != inspect.Parameter.empty:
                    kwargs[kwarg_name] = param.default
                    kwarg_names[param.name] = DEFAULT_VALUE_PLACEHOLDER
                else:
                    kwargs[kwarg_name] = None
            else:
                value_name_tuple = convert_string_to_tuple(value_string_name_tuple)
                value_name = value_name_tuple[0]
                next_arg = get_instance(module, value_name)
                if next_arg:
                    kwargs[kwarg_name] = next_arg
                    cross_compute_module_logger.info(f"next_arg: {next_arg}")
                    kwarg_names[kwarg_name] = value_name
                    input_names.add(value_name)
                else:
                    cross_compute_module_logger.info(f"next_arg: {next_arg}")
                    return None
                      # 'get_instance' returns None when it cannot find the object.
                        # Cannot re-compute a prediction if an argument object is not available in the module.

        return args, arg_names, kwargs, kwarg_names, input_names

    def check_types(func: Callable, args: list, kwargs: dict, method: Optional[Type] = None) -> bool:
        """
        Check if provided args and kwargs match the type hints of the function.
        Returns True if all types match or if no type hint exists for a parameter.
        Returns False if any type hint exists and doesn't match the provided object.
        
        Args:
            func: Function or method to check types against
            args: List of positional arguments
            kwargs: Dictionary of keyword arguments
            method: Class type to check 'self' parameter against for methods, if provided
        """
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)
        params = list(sig.parameters.items())
        cross_compute_module_logger.info(f"Running 'check_types' with:\n  func: {func}\n  args: {args}\n  kwargs: {kwargs}\n  method: {method}")
        
        arg_index = 0
        for param_name, param in params:
            # Special handling for 'self' parameter
            if param_name == 'self' and method:
                if arg_index < len(args):
                    self_arg = args[arg_index]
                else:
                    # This line may be unnecessary. I don't think 'self' would be assigned to a kwarg.
                    self_arg = kwargs.get('self')
                    
                if self_arg is not None:                   
                    # Check against method type if provided
                    if method is not None and not isinstance(self_arg, method):
                        cross_compute_module_logger.info(f"Types do not match: 'self' arg does not match input method type")
                        return False
                        
                arg_index += 1
                continue
                    
            # Handle positional args
            if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                if arg_index < len(args):
                    expected_type = type_hints.get(param_name, Any)
                    if expected_type is not Any and not isinstance(args[arg_index], expected_type):
                        cross_compute_module_logger.info(f"Types do not match: expected_type '{expected_type}' does not match args[arg_index] '{args[arg_index]}'")
                        return False
                    arg_index += 1
                    
            # Handle *args
            elif param.kind == Parameter.VAR_POSITIONAL:
                expected_type = type_hints.get(param_name, Any)
                if expected_type is not Any:
                    varargs = args[arg_index:]
                    if not all(isinstance(arg, expected_type) for arg in varargs):
                        cross_compute_module_logger.info(f"Types do not match: varargs '{varargs}' don't match expected_type '{expected_type}'")
                        return False
                arg_index = len(args)  # All remaining args consumed
                
        # Handle regular kwargs
        regular_kwargs = {k: v for k, v in kwargs.items() 
                        if any(p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY) 
                            and p.name == k for _, p in params)}
        
        for kwarg_name, kwarg_value in regular_kwargs.items():
            if kwarg_name not in sig.parameters:
                cross_compute_module_logger.info(f"Types do not match: kwarg_name '{kwarg_name}' not found in sig.parameters '{sig.parameters}'")
                return False
            expected_type = type_hints.get(kwarg_name, Any)
            if expected_type is not Any and not isinstance(kwarg_value, expected_type):
                cross_compute_module_logger.info(f"Types do not match: kwarg_value '{kwarg_value}' does not match expected_type '{expected_type}'")
                return False
                
        # Handle **kwargs
        varkwargs_param = next((param for _, param in params 
                            if param.kind == Parameter.VAR_KEYWORD), None)
        if varkwargs_param:
            remaining_kwargs = {k: v for k, v in kwargs.items() 
                            if k not in regular_kwargs}
            expected_type = type_hints.get(varkwargs_param.name, Any)
            if expected_type is not Any:
                if not all(isinstance(v, expected_type) for v in remaining_kwargs.values()):
                    cross_compute_module_logger.info(f"Types do not match: remaining_kwargs.values() '{remaining_kwargs.values()}' do not match expected_type '{expected_type}'")
                    return False
        elif any(k not in regular_kwargs for k in kwargs):  # Unknown kwargs with no **kwargs parameter
            cross_compute_module_logger.info(f"Types do not match: one or more kwarg input not found in signature.")
            return False
        
        cross_compute_module_logger.info(f"Types match!")
        return True

    def get_function_name(method_or_function, class_if_method, function_name):
        """
        Returns the function name, whether it is a function or a method.
        """
        if method_or_function == 'function':
            function = function_name
        elif method_or_function == 'method':
            function = f"{class_if_method}.{function_name}"
        else:
            function = None  # Handle unexpected cases if necessary
            cross_compute_module_logger.error(f"Warning: function or method name could not be located.\n  function_name: {function_name}\n  class_if_method: {class_if_method}\n  method_or_function: {method_or_function}")
            print(f"Warning: function or method name could not be located.\n  function_name: {function_name}\n  class_if_method: {class_if_method}\n  method_or_function: {method_or_function}")
        return [function]

    def compute(module, computation_dict, prediction_counter):
        # Re-load the module before each computation, resetting the module's namespace, effectively isolating each computation.
        reimported_ewm_module = reimport_module(module)

        arg_name_type_tuple_strings = computation_dict['instance_name(s)']['args']
        kwarg_name_type_typle_strings = computation_dict['instance_name(s)']['kwargs']
        func = None
        cls = None
        
        func_name = computation_dict['function_name']
        if computation_dict['method_or_function'] == 'function':
            # func is None if a callable object could not be found in the module
            func = getattr(reimported_ewm_module, func_name, None)
            if not func:
                cross_compute_module_logger.warning(f"Function object '{func_name}' not found in '{reimported_ewm_module}'. Computation skipped.")
                return None # If the funciton object is not available 'compute' returns None.
                              # If the computation data will not be added to the observation JSON
        elif computation_dict['method_or_function'] == 'method':
            class_path = computation_dict['class_if_method'].split('.')
            cls = reimported_ewm_module
            try:
                for part in class_path:
                    cls = getattr(cls, part)
                      # Note that 'cls' variable is also used for type checking of 'self' parameter when computing with methods
            except:
                cross_compute_module_logger.warning(f"Method object '{func_name}' not found in '{reimported_ewm_module}'. Computation skipped.")
                return None # If the method object is not available 'compute' returns None.
                              # If the computation data will not be added to the observation JSON
            func = create_method_caller(cls, computation_dict['function_name'])
            if not func:
                cross_compute_module_logger.warning(f"Method object '{func_name}' not found in class '{cls}' in '{reimported_ewm_module}'. Computation skipped.")
                return None # If the method caller cannot be created 'compute' returns None.
                              # The computation data will not be added to the observation JSON
        else:
            cross_compute_module_logger.warning(f"Invalid method_or_function value '{computation_dict['method_or_function']}' in computation dict:\n{computation_dict}")
            return None
        
        arg_tuple = generate_args_and_kwargs_from_names(reimported_ewm_module, func, arg_name_type_tuple_strings, kwarg_name_type_typle_strings)
        if arg_tuple == None:
            cross_compute_module_logger.warning("Skipping computation because of missing arguments:\n")
            return None
        else:
            # Unpack the tuple
            args, arg_names, kwargs, kwarg_names, input_names = arg_tuple
            cross_compute_module_logger.info(f"arg_tuple generated:\nargs:{args}\narg_names:{arg_names}\nkwargs:{kwargs}\nkwarg_names:{kwarg_names}\ninput_names:{input_names}")
        
        # Check that the arguments match type hints (when available) of the function or method
        type_match_bool = check_types(func, args, kwargs, cls)

        if not type_match_bool:
            cross_compute_module_logger.warning("Skipping computation because of argument types incompatible with function type hints\n")
            return None
        
        # Generate 'all_objects' value (accessed_for_prediction) for observation dict using only input and function FQ names 
            # If the computation succeeds, all_objects will be extended with 'accessed_objects'; if not, no accessed objects available and only inputs + function will be used
            # 'all_objects' will be used to sort observation data (used for context) by object name when generating conjecture questions.
        accessed_for_prediction=set()
        computation_function = get_function_name(computation_dict['method_or_function'], computation_dict['class_if_method'], computation_dict['function_name'])
        accessed_for_prediction.update(input_names)
        accessed_for_prediction.update(computation_function) 
        
        try:
            # Try to compute if the function and argument objects were found in the module
            result, variable_accesses, filtered_vars, access_graph, accessed_for_prediction = func_tracer_module.main_function(
                func=func,
                args=args, 
                kwargs=kwargs,
                module=reimported_ewm_module,
                arg_names=arg_names,
                kwarg_names=kwarg_names,
                all_objects=accessed_for_prediction,
                max_depth=128
                )
                
            cross_compute_module_logger.info(f"RESULT: {result}")
            output_dict = {
                    "prediction_id": prediction_counter,
                    "ewm_version": module_name,
                    'method_or_function': computation_dict['method_or_function'],
                    "function_name": computation_dict['function_name'],
                    'class_if_method': computation_dict['class_if_method'],
                    "instance_name(s)": computation_dict['instance_name(s)'],
                    "accessed_objects": access_graph,
                    "all_objects": accessed_for_prediction,
                    "result": result
                }
            return output_dict

        except Exception as e:
            cross_compute_module_logger.warning(f"An error occurred: {e}")
            cross_compute_module_logger.warning(f"Error type: {type(e)}")
            cross_compute_module_logger.warning(f"Error occurred while computing '{computation_dict['function_name']}' with inputs: {computation_dict['instance_name(s)']}",  exc_info=True)
            output_dict={
                "prediction_id": prediction_counter,
                "ewm_version": module_name,
                'method_or_function': computation_dict['method_or_function'],
                "function_name": computation_dict['function_name'],
                'class_if_method': computation_dict['class_if_method'],
                "instance_name(s)": computation_dict['instance_name(s)'],
                "all_objects": accessed_for_prediction,
                "computation_error": (f"{type(e)}", f"{e}") # Create a 'computation_error' key and store (error type, error message)
            }
            return output_dict

    return compute(module, computation_dict, prediction_counter)

"""
HELPER FUNCTIONS
"""
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
        del sys.modules[module_name]  # Clean up in case of error
        raise ImportError(f"Error during import of module '{module_name}': {e}")
    
    return module

def reimport_module(module):
    # Remove the module from sys.modules if it exists
    if module.__name__ in sys.modules:
        del sys.modules[module.__name__]

    # Re-import the module using your custom import function
    reloaded_module = import_module_from_path(module.__name__, module.__file__)
    
    return reloaded_module

def find_duplicate_dicts(data, source_name):
    if not isinstance(data, list):
        raise ValueError("Input should be a list of dictionaries")
    
    # Convert each dictionary to a string representation
    # If error, can use json.dumps(d, cls=CustomJSONEncoder, sort_keys=True)
    dict_strings = [json.dumps(d, sort_keys=True) for d in data]
    
    # Use defaultdict to group identical dictionaries
    duplicates = defaultdict(list)
    for i, dict_str in enumerate(dict_strings):
        duplicates[dict_str].append(i)
    
    # Print duplicates
    found_duplicates = False
    for dict_str, indices in duplicates.items():
        if len(indices) > 1:
            found_duplicates = True
            cross_compute_module_logger.warning(f"Error: Found duplicate dictionary in '{source_name}':")
            cross_compute_module_logger.warning(json.dumps(json.loads(dict_str), indent=2))
            cross_compute_module_logger.warning(f"Occurs at indices: {indices}\n")
    
    if not found_duplicates:
        cross_compute_module_logger.info(f"No duplicate dictionaries found in '{source_name}'.")
    else:
        print(f"Warning: Duplicate dictionaries found in '{source_name}'.\n")
        cross_compute_module_logger.warning(f"Duplicate dictionaries found in '{source_name}'.")

def inherit_observations(json_A_data: List[Dict[str, Any]], json_B_data: List[Dict[str, Any]], json_A_path: str, json_B_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    # Generate unique keys for each dictionary in the JSON lists
    # Note that these new dictionaries reference the input dictionaries. Modifying items in these modifies the input dictionaries. 
    json_A_key_dict = {create_prediction_key(item): item for item in json_A_data if 'computation_error' not in item}
    cross_compute_module_logger.debug(f"json_A_key_dict: {json.dumps(json_A_key_dict, indent=2)}")
    json_B_key_dict = {create_prediction_key(item): item for item in json_B_data if 'computation_error' not in item}
    cross_compute_module_logger.debug(f"json_B_key_dict: {json.dumps(json_B_key_dict, indent=2)}")

    # Find keys that are in both JSONs (shared predictions)
    keys_in_both_jsons = set(json_A_key_dict.keys()) & set(json_B_key_dict.keys())
    cross_compute_module_logger.info(f"keys_in_both_jsons:\n  {keys_in_both_jsons}")

    # List of observation keys to transfer
    observation_keys = ["prompt", "physically_consistent_observation", "observer_software", "observer_host", "observation_time", "pco_bool"]

    # Lists to store inheritance counts
    A_to_B_inheritance = []
    B_to_A_inheritance = []

    for pred_key in keys_in_both_jsons:
        item_A = json_A_key_dict[pred_key]
        cross_compute_module_logger.debug(f"item_A: {item_A}")
        item_B = json_B_key_dict[pred_key]
        cross_compute_module_logger.debug(f"item_B: {item_B}")

        # Check if all observation keys are present in either JSON A or JSON B
        json_A_has_observation = all(obs_key in item_A for obs_key in observation_keys)
        cross_compute_module_logger.debug(f"json_A_has_observation: {json_A_has_observation}")
        json_B_has_observation = all(obs_key in item_B for obs_key in observation_keys)
        cross_compute_module_logger.debug(f"json_B_has_observation: {json_B_has_observation}")

        # Transfer observation data when it exists only in one JSON
        if json_A_has_observation and not json_B_has_observation:
            cross_compute_module_logger.debug("json_A_has_observation and not json_B_has_observation")
            for obs_key in observation_keys:
                item_B[obs_key] = item_A[obs_key]
            A_to_B_inheritance.append(1)
            B_to_A_inheritance.append(0)
        elif json_B_has_observation and not json_A_has_observation:
            cross_compute_module_logger.debug("json_B_has_observation and not json_A_has_observation")
            for obs_key in observation_keys:
                item_A[obs_key] = item_B[obs_key]
            A_to_B_inheritance.append(0)
            B_to_A_inheritance.append(1)
        elif not json_A_has_observation and not json_B_has_observation:
            cross_compute_module_logger.debug("not json_A_has_observation and not json_B_has_observation")
            # No inheritance occurs for this prediction because there is no observation data to share
            A_to_B_inheritance.append(0)
            B_to_A_inheritance.append(0)
        elif json_A_has_observation and json_B_has_observation:
            cross_compute_module_logger.debug("json_A_has_observation and json_B_has_observation")
            # Print a warning when predictions are the same but observations are different (shouldn't happen when observations are being inherited).
            if item_A['pco_bool'] != item_B['pco_bool']:
                cross_compute_module_logger.warning(f"Warning: inherit_observations found same prediction but different observations for:\n{item_A}\nFrom: {json_A_path}\n{item_B}\nFrom: {json_B_path}")
            # No inheritance occurs for this prediction because both JSONs already have observation data
            A_to_B_inheritance.append(0)
            B_to_A_inheritance.append(0)

    update_json(json_path=json_A_path, new_json_data=json_A_data, extend_or_inherit='inherit')
    update_json(json_path=json_B_path, new_json_data=json_B_data, extend_or_inherit='inherit')

    # Return the updated JSON data and inheritance counts
    return json_A_data, json_B_data, A_to_B_inheritance, B_to_A_inheritance

def find_max_id(dict_list):
    max_id = float('-inf')  # Initialize to negative infinity to handle all numbers
    
    def search_dict(d):
        nonlocal max_id  # Pass 'max_id' variable from outer function to inner function
        for value in d:
            # If an 'prediction_id' key is in current dictionary
            if isinstance(value, dict):
                if 'prediction_id' in value:
                    max_id = max(max_id, value['prediction_id'])
    
    search_dict(dict_list)
    return max_id if max_id != float('-inf') else None  # Return None if no ids found


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

def convert_keys_to_strings(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            # Convert the key to a string if it's not already an acceptable type
            if not isinstance(key, (str, int, float, bool, type(None))):
                key_str = str(key)
            else:
                key_str = key
            # Recursively process the value
            new_dict[key_str] = convert_keys_to_strings(value)
        return new_dict
    elif isinstance(obj, (list, tuple, set)):
        # Recursively process iterable items
        return [convert_keys_to_strings(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Convert objects with __dict__ attribute
        return convert_keys_to_strings(vars(obj))
    else:
        # Return the object as is
        return obj

def update_json(json_path, new_json_data, extend_or_inherit):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Convert keys to strings before serialization
    computation_results_serializable = convert_keys_to_strings(new_json_data)

    # Load existing data if the file exists, otherwise start with an empty list
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    raise ValueError("Existing data is not a list")
            except json.JSONDecodeError:
                cross_compute_module_logger.warning(f"Error reading existing JSON from {json_path}. Starting with empty list.")
                existing_data = []
    else:
        existing_data = []

    if extend_or_inherit == "inherit":
        existing_data = computation_results_serializable
    elif extend_or_inherit == "extend":
        # Extend existing data with new computation results
        existing_data.extend(computation_results_serializable)
    else:
        cross_compute_module_logger.warning(f"Warning: problem occurred during update_json.")

    # Write the updated results back to the file
    with open(json_path, 'w') as f:
        json.dump(existing_data, f, cls=CustomJSONEncoder, indent=2)

    if extend_or_inherit == "inherit":
        print(f"Inherited observations saved to {json_path}\n")
    elif extend_or_inherit == "extend":
        print(f"{len(new_json_data)} new predictions added. Total: {len(existing_data)} predictions at {json_path}\n")

def print_summary_table(prev_json_data, new_json_data, in_both, only_in_prev_json, only_in_new_json, different_results):
    def split_pco_bools(dict_list):
        true_obs_dicts = []
        false_obs_dicts = []
        none_obs_dicts = []
        computation_error_dicts = []

        for item in dict_list:
            if 'pco_bool' in item:
                pco_bool_value = item.get('pco_bool')
                if pco_bool_value is True:
                    true_obs_dicts.append(item)
                elif pco_bool_value is False:
                    false_obs_dicts.append(item)
                elif pco_bool_value is None:
                    none_obs_dicts.append(item)
                # If 'pco_bool' key doesn't exist or has an unexpected value, the item is not added to any list
            elif 'computation_error' in item:
                computation_error_dicts.append(item)

        return true_obs_dicts, false_obs_dicts, none_obs_dicts, computation_error_dicts
    
    def split_sublists_into_two_lists(main_list):
        first_items = []
        second_items = []

        for sublist in main_list:
            if len(sublist) == 2:
                first_items.append(sublist[0])
                second_items.append(sublist[1])
            else:
                cross_compute_module_logger.warning(f"Warning: problem with output from compare_json_files: {sublist}")

        return first_items, second_items
    
    same_pred_prev, same_pred_new = split_sublists_into_two_lists(in_both)
    diff_pred_prev, diff_pred_new = split_sublists_into_two_lists(different_results)
    
    # Note that 'same_pred_new' is not used because 'same_pred_prev' is expected to contain the same predictions.
    in_prev_true, in_prev_false, in_prev_none, in_prev_computation_error = split_pco_bools(only_in_prev_json)
    in_new_true, in_new_false, in_new_none, in_new_computation_error = split_pco_bools(only_in_new_json)
    in_both_true, in_both_false, in_both_none, in_both_computation_error = split_pco_bools(same_pred_prev)
    diff_prev_true, diff_prev_false, diff_prev_none, diff_prev_computation_error = split_pco_bools(diff_pred_prev)
    diff_new_true, diff_new_false, diff_new_none, diff_new_computation_error = split_pco_bools(diff_pred_new)
    total_prev_true, total_prev_false, total_prev_none, total_prev_computation_error = split_pco_bools(prev_json_data)
    total_new_true, total_new_false, total_new_none, total_new_computation_error = split_pco_bools(new_json_data)

    table = [
        [f"Computations only in Last EWM", len(only_in_prev_json), len(in_prev_true), len(in_prev_false), len(in_prev_none), len(in_prev_computation_error)],
        [f"Computations only in Next EWM", len(only_in_new_json), len(in_new_true), len(in_new_false), len(in_new_none), len(in_new_computation_error)],
        [f"Shared Computations and Predictions", len(same_pred_prev), len(in_both_true), len(in_both_false), len(in_both_none), len(in_both_computation_error)],
        [f"Shared Computations, Different Predictions (Last)", len(diff_pred_prev), len(diff_prev_true), len(diff_prev_false), len(diff_prev_none), len(diff_prev_computation_error)],
        [f"Shared Computations, Different Predictions (Next)", len(diff_pred_new), len(diff_new_true), len(diff_new_false), len(diff_new_none), len(diff_new_computation_error)],
        [f"All Last EWM Predictions", len(prev_json_data), len(total_prev_true), len(total_prev_false), len(total_prev_none), len(total_prev_computation_error)],
        [f"All Next EWM Predictions", len(new_json_data), len(total_new_true), len(total_new_false), len(total_new_none), len(total_new_computation_error)]
    ]

    # Print the table; note that None PCO values are printed under 'Inconsistent with Observation" in brackets.
    print(tabulate(table, headers=["Prediction Outcomes", "Total Predictions", "PCO True", "PCO False", "PCO None", "Computation Error"], tablefmt="simple"))
    print()

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
MAIN
"""
def compare_predictions(func_tracer_module,
                        question_generator_module,
                        observation_generator_module,
                        previous_ewm_name,
                        new_ewm_name,
                        ewm_dir,
                        obs_json_dir,
                        ann_obj_dir,
                        log_dir,
                        obs_api_key,
                        obs_llm_name,
                        verbose=False):
    
    global cross_compute_module_logger
    cross_compute_module_logger = set_logging(verbose, log_dir, 'cross_compute_module')

    # Generate the observation JSON paths
    previous_json_path = f"{obs_json_dir}/{previous_ewm_name}_obs.json"
    new_json_path = f"{obs_json_dir}/{new_ewm_name}_obs.json"
    previous_ewm_path = f"{ewm_dir}/{previous_ewm_name}.py"
    new_ewm_path = f"{ewm_dir}/{new_ewm_name}.py"

    # Load JSONs
    try:
        prev_json_data = load_json(previous_json_path)
        new_json_data = load_json(new_json_path)

    except () as e:
        cross_compute_module_logger.error(f"Error: {e}")
    
    try:
        # Import EWM modules
        previous_ewm_module = import_module_from_path(previous_ewm_name, previous_ewm_path)
        new_ewm_module = import_module_from_path(new_ewm_name, new_ewm_path)

    except (ImportError, FileNotFoundError) as e:
        cross_compute_module_logger.error(f"Error: {e}")

    # Check if there are duplicate computations in the JSONs
    find_duplicate_dicts(prev_json_data, previous_ewm_name)
    find_duplicate_dicts(new_json_data, new_ewm_name)

    # Inherit existing observations (observations for shared predictions in old and new JSON are transferred from old JSON to new JSON)
    prev_json_data, new_json_data, old_to_new_inheritance, new_to_old_inheritance = inherit_observations(prev_json_data, new_json_data, previous_json_path, new_json_path)
    print(f"Observation inheritance completed. \n  {sum(old_to_new_inheritance)} observations inherited from last EWM to next EWM JSON. \n  {sum(new_to_old_inheritance)} observations inherited from next EWM to last EWM JSON.\n")
    cross_compute_module_logger.info(f"Observation inheritance completed. \n  {sum(old_to_new_inheritance)} observations inherited from last EWM to next EWM JSON. \n  {sum(new_to_old_inheritance)} observations inherited from next EWM to last EWM JSON.")

    # Generate questions for all predictions in new JSON that did not inherit observations
    question_generator_module.process_json(new_ewm_name, obs_json_dir, ann_obj_dir)
        
    # Generate observations for all predicitons in new JSON that have not inherited observations
    observation_generator_module.main_function(
        ewm_name=new_ewm_name, 
        obs_json_dir=obs_json_dir, 
        log_dir=log_dir,
        obs_api_key=obs_api_key, 
        obs_llm_name=obs_llm_name,
        verbose=False)

    # Re-load updated JSONs from paths (replaces old contents of json_data variables)
    try:
        prev_json_data = load_json(previous_json_path)
        new_json_data = load_json(new_json_path)

    except () as e:
        cross_compute_module_logger.error(f"Error: {e}")

    # Compare predictions
    cross_compute_module_logger.info("*** running 'compare_json_files' ***")
    only_in_prev_json, only_in_new_json, different_results, in_both = compare_json_files(prev_json_data, new_json_data)

    # Print summary table
    print("Before cross-computing predictions:")
    print_summary_table(prev_json_data, new_json_data, in_both, only_in_prev_json, only_in_new_json, different_results)

    print("Attempting to reproduce different computations across old and new EWMs...\n")

    # Use previous EWM to compute any reproducible computations
    reproducible_computations_from_new_ewm = []
    # Get the id of the last prediction in the last EWM's obs JSON
    last_prediction_id = find_max_id(prev_json_data)
    if last_prediction_id == None:
        cross_compute_module_logger.error(f"Unable to find previous prediction id values in last EWM ('{previous_ewm_name}') observation dict: {previous_json_path}")
    prediction_counter = last_prediction_id+1

    for computation_dict in only_in_new_json:
        reproduced_computation_dict = compute_across_modules(previous_ewm_module, previous_ewm_name, computation_dict, func_tracer_module, prediction_counter=prediction_counter)

        if reproduced_computation_dict is not None:
            reproducible_computations_from_new_ewm.append(reproduced_computation_dict)
            prediction_counter += 1
    
    # Extend the previous JSON file with the reproduced computations
    update_json(previous_json_path, reproducible_computations_from_new_ewm, extend_or_inherit='extend')

    # Use new EWM to compute any reproducible computations
    reproducible_computations_from_old_ewm = []
    # Get the id of the last prediction in the last EWM's obs JSON
    last_prediction_id = find_max_id(new_json_data)
    if last_prediction_id == None:
        cross_compute_module_logger.error(f"Unable to find next prediction id values in next EWM ('{new_ewm_name}') observation dict: {new_json_path}")
    prediction_counter = last_prediction_id +1

    for computation_dict in only_in_prev_json:
        reproduced_computation_dict = compute_across_modules(new_ewm_module, new_ewm_name, computation_dict, func_tracer_module, prediction_counter=prediction_counter)

        if reproduced_computation_dict is not None:
            reproducible_computations_from_old_ewm.append(reproduced_computation_dict)
            prediction_counter += 1
   
    # Extend the new JSON file with the reproduced computations
    update_json(new_json_path, reproducible_computations_from_old_ewm, extend_or_inherit='extend')
   
    # Reload the updated JSON files and compare them (after cross-computation)
    try:
        # Load JSONs
        prev_json_data = load_json(previous_json_path)
        new_json_data = load_json(new_json_path)

    except () as e:
        cross_compute_module_logger.error(f"Error: {e}")

    # Check if there are duplicate computations in the JSONs
    find_duplicate_dicts(prev_json_data, previous_ewm_name)
    find_duplicate_dicts(new_json_data, new_ewm_name)

    # Inherit existing observations (moves any observations for shared predictions from new JSON that were just cross-computed in old JSON)
    prev_json_data, new_json_data, old_to_new_inheritance, new_to_old_inheritance = inherit_observations(prev_json_data, new_json_data, previous_json_path, new_json_path)
    print(f"Observation inheritance completed. \n{sum(old_to_new_inheritance)} observations inherited from old to new JSON. \n{sum(new_to_old_inheritance)} observations inherited from new to old JSON.")

    # Generate questions for all new predictions in old JSON that did not inherit observations
    question_generator_module.process_json(previous_ewm_name, obs_json_dir, ann_obj_dir)
        
    # Generate observations for all new predicitons in old JSON that have not inherited observations
    observation_generator_module.main_function(
        ewm_name=previous_ewm_name, 
        obs_json_dir=obs_json_dir, 
        log_dir=log_dir,
        obs_api_key=obs_api_key, 
        obs_llm_name=obs_llm_name,
        verbose=False)

    # Re-load updated JSONs from paths
    try:
        prev_json_data = load_json(previous_json_path)
        new_json_data = load_json(new_json_path)

    except () as e:
        cross_compute_module_logger.error(f"Error: {e}")

    # Inherit existing observations (this is expected to do nothing unless the old JSON lacked observations before 
    # being passed to ewm_evaluator - should only occur with seed EWM)
    prev_json_data, new_json_data, old_to_new_inheritance, new_to_old_inheritance = inherit_observations(prev_json_data, new_json_data, previous_json_path, new_json_path)
    print(f"Observation inheritance completed. \n{sum(old_to_new_inheritance)} observations inherited from old to new JSON. \n{sum(new_to_old_inheritance)} observations inherited from new to old JSON.\n")

    # Compare JSON data
    cross_compute_module_logger.info("*** running 'compare_json_files' ***")
    only_in_prev_json, only_in_new_json, different_results, in_both = compare_json_files(prev_json_data, new_json_data)

    # Print second summary table
    print("After cross-computing predictions:")
    print_summary_table(prev_json_data, new_json_data, in_both, only_in_prev_json, only_in_new_json, different_results)
    
    print(f"Prediction comparison completed. JSON files updated with new and inherited observations.\n  Old JSON: {previous_json_path}\n  New JSON: {new_json_path}\n")
    
    return (prev_json_data, new_json_data)

if __name__ == "__main__":
    print("Test prediction comparison:")

    last_ewm_name = '20241217_114911_525850'
    next_ewm_name = 'REJECTED_20241217_142901_039977'

    """
    PATHS TO EWM DIRECTORIES
    """
    # Path to EWM builder module directory
    learning_module_dir = ".../generator_functions"
    # Path to EWM python file directory
    ewm_dir = ".../generator_functions/ewm_files"
    # Path to observation json directory
    obs_dir = ".../generator_functions/obs_ann_conj_files"
    # Path to annotated object json directory
    ann_obj_dir = ".../generator_functions/obs_ann_conj_files"
    # Path to log file directory
    log_dir = ".../generator_functions/log_files"

    """
    PATHS TO PROCESSING MODULES
    """
    function_tracer_name = "4_func_tracer_02_28"
    function_tracer_path = f"{learning_module_dir}/{function_tracer_name}.py"

    question_generator_name = "o_question_generator_06"
    question_generator_path = f"{learning_module_dir}/{question_generator_name}.py"

    observation_generator_name = "observer_interface_huggingface_api_07"
    observation_generator_path = f"{learning_module_dir}/{observation_generator_name}.py"

    """
    Import EWM learning modules
    """
    try:
        function_tracer_module = import_module_from_path(function_tracer_name, function_tracer_path)
        question_generator_module = import_module_from_path(question_generator_name, question_generator_path)
        observation_generator_module = import_module_from_path(observation_generator_name, observation_generator_path)

    except (ImportError, FileNotFoundError) as e:
        print(f"Error: {e}")

    pred_comparison_message = compare_predictions(
        # Modules
        func_tracer_module=function_tracer_module,  
        question_generator_module=question_generator_module,
        observation_generator_module=observation_generator_module,
        # EWM names
        previous_ewm_name=last_ewm_name, 
        new_ewm_name=next_ewm_name, 
        # Directories
        ewm_dir=ewm_dir, 
        obs_json_dir=obs_dir,
        ann_obj_dir=ann_obj_dir,
        log_dir=log_dir,
        # LLM info
        obs_api_key='test_key',
        obs_llm_name='test_name',
        # Logging style
        verbose=False)  
    
    # print(pred_comparison_message)