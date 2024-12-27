import inspect
import hashlib
from typing import get_origin, get_args, get_type_hints, Union, Callable, Tuple, Generator, Optional, Type, Dict, Set, Any, List
import types
from types import MappingProxyType, GetSetDescriptorType
import sys
import importlib
import logging
import json
import os
import random
import functools
from datetime import datetime

"""
COMPUTE EVERY FUNCITON WITH EVERY INSTANCE AND SAVE RESULTS 
"""
prediction_generator_module_logger = None
DEFAULT_VALUE_PLACEHOLDER = 'l8jNWV52p34HjK'

def auto_compute(module, ewm_version: str, max_results_per_func: int = 8, func_tracer_module=None):

    # Generate a tuple of all class objects (module-level and nested), excluding imported classes. 
    def get_nested_classes(cls):
        nested = []
        for name, obj in inspect.getmembers(cls):
            if inspect.isclass(obj) and obj.__module__ == cls.__module__:
                nested.append(obj)
                nested.extend(get_nested_classes(obj))
        return nested
    
    def get_all_classes(module) -> tuple:
        class_types = []
        for name, cls in inspect.getmembers(module, inspect.isclass):
            class_types.append(cls)
            # Recursively get nested classes
            class_types.extend(get_nested_classes(cls))
        return tuple(class_types)

    """
    # Note that it does not store built-in classes (list, str, int, etc) which are not defined in the module itself. Built-in classes should not be treated as knowledge because they cannot be updated. 
    # Main disadvantage of using nested classes seems to be more complex code and less tool support.
    # Some advantages of using nested classes:
    # Namespace pollution: Without nesting, you might end up with more classes in the global namespace of a module.
    # Less expressive design: In some cases, nested classes can express relationships between classes more clearly.
    # Potential for longer names: To avoid naming conflicts, you might need to use longer, more descriptive names for classes that could otherwise be nested.
    """

    # Generate a dictionary of all module level instances (name, obj) tuples, excluding imported instances, grouped by their type. 
    def create_instance_dict(module, class_types) -> Dict[type, List[Tuple[str, Any]]]:
        instance_dict = {}
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, class_types) and obj.__module__ == module.__name__:
                obj_type = type(obj)
                if obj_type not in instance_dict:
                    instance_dict[obj_type] = []
                instance_dict[obj_type].append((name, obj))
        
        # Print summary
        total_instances = sum(len(instances) for instances in instance_dict.values())
        instance_names = [name for instances in instance_dict.values() for name, _ in instances]
        prediction_generator_module_logger.info(f"Found {total_instances} instances: {', '.join(instance_names)}\n")
        
        return instance_dict

    """
    # Note 1: Instances created within functions or methods won't be captured unless they're assigned to module-level variables. 
          This should not be a problem because with optimal EWM connection, functions will call outside instances rather than defining them internally.
    # Note 2: Instance objects do not store their own varible name data (unlike classes and functions which have the attribute __name__). Therefore, we store tuples.
    """

    # Use 'instance_dict' to generate a dictionary of all attributes of the module level instances.
    #   The format is the same as instance_dict: (name, obj) tuples, excluding imported instances, grouped by their type. 
    #   Max depth is set to 128 by default
    def create_attribute_dict(instance_dict: Dict[type, List[Tuple[str, Any]]], max_depth: int = 128) -> Dict[type, List[Tuple[str, Any]]]:
        attribute_dict = {}

        def add_attributes(obj, prefix='', current_depth=0):
            if current_depth >= max_depth:
                return
            if not hasattr(obj, '__dict__'):
                return
            for attr_name, value in obj.__dict__.items():
                if not attr_name.startswith('_'):  # Skip private attributes
                    full_attr_name = f"{prefix}.{attr_name}" if prefix else attr_name
                    value_type = type(value)
                    # Add the attribute to the attribute_dict under its value's type
                    if value_type not in attribute_dict:
                        attribute_dict[value_type] = []
                    attribute_dict[value_type].append((full_attr_name, value))

                    # Recursively add attributes of the current value
                    add_attributes(value, full_attr_name, current_depth + 1)

        # Iterate over all instances in instance_dict
        for instances in instance_dict.values():
            for instance_name, instance_obj in instances:
                add_attributes(instance_obj, instance_name)

        return attribute_dict

    def get_nested_functions(func_obj):
        """Get nested functions with their proper qualified names"""
        nested = []
        for name, obj in inspect.getmembers(func_obj):
            if (isinstance(obj, types.FunctionType) and 
                obj.__module__ == func_obj.__module__):
                nested.append(obj)
                nested.extend(get_nested_functions(obj))
        return nested

    def get_all_functions(module):
        """Get all functions including nested ones from module"""
        functions = []
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                functions.append(obj)
                functions.extend(get_nested_functions(obj))
        return functions

    # Generate a set of tuples of all class names and names of methods defined directly in the class (module-level and nested), excluding imported functions.
    def get_direct_methods(class_types) -> set:
        methods = set()
        for cls in class_types:
            for name, method in cls.__dict__.items():
                if callable(method) and not name.startswith("__"):
                    methods.add((cls, name))
        return methods

    """
    Note that the current method:
    - will include inherited methods, not just methods defined in the class itself.
    - will catch static methods and class methods as well as instance methods.
    - might catch other callable attributes that aren't strictly methods (like a callable object stored as a class attribute).
    
    """

    def create_method_caller(cls, method_name):
        method = getattr(cls, method_name)
        
        @functools.wraps(method)
        def method_caller(*args, **kwargs):
            return method(*args, **kwargs)
        
        return method_caller
    
    def generate_combinations(all_arg_compatible_instances: List[List[Any]], var_arg_indices: List[int], max_results_per_func: int, func:Tuple) -> Generator[Tuple[Any, ...], None, None]:
        seen = set()
        total_possible_combinations = 1
        for single_arg_compatible_instances in all_arg_compatible_instances:
            total_possible_combinations *= len(single_arg_compatible_instances)
            random.shuffle(single_arg_compatible_instances)  # Shuffle each list of compatible instances
        
        if var_arg_indices:
            prediction_generator_module_logger.warning(f"Warning: Not all combinations for {func[0]} '{func[1].__name__}' will be computed. Function contains a VARIABLE ARGUMENT. Infinite possible combinations, {total_possible_combinations} compatible values. Max results: {max_results_per_func}")

        if total_possible_combinations > max_results_per_func:
            prediction_generator_module_logger.warning(f"Warning: Not all combinations for {func[0]} '{func[1].__name__}' will be computed. Possible combinations: {total_possible_combinations}. Max results: {max_results_per_func}")
        
        count = 0
        attempts = 0
        max_attempts = max_results_per_func * 8  # Limit the number of attempts to avoid infinite loops

        while count < max_results_per_func and attempts < max_attempts:
            # Generate a single random combination
            # Note that each item in arg_list is a tuple of (instance_name, instance_obj)
            comb = tuple(random.choice(arg_list) for arg_list in all_arg_compatible_instances)
            prediction_generator_module_logger.info(f"comb: {comb}")

            # Creat 'modified combination' for variable arguments by converting 'comb' into a list, changing the argument from one item to a tuple of two items, then converting the list back to a tuple.
            modified_comb = list(comb)
            for i in var_arg_indices:
                modified_comb[i] = (comb[i], random.choice(all_arg_compatible_instances[i]))
            
            modified_comb = tuple(modified_comb)
            
            # Check for duplicates
            # Convert to JSON string (sorts dictionary keys for consistency)
            json_str = json.dumps(modified_comb, cls=CustomJSONEncoder)
            # Create hash using the JSON string
            comb_hash=hashlib.md5(json_str.encode()).hexdigest()

            if comb_hash not in seen:
                seen.add(comb_hash)
                count += 1
                ## A printout to detect infinite loops in suspicious functions. 
                # if func[1].__name__ == "sqrt":
                #     print(f"yielding modified_comb: {modified_comb}")
                yield modified_comb
            
            attempts += 1
        
        if attempts >= max_attempts:
            prediction_generator_module_logger.warning(f"Warning: Reached maximum attempts ({max_attempts}) for function {func[1].__name__}. Generated {count} unique combinations.")

    def find_compatible_items_in_type_dict(arg_type, type_dict):
        compatible_for_arg = []
        # prediction_generator_module_logger.info(f"arg_type: {arg_type}\nid(arg_type):{id(arg_type)}")
        # prediction_generator_module_logger.info(f"arg_type.__module__: {arg_type.__module__}")
        # prediction_generator_module_logger.info(f"id(arg_type.__module__): {id(sys.modules[arg_type.__module__])}")

        for type_name, instances in type_dict.items():
            # prediction_generator_module_logger.info(f"type_name: {type_name}\nid(type_name): {id(type_name)}")
            # prediction_generator_module_logger.info(f"type_name.__module__: {type_name.__module__}")
            # prediction_generator_module_logger.info(f"id(type_name.__module__): {id(sys.modules[type_name.__module__])}")
            for instance_name, instance_obj in instances:
                # prediction_generator_module_logger.info(f"instance_name: {instance_name}\ninstance_obj:{instance_obj}\nid(instance_obj.__class__): {id(instance_obj.__class__)}")
                # prediction_generator_module_logger.info(f"instance_obj.__class__.__module__: {instance_obj.__class__.__module__}")
                # prediction_generator_module_logger.info(f"id(instance_obj.__class__.__module__): {id(sys.modules[instance_obj.__class__.__module__])}")
                # If the argument type hint is 'Any' add every instance in instance_dict to 'compatible_for_arg'.
                if arg_type is Any:
                    compatible_for_arg.append((instance_name, instance_obj))
                # Only add instance objects that match the argument type hint
                elif isinstance(instance_obj, arg_type):
                    # Note that isinstance() has special handling for Union types and should match any of the types in the Union.
                    compatible_for_arg.append((instance_name, instance_obj))        
        prediction_generator_module_logger.info(f"compatible_for_arg: {compatible_for_arg}")
        return compatible_for_arg
    
    def get_instance(module, instance_name):
        parts = instance_name.split('.')
        obj = module
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise AttributeError(f"{obj} does not have attribute '{part}'")
        return obj
    
    def generate_args_and_kwargs(func, combo, module):
        """
        Generate formatted argument lists and kwarg dictionares for the function.
        Default values are used as a last option if no compatible defined objects can be found.
        When default values are used, a placeholder is inserted into the arg or kwarg variables.
        """
        # Get the function's signature
        sig = inspect.signature(func)
        parameters = list(sig.parameters.values())
        
        arg_name_type_tuples = []
        arg_names = []
        kwarg_name_type_tuples = {}
        kwarg_names = {}
        param_index = 0
        
        # Generate list of argument names and dictionary of kwarg names
        for arg in combo:
            if param_index >= len(parameters):
                break
            param = parameters[param_index]
            
            # Default or positional-only argument types append to args
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or inspect.Parameter.POSITIONAL_ONLY:
                if arg == DEFAULT_VALUE_PLACEHOLDER:
                    # Append the default value of the parameter
                    arg_name_type_tuples.append(DEFAULT_VALUE_PLACEHOLDER)
                    arg_names.append(DEFAULT_VALUE_PLACEHOLDER)
                else:
                    arg_name = arg[0]
                    arg_object = arg[1]
                    # Each item in arg_list is a tuple of (instance_name, instance_obj)
                    # Update arg_names list to store tuples of (instance name, instance class)
                    if type(arg_object):
                        arg_name_type_tuples.append((arg_name, type(arg_object).__name__))
                    else:
                        arg_name_type_tuples.append((arg_name,))
                        prediction_generator_module_logger.warning(f"Unable to find type for arg_name: {arg_name}; arg_object: {arg_object}")
                    arg_names.append(arg_name)
            
            # Note that variable arguments (*args and *kwargs) cannot accept default values, so DEFAULT_VALUE_PLACEHOLDER should not appear here and special handling is not needed        
            # *args parameter 
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                # check if the value in arg_list is a tuple of value tuples    
                if isinstance(arg, tuple) and any(isinstance(item, tuple) for item in arg):
                    # add the instance values from the args tuple to the args list
                    for single_arg_instance_tuple in arg:
                        single_arg_instance_tuple_name = single_arg_instance_tuple[0]
                        single_arg_instance_tuple_object = single_arg_instance_tuple[1]
                        if type(single_arg_instance_tuple_object):
                            arg_name_type_tuples.append((single_arg_instance_tuple_name, type(single_arg_instance_tuple_object)))
                        else:
                            arg_name_type_tuples.append((single_arg_instance_tuple_name,))
                        arg_names.append(single_arg_instance_tuple_name)
                else: 
                    arg_name_type_tuples.append((arg[0],))
                    arg_names.append(arg[0])

            # keyword arguments should probably be ignored (attributes passed to func will not have keywords and param.name may be a useless/conflicting keyword)
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                if arg == DEFAULT_VALUE_PLACEHOLDER:  # meaningless string 'l8jNWV52p34HjK' is placeholder for default value
                    kwarg_name_type_tuples[param.name] = DEFAULT_VALUE_PLACEHOLDER
                    kwarg_names[param.name] = DEFAULT_VALUE_PLACEHOLDER
                else:
                    arg_name = arg[0]
                    arg_object = arg[1]
                    if type(arg_object):
                        kwarg_name_type_tuples[param.name] = (arg_name, type(arg_object).__name__)
                    else:
                        kwarg_name_type_tuples[param.name] = (arg_name,)
                        prediction_generator_module_logger.warning(f"Unable to find type for kwarg: {param.name}; arg_name: {arg_name}; arg_object: {arg_object}")
                    kwarg_names[param.name] = arg_name
                
            # **kwargs parameter
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                # Ignoring variable kwargs for now (unclear how to get param.name for these argments)
                pass

            else:
                prediction_generator_module_logger.warning(f"Warning: something went wrong in call_function_with_arg_list")
            
            param_index += 1

        # arg and kwarg objects must be regenerated from names because module is re-imported after each prediction
        args = []
        kwargs = {}

        # Generate list of arg objects
        for i, arg_name_tuple in enumerate(arg_name_type_tuples):
            if arg_name_tuple == DEFAULT_VALUE_PLACEHOLDER:
                if i < len(parameters) and parameters[i].default != inspect.Parameter.empty:
                    default_arg_object = parameters[i].default
                    args.append(default_arg_object)
                else:
                    args.append(None)
            else:
                arg_name = arg_name_tuple[0]
                try:
                    arg_object = get_instance(module, arg_name)
                    args.append(arg_object)
                except AttributeError as e:
                    raise AttributeError(f"Error in args: {str(e)}")
        
        # Generate dictionary of kwarg objects
        for kwarg_name, value_name_tuple in kwarg_name_type_tuples.items():
            param = sig.parameters.get(kwarg_name)
            if value_name_tuple == DEFAULT_VALUE_PLACEHOLDER:
                if param and param.default != inspect.Parameter.empty:
                    kwargs[kwarg_name] = param.default
                else:
                    kwargs[kwarg_name] = None
            else:
                value_name = value_name_tuple[0]
                try:
                    kwargs[kwarg_name] = get_instance(module, value_name)
                except AttributeError as e:
                    raise AttributeError(f"Error in kwargs: {str(e)}")

        return args, arg_name_type_tuples, arg_names, kwargs, kwarg_name_type_tuples, kwarg_names

    def get_inputs(arg_names, kwarg_names):
        """
        Returns a list of FQ names of all input objects.
        """
        inputs = []
        # Get positional arguments
        inputs.extend(arg_names)
        # Get keyword argument values
        inputs.extend(kwarg_names.values())
        return inputs

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
            prediction_generator_module_logger.error(f"Warning: function or method name could not be located.\n  function_name: {function_name}\n  class_if_method: {class_if_method}\n  method_or_function: {method_or_function}")
        return [function]

    def generate_arg_combos(
        func: Tuple[str, Callable, Optional[Type]], # The str designates "method" or "funcion", the optional type is for storing classes of wrapped methods.
        instance_dict: Dict[type, List[Tuple[str, Any]]],
        max_results_per_func: int,
        ) -> List[Dict[str, Any]]:

        # Get function name and parameters
        method_or_function = func[0]
        method_class = func[2]
        func_name = func[1].__name__
        signature = inspect.signature(func[1])

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
        parameters = signature.parameters

        # Get type hints for the function.  For a Union type, this will return the Union object itself (including the types it stores).
        type_hints = get_type_hints(func[1])

        prediction_generator_module_logger.info(f"Searching for matching instances.\n  func_name: {func_name}\n  func:{func}\n  signature:{signature}\n  type_hints: {type_hints}")

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

        # This step may be unnecessary: Remove 'return' from type hints if present
        if 'return' in type_hints:
            del type_hints['return']

        # Prepare a list to store compatible instances for each argument
        compatible_instances = []
        var_arg_indices: List[int] = []  # To keep track of which arguments are variable

        for i, (param_name, param) in enumerate(parameters.items()):
            # Get types for parameters from type_hints (this gets the type objects, not string names)
            # When type hint is absent, default to Any type
            arg_type = type_hints.get(param_name, Any)
            # prediction_generator_module_logger.info(f"arg_type: {arg_type}")
            # prediction_generator_module_logger.info(f"id(arg_type): {id(arg_type)}")
            # Check if this is the 'self' parameter of a method
            if param_name == 'self' and method_or_function == 'method':
                if method_class is not None: 
                    arg_type = method_class
                    # prediction_generator_module_logger.info(f"method_class: {method_class}")
                    # prediction_generator_module_logger.info(f"id(method_class): {id(method_class)}")
                else:
                    prediction_generator_module_logger.warning(f"Method class type is not provided for method '{func_name}'.")
            
            prediction_generator_module_logger.info(f"Searching instance_dict for any object compatible with type '{arg_type}'")
            compatible_for_arg = find_compatible_items_in_type_dict(arg_type, instance_dict)
            if not compatible_for_arg:
                prediction_generator_module_logger.info(f"No compatible instances found for '{func_name}', parameter '{param_name}' of type '{arg_type}'\n\ninstance_dict:\n{instance_dict}")

            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                var_arg_indices.append(i)
                prediction_generator_module_logger.info(f"Variable argument present in {func_name}, tuples will be generated for parameter {param_name}")
            
            compatible_instances.append(compatible_for_arg)

        # If any arguments lack compatible instances and lack default values, list any compatible attributes in existing instances.
        for i, arg_instances in enumerate(compatible_instances):
            # Check if no instances matching the argument type were found. 
            if not arg_instances:
                param_items = list(parameters.items())
                param_name, param = param_items[i]
                arg_type = type_hints.get(param_name, Any)
                
                # Check attribute dict for compatible attributes
                prediction_generator_module_logger.info(f"Searching attribute_dict for any object compatible with type '{arg_type}'")
                compatible_for_arg =  find_compatible_items_in_type_dict(arg_type, attribute_dict)
                
                # Look for compatible attributes in the instances
                if compatible_for_arg:
                    compatible_instances[i] = compatible_for_arg

                # If no attributes found, check if the argument has a default value.
                  # This implies that the default value is not a defined instance or attribute and does not need to be included in tracked objects (e.g. 'inputs' or 'all_objects')
                else:
                    prediction_generator_module_logger.info(f"No compatible attributes found for '{func_name}', parameter '{param_name}' of type '{arg_type}'")
                    if param.default is not param.empty:
                        # If a default arg exists, insert meaningless string ('l8jNWV52p34HjK') as placeholder for a default value
                        compatible_instances[i] = [(DEFAULT_VALUE_PLACEHOLDER, param.default)]
                        prediction_generator_module_logger.info(f"Parameter '{param_name}' of type '{arg_type}' for function {func_name} has a default value: {param.default}")
                    else:
                        prediction_generator_module_logger.warning(f"No compatible instances, attributes, or default value found for '{func_name}', parameter '{param_name}' of type '{arg_type}'")
                        return None

        # Generate combinations using generate_combinations
        selected_combinations = list(generate_combinations(compatible_instances, var_arg_indices, max_results_per_func, func))

        return selected_combinations

    def apply_function_to_combos(
        method_or_function,
        func_name,
        method_class,
        selected_combinations,
        prediction_counter: int
        ) -> List[Dict[str, Any]]:

        class_name_if_method = str(method_class.__qualname__) if method_class else method_class
        prediction_generator_module_logger.info(f"Applying function to arg combos:\nmethod_or_function: {method_or_function}\n  func_name: {func_name}\n  class_name_if_method: {class_name_if_method}")

        # Apply argument combinations to function and return details of computation
        results = []

        prediction_generator_module_logger.info(f"selected_combinations: {selected_combinations}")
        if not selected_combinations:
            return None
        
        # Record computation details
        for combination in selected_combinations:
            prediction_generator_module_logger.info(f"combination: {combination}")
            # Re-import the module before each computation. Note that the more efficient 'reload' of the module did not work,
            # likely because of the custom module paths used, Reloading resets the module's namespace effectively isolating each computation.
            reloaded_ewm_module = reimport_module(module)

            reloaded_callable_func = None
            args = []
            kwargs = {}

            if method_or_function == 'function':
                # func is None if a callable object could not be found in the module
                reloaded_callable_func = getattr(reloaded_ewm_module, func_name, None)
            elif method_or_function == 'method':
                class_path = class_name_if_method.split('.')
                cls = reloaded_ewm_module
                for part in class_path:
                    cls = getattr(cls, part)
                reloaded_callable_func = create_method_caller(cls, func_name)
            else:
                prediction_generator_module_logger.warning(f"Invalid method_or_function: {method_or_function}")
                return None

            if reloaded_callable_func is None:
                prediction_generator_module_logger.warning(f"{method_or_function} '{func_name}' not found.")
                return None
                
            # Unpack the combination tuples to get the instance objects and names, separated as args and kwargs
            args, arg_name_type_tuples, arg_names, kwargs, kwarg_name_type_tuples, kwarg_names = generate_args_and_kwargs(reloaded_callable_func, combination, reloaded_ewm_module)
            string_tuple_arg_names = [str(arg_tuple) for arg_tuple in arg_name_type_tuples]
            string_tuple_kwarg_names = {key: str(value) for key, value in kwarg_name_type_tuples.items()}

            # Generate 'all_objects' value (accessed_for_prediction) for observation dict using only input and function FQ names 
                # If the computation succeeds, all_objects will be extended with 'accessed_objects'; if not, no accessed objects available and only inputs + function will be used
                # 'all_objects' will be used to sort observation data (used for context) by object name when generating conjecture questions.
            accessed_for_prediction=set()
            computation_function = get_function_name(method_or_function, class_name_if_method, func_name)
            computation_inputs= get_inputs(arg_names, kwarg_names)
            accessed_for_prediction.update(computation_inputs)
            accessed_for_prediction.update(computation_function)
            
            try:
                # Compute function with object access tracing
                result, variable_accesses, filtered_vars, access_graph, accessed_for_prediction = func_tracer_module.main_function(
                    func=reloaded_callable_func,
                    args=args, 
                    kwargs=kwargs,
                    module=reloaded_ewm_module,
                    arg_names = arg_names, # A list or argument names (e.g. ['earth', 'earth_radius.value'])
                    kwarg_names = kwarg_names, # A dictionary of kwargs where keys are parameter names and values are the names of input objects (e.g. {'radius':'earth_radius'})
                    all_objects = accessed_for_prediction,
                    max_depth=128) 
                   # Note that instance_dict and attribute_dict are recreated inside func_tracer each time it is called. The correct object IDs must be found after each time the module is re-imported

                prediction_generator_module_logger.info(f"RESULT: {result}")
                results.append({
                    "prediction_id":prediction_counter,
                    "ewm_version": ewm_version,
                    'method_or_function': method_or_function,
                    "function_name": func_name,
                    'class_if_method': class_name_if_method,
                    "instance_name(s)": {
                            "args": string_tuple_arg_names,
                            "kwargs": string_tuple_kwarg_names
                        },
                    "accessed_objects": access_graph,
                    "all_objects": accessed_for_prediction,
                    "result": result
                })
                prediction_counter +=1

            except Exception as e:
                prediction_generator_module_logger.warning(f"Error occurred while computing '{reloaded_callable_func}' with inputs: args:{arg_name_type_tuples}; kwargs: {kwarg_name_type_tuples}\nError type: {type(e)}\nError: {e}",  exc_info=True)   
                   
                results.append({
                    "prediction_id":prediction_counter,
                    "ewm_version": ewm_version,
                    'method_or_function': method_or_function,
                    "function_name": func_name,
                    'class_if_method': class_name_if_method,
                    "instance_name(s)": {
                            "args": string_tuple_arg_names,
                            "kwargs": string_tuple_kwarg_names
                        },
                    "all_objects": accessed_for_prediction,
                    "computation_error": (f"{type(e)}", f"{e}") # Create a 'computation_error' key and store (error type, error message)
                })
                prediction_counter +=1
                continue

        return prediction_counter, results

    prediction_generator_module_logger.info(f"Starting auto_compute for module: {module.__name__}\n")

    # class_types is a tuple of class objects (e.g. "(<class 'tracer_test_02.Mass'>, <class 'tracer_test_02.Name'>,...)")
    class_types = get_all_classes(module)
    
    # instance_dict is a dictionary with class keys and list values. The lists contain tuples of instance names and objects.
    #  e.g. "{<class 'tracer_test_02.Planet'>: [('earth', <tracer_test_02.Planet object at 0x7f7ece8b1870>)],..."
    instance_dict = create_instance_dict(module, class_types)
    prediction_generator_module_logger.info(f"instance_dict: \n{instance_dict}")

    # attribute_dict is a dictionary with class keys and list values. The lists contain tuples of attribute fq names and objects.
    # e.g. "{<class 'str'>: [('earth.name', 'Earth'), ('earth.mass.value.unit.name.name', 'kilogram'),..."
    attribute_dict = create_attribute_dict(instance_dict, max_depth=8)
    prediction_generator_module_logger.info(f"attribute_dict: \n{attribute_dict}")

    # Functions is a list of function objects
    functions = get_all_functions(module)
    prediction_generator_module_logger.info(f"Found {len(functions)} functions:")
    prediction_generator_module_logger.info(f"functions: \n{functions}")
    for func in sorted(functions, key=lambda x: x.__name__):
        prediction_generator_module_logger.info(f"  {func.__name__}")
        
    # Use the class_types already collected to generate a set of all methods.
    # methods is a set of tuples of class object and method name (e.g. "{(<class 'tracer_test_02.Planet'>, 'calculate_gravity'),...")
    methods = get_direct_methods(class_types)

    prediction_generator_module_logger.info(f"Found {len(methods)} methods:")
    for cls, method in sorted(methods, key=lambda x: (x[0].__name__, x[1])):
        prediction_generator_module_logger.info(f"  {cls.__name__}.{method}")

    # A list that stores function and argument combos
    list_of_func_combos_lists = []
    for func in functions:
        func_tuple = ('function', func, None)
        selected_combinations = generate_arg_combos(
                                    func_tuple, # The str designates "method" or "funcion", the optional type is for storing classes of wrapped methods.
                                    instance_dict,
                                    max_results_per_func)
        list_of_func_combos_lists.append([func_tuple, selected_combinations])
    for cls, method in methods:
        callable_method = create_method_caller(cls, method)
        func_tuple = ('method', callable_method, cls)
        selected_combinations = generate_arg_combos(
                                    func_tuple, # The str designates "method" or "funcion", the optional type is for storing classes of wrapped methods.
                                    instance_dict,
                                    max_results_per_func)
        list_of_func_combos_lists.append([func_tuple, selected_combinations])

    # Compute results for all compatible combinations of functions/methods and instances
    computed_results = []
    # Initialize a counter to assign a sequential ID to each prediction
    prediction_counter = 1
    
    for func_combos_list in list_of_func_combos_lists:
        func_tuple = func_combos_list[0]
        method_or_function = func_tuple[0]
        func_name = func_tuple[1].__name__
        method_class = func_tuple[2]
        selected_combinations = func_combos_list[1]

        results = apply_function_to_combos(method_or_function,
                                        func_name,
                                        method_class,
                                        selected_combinations,
                                        prediction_counter)
        if results:
            last_id_num, func_results = results
            computed_results.extend(func_results)
            prediction_counter = last_id_num
        else:
            prediction_generator_module_logger.warning(f"No computations could be generated for '{func}")

    return computed_results


"""
SAVE RESULTS TO JSON
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

def save_results(temp_json_path, computation_results):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(temp_json_path), exist_ok=True)

    # Convert keys to strings before serialization
    computation_results_serializable = convert_keys_to_strings(computation_results)

    # Write the results to the file, overwriting if it already exists
    with open(temp_json_path, 'w') as f:
        json.dump(
            computation_results_serializable,
            f,
            indent=2,
            cls=CustomJSONEncoder
        )

    prediction_generator_module_logger.info(f"{len(computation_results)} predictions saved at {temp_json_path}\n")
    print(f"{len(computation_results)} predictions saved at {temp_json_path}\n")

"""
SUMMARIZE COMPUTATION RESULTS
"""
def analyze_computation_results(computation_results) -> None:
    # Count the number of results for each function/method name
    function_counts = {}
    
    # Count the number of times each instance was used
    instance_counts = {}

    def increment_count(dictionary, key):
        dictionary[key] = dictionary.get(key, 0) + 1

    ## Activate this if you only want to track the highest level instance name
    # def process_instance_name(name):
    #     # Extract the highest level name (before the first dot)
    #     return name.split('.')[0]

    for computation_result in computation_results:
        # Count function/method occurrences
        function_name = computation_result['function_name']
        increment_count(function_counts, function_name)

        # Process instance names
        instance_names = computation_result['instance_name(s)']
        
        # Process args
        for arg_name in instance_names.get('args', []):
            if arg_name == DEFAULT_VALUE_PLACEHOLDER:
                continue
            if isinstance(arg_name, str):
                # highest_level_name = process_instance_name(arg_name)
                increment_count(instance_counts, arg_name)
            elif isinstance(arg_name, tuple):
                # Process each element in the tuple
                for name in arg_name:
                    if isinstance(name, str):
                        # highest_level_name = process_instance_name(name)
                        increment_count(instance_counts, name)
            else:
                # If it's neither string nor tuple, use it as is (though this shouldn't happen)
                increment_count(instance_counts, str(arg_name))
        
        # Process kwargs
        for kwarg_name in instance_names.get('kwargs', {}).values():
            if kwarg_name == DEFAULT_VALUE_PLACEHOLDER:
                continue
            if isinstance(kwarg_name, str):
                # highest_level_name = process_instance_name(kwarg_name)
                increment_count(instance_counts, kwarg_name)
            elif isinstance(kwarg_name, tuple):
                # Process each element in the tuple
                for name in kwarg_name:
                    if isinstance(name, str):
                        # highest_level_name = process_instance_name(name)
                        increment_count(instance_counts, name)
            else:
                # If it's neither string nor tuple, use it as is (though this shouldn't happen)
                increment_count(instance_counts, str(kwarg_name))

    # Print the results
    prediction_generator_module_logger.info("\nNumber of results for each function/method:")
    for func, count in function_counts.items():
        prediction_generator_module_logger.info(f"  {func}: {count}")

    prediction_generator_module_logger.info("\nNumber of times each instance was used:")
    for instance, count in sorted(instance_counts.items()):
        prediction_generator_module_logger.info(f"  {instance}: {count}")

def import_module_from_path(module_name, module_path):
    # Ensure the module path exists
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"No file found at {module_path}")
    
    # If module_path is a directory, assume we are looking for a .py file with the module_name
    if os.path.isdir(module_path):
        module_path = os.path.join(module_path, f"{module_name}.py")

    # Get the directory part of the module path
    module_dir = os.path.dirname(module_path)
    
    # Temporarily add the module's directory to sys.path
    sys.path.insert(0, module_dir)
    
    try:
        # Import the module using its name
        module = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Error during import of module '{module_name}': {e}")
    finally:
        # Always remove the directory from sys.path after import
        sys.path.pop(0)
    
    return module

def reimport_module(module):
    # Remove the module from sys.modules if it exists
    if module.__name__ in sys.modules:
        del sys.modules[module.__name__]

    # Re-import the module using your custom import function
    reloaded_module = import_module_from_path(module.__name__, module.__file__)
    
    return reloaded_module

# Checks if the prediction JSON exists and is not empty
def check_json_at_path(json_path):
    # Check if the file exists
    if not os.path.exists(json_path):
        return False
    
    try:
        # Open and load the JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # Check if the JSON is empty (None, {}, or [])
        if data is None or data == {} or data == []:
            return False

        # If the file contains valid and non-empty JSON, return True
        return True
    except (json.JSONDecodeError, IOError):
        # If there is an error reading the file or decoding JSON, return False
        return False

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
def main_function(ewm_name, 
                ewm_dir, # The directory the EWM '.py' files are stored in
                obs_json_dir, # The directory the observation JSON files are stored in
                log_dir,
                func_tracer_module,
                replace=False,  # Indicates if an existing prediction JSON file should be replaced
                verbose=False):
    
    # Generate the ewm_path
    ewm_filename = f"{ewm_name}.py"
    ewm_path = f"{ewm_dir}/{ewm_filename}"
    # Generate the output_json_path
    obs_filename = f"{ewm_name}_obs.json"
    output_json_path = f"{obs_json_dir}/{obs_filename}"

    # Configure logging
    global prediction_generator_module_logger
    prediction_generator_module_logger = set_logging(verbose, log_dir, 'prediction_generator_module')

    def generate_new_predictions(ewm_name, ewm_path, func_tracer_module, output_json_path):
        try:
            # Import EWM module
            ewm_module = import_module_from_path(ewm_name, ewm_path)

        except (ImportError, FileNotFoundError) as e:
            print(f"Error importing modules: {e}")
            return  # Stop further execution if an import fails

        # Compute predictions using imported EWM
        computation_results = auto_compute(module=ewm_module,ewm_version=ewm_name, max_results_per_func=8, func_tracer_module=func_tracer_module)

        # Print a summary of the computation results
        analyze_computation_results(computation_results)

        if computation_results:
            save_results(output_json_path, computation_results)
            return computation_results
        else:
            print("No results to save.")
            return None

    if replace:
        computation_results = generate_new_predictions(ewm_name, ewm_path, func_tracer_module, output_json_path)

    else:
        json_with_content_exists = check_json_at_path(output_json_path)
        if not json_with_content_exists:
            computation_results = generate_new_predictions(ewm_name, ewm_path, func_tracer_module, output_json_path)
        else:
            print(f"Predictions already exist at '{output_json_path}'. No new predictions generated.\n To generate new predictions anyway, set 'replace' argument to False.")

    return computation_results

if __name__ == "__main__":

    # Example for testing:
    test_ewm_name = "20241221_202418_784842"
    test_ewm_dir = ".../generator_functions/ewm_files"
    test_obs_json_dir = ".../generator_functions/obs_ann_conj_files"
    log_dir = ".../generator_functions/log_files"
    test_func_tracer_name = "4_func_tracer"
    test_func_tracer_path = f".../generator_functions/{test_func_tracer_name}.py"
    verbose = False

    func_tracer_module = import_module_from_path(test_func_tracer_name, test_func_tracer_path)
    func_tracer_module.configure_logging_by_call(verbose, log_dir)

    main_function(ewm_name=test_ewm_name, 
                  ewm_dir=test_ewm_dir,
                  obs_json_dir=test_obs_json_dir, 
                  log_dir=log_dir, 
                  func_tracer_module=func_tracer_module, 
                  replace=True, 
                  verbose=verbose)

