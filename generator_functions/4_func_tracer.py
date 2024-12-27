import sys
import types
import logging
from typing import Optional, Union, List
from collections import defaultdict
from typing import Union, Tuple, Optional, Type, Type,  Dict, Any, List
import types
from dis import get_instructions
import inspect
from datetime import datetime
import os
import builtins

tracer_module_logger = None

# Constants for opcodes
LOAD_FAST = 124    # Loads a local variable
STORE_FAST = 125   # Stores a value into a local variable
STORE_NAME = 90    # Stores a value into a named variable
STORE_ATTR = 91    # Stores a value as an attribute of an object
LOAD_GLOBAL = 116  # Loads a global variable
LOAD_ATTR = 106    # Loads an attribute from an object

class VariableAccess:
    def __init__(self, name, var_object, access_type, accessed_in, line):
        self.name = name
        self.var_object = var_object
        self.access_type = access_type
        self.accessed_in: VariableAccess = accessed_in
        self.line = line
    
class DynamicTracer:
    def __init__(self, 
                 module: Optional[Union[str, types.ModuleType]] = None,
                 max_depth: int = sys.getrecursionlimit(),
                 all_module_names_ids: list = None,
                 implicit_obj_dicts: list = None,
                 arg_names: list = None,
                 kwarg_names: dict = None,
                 args: list = None,  # Add these parameters
                 kwargs: dict = None):
        self.module = module
        self.max_depth = max_depth
        self.all_name_id_in_module = all_module_names_ids
        self.implicit_obj_dicts = implicit_obj_dicts
        self.arg_names = arg_names or []
        self.kwarg_names = kwarg_names or {}
        self.original_args = {id(arg): name for arg, name in zip(args, arg_names)}
        self.original_kwargs = {id(val): name for param, name in kwarg_names.items() 
                              for val in [kwargs.get(param)] if val is not None}
        self.current_depth = 0
        self.call_stack = []
        self.variable_accesses: List[VariableAccess] = []
        self.access_graph = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.filtered_vars: list = [] 
        self.parameter_mappings = {}  # Stack to store parameter mappings for each frame (for tracking original argument names)

    def not_in_module_defs(self, var_name, var_object):
        name_id_tuple = (var_name, id(var_object))
        not_in_module_defs = True # Bool indicating if object was found in module definitions
        if name_id_tuple in self.all_name_id_in_module:
            not_in_module_defs = False
        return not_in_module_defs

    def _update_parameter_mappings(self, frame, var_name, var_obj):
        if not inspect.getmodule(frame ) is self.module: # Do not update parameter mappings while the frame is outside the EWM
            return
        # Populate 'self.parameter_mappings' with original input arguments if this has not been done:
        original_args_and_kwargs= self.original_args|self.original_kwargs
        if not self.parameter_mappings:
            self.parameter_mappings[frame]=original_args_and_kwargs

        # Update parameter mappings with var_name and var_obj:
        if frame in self.parameter_mappings:
            if id(var_obj) in self.parameter_mappings[frame]:
                # Append the new name if an original argument name is not stored under the id
                # Note that there may be conflicts if a variable is reassigned in a frame; however, the first name should be reliable to match the original defined name of the variable
                if isinstance(self.parameter_mappings[frame][id(var_obj)], str):
                    # Convert string values to lists (original_args and original_kwargs store strings)
                    self.parameter_mappings[frame][id(var_obj)]=[self.parameter_mappings[frame][id(var_obj)]]
                self.parameter_mappings[frame][id(var_obj)].append(var_name)
                tracer_module_logger.info(f"Appended var_name {var_name} to self.parameter_mappings.") 
                tracer_module_logger.info(f"self.parameter_mappings: {self.parameter_mappings}")
            # If the id key is not in self.parameter_mappings, add the mapping info
            else: self.parameter_mappings[frame][id(var_obj)]=[var_name]
        # If the frame key is not in self.parameter_mappings, add the mapping info
        else:
            self.parameter_mappings[frame]={id(var_obj):[var_name]}

    def _get_original_name(self, frame, var_name, var_obj):
        """Get the original name for a variable in the current frame."""
        tracer_module_logger.info(f"Getting original name for var_name: {var_name}, id: {id(var_obj)}.")
        tracer_module_logger.info(f"frame: {frame}")
        tracer_module_logger.info(f"self.parameter_mapping: {self.parameter_mappings}")
        if inspect.getmodule(frame) is self.module:
            var_name = self.parameter_mappings[frame][id(var_obj)][0]
            # Check if the object has been named in previous frames; if so, inherit name
            caller_frame = frame.f_back # Get the frame of the caller function
            caller_frame_module = inspect.getmodule(caller_frame)
            while caller_frame:
                if caller_frame_module is self.module:
                    tracer_module_logger.info(f"caller_frame: {caller_frame}")
                    if caller_frame in self.parameter_mappings: # Check if the id had a name in the caller frame
                        if id(var_obj) in self.parameter_mappings[caller_frame]:
                            var_name = self.parameter_mappings[caller_frame][id(var_obj)][0] # Rename var_name to the first name given to that id in the caller frame
                caller_frame=caller_frame.f_back
                caller_frame_module = inspect.getmodule(caller_frame)
        return var_name            
    
    def _check_assignments(self, frame):
        code = frame.f_code
        instructions = list(get_instructions(code))
        current_line = frame.f_lineno

        tracer_module_logger.debug(f"Checking line {current_line} in {code.co_name}")
        tracer_module_logger.debug(f"Local variables: {frame.f_locals}")
        tracer_module_logger.debug(f"Global variables: {frame.f_globals.keys()}")

        for i, instruction in enumerate(instructions):

            tracer_module_logger.debug(f"*** Instruction: {instruction}")

            var_name = None
            var_object = None
            access_type = None

            if instruction.opname == 'LOAD_FAST':
                var_name = code.co_varnames[instruction.arg]
                # Skip recording if this is the beginning of an attribute chain
                if i < len(instructions) - 1:
                    next_instr = instructions[i + 1]
                    if next_instr.opname == 'LOAD_ATTR':
                        # This is the beginning of an attribute chain
                        tracer_module_logger.debug(f"More attributes ahead, not recording access: {var_name}")
                        continue  # Analyse the next instruction (skip below processing)
                var_object = frame.f_locals.get(var_name)
                access_type = "read"
                tracer_module_logger.debug(f"LOAD_FAST detected: {var_name}, {var_object}")
                # Update var_name to match the name in the original module (e.g. if it was assigned a new name during computation)
                # var_name, found_object_in_module = self.find_original_name_of_obj(var_object, var_name)
                self._update_parameter_mappings(frame, var_name, var_object)
                var_name = self._get_original_name(frame, var_name, var_object) # Get the original name of the object if it was reassigned during funciton calling

            elif instruction.opname == 'LOAD_GLOBAL':
                var_name = code.co_names[instruction.arg]
                # Skip recording if this is the beginning of an attribute chain
                if i < len(instructions) - 1:
                    next_instr = instructions[i + 1]
                    if next_instr.opname == 'LOAD_ATTR':
                        # This is the beginning of an attribute chain
                        tracer_module_logger.debug(f"More attributes ahead, not recording access: {var_name}")
                        continue  # Analyse the next instruction (skip below processing)
                var_object = frame.f_globals.get(var_name)
                access_type = "read" 
                tracer_module_logger.debug(f"LOAD_GLOBAL or STORE_GLOBAL detected: {var_name}, {var_object}")
                # Update var_name to match the name in the original module (e.g. if it was assigned a new name during computation)
                # var_name, found_object_in_module = self.find_original_name_of_obj(var_object, var_name)
                self._update_parameter_mappings(frame, var_name, var_object)
                var_name = self._get_original_name(frame, var_name, var_object) # Get the original name of the object if it was reassigned during funciton calling

            elif instruction.opname == 'LOAD_NAME':
                var_name = code.co_varnames[instruction.arg]
                # Skip recording if this is the beginning of an attribute chain
                if i < len(instructions) - 1:
                    next_instr = instructions[i + 1]
                    if next_instr.opname == 'LOAD_ATTR':
                        # This is the beginning of an attribute chain
                        tracer_module_logger.debug(f"More attributes ahead, not recording access: {var_name}")
                        continue  # Analyse the next instruction (skip below processing)
                var_object = frame.f_locals.get(var_name, 
                 frame.f_globals.get(var_name, 
                 frame.f_builtins.get(var_name)))
                tracer_module_logger.debug(f"LOAD_NAME detected: {var_name}, {var_object}")
                # Update var_name to match the name in the original module (e.g. if it was assigned a new name during computation)
                # var_name, found_object_in_module = self.find_original_name_of_obj(var_object, var_name)
                self._update_parameter_mappings(frame, var_name, var_object)
                var_name = self._get_original_name(frame, var_name, var_object) # Get the original name of the object if it was reassigned during funciton calling

            elif instruction.opname == 'LOAD_ATTR':
                var_name = code.co_names[instruction.arg]
                access_type = "read"
                tracer_module_logger.debug(f"LOAD_ATTR detected. Name: {var_name}.")

                # Skip recording if this is not the end of an attribute chain
                if i < len(instructions) - 1:
                    next_instr = instructions[i + 1]
                    if next_instr.opname == 'LOAD_ATTR':
                    # if next_instr.opname in ('LOAD_ATTR', 'STORE_ATTR'):
                        # This is NOT the end of an attribute chain
                        tracer_module_logger.debug(f"More attributes ahead, not recording access: {var_name}")
                        continue  # Analyse the next instruction (skip below processing)
                
                # List the previous object names from chains of LOAD_ATTR instructions
                prev_instr_num = i-1
                prev_instr = instructions[prev_instr_num]
                attr_stack = [var_name]
                tracer_module_logger.debug(f"prev_instr: {prev_instr}")
                prev_obj_name = None
                
                while prev_instr.opname == 'LOAD_ATTR':
                    prev_obj_name = code.co_names[prev_instr.arg]
                    attr_stack.insert(0,prev_obj_name)
                    prev_instr_num -= 1
                    prev_instr = instructions[prev_instr_num]
                    tracer_module_logger.debug(f"prev_instr: {prev_instr}")
                # Record the instruction of the base object
                tracer_module_logger.debug(f"prev_instr: {prev_instr}")
                
                # Get the base instance name and object 
                base_obj = None
                base_obj_name = None
                if prev_instr.opname in ('LOAD_FAST', 'LOAD_NAME'):
                    base_obj_name = code.co_varnames[prev_instr.arg]
                    base_obj = frame.f_locals.get(base_obj_name)
                    # Update base_obj_name to match the name in the original module (e.g. if it was assigned a new name during computation)
                    # base_obj_name = self.find_original_name_of_obj(base_obj, base_obj_name)[0]
                elif prev_instr.opname == 'LOAD_GLOBAL':
                    base_obj_name = code.co_names[prev_instr.arg]
                    base_obj = frame.f_globals.get(base_obj_name, frame.f_builtins.get(base_obj_name))
                    # Update base_obj_name to match the name in the original module (e.g. if it was assigned a new name during computation)
                    # base_obj_name = self.find_original_name_of_obj(base_obj, base_obj_name)[0]
                self._update_parameter_mappings(frame, base_obj_name, base_obj)
                base_obj_name = self._get_original_name(frame, base_obj_name, base_obj) # Get the original name of the object if it was reassigned during funciton calling
                # Insert the instance name in the attr_stack
                attr_stack.insert(0,base_obj_name)
                tracer_module_logger.debug(f"prev_attr_list: {attr_stack}") 
                tracer_module_logger.debug(f"base object name: {base_obj_name}, base object: {base_obj}")

                # Now use the attr_stack to try to get the current attr object.
                  # I do it this way because sometimes he instructions do not store attribute objects.
                next_attr_obj = base_obj
                for attr_name in attr_stack[1:]:
                    try:
                        next_attr_obj = getattr(next_attr_obj, attr_name)
                        tracer_module_logger.debug(f"next attr name: {attr_name}, next_attr_obj: {next_attr_obj}")
                    except:
                        tracer_module_logger.info(f"Failed to get next attribute in 'attr_stack':\nattr_stack: {attr_stack}\nnext_attr_obj: {next_attr_obj}\nattr_name: {attr_name}")
                        continue  # Go to the next instruction

                # Assign the last attribute object to var_obj
                var_object = next_attr_obj 
                # Update var_name to fq name
                var_name = '.'.join(attr for attr in attr_stack)
                # var_name, found_object_in_module = self.find_original_name_of_obj(next_attr_obj, var_name)
                  
            if var_name: # and found_object_in_module
                tracer_module_logger.debug(f"Found variable: {var_name}")
                tracer_module_logger.debug(f"Found value: {var_object}")
                tracer_module_logger.debug(f"Access type: {access_type}")

                # Record only the full attribute access
                self._record_access(var_name, var_object, access_type, frame)
    
    def _record_access(self, var_name, var_object, access_type, frame):
        current_line = frame.f_lineno

        current_function_access = self.call_stack[-1] if self.call_stack else None
        accessed_in = current_function_access

        if self._should_include_object(var_name, var_object, access_type, accessed_in, current_line):
            variable_access = VariableAccess(var_name, var_object, access_type, accessed_in, current_line)
            self.variable_accesses.append(variable_access)
            
            # Generate debug statement
            debug_str = f"Added variable access: {var_name}"
            if accessed_in:
                if accessed_in.name:
                    debug_str += f" in {accessed_in.name}"
            else:
                debug_str += f" with no higher 'accessed_in' function"
            debug_str += f" at line {current_line}"
            tracer_module_logger.debug(debug_str)

            # Update the access graph
            self._update_access_graph(variable_access)
        else:
            tracer_module_logger.debug(f"Skipped variable: {var_name} (not included based on filters)")
            
    ## THIS METHOD REMOVED IN VERSION 02_18: not needed. there is some difference between 'defined_in' and 'accessed_in' but I don't 
        ## understand it and I don't think I need the extra information.
    # def _get_variable_definition_context(self, var_name, frame):
    #     if var_name in frame.f_locals:
    #         return frame.f_code.co_name
    #     elif var_name in frame.f_globals:
    #         return 'global'
    #     else:
    #         # Return the current function name (e.g. for local variables that might not be in f_locals yet (like function parameters at the start of the function))
    #         return frame.f_code.co_name

    def _update_access_graph(self, var_access):
        chain = []
        current = var_access

        while current is not None:
        # Insert var_access tuple in chain, then iterate through 'accessed_in', 
        #   inserting parent to front of chain until reaching 'None'
            var_tuple = (current.name, type(current.var_object).__name__) #  current.var_object, current.access_type, current.accessed_in, current.line
            tracer_module_logger.debug(f"var_tuple: {var_tuple}")
            tracer_module_logger.debug(f"current.var_object: {current.var_object}")
            chain.insert(0, var_tuple)
            current = current.accessed_in

        # Now, starting from the root of accessed_graph, traverse or create the nested dictionaries
        #   needed to store the chain.
        current_dict = self.access_graph
        for var_tuple in chain:
            if var_tuple not in current_dict:
                current_dict[var_tuple] = {}
            current_dict = current_dict[var_tuple]
        
        # This section is added to prevent an error message when logging the root accessed_in.name (which is None)
        if var_access.accessed_in is not None:
            parent_name = var_access.accessed_in.name
        else:
            parent_name = 'None'

        tracer_module_logger.debug(f"Updated access graph: {parent_name} -> {var_access.name}")


    def _should_include_object(self, var_name, var_object, access_type, accessed_in, line) -> bool:
        failed_filters = set()
        passed_filters = set()

        # Filter dunder names (magic methods, built-in attributes, private name mangling)
        if var_name.startswith('__'):
            tracer_module_logger.info(f"'Starts with __' filter failed for: {var_name}, {var_object}")
            failed_filters.add("Starts with __")
        else:
            tracer_module_logger.debug(f"'Starts with __' filter passed for: {var_name}, {var_object}")
            passed_filters.add("Starts with __")

        # Filter implicit objects
        is_implicit_obj = is_implicit_object(self.module, var_object)
        if is_implicit_obj:
            tracer_module_logger.info(f"'is_implicit_obj' filter failed for: {var_name}, {var_object}")
            failed_filters.add("is_implicit_obj")
        else:
            tracer_module_logger.debug(f"'is_implicit_obj' filter passed for: {var_name}, {var_object}")
            passed_filters.add("is_implicit_obj")
        
        # Filter objects that do not match any recognized classes, methods, functions, instances, or attributes (e.g. transient objects defined locally)
        not_in_module_defs = self.not_in_module_defs(var_name, var_object)
        if not_in_module_defs:
            tracer_module_logger.info(f"'not_in_module_defs' filter failed for: {var_name}, {var_object}")
            failed_filters.add("not_in_module_defs")
        else:
            tracer_module_logger.debug(f"'not_in_module_defs' filter passed for: {var_name}, {var_object}")
            passed_filters.add("not_in_module_defs")

        if not failed_filters:
            return True
        else:
            self.filtered_vars.append(((VariableAccess(var_name, var_object, access_type, accessed_in, line)), failed_filters, passed_filters))
            return False

    def _handle_function_call(self, frame):
        code = frame.f_code
        func_name = code.co_name
        class_name = None
        caller_frame = frame.f_back

        # Setup class info for methods
        if 'self' in frame.f_locals or 'cls' in frame.f_locals:
            instance = frame.f_locals.get('self') or frame.f_locals.get('cls')
            for cls in instance.__class__.__mro__:
                if func_name in cls.__dict__:
                    class_name = cls.__qualname__
                    break
            if not class_name:
                class_name = instance.__class__.__qualname__

        # Construct qualified name
        qualified_name = f"{class_name}.{func_name}" if class_name else func_name
        
        # Get function object 
        func_object = None
        if class_name:
            # For methods, look up in the class
            instance = frame.f_locals.get('self') or frame.f_locals.get('cls')
            for cls in instance.__class__.__mro__:
                if func_name in cls.__dict__:
                    func_object = cls.__dict__[func_name]
                    break
        else:
            # For regular functions, look in globals/locals
            func_object = frame.f_globals.get(func_name) or frame.f_locals.get(func_name)

        # Fall back to code object only if we couldn't find the function
        if func_object is None:
            func_object = code

        # Create function access record
        function_access = VariableAccess(
            name=qualified_name,
            var_object=func_object,
            access_type='function',
            accessed_in=self.call_stack[-1] if self.call_stack else None,
            line=code.co_firstlineno)
        
        if self._should_include_object(function_access.name, function_access.var_object, 
                                    function_access.access_type, function_access.accessed_in, 
                                    function_access.line):
            self.call_stack.append(function_access)

    def _handle_function_return(self, frame):
        if self.call_stack:
            tracer_module_logger.info(f"self.call_stack PRE-pop: {self.call_stack}")
            self.call_stack.pop()
            tracer_module_logger.info(f"self.call_stack POST-pop: {self.call_stack}")
        # Clean up parameter mapping when function returns.
        tracer_module_logger.info(f"self.parameter_mappings PRE-pop: {self.parameter_mappings}")
        self.parameter_mappings.pop(frame, None)
        tracer_module_logger.info(f"self.parameter_mappings POST-pop: {self.parameter_mappings}")

    def trace_function(self, frame, event, arg):
        if self.current_depth >= self.max_depth:
            print(f"Warning: reached max tracing depth ({self.max_depth}) for function. Tracing terminated before result.")
            return
        tracer_module_logger.debug(f"Event: {event}, Function: {frame.f_code.co_name}, Line: {frame.f_lineno}")
        
        if event == 'call':
            self.current_depth += 1
            self._handle_function_call(frame)
            tracer_module_logger.debug(f"Entering function: {frame.f_code.co_name}")
            return self.trace_function  # Continue tracing into the called function
        elif event == 'return':
            self.current_depth -= 1
            self._handle_function_return(frame)
            tracer_module_logger.debug(f"Returning from function: {frame.f_code.co_name}")
        elif event == 'line':
            self._check_assignments(frame)
        elif event == 'exception':
            tracer_module_logger.warning(f"Exception in {frame.f_code.co_name}: {arg}")

        if event != 'call':
            return self.trace_function  # Continue tracing for non-call events
    
def get_nested_classes(cls: Type, parent_name: str = "") -> List[Tuple[str, Type]]:
    """Recursively get nested classes with their names and class objects
    
    Args:
        cls: Class to inspect
        parent_name: Name of parent class for building path
    Returns: List of (qualified_name, class_object) tuples
    """
    nested: List[Tuple[str, Type]] = []
    base_name = parent_name + "." + cls.__name__ if parent_name else cls.__name__
    
    for name, obj in inspect.getmembers(cls):
        if inspect.isclass(obj) and obj.__module__ == cls.__module__:
            nested.append((base_name + "." + name, obj))
            nested.extend(get_nested_classes(obj, base_name + "." + name))
    
    return nested

def get_all_classes(module: types.ModuleType) -> List[Tuple[str, Type]]:
   """Get all classes in a module including nested ones
   
   Args:
       module: Module to inspect
   Returns: List of (qualified_name, class_object) tuples 
   """
   class_names_and_objs: List[Tuple[str, Type]] = []
   
   for name, cls in inspect.getmembers(module, inspect.isclass):
       if cls.__module__ == module.__name__:
           class_names_and_objs.append((name, cls))
           class_names_and_objs.extend(get_nested_classes(cls, name))
           
   return class_names_and_objs

# Generate a dictionary of all module level instances (name, obj) tuples, excluding imported instances, grouped by their type. 
def create_instance_dict(module, class_names_and_objs) -> Dict[type, List[Tuple[str, Any]]]:
    class_objs_tuple = tuple(name_obj[1] for name_obj in class_names_and_objs)
    instance_dict = {}
    for name, obj in inspect.getmembers(module):
        if isinstance(obj, class_objs_tuple) and obj.__module__ == module.__name__:
            obj_type = type(obj)
            if obj_type not in instance_dict:
                instance_dict[obj_type] = []
            instance_dict[obj_type].append((name, obj))
    return instance_dict

# Use 'instance_dict' to generate a dictionary of all attributes of the module level instances.
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
    for instance_tuple in instance_dict.values():
        for instance_name, instance_obj in instance_tuple:
            add_attributes(instance_obj, instance_name)

    return attribute_dict

def get_nested_functions(func_obj, parent_name=""):
    """Get nested functions with their proper qualified names"""
    nested = []
    for name, obj in inspect.getmembers(func_obj):
        if (isinstance(obj, types.FunctionType) and 
            obj.__module__ == func_obj.__module__):
            qualified_name = f"{parent_name}.<locals>.{name}" if parent_name else name
            nested.append((qualified_name, obj))
            nested.extend(get_nested_functions(obj, qualified_name))
    return nested

def get_all_functions(module):
    """Get all functions including nested ones from module"""
    functions = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            functions.append((name, obj))
            functions.extend(get_nested_functions(obj, name))
    return functions

def get_direct_methods(class_names_and_objs) -> list:
    """Generate a list of tuples of method name and objects."""
    method_name_obj_tuples = set()
    
    for class_name, cls in class_names_and_objs:
        for method_name, obj in cls.__dict__.items():
            if callable(obj): # Optionally: " and not method_name.startswith("__"):""
                qualified_name = f"{class_name}.{method_name}"
                method_name_obj_tuples.add((qualified_name, obj))
                
    return list(method_name_obj_tuples)

def build_implicit_object_list(module):
    """
    Generate a list of dictionaries of implicit objects in the module (e.g builtins) - objects present but not explicitly defined
        Output Format: 
        [{'name': name, 'type': type(obj).__name__, 'id': id(obj), 'module': type(obj).__module__}, ...]
    """
    implicit_objects = []
    try:
        module_file = inspect.getsourcefile(module)
    except TypeError:
        module_file = None
        tracer_module_logger.warning(f"find_module_implicit_objects unable to find source file of module: {module}")

    for name, obj in inspect.getmembers(module):
        # Skip private/special attributes
        if name.startswith('_'):
            continue
            
        if is_implicit_object(module, obj):
            implicit_objects.append({
                'name': name,
                'type': type(obj).__name__,
                'id': id(obj),
                'module': type(obj).__module__,
            })
    
    return implicit_objects  # Added return statement

def is_implicit_object(module, obj):
    try:
        module_file = inspect.getsourcefile(module)
    except TypeError:
        module_file = None
        tracer_module_logger.warning(f"find_module_implicit_objects unable to find source file of module: {module}")

    # Skip if it's a module
    if inspect.ismodule(obj):
        tracer_module_logger.debug(f"is_implicit_object: '{obj}' is module")
        return False
    
    # Check if it's a code object (listcomp, genexpr)
    if type(obj).__name__ == 'code' and getattr(obj, 'co_name', '').startswith('<'):
        tracer_module_logger.debug(f"is_implicit_object: '{obj}' is compiler-generated code object")
        return True
    
    # Check if object's type is defined in our module
    if type(obj).__module__ == module.__name__:
        tracer_module_logger.debug(f"is_implicit_object: '{obj}' type defined in module")
        return False

    # Check for built-in functions and names
    if isinstance(obj, (types.BuiltinFunctionType, types.BuiltinMethodType)) or \
       (isinstance(obj, type(None)) and hasattr(builtins, str(obj))):
        tracer_module_logger.debug(f"is_implicit_object: '{obj}' is builtin function/name")
        return True

    # Check if this is a primitive type value (int, float, str, bool, etc.)
    if isinstance(obj, (int, float, str, bool, type(None))):
        tracer_module_logger.debug(f"is_implicit_object: '{obj}' is primitive value")
        return False
     
    try:
        obj_file = inspect.getfile(obj)
        if obj_file == module_file:
            tracer_module_logger.debug(f"is_implicit_object: '{obj}' getfile is module_file")
            return False
        else:
            tracer_module_logger.debug(f"is_implicit_object: '{obj}' from different file: {obj_file}")
            return True
    except (TypeError, ValueError):
        tracer_module_logger.debug(f"is_implicit_object: '{obj}' no file - checking type")
        
        # Explicitly check for known implicit types first
        if isinstance(obj, (
            types.CodeType,  # Catches <genexpr>, <listcomp>
            types.GeneratorType,
            types.CoroutineType,
            types.AsyncGeneratorType,
        )) or type(obj).__name__.endswith('_iterator'):  # Catches tuple_iterator etc.
            tracer_module_logger.debug(f"is_implicit_object: '{obj}' is generator/iterator/code object")
            return True

        # If it's from typing module, it's implicit
        if type(obj).__module__ == 'typing':
            tracer_module_logger.debug(f"is_implicit_object: '{obj}' is typing object")
            return True
        
        # Check special descriptor types
        if isinstance(obj, (
            types.MethodType,
            types.GetSetDescriptorType,
            types.MemberDescriptorType,
            property,
            classmethod,
            staticmethod,
            types.BuiltinFunctionType,
            types.BuiltinMethodType
        )):
            tracer_module_logger.debug(f"is_implicit_object: '{obj}' is special type")
            return True

        # Check for code objects (list comprehensions, generator expressions)
        if isinstance(obj, types.CodeType) or \
           (hasattr(obj, '__name__') and obj.__name__ in ('<listcomp>', '<genexpr>')):
            tracer_module_logger.debug(f"is_implicit_object: '{obj}' is code object")
            return True
            
    return False

def get_other_accessed_objects(accessed_objects):
    """
    Recursively retrieves all accessed object names from the 'accessed_objects' dictionary.
    Each key is expected to be a tuple where the first element is the object name.
    
    Args:
        accessed_objects (dict): A nested dictionary where keys are tuples containing
            (object_name, type)
            
    Returns:
        list: A list of all object names extracted from the dictionary keys
    """
    other_accessed_objects = []
    
    def recursive_extract(obj_dict):
        for key, value in obj_dict.items():
            # Extract object name (first element of the tuple key)
            object_name = key[0]
            other_accessed_objects.append(object_name)
            
            # Recursively process nested dictionaries
            if isinstance(value, dict):
                recursive_extract(value)
    
    recursive_extract(accessed_objects)
    return other_accessed_objects


"""
HELPER FUNCTIONS
"""
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
def configure_logging_by_call(verbose, log_dir):
    global tracer_module_logger
    tracer_module_logger = set_logging(verbose, log_dir, module_name="tracer_module")

def main_function(func: callable, 
                    args: list, 
                    kwargs: dict,
                    module: Optional[types.ModuleType], 
                    arg_names: list,  # A list or argument names (e.g. ['earth', 'earth_radius.value'])
                    kwarg_names: dict,  # A dictionary of kwargs where keys are parameter names and values are the names of input objects (e.g. {'radius':'earth_radius'})
                    all_objects: set,  
                    max_depth: int = sys.getrecursionlimit()) -> tuple:
    
    # Logging configuration is run separately from the main function because "trace_function_execution" must be run separately for each prediction
    """
    Replicate the object dictionaries from the p_generator
    These are used to override reassigned names of objects during the computation (e.g. 'self' in method calls)
    """
 
    # class_name_obj_tuples is a list of tuples of class names and objects
    class_list = get_all_classes(module)
    # tracer_module_logger.info(f"class_list: {class_list}")

    # instance_dict is a dictionary with class keys and list values. The lists contain tuples of instance names and objects.
        # e.g. "{<class 'tracer_test_02.Planet'>: [('earth', <tracer_test_02.Planet object at 0x7f7ece8b1870>)],..."
    instance_dict = create_instance_dict(module, class_list)
    # tracer_module_logger.info(f"instance_dict:\n{instance_dict}")

    # attribute_dict is a dictionary with class keys and list values. The lists contain tuples of attribute fq names and objects.
    # e.g. "{<class 'str'>: [('earth.name', 'Earth'), ('earth.mass.value.unit.name.name', 'kilogram'),..."
    attribute_dict = create_attribute_dict(instance_dict, max_depth=16)
    # tracer_module_logger.info(f"attribute_dict:\n{attribute_dict}")

    # function_dict is a list of tuples of (name, obj) for all functions and nested functions
    functions_list = get_all_functions(module)
    # tracer_module_logger.info(f"functions_list: {functions_list}")

    # Use the class_types already collected to generate a set of all methods.
    methods_list = get_direct_methods(class_list)
    # tracer_module_logger.info(f"methods_list: {methods_list}")

    all_module_names_objs = (class_list + functions_list + methods_list + 
            [name_obj_tuple for obj_dict in (instance_dict, attribute_dict) for obj_list in obj_dict.values() for name_obj_tuple in obj_list])
    # tracer_module_logger.info(f"all_module_names_objs: {all_module_names_objs}")

    all_module_names_ids = [(name, id(obj)) for name, obj in all_module_names_objs]
    # tracer_module_logger.info(f"all_module_names_ids: {all_module_names_ids}")

    implicit_obj_dicts = build_implicit_object_list(module)
    # tracer_module_logger.info(f"implicit_obj_dicts:\n{implicit_obj_dicts}")

    tracer = DynamicTracer(module, 
                           max_depth, 
                           all_module_names_ids,
                           implicit_obj_dicts,
                           arg_names=arg_names,
                           kwarg_names=kwarg_names,
                           args=args,
                           kwargs=kwargs)

    def recursive_trace(frame, event, arg):
        if event == 'call':
            sys.settrace(recursive_trace)
        result = tracer.trace_function(frame, event, arg)
        if event == 'return':
            sys.settrace(None)
        return result

    def traced_func(func, args, kwargs):
        sys.settrace(recursive_trace)
        try:
            return func(*args, **kwargs)
        finally:
            sys.settrace(None)

    result = traced_func(func, args, kwargs)

    # Generate the set of FQ names of all objects accessed during the computation
    accessed_for_prediction = all_objects
    other_accessed_objects = get_other_accessed_objects(tracer.access_graph)
    accessed_for_prediction.update(other_accessed_objects)
    tracer_module_logger.info(f"accessed_for_prediction: {accessed_for_prediction}")

    return result, tracer.variable_accesses, tracer.filtered_vars, tracer.access_graph, accessed_for_prediction


# Example usage
if __name__ == "__main__":
    log_dir = ".../generator_functions/log_files"
    import sys
    sys.path.append('.../graph_visualization/test_ewms')
    import tracer_test_02
    
    module = tracer_test_02
    function = tracer_test_02.Planet.calculate_gravity
    args = [tracer_test_02.earth, tracer_test_02.earth_radius]
    kwargs = {}
    set_logging(verbose_bool=False, log_dir=log_dir)

    """
    Trace a function
    """
    result, variable_accesses, filtered_vars, access_graph = main_function(
        function, 
        args=args,
        kwargs=kwargs,
        module=tracer_test_02,
        arg_names=['earth', 'earth_radius'],
        kwarg_names={},
        max_depth=128)

    # Print Result
    tracer_module_logger.info(f"\nResult: {str(result)}")

    # Print Accessed Variables
    tracer_module_logger.debug("\nAccessed Variable Set:")
    accessed_dict = {}
    for access in variable_accesses:
        if access.access_type == "read":
            accessed_dict.setdefault(access.name, []).append(access.line)
    for name, line in accessed_dict.items():
        tracer_module_logger.debug(f"  {name}\n    On Line(s): {line}")
       
    # Create a dictionary from filtered_vars
    filtered_vars_dict = {}
    for var, failed_filters, passed_filters in filtered_vars:
        if var.name not in filtered_vars_dict:
            filtered_vars_dict[var.name] = set()
        filtered_vars_dict[var.name].add((var.line, tuple(failed_filters)))

    # Print the filtered vars dictionary
    tracer_module_logger.debug("\nFiltered Vars:")
    for key, value in filtered_vars_dict.items():
        tracer_module_logger.debug(f"{key}:")
        for set_item in value:
            tracer_module_logger.debug(f"  Line No.:{set_item[0]}\n  Failed Filters: {set_item[1]}")

    # Print the graph of variable connections
    def print_access_graph(graph, indent=''):
        for key, value in graph.items():
            if isinstance(value, dict):
                tracer_module_logger.info(f"{indent}{key}:")
                print_access_graph(value, indent + '  ')
            else:
                tracer_module_logger.info(f"{indent}Accessed: {key}")

    tracer_module_logger.info("\nAccess Graph:")
    for context, accesses in access_graph.items():
        tracer_module_logger.info(f"{context}:")
        print_access_graph(accesses, '  ')
    
    tracer_module_logger.debug(F"\n{access_graph}")
    

    """
    Other tests:
    -----
    import test_prediction_environment_09

    function = test_prediction_environment_09.Planet.calculate_density
    args = [test_prediction_environment_09.earth_planet]
    kwargs = {}
    -----
    import tracer_test_01

    function = tracer_test_01.Planet.get_highest_layer
    args = [tracer_test_01.earth]
    kwargs = {}
    -----
    import tracer_test_02

    function = tracer_test_02.Planet.calculate_gravity
    args = [tracer_test_02.earth, tracer_test_02.radius]
    kwargs = {}
    -----
    
    """