"""
generate_ann_obj_dicts.py

- Generates a dictionary of objects defined in the EWM and information about each definition
- Saves dictionary as a JSON

"""
import sys
import json
import importlib.util
from typing import List, Dict, Any, Optional, get_type_hints, Union, get_args, Tuple, get_origin
import os
import types
from types import MappingProxyType, GetSetDescriptorType
import logging
import inspect
import ast
from typing import get_type_hints
from dataclasses import dataclass
from datetime import datetime

ann_obj_dict_generator_module_logger = None
PRUNED_SCORE_PLACEHOLDER = 'N8cT89C044lqvJ'
DEFINED_OBJECTS_PLACEHOLDER = "vcM/juON9%1gv!lLp"
ANY_TYPE_PLACEHOLDER = "kHNMWb9b^1!gzcx"

# Helper function to get the base object of an attribute
def get_base_name(node):
    if isinstance(node, ast.Attribute):
        return get_base_name(node.value)  # Recur to find the base
    elif isinstance(node, ast.Name):
        return node.id  # Return the base object name
    return None

def get_func_obj(fq_name, module):
    func_obj = module
    for attr in fq_name.split('.'):
        func_obj = getattr(func_obj, attr)
    return func_obj

@dataclass
class ParameterInfo:
    name: str
    type: Optional[Union[type, Tuple[type, ...]]]
    default: Any = inspect.Parameter.empty
    
def resolve_parameter_types(func_obj) -> Dict[str, ParameterInfo]:
    """
    Resolves parameter types for a function or method, including 'self' parameter.
    Extracts individual types from Union types.
    """
    # Get signature and type hints
    signature = inspect.signature(func_obj)
    type_hints = get_type_hints(func_obj)
    
    def extract_types(type_hint):
        """Extract types from Union and Optional types."""
        if type_hint is None:
            return None
            
        # Get the origin (e.g., Union, Optional, List, etc.)
        origin = get_origin(type_hint)
        
        if origin is Union:
            # Get all types from the Union
            types_tuple = get_args(type_hint)
            # Remove None type for Optional (Union[X, None])
            types_without_none = tuple(t for t in types_tuple if t is not type(None))
            return types_without_none if len(types_without_none) > 1 else types_without_none[0]
        
        return type_hint
    
    # Function to get the class type for methods
    def get_self_class():
        if isinstance(func_obj, types.MethodType):
            # Bound method - get from instance
            return func_obj.__self__.__class__
        elif hasattr(func_obj, '__qualname__'):
            # Unbound method - get from qualname
            class_path = func_obj.__qualname__.split('.')[:-1]
            if class_path:
                # Get module
                module = inspect.getmodule(func_obj)
                if module:
                    current = module
                    for part in class_path:
                        current = getattr(current, part)
                    return current
        return None
    
    # Process each parameter
    params = {}
    for name, param in signature.parameters.items():
        # If it's 'self' and has no type hint, try to get the class
        if name == 'self' and name not in type_hints:
            param_type = get_self_class()
        else:
            param_type = extract_types(type_hints.get(name, None))
            
        params[name] = ParameterInfo(
            name=name,
            type=param_type,
            default=param.default
        )
    
    return params

def get_nested_attribute_type_safe(full_path, module, default=None):
    """
    Safely get the type of a nested attribute using its full path and module context.
    
    Args:
        full_path: Complete dot-separated path to the attribute (e.g., "instance.attribute_a.attribute_b")
        module: The module object where the instance is defined
        default: Value to return if the attribute doesn't exist
        
    Returns:
        Type of the attribute or default value if not found
        
    Example:
        class Address:
            def __init__(self):
                self.zip_code = 12345
                
        class Person:
            def __init__(self):
                self.address = Address()
                
        person = Person()
        
        # Get type using the full path
        type_name = get_nested_attribute_type_safe("person.address.zip_code", globals())  # Returns int
    """
    try:
        # Split the path into instance name and attribute path
        parts = full_path.split('.')
        instance_name = parts[0]
        attr_path = parts[1:]
        
        # Get the instance from the module
        current_obj = getattr(module, instance_name)
        
        # Traverse through the remaining attributes
        for attr in attr_path:
            current_obj = getattr(current_obj, attr)
        
        return type(current_obj)
    except (AttributeError, KeyError):
        return default
 
def track_definitions_and_imports(module):
    """
    Function to walk the AST and track defined and imported objects.
    """
    module_name = module.__name__
    try:
        # Get the source code of the module
        module_source = inspect.getsource(module)
    except Exception as e:
        ann_obj_dict_generator_module_logger.warning(f"Could not get source for module '{module_name}': {e}")

    # Parse the source code into an Abstract Syntax Tree (AST)
    module_ast = ast.parse(module_source)

    object_dict = {}

    class ObjectCollector(ast.NodeVisitor):
        """
        AST Node Visitor class that collects names of functions, classes,
        and instances (variables) from the module source code, excluding special
        methods and attributes.
        """
        def __init__(self):
            # Dictionary to store collected definitions
            self.defined_objects = {
                'in_module': {
                    'instance_node_names': set(),
                    'instance_fq_names': set(),
                    'function_node_names': set(),
                    'function_fq_names': set(),
                    'class_node_names': set(),
                    'class_fq_names': set(),
                },
                'outside_module': set()  # For base names of imported modules
            }
            # Stacks to keep track of nested scopes (e.g., classes, functions)
            self.parent_stack = []
            self.context_stack = []

        def _is_special_name(self, name):
            """
            Determine if a name is a special (magic) method or attribute.
            """
            # Special names start and end with '__', and are longer than '__'
            return name.startswith('__') and name.endswith('__') and len(name) > 4
        
        def _get_fq_name(self, name):
            """
            Construct the fully qualified name of an object by combining
            the parent stack with the current name.
            """
            if self.parent_stack:
                # Join parent names with the current name (e.g., ClassName.methodName)
                return '.'.join(self.parent_stack + [name])
            else:
                return name  # Top-level name

        def visit_FunctionDef(self, node):
            """
            Visit a function definition node.
            """
            # Skip special methods like '__init__' or '__str__'
            if self._is_special_name(node.name):
                return  # Skip processing this node
            
            # Store node name
            self.defined_objects['in_module']['function_node_names'].add(node.name)

            # Store fully qualified name (including parent scopes)
            fq_name = self._get_fq_name(node.name)
            self.defined_objects['in_module']['function_fq_names'].add(fq_name)
        
            # Update stacks and recursively visit child nodes
            self.parent_stack.append(node.name)
            self.context_stack.append('function')
            self.generic_visit(node)  # Continue traversing child nodes
            self.context_stack.pop()
            self.parent_stack.pop()

        def visit_ClassDef(self, node):
            """
            Visit a class definition node.
            """
            # Skip special methods like '__init__' or '__str__'
            if self._is_special_name(node.name):
                return  # Skip processing this node
            
            # Store node name
            self.defined_objects['in_module']['class_node_names'].add(node.name)

            # Store fully qualified name (including parent scopes)
            fq_name = self._get_fq_name(node.name)
            self.defined_objects['in_module']['class_fq_names'].add(fq_name)

            # Update stacks and visit child nodes (e.g., methods)
            self.parent_stack.append(node.name)
            self.context_stack.append('class')
            self.generic_visit(node)
            self.context_stack.pop()
            self.parent_stack.pop()

        def visit_AsyncFunctionDef(self, node):
            """
            Visit an asynchronous function definition node.
            """
            # Skip special methods like '__init__' or '__str__'
            if self._is_special_name(node.name):
                return  # Skip processing this node
            
            # Store funciton node name
            self.defined_objects['in_module']['function_node_names'].add(node.name)

            # Store function fully qualified name (including parent scopes)
            fq_name = self._get_fq_name(node.name)
            self.defined_objects['in_module']['function_fq_names'].add(fq_name)
        
            # Update stacks and visit child nodes
            self.parent_stack.append(node.name)
            self.context_stack.append('function')
            self.generic_visit(node)
            self.context_stack.pop()
            self.parent_stack.pop()

        def visit_Assign(self, node):
            """
            Visit an assignment node.
            """
            # Handle variable assignments at the module or class level
            self._handle_assign(node)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            """
            Visit an annotated assignment node (e.g., variable with type hint).
            """
            # Handle annotated assignments
            self._handle_assign(node)
            self.generic_visit(node)

        def _handle_assign(self, node):
            """
            Handle assignment nodes to collect variable definitions.
            """
            # Skip assignments inside functions (local variables) or classes (class attributes)
            if self.context_stack and self.context_stack[-1] == 'function':
                return
            if self.context_stack and self.context_stack[-1] == 'class':
                return
            
            # Get the list of targets being assigned to
            targets = node.targets if hasattr(node, 'targets') else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # Skip special names like '__all__' or '__version__'
                    if self._is_special_name(name):
                        continue
                    
                    # Store node name
                    self.defined_objects['in_module']['instance_node_names'].add(name)

                    # Store fully qualified name (including parent scopes)
                    fq_name = self._get_fq_name(name)
                    self.defined_objects['in_module']['instance_fq_names'].add(fq_name)

                elif isinstance(target, ast.Tuple):
                    # Handle tuple unpacking assignments (e.g., a, b = 1, 2)
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            name = elt.id
                            if self._is_special_name(name):
                                continue

                            # Store node name
                            self.defined_objects['in_module']['instance_node_names'].add(name)

                            # Store fully qualified name (including parent scopes)
                            fq_name = self._get_fq_name(name)
                            self.defined_objects['in_module']['instance_fq_names'].add(fq_name)

        def visit_Import(self, node):
            """
            Track imported names (e.g., from external_module import Planet)
            """
            for alias in node.names:
                base_name = alias.asname or alias.name.split('.')[0]
                self.defined_objects['outside_module'].add(base_name)
        
            # No not continue to vist child nodes stacks and visit child nodes (self.generic_visit(node))

        def visit_ImportFrom(self, node):
            """
            Track imports from specific modules (e.g., import some_other_module)
            """
            for alias in node.names:
                base_name = alias.asname or alias.name
                self.defined_objects['outside_module'].add(base_name)
        
            # No not continue to vist child nodes stacks and visit child nodes (self.generic_visit(node))

    # Instantiate the DefinitionCollector and traverse the AST
    collector = ObjectCollector()
    collector.visit(module_ast)
    # Update the object dictionary with collected definitions
    object_dict.update(collector.defined_objects)

    return object_dict

def log_node_and_score(node, score, pruned=False):  
    """
    Logging used by AST Scorer.
    Print node type and score with a fallback for specific attributes
    """
    if hasattr(node, 'name'):
        if pruned:
            ann_obj_dict_generator_module_logger.info(f"Pruned Node:  Node: {type(node).__name__}, Name: {node.name}, Score: {score}")
        else:
            ann_obj_dict_generator_module_logger.info(f"Node: {type(node).__name__}, Name: {node.name}, Score: {score}")
    elif hasattr(node, 'id'):
        if pruned:
            ann_obj_dict_generator_module_logger.info(f"Pruned Node:  Node: {type(node).__name__}, ID: {node.id}, Score: {score}")
        else:
            ann_obj_dict_generator_module_logger.info(f"Node: {type(node).__name__}, ID: {node.id}, Score: {score}")
    elif hasattr(node, 'value'):
        if pruned:
            ann_obj_dict_generator_module_logger.info(f"Pruned Node:  Node: {type(node).__name__}, Value: {node.value}, Score: {score}")
        else:
            ann_obj_dict_generator_module_logger.info(f"Node: {type(node).__name__}, Value: {node.value}, Score: {score}")
    else:
        if pruned:
            ann_obj_dict_generator_module_logger.info(f"Pruned Node:  Node: {type(node).__name__}, Score: {score}")

        else:
            ann_obj_dict_generator_module_logger.info(f"Node: {type(node).__name__}, Score: {score}")

def compare_static_definitions(dict1, dict2):
    """
    Check if the definitions are identical (return 'True'), or if they have changed (return 'False').
    Note: 'compare_nested_dicts' (used to compare 'initialized_definition' will not work for 'static_definition'
      dictionaries because the 'location' information must be excluded from the comparison)
    """
    def recursive_compare(value1, value2):
        # Handle None values
        if value1 is None or value2 is None:
            return value1 == value2
        # Compare strings directly
        if isinstance(value1, str) and isinstance(value2, str):
            return value1 == value2
        # Compare lists of strings
        if isinstance(value1, list) and isinstance(value2, list):
            return sorted(value1) == sorted(value2)
        # Otherwise, not comparable
        return False
    
    # Check if the keys are both storing None
    if dict1 is None and dict2 is None:
        # Return True when the definitions match and neither have values
        return True
    
    if (dict1 is None and dict2 is not None) or (dict1 is not None and dict2 is None):
        # Return False if one is None and the other is not None
        return False
    
    if len(dict1) != len(dict1):
        # Return False if they are different lengths
        return False
    
    for (mod_dict1, mod_dict2) in (zip(dict1, dict2)):
        if not recursive_compare(mod_dict1['text'], mod_dict2["text"]):
            return False

    return True

def compare_nested_dicts(dict1, dict2):
    """
    Recursively compare two nested dictionaries for equality.

    Args:
        dict1 (dict): The first dictionary to compare.
        dict2 (dict): The second dictionary to compare.

    Returns:
        bool: True if dictionaries are identical, False otherwise.
    """
    # If both are not dictionaries, compare them directly
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict1 == dict2

    # If the keys differ, dictionaries are not identical
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]

        # Recursively compare nested dictionaries
        if isinstance(value1, dict) and isinstance(value2, dict):
            if not compare_nested_dicts(value1, value2):
                return False
        # If values are lists, compare them element-wise
        elif isinstance(value1, list) and isinstance(value2, list):
            if len(value1) != len(value2):
                return False
            for item1, item2 in zip(value1, value2):
                if not compare_nested_dicts(item1, item2):
                    return False
        else:
            # Directly compare the values
            if value1 != value2:
                return False
    return True


# Function to generate the path for each node
def generate_paths(node, current_path=()):
    """
    Recursively walk the AST and generate a unique path information for each node.
    The path consists of tuples indicating the relationship and index in that relationship.
    This path information is stored in annotated object definitions, in case nodes must be found later. 
    """
    # List to store paths and corresponding nodes
    paths = [(current_path, node)]

    # Recursively go over child nodes
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            # If it's a list, traverse each element with its index
            for index, item in enumerate(value):
                if isinstance(item, ast.AST):
                    # Add the field name and index to the path
                    paths.extend(generate_paths(item, current_path + ((field, index),)))
        elif isinstance(value, ast.AST):
            # If it's a single AST node, recurse with the field name as the next step in the path
            paths.extend(generate_paths(value, current_path + ((field,),)))
    
    return paths

def generate_annotated_object_dict(module, defined_objects_dict):
    object_dict = {}
    module_name = module.__name__
    try:
        # Get the source code of the module
        module_source = inspect.getsource(module)
    except Exception as e:
        ann_obj_dict_generator_module_logger.warning(f"Could not get source for module '{module_name}': {e}")
        return object_dict

    # Parse the source code into an Abstract Syntax Tree (AST)
    module_ast = ast.parse(module_source)
    source_lines = module_source.splitlines()

    # Generate paths for the entire AST
    all_node_paths = generate_paths(module_ast)

    # Class for generating dictionary of all instances of a given fully qualified class name (for annotating classes)
    class ClassConnectionCollector(ast.NodeVisitor):
        def __init__(self):
            self.class_info = {}    # Mapping from class fq name to {"instances": [], "methods": [], "subclasses": []}
            self.class_bases = {}   # Mapping from class fq name to list of base class fq names
            self.scope_stack = []
            self.class_stack = []
            self.imports = {}       # Mapping from alias to module or class name

        def visit_Import(self, node):
            # Handle 'import module' or 'import module as alias'
            for alias in node.names:
                name = alias.name
                asname = alias.asname or alias.name
                self.imports[asname] = name

        def visit_ImportFrom(self, node):
            # Handle 'from module import name' or 'from module import name as alias'
            module = node.module or ''
            for alias in node.names:
                name = alias.name
                asname = alias.asname or name
                if module:
                    full_name = module + '.' + name
                else:
                    full_name = name
                self.imports[asname] = full_name

        def visit_Module(self, node):
            self.scope_stack.append('<module>')
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_ClassDef(self, node):
            # Build the fully-qualified class name
            if self.class_stack:
                fq_class_name = '.'.join(self.class_stack + [node.name])
            else:
                fq_class_name = node.name

            # Initialize the class info if not already present
            if fq_class_name not in self.class_info:
                self.class_info[fq_class_name] = {"instances": set(), "methods": set(), "subclasses": set()}

            # Process base classes to collect inheritance information
            base_names = []
            for base in node.bases:
                base_name = self.get_full_name(base)
                base_name_resolved = self.imports.get(base_name, base_name)
                if base_name_resolved:
                    base_names.append(base_name_resolved)
            self.class_bases[fq_class_name] = base_names

            # Push the class name onto the class stack and scope stack
            self.class_stack.append(node.name)
            self.scope_stack.append(node.name)
            # Visit methods and assignments in the class body
            self.generic_visit(node)
            # Pop the class name from the stacks
            self.scope_stack.pop()
            self.class_stack.pop()

        def visit_FunctionDef(self, node):
            # Build the fully-qualified method name
            fq_method_name = '.'.join(self.scope_stack[1:] + [node.name])
            # If we are inside a class, add the method to that class's methods list
            if self.class_stack:
                fq_class_name = '.'.join(self.class_stack)
                # Ensure the class is in the class_info dictionary
                if fq_class_name not in self.class_info:
                    self.class_info[fq_class_name] = {"instances": set(), "methods": set(), "subclasses": set()}
                self.class_info[fq_class_name]["methods"].add(fq_method_name)
            self.scope_stack.append(node.name)
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_Assign(self, node):
            # Check if the value is a Call (i.e., an instance creation)
            if isinstance(node.value, ast.Call):
                # Get the fully-qualified name of the class being instantiated
                class_name = self.get_full_name(node.value.func)
                if class_name:
                    # Resolve any imports
                    class_name_resolved = self.imports.get(class_name, class_name)
                    # Collect variable names assigned to instances of the class
                    for target in node.targets:
                        var_name = self.get_full_name(target)
                        fq_var_name = '.'.join(self.scope_stack[1:] + [var_name])
                        # Add to the mapping
                        if class_name_resolved not in self.class_info:
                            self.class_info[class_name_resolved] = {"instances": set(), "methods": set(), "subclasses": set()}
                        self.class_info[class_name_resolved]["instances"].add(fq_var_name)
            self.generic_visit(node)

        def get_full_name(self, node):
            # Recursively get the fully-qualified name from an AST node
            if isinstance(node, ast.Attribute):
                value = self.get_full_name(node.value)
                attr_name = node.attr
                full_name = value + '.' + attr_name if value else attr_name
                return full_name
            elif isinstance(node, ast.Name):
                name = node.id
                if name == 'self':
                    # Return the current class fully-qualified name
                    return '.'.join(self.class_stack)
                else:
                    # Resolve the name using the imports mapping
                    return self.imports.get(name, name)
            elif isinstance(node, ast.Call):
                return self.get_full_name(node.func)
            else:
                return ''

        def build_subclasses(self):
            # Build the subclasses relationships after traversal
            for subclass, bases in self.class_bases.items():
                for base in bases:
                    # Ensure the base class is in class_info
                    if base not in self.class_info:
                        self.class_info[base] = {"instances": set(), "methods": set(), "subclasses": set()}
                    self.class_info[base]["subclasses"].add(subclass)

        def visit(self, node):
            super().visit(node)
            self.build_subclasses()

    # Class for updating 'static_definition'
    class ModificationVisitor(ast.NodeVisitor):
        def __init__(self, annotated_obj_dict, module_source, source_lines):
            self.annotated_obj_dict = annotated_obj_dict
            self.module_source = module_source
            self.source_lines = source_lines

        def _get_source_segment(self, node):
            """
            Retrieve the source code segment corresponding to the AST node.
            """
            try:
                # For Python 3.8 and newer, use ast.get_source_segment
                return ast.get_source_segment(self.module_source, node)
            except AttributeError:
                # For older Python versions, reconstruct source using line numbers
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    start = node.lineno - 1  # Line numbers start at 1
                    end = node.end_lineno
                    return '\n'.join(self.source_lines[start:end])
                else:
                    ann_obj_dict_generator_module_logger.warning(f"Cannot get source segment for node: Missing 'lineno' or 'end_lineno'")
                    return None

        def _get_nested_class_or_func_def_name (self, node):
            name = node.name if isinstance(node, (ast.ClassDef, ast.FunctionDef)) else ""
            # Traverse up the tree to find parent nodes
            parent = getattr(node, 'parent', None)
            if parent:
                parent_name = self._get_nested_class_or_func_def_name(parent)
                if parent_name:
                    return f"{parent_name}.{name}"
            return name
            
        def _get_full_name(self, node):
            """
            Recursively retrieve the fully qualified name from the AST node.
            """
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                value = self._get_full_name(node.value)
                if value:
                    return f"{value}.{node.attr}"
                else:
                    return node.attr
            elif isinstance(node, ast.Call):
                return self._get_full_name(node.func)
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                return self._get_nested_class_or_func_def_name(node)
            else:
                return None   

        def _add_modification(self, name, node):
            """
            Add modification details to the 'static_definition' list in names_dict.
            """
            text = self._get_source_segment(node)
            location = {
                'start': (node.lineno, node.col_offset),
                'end': (node.end_lineno, node.end_col_offset)
            }
            node_path = None
            for path, stored_node in all_node_paths:
                if stored_node is node:  # Compare the node objects
                    node_path = path
            
            # Get definition score
              # Instantiate the ASTScorer and traverse the AST
            scorer = ASTScorer()
            scorer.visit(node)
            # Update the object dictionary with collected definitions
            adjusted_length = scorer.score

            if 'static_definition' not in self.annotated_obj_dict[name]:
                self.annotated_obj_dict[name]['static_definition'] = []
            self.annotated_obj_dict[name]['static_definition'].append({'text': text, 
                                                                       'adjusted_length': adjusted_length,
                                                                        'location': location, 
                                                                        'ast_node' : node,  # A temporary key used for generating a list of all definition nodes
                                                                        'node_path': node_path, 
                                                                        'node_type': type(node).__name__})

        def visit_Assign(self, node):
            """
            Handle assignment operations in the AST.
            """
            for target in node.targets:
                self._handle_assignment(target, node)
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            """
            Handle augmented assignment operations in the AST.
            """
            self._handle_assignment(node.target, node)
            self.generic_visit(node)

        def visit_Call(self, node):
            """
            Handle function and method calls in the AST.
            """
            if isinstance(node.func, ast.Attribute):
                obj_name = self._get_full_name(node.func.value)
                inst_and_attr_names = self._find_inst_and_attr_names(obj_name)
                if inst_and_attr_names:
                    for name in inst_and_attr_names:
                        if name in self.annotated_obj_dict:
                            self._add_modification(name, node)
                            # Note that we are adding duplicat static definition details to instance 
                            # keys and attribute keys (prevent duplicate scoring by excluding attributes)
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            """
            Handle Class definitions and store them in 'static_definition'.
            """
            # Get the class name
            full_class_name = self._get_full_name(node)
            # Capture the full class definition
            if full_class_name and full_class_name in self.annotated_obj_dict:
                self._add_modification(full_class_name, node)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            """
            Handle Function or Method definitions and store them in 'static_definition'.
            """
            # Find the fully qualified name for the method if it's part of a class
            full_func_name = self._get_full_name(node)
            if full_func_name and full_func_name in self.annotated_obj_dict:
                self._add_modification(full_func_name, node)
            self.generic_visit(node)

        def visit_Delete(self, node):
            """
            Handle deletion operations in the AST.
            """
            for target in node.targets:
                self._handle_deletion(target, node)
            self.generic_visit(node)

        def _handle_assignment(self, target, node):
            """
            Process assignment targets and check for modifications.
            """
            full_name = self._get_full_name(target)          
            inst_and_attr_names = self._find_inst_and_attr_names(full_name)
            if inst_and_attr_names:
                for name in inst_and_attr_names:
                    if name in self.annotated_obj_dict:
                        self._add_modification(name, node)
                        # Note that we are adding duplicat static definition details to instance 
                        # keys and attribute keys (prevent duplicate scoring by excluding attributes)

        def _handle_deletion(self, target, node):
            """
            Process deletion targets and check for modifications.
            """
            full_name = self._get_full_name(target)
            inst_and_attr_names = self._find_inst_and_attr_names(full_name)
            if inst_and_attr_names:
                for name in inst_and_attr_names:
                    if name in self.annotated_obj_dict:
                        self._add_modification(name, node)
                        # Note that we are adding duplicat static definition details to instance 
                        # keys and attribute keys (prevent duplicate scoring by excluding attributes)

        def _find_inst_and_attr_names(self, full_name):
            """
            Find the base object name from a fully qualified name.
            """
            base_and_attr_names = []

            if full_name:
                parts = full_name.split('.')
                for i in range(len(parts), 0, -1):
                    # Iterating over progressively shorter slices of the parts list
                    base_name = '.'.join(parts[:i])
                    if base_name in self.annotated_obj_dict and self.annotated_obj_dict[base_name]['type'] in ['instance', 'attribute']:
                        # Collect names of the base instance and the attribute(s) if they are in annotated_obj_dict
                        base_and_attr_names.insert(0, base_name)                        
                return base_and_attr_names
            return None

    # Class for scoring function definitions
    class ASTScorer(ast.NodeVisitor):
        def __init__(self, skip_nodes_list = None):
            self.skip_nodes_list = skip_nodes_list
            self.total_pruned_score = 0
            self.score = 0
            self.score_coefficient = [1]
            self.current_args = []

        def visit_Module(self, node):
            for stmt in node.body:
                self.visit(stmt)

        def visit_FunctionDef(self, node):
            # Collect argument names
            arg_names = [arg.arg for arg in node.args.args]
            if node.args.vararg:
                arg_names.append(node.args.vararg.arg)
            if node.args.kwarg:
                arg_names.append(node.args.kwarg.arg)

            # Temporarily store current arguments
            old_args = self.current_args
            self.current_args = arg_names

            # Skip the docstring if present
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, (ast.Str, ast.Constant)):
                body = body[1:]

            # Visit the function body
            for stmt in body:
                self.visit(stmt)

            # Restore previous arguments
            self.current_args = old_args

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

        def visit_ExceptHandler(self, node):
            # Do not score exception clauses
            pass

        def visit_Raise(self, node):
            # Do not score 'raise' clauses
            pass

        def visit_Constant(self, node):
            # This method covers the majority of literal types.
            # E.g. string, int, float, bool, None, byte literals, complex numbers
            score_increment = 1
            if self.skip_nodes_list == None:
                self.score += score_increment
                log_node_and_score(node, self.score, pruned=False)
            else:
                self.total_pruned_score += score_increment
                log_node_and_score(node, self.total_pruned_score, pruned=True)

            self.generic_visit(node)

        def visit_Call(self, node):
            func_name = self.get_base_name(node.func)

            # Update set coefficient depending on if name is in module
            # Because names are overrided by scope, first check args (override module names), 
            # then module names (override imported names)
            if func_name in self.current_args:
                self.score_coefficient.append(0)
                if self.skip_nodes_list == None:
                    log_node_and_score(node, self.score, pruned=False)
                else:
                    log_node_and_score(node, self.total_pruned_score, pruned=True)
                # Do not score 'call' nodes. The names of the called nodes will be scored.
                self.generic_visit(node)
                self.score_coefficient.pop()

            elif self.is_in_module(func_name):
                self.score_coefficient.append(0.01)
                if self.skip_nodes_list == None:
                    log_node_and_score(node, self.score, pruned=False)
                else:
                    log_node_and_score(node, self.total_pruned_score, pruned=True)
                self.generic_visit(node)
                self.score_coefficient.pop()
            
            elif self.from_imported_module(func_name):
                self.score_coefficient.append(100)
                if self.skip_nodes_list == None:
                    log_node_and_score(node, self.score, pruned=False)
                else:
                    log_node_and_score(node, self.total_pruned_score, pruned=True)
                self.generic_visit(node)
                self.score_coefficient.pop()

            else:
                self.score_coefficient.append(1)
                if self.skip_nodes_list == None:
                    log_node_and_score(node, self.score, pruned=False)
                else:
                    log_node_and_score(node, self.total_pruned_score, pruned=True)
                self.generic_visit(node)
                self.score_coefficient.pop()  

        def visit_Attribute(self, node):
            base_name = self.get_base_name(node)

            # Update coefficient depending on if name is in module
            # Because names are overrided by scope, first check args (override module names), 
            # then module names (override imported names)
            if base_name in self.current_args:
                score_increment = 0
                if self.skip_nodes_list == None:
                    self.score += score_increment
                    log_node_and_score(node, self.score, pruned=False)
                else:
                    self.total_pruned_score += score_increment
                    log_node_and_score(node, self.total_pruned_score, pruned=True)
                # Do not visit child nodes (chained attributes only scored once).


            elif self.is_in_module(base_name):
                score_increment = 0.01
                if self.skip_nodes_list == None:
                    self.score += score_increment
                    log_node_and_score(node, self.score, pruned=False)
                else:
                    self.total_pruned_score += score_increment
                    log_node_and_score(node, self.total_pruned_score, pruned=True)
                # Do not visit child nodes (chained attributes only scored once).

            elif self.from_imported_module(base_name):
                score_increment = 100
                if self.skip_nodes_list == None:
                    self.score += score_increment
                    log_node_and_score(node, self.score, pruned=False)
                else:
                    self.total_pruned_score += score_increment
                    log_node_and_score(node, self.total_pruned_score, pruned=True)
                # Do not visit child nodes (chained attributes only scored once).

            else:
                score_increment = 1
                if self.skip_nodes_list == None:
                    self.score += score_increment
                    log_node_and_score(node, self.score, pruned=False)
                else:
                    self.total_pruned_score += score_increment
                    log_node_and_score(node, self.total_pruned_score, pruned=True)
                # Do not visit child nodes (chained attributes only scored once). 

        def visit_Assign(self, node):
        # Visit the 'value' part of the assignment (not the 'target')
            self.visit(node.value)

        def visit_For(self, node):
            # Skip visiting the 'target', but visit other children (like 'iter', 'body', etc.)
            self.visit(node.iter)
            for stmt in node.body:
                self.visit(stmt)
            for stmt in node.orelse:
                self.visit(stmt)

        def visit_With(self, node):
            # Skip 'optional_vars' in the 'With' statement
            for item in node.items:
                self.visit(item.context_expr)
            for stmt in node.body:
                self.visit(stmt)

        def visit_Name(self, node):
            # Name nodes are where most scoring happens. Score depends on 
            # whether names are in the in-module dictionary.
            if node.id in self.current_args:
                # Update set coefficient depending on if name is in module
                self.score_coefficient.append(0)

                score_increment = 1 * self.score_coefficient[-1]
                if self.skip_nodes_list == None:
                    self.score += score_increment
                    log_node_and_score(node, self.score, pruned=False)
                else:
                    self.total_pruned_score += score_increment
                    log_node_and_score(node, self.total_pruned_score, pruned=True)

                self.generic_visit(node)
                self.score_coefficient.pop()

            elif self.is_in_module(node.id):
                self.score_coefficient.append(0.01)

                score_increment = 1 * self.score_coefficient[-1]
                if self.skip_nodes_list == None:
                    self.score += score_increment
                    log_node_and_score(node, self.score, pruned=False)
                else:
                    self.total_pruned_score += score_increment
                    log_node_and_score(node, self.total_pruned_score, pruned=True)

                self.generic_visit(node)
                self.score_coefficient.pop()
            
            elif self.from_imported_module(node.id):
                self.score_coefficient.append(100)

                score_increment = 1 * self.score_coefficient[-1]
                if self.skip_nodes_list == None:
                    self.score += score_increment
                    log_node_and_score(node, self.score, pruned=False)
                else:
                    self.total_pruned_score += score_increment
                    log_node_and_score(node, self.total_pruned_score, pruned=True)

                self.generic_visit(node)
                self.score_coefficient.pop()

            else:
                self.score_coefficient.append(1)

                score_increment = 1 * self.score_coefficient[-1]
                if self.skip_nodes_list == None:
                    self.score += score_increment
                    log_node_and_score(node, self.score, pruned=False)
                else:
                    self.total_pruned_score += score_increment
                    log_node_and_score(node, self.total_pruned_score, pruned=True)

                self.generic_visit(node)
                self.score_coefficient.pop()      

        def get_base_name(self, node):
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                value_name = self.get_base_name(node.value)
                return value_name
            else:
                return ""
        
        def get_base_name(self, node):
            if isinstance(node, ast.Name):
                # Base case: if the node is a variable (Name node), return its ID
                return node.id
            elif isinstance(node, ast.Attribute):
                # If it's an Attribute node, recurse on its 'value' field
                return self.get_base_name(node.value)
            elif isinstance(node, ast.Subscript):
                # If it's a Subscript (e.g., list_a[0]), recurse on its 'value' field
                return self.get_base_name(node.value)
            elif isinstance(node, ast.Call):
                # For function calls, recurse into the 'func' part of the call
                return self.get_base_name(node.func)
            else:
                # If it's an unhandled node type, return an empty string
                return ""

        def get_full_name(self, node):
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                value_name = self.get_full_name(node.value)
                return f"{value_name}.{node.attr}"
            else:
                return ""

        def is_in_module(self, name):
            return (
                name in defined_objects_dict['in_module']['instance_node_names'] or
                name in defined_objects_dict['in_module']['instance_fq_names'] or
                name in defined_objects_dict['in_module']['function_node_names'] or
                name in defined_objects_dict['in_module']['function_fq_names'] or
                name in defined_objects_dict['in_module']['class_node_names'] or
                name in defined_objects_dict['in_module']['class_fq_names']
            )
        
        def from_imported_module(self, name):
            return name in defined_objects_dict['outside_module']
        
        def visit(self, node):
            if self.skip_nodes_list is not None and node in self.skip_nodes_list:
                return  # Skip this node and its children by returning early
            else:
                super().visit(node)  # Use the default dispatching to visit_* methods


    # Class to build dictionary of definitions, store location and content information
    class DefinitionCollector(ast.NodeVisitor):
        """
        AST Node Visitor class that collects definitions of functions, classes,
        and instances (variables) from the module source code, excluding special
        methods and attributes.
        """
        def __init__(self):
            # Dictionary to store collected definitions
            self.definitions = {}
            # Stacks to keep track of nested scopes (e.g., classes, functions)
            self.parent_stack = []
            self.context_stack = []

        def _is_special_name(self, name):
            """
            Determine if a name is a special (magic) method or attribute.
            """
            # Special names start and end with '__', and are longer than '__'
            return name.startswith('__') and name.endswith('__') and len(name) > 4

        def visit_FunctionDef(self, node):
            """
            Visit a function definition node.
            """
            # Skip special methods like '__init__' or '__str__'
            if self._is_special_name(node.name):
                return  # Skip processing this node
            # Get the fully qualified name (including parent scopes)
            fq_name = self._get_fq_name(node.name)

            # Collect information about the function
            obj_info = {
                'type': 'function',
                'ast_node': node,  # Store the AST node,
                'static_definition': []
            }

            # Add the function to the definitions dictionary
            self.definitions[fq_name] = obj_info
        
            # Update stacks and recursively visit child nodes
            self.parent_stack.append(node.name)
            self.context_stack.append('function')
            self.generic_visit(node)  # Continue traversing child nodes
            self.context_stack.pop()
            self.parent_stack.pop()

        def visit_ClassDef(self, node):
            """
            Visit a class definition node.
            """
            # Skip special classes (if any exist with special names)
            if self._is_special_name(node.name):
                return
            fq_name = self._get_fq_name(node.name)

            if fq_name in class_connections_dict.class_info.keys():
                class_connections = class_connections_dict.class_info[fq_name]
        
            obj_info = {
                'type': 'class',
                'ast_node': node,  # Store the AST node
                'static_definition': [],
                'class_connections': class_connections
            }
            self.definitions[fq_name] = obj_info

            # Update stacks and visit child nodes (e.g., methods)
            self.parent_stack.append(node.name)
            self.context_stack.append('class')
            self.generic_visit(node)
            self.context_stack.pop()
            self.parent_stack.pop()

        def visit_AsyncFunctionDef(self, node):
            """
            Visit an asynchronous function definition node.
            """
            # Skip special async functions
            if self._is_special_name(node.name):
                return
            fq_name = self._get_fq_name(node.name)

            obj_info = {
                'type': 'async_function',
                'ast_node': node,  # Store the AST node
                'static_definition': []
            }
            self.definitions[fq_name] = obj_info

            # Update stacks and visit child nodes
            self.parent_stack.append(node.name)
            self.context_stack.append('function')
            self.generic_visit(node)
            self.context_stack.pop()
            self.parent_stack.pop()

        def visit_Call(self, node):
            """
            This was added to collect attribute information when attributes 
            are assigned using a constructor (i.e., via keyword arguments in an instance creation).
            """
            ann_obj_dict_generator_module_logger.info("\nInside 'visit_Call'")
            # Skip assignments inside function (local variables) or class (class attributes) definitions
            ann_obj_dict_generator_module_logger.info(f"self.context_stack:{self.context_stack}")
            if self.context_stack and self.context_stack[-1] == 'function':
                return
            if self.context_stack and self.context_stack[-1] == 'class':
                return

            # Check if the function being called is a class constructor
            if isinstance(node.func, ast.Name):
                # class_name = node.func.id
                # Optionally, only process classes defined in the module.

                # Get the object being assigned to, if this Call is part of an assignment
                parent = getattr(node, 'parent', None)
                if isinstance(parent, ast.Assign):
                    # Get the targets of the assignment
                    targets = parent.targets
                    for target in targets:
                        if isinstance(target, ast.Name):
                            instance_name = target.id
                            fq_instance_name = self._get_fq_name(instance_name)

                            # Process keyword arguments as attributes
                            for keyword in node.keywords:
                                attr_name = keyword.arg
                                fq_attr_name = f"{fq_instance_name}.{attr_name}"

                                # Skip special names
                                if self._is_special_name(attr_name):
                                    continue

                                # Update obj_info for the attribute
                                obj_info = {
                                    'type': 'attribute',
                                    'ast_node': node,  # Store the AST node
                                    'initialized_definition': None,
                                    'static_definition': []
                                }
                                ann_obj_dict_generator_module_logger.info(f"fq_attr_name: {fq_attr_name}\nattr_name: {attr_name}\nassigned type: 'attribute'\n_get_source_segment(node):{self._get_source_segment(node)}")

                                self.definitions[fq_attr_name] = obj_info

            # Continue visiting child nodes
            self.generic_visit(node)

        def visit_Assign(self, node):
            """
            Visit an assignment node.
            """
            # Handle variable assignments at the module or class level
            self._handle_assign(node)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            """
            Visit an annotated assignment node (e.g., variable with type hint).
            """
            # Handle annotated assignments
            self._handle_assign(node)
            self.generic_visit(node)

        def _handle_assign(self, node):
            """
            Handle assignment nodes to collect variable definitions.
            """
            # Get the list of targets being assigned to
            targets = node.targets if hasattr(node, 'targets') else [node.target]

            # Log the targets to view the ast.Attribute nodes
            for target in targets:
                ann_obj_dict_generator_module_logger.info(f"Processing target: {ast.dump(target)}")

            # Skip assignments inside function (local variables) or class (class attributes) definitions
            ann_obj_dict_generator_module_logger.info(f"self.context_stack:{self.context_stack}")
            if self.context_stack and self.context_stack[-1] == 'function':
                return
            if self.context_stack and self.context_stack[-1] == 'class':
                return
            
            for target in targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # Skip special names like '__all__' or '__version__'
                    if self._is_special_name(name):
                        continue
                    fq_name = self._get_fq_name(name)

                    obj_info = {
                        'type': 'instance',
                        'ast_node': node,  # Store the AST node
                        'initialized_definition': None,
                        'static_definition': []
                    }
                    self.definitions[fq_name] = obj_info
    
                elif isinstance(target, ast.Attribute):
                    # Handle assignment to attributes
                    attr_name = self._get_full_attribute_name(target)
                    # Skip special names
                    if self._is_special_name(attr_name):
                        continue
                    obj_info = {
                        'type': 'attribute',
                        'ast_node': node,  # Store the AST node
                        'initialized_definition': None,
                        'static_definition': []
                    }
                    ann_obj_dict_generator_module_logger.info("Inside '_handle_assign (ast.Attribute)")
                    ann_obj_dict_generator_module_logger.info(f"attr_name: {attr_name}\nassigned type: 'attribute'\n_get_source_segment(node):{self._get_source_segment(node)}")
                    self.definitions[attr_name] = obj_info

                elif isinstance(target, ast.Tuple):
                    # Handle tuple unpacking assignments (e.g., a, b = 1, 2)
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            name = elt.id
                            if self._is_special_name(name):
                                continue
                            fq_name = self._get_fq_name(name)

                            obj_info = {
                                'type': 'instance',
                                'ast_node': node,  # Store the AST node
                                'initialized_definition': None,
                                'static_definition': []
                            }
                            self.definitions[fq_name] = obj_info
                        elif isinstance(elt, ast.Attribute):
                            # Handle attribute in tuple unpacking
                            attr_name = self._get_full_attribute_name(elt)
                            if self._is_special_name(attr_name):
                                continue
                            obj_info = {
                                'type': 'attribute',
                                'ast_node': node,  # Store the AST node
                                'initialized_definition': None,
                                'static_definition': []
                            }
                            ann_obj_dict_generator_module_logger.info("Inside '_handle_assign (ast.Tuple(ast.Attribute))")
                            ann_obj_dict_generator_module_logger.info(f"attr_name: {attr_name}\nassigned type: 'attribute'\n_get_source_segment(node):{self._get_source_segment(node)}")
                            self.definitions[attr_name] = obj_info

        def _get_full_attribute_name(self, node):
            """
            Recursively get the fully-qualified name of an attribute node.
            """
            if isinstance(node, ast.Attribute):
                value_name = self._get_full_attribute_name(node.value)
                return f"{value_name}.{node.attr}" if value_name else node.attr
            elif isinstance(node, ast.Name):
                return node.id
            else:
                return ''
    
        def _get_fq_name(self, name):
            """
            Construct the fully qualified name of an object by combining
            the parent stack with the current name.
            """
            if self.parent_stack:
                # Join parent names with the current name (e.g., ClassName.methodName)
                return '.'.join(self.parent_stack + [name])
            else:
                return name  # Top-level name

        def _get_source_segment(self, node):
            """
            Retrieve the source code segment corresponding to the AST node.
            """
            try:
                # For Python 3.8 and newer, use ast.get_source_segment
                return ast.get_source_segment(module_source, node)
            except AttributeError:
                # For older Python versions, reconstruct source using line numbers
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    start = node.lineno - 1  # Line numbers start at 1
                    end = node.end_lineno
                    return '\n'.join(source_lines[start:end])
                else:
                    ann_obj_dict_generator_module_logger.warning(f"Cannot get source segment for node: Missing 'lineno' or 'end_lineno'")
                    return None

        def visit(self, node):
            # Add parent references
            for child in ast.iter_child_nodes(node):
                child.parent = node
            super().visit(node)


    # Given an instance and it's attributes, the original object sources for any references.
    def serialize_object(obj, max_depth=16, current_depth=0, root_obj_id=None):
        """
        Serialize an object's attributes, replacing references to other objects
        with their fully qualified names, and clearly distinguishing references
        from basic data types.
        """
        obj_id = id(obj)

        if current_depth >= max_depth:
            ann_obj_dict_generator_module_logger.warning(f"Reached max depth while serializing object: {obj}")
            return '<Max Depth Reached>'

        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple, set)):
            return [serialize_object(item, max_depth, current_depth + 1, root_obj_id) for item in obj]
        elif isinstance(obj, dict):
            return {key: serialize_object(value, max_depth, current_depth + 1, root_obj_id) for key, value in obj.items()}
        elif obj_id in obj_id_to_fq_name and obj_id != root_obj_id:
            # Object is a known object from the module and not the root object
            return {'__reference__': obj_id_to_fq_name[obj_id]}
        elif hasattr(obj, '__dict__'):
            result = {'__class__': obj.__class__.__name__}
            attrs = {}
            for attr_name, attr_value in obj.__dict__.items():
                if not attr_name.startswith('__'):
                    attr_value_id = id(attr_value)
                    if attr_value_id in obj_id_to_fq_name and attr_value_id != root_obj_id:
                        # Replace attribute value with its FQN, marking it as a reference
                        attrs[attr_name] = {'__reference__': obj_id_to_fq_name[attr_value_id]}
                    else:
                        # Recursively serialize the attribute
                        attrs[attr_name] = serialize_object(attr_value, max_depth, current_depth + 1, root_obj_id)
            result['__attributes__'] = attrs
            return result
        else:
            return repr(obj)  # Fallback for objects without __dict__

    # Instantiate ClassConnectionCollector to find all class connections
    class_connections_dict = ClassConnectionCollector()
    # Visit the AST nodes
    class_connections_dict.visit(module_ast)

    # Instantiate the DefinitionCollector and traverse the AST
    definition_collector = DefinitionCollector()
    definition_collector.visit(module_ast)
    # Update the object dictionary with collected definitions
    object_dict.update(definition_collector.definitions)

    # Initialize dictionary for reverse mapping
    obj_id_to_fq_name = {}

    # Retrieve all objects and build the reverse mapping from object IDs to fully qualified names
    for fq_name, obj_info in object_dict.items():
        try:
            obj = module
            parts = fq_name.split('.')
            for part in parts:
                obj = getattr(obj, part)
            if obj_info['type'] != 'attribute':
                # Skip mapping attributes from obj_info;
                # the initialized defintion will then be stored in the instance.
                obj_id_to_fq_name[id(obj)] = fq_name
            obj_info['object'] = obj  # Store the object for later use
        except Exception as e:
            ann_obj_dict_generator_module_logger.warning(f"Could not retrieve object for '{fq_name}': {e}")
            obj_info['object'] = None

    # Update 'initialized_definition' with the initialized state for instance objects
      # Note: this cannot be done inside DefinitionCollector because it requires a complete obj_id_to_fq_name dict.
    for fq_name, obj_info in object_dict.items():
        try:
            if obj_info['type'] in ['instance', 'attribute']:
                # Retrieve the object
                obj = obj_info.get('object')
                if not obj:
                    continue  # Skip if we couldn't retrieve the object
                # Serialize the object's state, avoiding replacing the root object
                obj_serialized = serialize_object(obj, root_obj_id=id(obj))
                obj_info['initialized_definition'] = obj_serialized  # Keep as dict
        except Exception as e:
            ann_obj_dict_generator_module_logger.warning(f"Could not retrieve initialized definition for '{fq_name}': {e}")

        # Clean up the temporary 'object' key
        obj_info.pop('object', None)
        obj_info.pop('ast_node', None)  # Remove the AST node to prevent serialization issues

    # Get modifications and update object_dict
    # Instantiate the ModificationVisitor and traverse the AST
    modification_visitor = ModificationVisitor(annotated_obj_dict=object_dict, module_source=module_source, source_lines=source_lines)
    modification_visitor.visit(module_ast)

    # Generate list of all the scored nodes (every node in 'static_definition')
    scored_node_list = []
    for key, obj_info in object_dict.items():
        if obj_info['type'] != 'attribute': # Note that attribute definitions are excluded because these sould
                                            # already be in instance definitions, and we will not be counting their
                                            # scores when evaluating conjectures (to avoid double counting); therefore,
                                            # if an attribute definition is missed in the instance definition, we want it
                                            # to be scored in 'PRUNED_SCORE_PLACEHOLDER'
            for segment in obj_info['static_definition']:
                scored_node_list.append(segment['ast_node'])
                # Clean up the temporary 'ast_node' key
                segment.pop('ast_node', None)  # Remove the AST node
        else:
            for segment in obj_info['static_definition']:
                segment.pop('ast_node', None)  # Remove the AST node

    # Pass node list to ASTScorer to score remaining AST (passing over each node already scored and in 'static_definition')
      # Instantiate the ASTScorer and traverse the AST
    pruned_tree_scorer = ASTScorer(skip_nodes_list = scored_node_list)
    pruned_tree_scorer.visit(module_ast)
    # Update the object dictionary with collected definitions
    pruned_score = pruned_tree_scorer.total_pruned_score

    # Store the pruned score in the object_dict...
    object_dict[PRUNED_SCORE_PLACEHOLDER] = pruned_score
    
    # Add 'attributes' to each instance dict in object_dict
        # This will store names of all the attributes of that instance
    for instance, inst_info in object_dict.items():
        if instance == PRUNED_SCORE_PLACEHOLDER:
            continue
        elif inst_info['type'] == 'instance':
            inst_info['instance_attributes'] = []
            for attribute, attr_info in object_dict.items():
                if attribute == PRUNED_SCORE_PLACEHOLDER:
                    continue
                elif attr_info['type'] == 'attribute':
                    # Check if the instance name matches the first part of the attribute name before the dot
                    if instance == attribute.split('.')[0]:
                        inst_info['instance_attributes'].append(attribute)
                    
    return object_dict

def generate_object_pco_dict(ewm_observation_data, annotated_obj_dict):
    """
    Generate a dictionary with a key for each defined object and counts for the number of 'true', 'false', and 'none' PCOs that the object was accessed in.
    """
    true_false_none_error_count_dict = {} # Initialize the dictionary
    
    for observation_dict in ewm_observation_data:
        if 'pco_bool' in observation_dict and observation_dict['pco_bool'] == False:
            for obj_name in observation_dict['all_objects']:
                # If the object_name string is not already a key in true_false_dicts, add it
                if obj_name not in true_false_none_error_count_dict:
                    true_false_none_error_count_dict[obj_name] = {'true': 0, 'false': 0, 'none': 0, 
                                                            'error': 0, 'total': 0}
                # Increment the false count for the string
                true_false_none_error_count_dict[obj_name]['false'] += 1
                true_false_none_error_count_dict[obj_name]['total'] += 1      
        elif 'pco_bool' in observation_dict and observation_dict['pco_bool'] == True:
            for obj_name in observation_dict['all_objects']:
                # If the object_name string is not already a key in true_false_dicts, add it
                if obj_name not in true_false_none_error_count_dict:
                    true_false_none_error_count_dict[obj_name] = {'true': 0, 'false': 0, 'none': 0, 
                                                            'error': 0, 'total': 0}
                # Increment the false count for the string
                true_false_none_error_count_dict[obj_name]['true'] += 1
                true_false_none_error_count_dict[obj_name]['total'] += 1    
        elif 'pco_bool' in observation_dict and observation_dict['pco_bool'] == None:
            for obj_name in observation_dict['all_objects']:
                # If the object_name string is not already a key in true_false_dicts, add it
                if obj_name not in true_false_none_error_count_dict:
                    true_false_none_error_count_dict[obj_name] = {'true': 0, 'false': 0, 'none': 0, 
                                                            'error': 0, 'total': 0}
                # Increment the false count for the string
                true_false_none_error_count_dict[obj_name]['none'] += 1
                true_false_none_error_count_dict[obj_name]['total'] += 1    
        elif 'computation_error' in observation_dict:
            for obj_name in observation_dict['all_objects']:
                # If the object_name string is not already a key in true_false_dicts, add it
                if obj_name not in true_false_none_error_count_dict:
                    true_false_none_error_count_dict[obj_name] = {'true': 0, 'false': 0, 'none': 0, 
                                                            'error': 0, 'total': 0}
                # Increment the false count for the string
                true_false_none_error_count_dict[obj_name]['error'] += 1
                true_false_none_error_count_dict[obj_name]['total'] += 1    

    # Update counts for instances using all attribute counts
    for fq_name, obj_info in annotated_obj_dict.items():
        if fq_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
            continue
        if obj_info['type'] == "instance":
            if fq_name not in true_false_none_error_count_dict:
                true_false_none_error_count_dict[fq_name] = true_false_none_error_count_dict[fq_name] = {'true': 0, 'false': 0, 'none': 0, 
                                                           'error': 0, 'total': 0}
            for attribute in obj_info['instance_attributes']:
                if attribute in true_false_none_error_count_dict:
                    true_false_none_error_count_dict[fq_name]['false'] += true_false_none_error_count_dict[attribute]['false']
                    true_false_none_error_count_dict[fq_name]['true'] += true_false_none_error_count_dict[attribute]['true']
                    true_false_none_error_count_dict[fq_name]['none'] += true_false_none_error_count_dict[attribute]['none']
                    true_false_none_error_count_dict[fq_name]['error'] += true_false_none_error_count_dict[attribute]['error']
                    true_false_none_error_count_dict[fq_name]['total'] += true_false_none_error_count_dict[attribute]['total']
                else:
                    ann_obj_dict_generator_module_logger.warning(f"Missing attribute {attribute} of instance {fq_name} in obseration dict. It is in annotated_obj_dict, 'instance_connections'.")

    # Update counts for classes using all instance and method counts
    for fq_name, obj_info in annotated_obj_dict.items():
        if fq_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
            continue
        if obj_info['type'] == "class":
            if fq_name not in true_false_none_error_count_dict:
                true_false_none_error_count_dict[fq_name] = true_false_none_error_count_dict[fq_name] = {'true': 0, 'false': 0, 'none': 0, 
                                                           'error': 0, 'total': 0}
            for instance in obj_info['class_connections']['instances']:
                if instance in true_false_none_error_count_dict:
                    true_false_none_error_count_dict[fq_name]['false'] += true_false_none_error_count_dict[instance]['false']
                    true_false_none_error_count_dict[fq_name]['true'] += true_false_none_error_count_dict[instance]['true']
                    true_false_none_error_count_dict[fq_name]['none'] += true_false_none_error_count_dict[instance]['none']
                    true_false_none_error_count_dict[fq_name]['error'] += true_false_none_error_count_dict[instance]['error']
                    true_false_none_error_count_dict[fq_name]['total'] += true_false_none_error_count_dict[instance]['total']
                else:
                    ann_obj_dict_generator_module_logger.warning(f"Missing instance {instance} of class {fq_name} in obseration dict. It is in annotated_obj_dict, 'class_connetions'.")

            for method in obj_info['class_connections']['methods']:
                if method in true_false_none_error_count_dict:
                    true_false_none_error_count_dict[fq_name]['false'] += true_false_none_error_count_dict[method]['false']
                    true_false_none_error_count_dict[fq_name]['true'] += true_false_none_error_count_dict[method]['true']
                    true_false_none_error_count_dict[fq_name]['none'] += true_false_none_error_count_dict[method]['none']
                    true_false_none_error_count_dict[fq_name]['error'] += true_false_none_error_count_dict[method]['error']
                    true_false_none_error_count_dict[fq_name]['total'] += true_false_none_error_count_dict[method]['total']
                else:
                    ann_obj_dict_generator_module_logger.warning(f"Missing method {method} of class {fq_name} in obseration dict. It is in annotated_obj_dict, 'class_connetions'.")
    
    ann_obj_dict_generator_module_logger.info(json.dumps(true_false_none_error_count_dict, indent=2, cls=CustomJSONEncoder)[:1000])
    
    # for fq_name, obj_info in annotated_obj_dict.items():
    #     if fq_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
    #         continue
    #     if obj_info['type'] == "class":
    #         for subclass in obj_info['class_connections']['subclasses']:
    #             true_false_none_error_count_dict[fq_name]['false'] += true_false_none_error_count_dict[subclass]['false']
    #             true_false_none_error_count_dict[fq_name]['true'] += true_false_none_error_count_dict[subclass]['true']
    #             true_false_none_error_count_dict[fq_name]['none'] += true_false_none_error_count_dict[subclass]['none']
    #             true_false_none_error_count_dict[fq_name]['error'] += true_false_none_error_count_dict[subclass]['error']
    #             true_false_none_error_count_dict[fq_name]['total'] += true_false_none_error_count_dict[subclass]['total']
    #         else:
    #             ann_obj_dict_generator_module_logger.warning(f"Missing method {method} of class {fq_name} in obseration dict. It is in annotated_obj_dict,'class_connections'.")

    # Compute 'false_none_and_error_to_true' values for each object in true_false_none_error_count_dict
    for object_name, value in true_false_none_error_count_dict.items():
        true_count = value['true']
        false_count = value['false']
        none_count = value['none']
        error_count = value['error']
        
        # Handle division by zero by setting the ratio to None or a custom value (e.g., 'undefined')
        if true_count == 0:
            false_none_and_error_to_true = None if (false_count + none_count + error_count) == 0 else float('inf')  # Set to infinity if only false_count is positive
        else:
            false_none_and_error_to_true = (false_count + none_count + error_count)/true_count
        
        # Store the calculated ratio in the dictionary
        true_false_none_error_count_dict[object_name]['false_none_and_error_to_true'] = false_none_and_error_to_true
    
    return true_false_none_error_count_dict


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
def initialize_ann_obj_dict(next_ewm_name, 
                ewm_dir, 
                ann_obj_dir,
                log_dir,
                last_ewm_name=None, # Default to 'None'; an optional input
                                    # If only one EWM is input, skip comparison steps 
                verbose_bool=False):
              
    # Configure logging
    global ann_obj_dict_generator_module_logger
    ann_obj_dict_generator_module_logger = set_logging(verbose_bool, log_dir, 'ann_obj_dict_generator_module')

    # Generate the file paths
    last_ann_obj_dicts_path = f"{ann_obj_dir}/{last_ewm_name}_ann_obj.json"
    ann_obj_dict_generator_module_logger.info(f"last_ann_obj_dicts_path: {last_ann_obj_dicts_path}")
    next_ann_obj_dicts_path = f"{ann_obj_dir}/{next_ewm_name}_ann_obj.json"
    next_ewm_path = f"{ewm_dir}/{next_ewm_name}.py"
    ann_obj_dict_generator_module_logger.info(f"next_ewm_path: {next_ewm_path}")

    try:
        # Import new EWM module
        next_ewm_module = import_module_from_path(next_ewm_name, next_ewm_path)
    except () as e:
        ann_obj_dict_generator_module_logger.error(f"Error: {e}")

    # Generate a dictionary storing names of all defined objects in the next EWM (and names of all imported objects)
    next_ewm_objects_dict = track_definitions_and_imports(next_ewm_module)
    ann_obj_dict_generator_module_logger.info("**** next_ewm_objects_dict (first 1000 char) ****")
    ann_obj_dict_generator_module_logger.info(json.dumps(next_ewm_objects_dict, indent=2, cls=CustomJSONEncoder)[:10000])

    # Generate the annotated object dictionary of next EWM (stores info for each definition and scores them for organization)
      # Lower score mean better organization for any given function (uses other internal functions when possible)
    next_annotated_object_dict = generate_annotated_object_dict(next_ewm_module, next_ewm_objects_dict)
    ann_obj_dict_generator_module_logger.info("**** next_annotated_object_dict (first 1000 char) ****")
    # ann_obj_dict_generator_module_logger.info(json.dumps(next_annotated_object_dict, indent=2, cls=CustomJSONEncoder)[:1000])

    # Update the annotated object dictionary with function argument types
    ann_obj_dict_generator_module_logger.info(f"Updating annotated object dictionary with function argument types:")
    for fq_name, obj_info in next_annotated_object_dict.items():
        if fq_name == PRUNED_SCORE_PLACEHOLDER:
            continue
        elif obj_info['type']=='function':
            ann_obj_dict_generator_module_logger.info(f"fq_name: {fq_name}")
            func_obj = get_func_obj(fq_name, next_ewm_module)
            param_type_info = resolve_parameter_types(func_obj)
            func_signature = inspect.signature(func_obj)
            param_type_list= list()

            for arg_name in func_signature.parameters:
                arg_info = param_type_info[arg_name]
                if isinstance(arg_info.type, tuple):
                    for t in arg_info.type:
                        param_type_list.append(t.__name__ if t else ANY_TYPE_PLACEHOLDER)
                else:
                    param_type_list.append(arg_info.type.__name__ if arg_info.type else ANY_TYPE_PLACEHOLDER)
                # default = "" if info.default is inspect.Parameter.empty else f" (default={info.default})"
            obj_info['arg_types'] = param_type_list  

    # Update the annotated object dictionary with attribute object types
    for fq_name, obj_info in next_annotated_object_dict.items():
        if fq_name == PRUNED_SCORE_PLACEHOLDER:
            continue
        elif obj_info['type']=='attribute':        
            # Get attribute type
            attr_type=get_nested_attribute_type_safe(fq_name, next_ewm_module, default=None)
            ann_obj_dict_generator_module_logger.info(f"fq_name: {fq_name}\n obj_info:{obj_info}\n attr_type:{attr_type}")
            obj_info['attribute_type'] = attr_type.__name__

    # If only one EWM is input skip comparisons.
    last_annotated_object_dict=None
    if not last_ewm_name:
        ann_obj_dict_generator_module_logger.info(f"No 'last_ewm_name' input. Skipping EWM comparisons.")

    # If 'last_ewm_name' is provided as input, update the annotated object dictionaries with difference info
    else:
        ann_obj_dict_generator_module_logger.info(f"'last_ewm_name' input found: {last_ewm_name}. Updating next_annotated_object_dict with EWM difference info.")
        # Load last EWMs annotated object dictionaries from JSON
        last_annotated_object_dict= load_json(last_ann_obj_dicts_path)

        # Update the new annotated defintions with 'added', 'unchanged', and 'removed' ('removed' will store the old annotated object info) 
        for fq_name, obj_info in next_annotated_object_dict.items():
            if fq_name == PRUNED_SCORE_PLACEHOLDER:
                continue
            elif fq_name in last_annotated_object_dict.keys():
                if 'initialized_definition' in obj_info and 'initialized_definition' in last_annotated_object_dict[fq_name]:
                    old_initialized_definition = last_annotated_object_dict[fq_name]['initialized_definition']
                    new_initialized_definition = obj_info['initialized_definition']
                    initialised_bool = compare_nested_dicts(old_initialized_definition, new_initialized_definition)
                    # initialised_bool is 'True' when the initialized definitions are identical 
                    obj_info['unchanged'] = initialised_bool
                    # Update 'unchanged' value ('True' when initialized, or static when no initialized, definitions are the same)
                else:
                    old_static_definitions = last_annotated_object_dict[fq_name]['static_definition']
                    new_static_definitions = obj_info['static_definition']
                    static_comparison_bool = compare_static_definitions(old_static_definitions, new_static_definitions)
                    # static_comparison_bool is 'True' when the static definitions are identical 
                    obj_info['unchanged'] = static_comparison_bool 
                    # Note that 'unchanged' only reads reference names (does not check if references contents changed)
                        # It seems possible to recursively update 'changed' to 'True' for every object that references a changed definition; but not necessary now.
            else:
                obj_info['added'] = True

        # Store 'removed' definitions
        for fq_name, obj_info in last_annotated_object_dict.items():
            if fq_name not in next_annotated_object_dict.keys():
                next_annotated_object_dict[fq_name]=obj_info
                next_annotated_object_dict[fq_name]['removed'] = True

    # Add the new_ewm_objects_dict to the new_annotated_object_dict
    next_annotated_object_dict[DEFINED_OBJECTS_PLACEHOLDER]=next_ewm_objects_dict

    # # Log warnig messages if annotations are missing for any fq names in the annotated object dict
    # for fq_name, obj_info in new_annotated_object_dict.items():
    #     if 'pco_val_counts' not in obj_info:
    #         ann_obj_dict_generator_module_logger.warning(f"Missing pco_val_counts for {fq_name} key in new_annotated_object_dict")

    # Save new conjecture assessment JSON
    save_results(next_ann_obj_dicts_path, next_annotated_object_dict)
    print(f"\nSaved new annotated_object_dict at: {next_ann_obj_dicts_path}\n")
    
    return (last_annotated_object_dict, next_annotated_object_dict)
   
def update_pco_data(ewm_name, 
                obs_json_dir,
                ann_obj_dir,
                log_dir,
                verbose_bool=False):
              
    # Configure logging
    global ann_obj_dict_generator_module_logger
    ann_obj_dict_generator_module_logger = set_logging(verbose_bool, log_dir, 'ann_obj_dict_generator_module')

    # Generate the file paths
    obs_dicts_path = f"{obs_json_dir}/{ewm_name}_obs.json"
    ann_obj_dicts_path = f"{ann_obj_dir}/{ewm_name}_ann_obj.json"

    # Generate a dictionary with a key for each defined object and counts for the number of 
      # 'true', 'false', and 'none' PCOs that the object was accessed in. 
        # Format:  {'object_name':{'true': 0, 'false': 0, 'none': 0, 'false_and_none_to_true': 0}, ...}
    ewm_obs_dicts= load_json(obs_dicts_path)
    ann_obj_dict = load_json(ann_obj_dicts_path)
    ewm_obj_pco_val_dict = generate_object_pco_dict(ewm_obs_dicts, ann_obj_dict)

    # Insert object_pco_val_dict into annotated object dict
    for fq_name, obj_info in ann_obj_dict.items():
        if fq_name in [PRUNED_SCORE_PLACEHOLDER, DEFINED_OBJECTS_PLACEHOLDER]:
            continue
        elif fq_name in ewm_obj_pco_val_dict.keys():
            obj_info['pco_val_counts'] = ewm_obj_pco_val_dict[fq_name]
        else:
            # Set all PCO value counts to zero and log warning.
            obj_info['pco_val_counts'] = {'true': 0, 'false': 0, 'none': 0, 
                                        'error': 0, 'total': 0, 'false_none_and_error_to_true':0}
            ann_obj_dict_generator_module_logger.warning(f"Missing pco_val_counts for {fq_name} key in new ann_obj_dict: '{ann_obj_dicts_path}'")

    ann_obj_dict_generator_module_logger.info(f"pco_val_counts added to annotated_object_dict.")


    # # Log warnig messages if annotations are missing for any fq names in the annotated object dict
    # for fq_name, obj_info in new_annotated_object_dict.items():
    #     if 'pco_val_counts' not in obj_info:
    #         ann_obj_dict_generator_module_logger.warning(f"Missing pco_val_counts for {fq_name} key in new_annotated_object_dict")

    # Save new conjecture assessment JSON
    save_results(ann_obj_dicts_path, ann_obj_dict)
    print(f"Updated ann_obj_dict with PCO data and saved at: {ann_obj_dicts_path}\n")
    
    return ann_obj_dict
 

if __name__ == "__main__":
    # EWM names:
    test_last_ewm_name = None  #  "test_dynamic_modification_01"
    test_next_ewm_name = "20241223_153122_270575"  
    # Directories:
    ewm_dir=".../generator_functions/ewm_files"
    ann_obj_dir=".../generator_functions/obs_ann_conj_files"
    obs_json_dir='.../generator_functions/obs_ann_conj_files'
    log_dir = '.../generator_functions/log_files'

    last_ewm_ann_obj_dicts, next_ewm_ann_obj_dicts=initialize_ann_obj_dict(
        next_ewm_name=test_next_ewm_name,
        last_ewm_name=None,  
        ewm_dir=ewm_dir, 
        ann_obj_dir=ann_obj_dir, 
        log_dir= log_dir,
        verbose_bool=False)        

    # ewm_ann_obj_dicts=update_pco_data(
    #     ewm_name=test_next_ewm_name, 
    #     obs_json_dir=obs_json_dir,
    #     ann_obj_dir=ann_obj_dir,
    #     log_dir= log_dir,
    #     verbose_bool=False)   

"""
Code to retrieve an AST node given its path:


# Function to retrieve a node from the AST given a path
def get_node_by_path(tree, path):

    # Given an AST and a path, return the node that corresponds to the path.
 
    current_node = tree
    for step in path:
        if len(step) == 1:
            # This is a single field (not a list)
            field = step[0]
            current_node = getattr(current_node, field)
        elif len(step) == 2:
            # This is a field that contains a list
            field, index = step
            current_node = getattr(current_node, field)[index]
    return current_node

# Example: Retrieve the node for the Assign node (x = 10) based on its path
path_for_assign_node = (('body', 0), ('body', 0), ('body', 0))

# Get the node from the AST based on the path
retrieved_node = get_node_by_path(tree, path_for_assign_node)

# Print information about the retrieved node
if retrieved_node:
    print(f"Retrieved node type: {type(retrieved_node).__name__}")
    if isinstance(retrieved_node, ast.Assign):
        print(f"Retrieved node represents an assignment on line {retrieved_node.lineno}")
else:
    print("Node not found")

    

"""