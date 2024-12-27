"""
main_function.py
"""
import sys
import os
import importlib.util
import logging
from datetime import datetime

# Global variable for this module's logger
main_module_logger = None

class LearningLoop():
    def __init__(self, 
                # LLM API info 
                conj_api_key, 
                conj_llm_name,
                obs_api_key, 
                obs_llm_name,
                # Directories
                ewm_dir, 
                obs_json_dir,
                ann_obj_dir,
                conj_dir,  
                log_dir,             
                # Seed knowledge
                seed_prompt=None, 
                seed_ewm_name=None, 
                # Logging bool (verbose or not)
                verbose=False,
                # Number of learning cycles
                num_cycles = 8):
        
        self.conj_api_key=conj_api_key
        self.conj_llm_name=conj_llm_name
        self.obs_api_key=obs_api_key
        self.obs_llm_name=obs_llm_name

        self.seed_prompt=seed_prompt
        self.seed_ewm_name=seed_ewm_name

        self.learning_module_dir=learning_module_dir
        self.ewm_dir=ewm_dir
        self.obs_json_dir=obs_json_dir
        self.ann_obj_dir=ann_obj_dir
        self.conj_dir=conj_dir
        self.log_dir=log_dir

        self.verbose=verbose
        self.num_cycles=num_cycles

        # Variables created while cycling
        self.last_ewm_name=None
        self.next_ewm_name=seed_ewm_name

        """
        Note that obs_dicts, ann_obj_dicts, and conj_dicts are not actually used inside 'main_function'
        and are not passed between modules. These variables can be removed in future update (modules don't need to return them).
        """
    
    def learn(self):
        """
        If a seed prompt is provided, generate a seed EWM (assign it to 'next_ewm')
        Begin with an EWM, then:
            1) Generate predictions (if not available)
                a) Generate questions
                b) Generate observations
            2) If a previous EWM exists
                a) Compare current EWM to 'last' ewm (if one exists); accept or reject 'next' EWM
            3) Generate conjecture (if 'next_ewm' was rejected use 'last_ewm', otherwise use 'next_ewm')
                a) Assign 'next_ewm' to 'last_ewm'
                b) Assign new EWM to 'next_ewm'
            [iterate until num_cycles == 0]
        """

        # Ensure there is one seed prompt, one seed EWM, or a last EWM
        if self.seed_prompt and self.seed_ewm_name:
            print(f"Error: Cannot start with both seed prompt and seed EWM; input only one:\n  - seed_prompt: {self.seed_prompt}\n  - seed_ewm: {self.seed_ewm_name}")
            return
        if not self.seed_prompt and not self.seed_ewm_name:
            # Check if 'learn' is being run with a given 'last_ewm' and 'next_ewm' (need to be manually assigned); 'next_ewm' defaults to 'seed_ewm' value.
            if self.last_ewm_name == None: 
                print(f"Error: No seed prompt and no seed EWM passed to 'LearningLoop.learn'.\nInput either seed prompt or seed EWM, or assign 'last_ewm' and 'next_ewm' to run using a an existing learning chain.")
                return
        
        # If there is a seed prompt, generate first EWM
        if self.seed_prompt:
            self.next_ewm_name = conjecture_and_updater_module.main_function(
                # LLM API info 
                conj_api_key=self.conj_api_key,
                conj_llm_name=self.conj_llm_name,
                # Seed knowledge
                seed_prompt=seed_prompt,
                last_ewm_name=None,
                next_ewm_name=None,
                # Directories
                ewm_dir=self.ewm_dir,
                obs_json_dir=self.obs_json_dir,
                ann_obj_dir=self.ann_obj_dir,
                conj_dir=self.conj_dir,
                log_dir=self.log_dir,
                # Logging bool
                verbose=self.verbose)

            # Count down one learning cycle (one new update was attempted)
            self.num_cycles -= 1
            main_module_logger.info(f"self.num_cycles: {self.num_cycles}")
            print(f"Learning cycles remaining {self.num_cycles}\n")
            # Set self.seed_prompt to none (it has been used to create an EWM)
            self.seed_prompt = None

        # Run designated number of learning cycles
        while self.num_cycles > 0:
            # Check if observation JSON data exists. If it's missing, generate it 
            obs_json_filename = f"{self.next_ewm_name}_obs.json"
            obs_json_path = os.path.join(self.obs_json_dir, obs_json_filename)
            obs_file_exists = os.path.exists(obs_json_path)
            print(f"\n------------------------\nBEGINNING LEARNING CYCLE\n------------------------\n  self.last_ewm_name: {self.last_ewm_name}\n  self.next_ewm_name: {self.next_ewm_name}\n\n")
            main_module_logger.error(f"\n------------------------\nBEGINNING LEARNING CYCLE\n------------------------\n  self.last_ewm_name: {self.last_ewm_name}\n  self.next_ewm_name: {self.next_ewm_name}\n\n")

            # Generate ann_obj_dicts
            ann_obj_generator_module.initialize_ann_obj_dict(
                next_ewm_name=self.next_ewm_name,
                last_ewm_name=self.last_ewm_name,
                ewm_dir=self.ewm_dir, 
                ann_obj_dir=self.ann_obj_dir, 
                log_dir=self.log_dir,
                verbose_bool=self.verbose)   
            main_module_logger.info(f"'ann_obj_generator_module.initialize_ann_obj_dict' executed with last_ewm_name '{self.last_ewm_name}', next_ewm_name '{self.next_ewm_name}'")

            # Check that the annotated object JSON was created successfully. 
            ann_obj_json_filename = f"{self.next_ewm_name}_ann_obj.json"
            ann_obj_json_path = os.path.join(self.ann_obj_dir, ann_obj_json_filename)
            ann_obj_file_exists = os.path.exists(ann_obj_json_path)
            if not ann_obj_file_exists:
                main_module_logger.error(f"Unexpected behavior: annotated object file should have been created for '{self.next_ewm_name}' but it was not found.\nobs_json_filename:{obs_json_filename}\nobs_json_path:{obs_json_path}\nobs_file_exists:{obs_file_exists}")
                sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.
       
            if not obs_file_exists:
                # Generate predictions.
                print(f"Generating predictions for EWM: {self.next_ewm_name}\n")
                main_module_logger.info(f"Generating preditions for EWM: {self.next_ewm_name}\n")
                prediction_generator_module.main_function(
                    ewm_name=self.next_ewm_name, 
                    ewm_dir=self.ewm_dir,
                    obs_json_dir=self.obs_json_dir, 
                    log_dir=self.log_dir,
                    func_tracer_module=function_tracer_module, 
                    replace=True,
                    verbose=self.verbose)
                main_module_logger.info(f"'prediction_generator_module.main_function' executed with ewm_name '{self.next_ewm_name}'")
          
                # Note: Questions and observations will be inherited via cross-compute module or generated after. 
                  # Predictions previously observed should receive old observation data.

            # Check that the observation JSON was created successfully (if it did not exist). 
            obs_file_exists = os.path.exists(obs_json_path)
            if not obs_file_exists:
                main_module_logger.error(f"Unexpected behavior: observation file should have been created for '{self.next_ewm_name}' but it was not found.\nobs_json_filename:{obs_json_filename}\nobs_json_path:{obs_json_path}\nobs_file_exists:{obs_file_exists}")
                sys.exit(1)  # Exit module. This kind of error is unexpected and not handled in module.
                              
            if self.last_ewm_name:
                # Cross-compute observations
                main_module_logger.info(f"Cross-computing observations for:\n  Last EWM: {self.last_ewm_name}\n  Next EWM: {self.next_ewm_name}\n")
                print(f"Cross-computing observations for:\n  Last EWM: {self.last_ewm_name}\n  Next EWM: {self.next_ewm_name}\n")
                cross_compute_observations_module.compare_predictions(
                    func_tracer_module=function_tracer_module,  
                    question_generator_module=question_generator_module,
                    observation_generator_module=observation_generator_module,
                    previous_ewm_name=self.last_ewm_name, 
                    new_ewm_name=self.next_ewm_name, 
                    ewm_dir=self.ewm_dir, 
                    obs_json_dir=self.obs_json_dir,
                    ann_obj_dir=self.ann_obj_dir,
                    log_dir=self.log_dir,
                    obs_api_key=self.obs_api_key,
                    obs_llm_name=self.obs_llm_name,
                    verbose=self.verbose)  
                main_module_logger.info(f"'cross_compute_observations_module.compare_predictions' executed with last_ewm_name '{self.last_ewm_name}', next_ewm_name '{self.next_ewm_name}'")

            else:
                main_module_logger.info(f"Warning: unable to cross-compute observations. Missing 'last_ewm_name' ({self.last_ewm_name}) or 'last_ewm_obs_dicts'.")
                print(f"Warning: unable to cross-compute observations. Missing 'last_ewm_name' ({self.last_ewm_name}) or 'last_ewm_obs_dicts'.\n") 
            
                # Generate question prompts.
                print(f"Generating question prompts for EWM: {self.next_ewm_name}\n")
                main_module_logger.info(f"Generating question prompts for EWM: {self.next_ewm_name}\n")
                question_generator_module.process_json(
                    ewm_name=self.next_ewm_name,
                    obs_json_dir=self.obs_json_dir,
                    ann_obj_dir=self.ann_obj_dir)
                main_module_logger.info(f"'question_generator_module.process_json' executed with ewm_name '{self.next_ewm_name}'")

                # Generate observation data.
                print(f"Generating observations for EWM: {self.next_ewm_name}\n")
                main_module_logger.info(f"Generating observations for EWM: {self.next_ewm_name}\n")
                observation_generator_module.main_function(
                    obs_api_key=self.obs_api_key,
                    obs_llm_name=self.obs_llm_name,
                    ewm_name=self.next_ewm_name, 
                    obs_json_dir=self.obs_json_dir,
                    log_dir=self.log_dir, 
                    verbose=self.verbose)
                main_module_logger.info(f"'observation_generator_module.main_function' executed with ewm_name '{self.next_ewm_name}'")
    
            # Update next EWMs annotated object dict with PCO data
                # self.last_ewm_ann_obj_dicts should remain 'None' if 'self.last_ewm_name' is None. 
            print(f"Updating next EWM's annotated object dict with PCO data: {self.next_ewm_name}\n") 
            main_module_logger.info(f"Updating next EWM's annotated object dict with PCO data: {self.next_ewm_name}")       
            ann_obj_generator_module.update_pco_data(
                ewm_name=self.next_ewm_name, 
                obs_json_dir=self.obs_json_dir,
                ann_obj_dir=self.ann_obj_dir,
                log_dir= self.log_dir,
                verbose_bool=self.verbose) 
            main_module_logger.info(f"'ann_obj_generator_module.update_pco_data' executed with ewm_name '{self.next_ewm_name}'")
            
            # Set 'conjecture_accepted_bool' default to None 
              # If there is no last EWM, it will remain None and the seed_ewm will be shifted from 'next_ewm' to 'last_ewm'
            conjecture_accepted_bool = None

            # If last EWM is present decide whether to accept or reject next EWM
            if self.last_ewm_name:
                # Evaluate update (accept or reject conjecture)
                conjecture_accepted_bool, conjecture_evaluation_summary_dict = evaluate_conjecture_quality_module.main_function(
                    last_ewm_name=self.last_ewm_name, 
                    next_ewm_name=self.next_ewm_name,                 
                    ann_obj_dir=self.ann_obj_dir,
                    obs_dir=self.obs_json_dir,
                    conj_dir = self.conj_dir,
                    log_dir=self.log_dir,
                    verbose_bool=self.verbose)
                main_module_logger.info(f"'evaluate_conjecture_quality_module'.main_function executed with last_ewm_name '{self.last_ewm_name}', next_ewm_name '{self.next_ewm_name}'")
                main_module_logger.info(f"conjecture_accepted_bool: {conjecture_accepted_bool}")
                print(f"conjecture_accepted_bool: {conjecture_accepted_bool}\n")

            # Generate next conjecture
            new_ewm_name = None # 'new_ewm_name' remains None until conjecture_and_updater_module succeeds along a question path
            
            while new_ewm_name is None and self.num_cycles > 0:
                main_module_logger.info(f"Calling 'conjecture_and_updater_module':\n  seed_prompt: {seed_prompt}\n  last_ewm_name: {self.last_ewm_name}\n  next_ewm_name: {self.next_ewm_name}")
                new_ewm_name = conjecture_and_updater_module.main_function(
                    # LLM API info 
                    conj_api_key=self.conj_api_key,
                    conj_llm_name=self.conj_llm_name,
                    # Seed knowledge
                    seed_prompt=None,
                    last_ewm_name=self.last_ewm_name,
                    next_ewm_name=self.next_ewm_name,
                    # Directories
                    ewm_dir=self.ewm_dir,
                    obs_json_dir=self.obs_json_dir,
                    ann_obj_dir=self.ann_obj_dir,
                    conj_dir=self.conj_dir,
                    log_dir=self.log_dir,
                    # Logging bool
                    verbose=False)

                # Count down one learning cycle (one new update was attempted)
                self.num_cycles -= 1
                main_module_logger.info(f"self.num_cycles: {self.num_cycles}")
                print(f"Learning cycles remaining {self.num_cycles}\n")

            # If this is the first conjecture for a seed_ewm (i.e. no 'last_ewm')
              # Or if the prior conjecture was evaluated and accepted:
            if conjecture_accepted_bool is None or conjecture_accepted_bool is True:
                # Reassign next_ewm data to last_ewm data and new_ewm to next_ewm 
                self.last_ewm_name = self.next_ewm_name
                self.next_ewm_name = new_ewm_name

            # Otherwise, the prior conjecture was evaluated and rejected:
            else:
                # Do not change 'last_ewm' (re-use the last ewm; it was better)
                # Replace 'next_ewm' (the rejected ewm) with new_ewm (the new conjecture)
                # The new conjecture ewm will be evaluated against the original ewm (not the rejected one)
                self.next_ewm_name = new_ewm_name


"""
HELPER FUNCTIONS
"""
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


def set_logging(verbose_bool, log_dir, module_name):
    """Creates and configures a logger for a specific module with its own log file"""
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
    # Construct the log filename with timestamp and module name
    log_filename = f'{timestamp}_{module_name}.log'
    log_filepath = os.path.join(log_dir, log_filename)

    # Get a named logger for this module
    logger = logging.getLogger(module_name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Create a file handler with the module-specific filename
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
if __name__ == "__main__":
    """
    Seed data (EWM or prompt)
    Run 'learn' with either a seed prompt or a seed EWM; 
    Or initialize 'learner' then manually assign 'self.last_ewm' and 'self.next_ewm' to run from a point in an existing EWM learning chain.
    """

    # Optional seed prompt
    seed_prompt = """Write a Python module that includes a class with at least one method, and a few instances that can be used as inputs to that method. Use instance names that correspond to real things. 
All the objects should be representations of specific real world 'things' and interactions. Use type hints where appropriate. Don't include print statements or example computations in the code. I will manually generate the computations later.
When you are ready to write the code, write it between triple backticks, like this:
```
[put ONLY your module code here, do not write "python" or a module name]
```
"""

    # Optional path to seed EWM 
    seed_ewm_name = None

    # Optional 'last_ewm' and 'next_ewm' if running 'learn' with an existing EWM chain
    last_ewm_name = '20241225_210944_080976'
    next_ewm_name = '20241225_211010_191301'

    # Set number of conjecture updates to learn
    NUM_CYCLES = 32

    # Set logging bool (verbose or not verbose). 
        # 'True' sets logging to Debug and prints logs to terminal.
        # 'False' sets logging to Info and saves log files in log_dir.
    verbose=False

    # API Keys and API info:
    TOGETHER_API_KEY = ''
    conjecture_model_name = 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo' #"meta-llama/Llama-3.3-70B-Instruct-Turbo" 

    HUGGING_FACE_HUB_TOKEN = ""
    observer_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Placeholder variables (used in JSON files)
    PRUNED_SCORE_PLACEHOLDER = 'N8cT89C044lqvJ'  # Used in the JSON list of annotated object dictionaries; stores a score of the code, excluding scored definitions
    DEFINED_OBJECTS_PLACEHOLDER = "vcM/juON9%1gv!lLp"
    ANY_TYPE_PLACEHOLDER = "kHNMWb9b^1!gzcx"

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
    # Path to conjecture json directory
    conj_dir = ".../generator_functions/obs_ann_conj_files"
    # Path to log file directory
    log_dir = ".../generator_functions/log_files"

    # Set logging
    main_module_logger = set_logging(verbose, log_dir, module_name='main_module') 
    
    """
    PATHS TO PROCESSING MODULES
    """

    ann_obj_generator_name = "2_generate_ann_obj_dicts"
    ann_obj_generator_path = f"{learning_module_dir}/{ann_obj_generator_name}.py"

    prediction_generator_name = "3_prediction_generator"
    prediction_generator_path = f"{learning_module_dir}/{prediction_generator_name}.py"

    function_tracer_name = "4_func_tracer"
    function_tracer_path = f"{learning_module_dir}/{function_tracer_name}.py"

    question_generator_name = "5_obs_ques_generator"
    question_generator_path = f"{learning_module_dir}/{question_generator_name}.py"

    observation_generator_name = "6_observation_generator"
    observation_generator_path = f"{learning_module_dir}/{observation_generator_name}.py"

    cross_compute_observations_name = "7_x_compute_obs"
    cross_compute_observations_path = f"{learning_module_dir}/{cross_compute_observations_name}.py"

    evaluate_conjecture_quality_name = "8_evaluate_conjecture"
    evaluate_conjecture_quality_path = f"{learning_module_dir}/{evaluate_conjecture_quality_name}.py"

    c_question_generator_name = "9_conjecture_generator"
    c_question_generator_path = f"{learning_module_dir}/{c_question_generator_name}.py"

    """
    Import EWM learning modules
    """
    try:
        function_tracer_module = import_module_from_path(function_tracer_name, function_tracer_path)
        prediction_generator_module = import_module_from_path(prediction_generator_name, prediction_generator_path)
        question_generator_module = import_module_from_path(question_generator_name, question_generator_path)
        observation_generator_module = import_module_from_path(observation_generator_name, observation_generator_path)
        cross_compute_observations_module = import_module_from_path(cross_compute_observations_name, cross_compute_observations_path)
        ann_obj_generator_module = import_module_from_path(ann_obj_generator_name, ann_obj_generator_path)
        evaluate_conjecture_quality_module = import_module_from_path(evaluate_conjecture_quality_name, evaluate_conjecture_quality_path)
        conjecture_and_updater_module = import_module_from_path(c_question_generator_name, c_question_generator_path)    

    except (ImportError, FileNotFoundError) as e:
        print(f"Error: {e}")

    # Configure logging in 'function_tracer module' and 'question_generator_module' (so verbose bool and log_dir don't have to get passed through all the funcitons that call it)
    function_tracer_module.configure_logging_by_call(verbose, log_dir)
    question_generator_module.configure_logging_by_call(verbose, log_dir)

    """
    Run Learning Loop
    """
    # Initialize the learning loop
    learner = LearningLoop(
        # LLM API info 
        conj_api_key= TOGETHER_API_KEY, 
        conj_llm_name=conjecture_model_name, 
        obs_api_key=HUGGING_FACE_HUB_TOKEN, 
        obs_llm_name=observer_model_name,
        # Directories
        ewm_dir=ewm_dir, 
        obs_json_dir=obs_dir,
        ann_obj_dir=ann_obj_dir,
        conj_dir=conj_dir,
        log_dir=log_dir,
        # Seed knowledge
        seed_prompt=None, 
        seed_ewm_name=seed_ewm_name,
        # Logging setting
        verbose=verbose, 
        # Number of learning cycles
        num_cycles=NUM_CYCLES)

    """ 
    Uncomment if you want to continue 'learn' from an existing position in an EWM learning chain
       Also set seed_prompt and seed_ewm_name to None 
    """
    learner.last_ewm_name=last_ewm_name
    learner.next_ewm_name=next_ewm_name

    learner.learn()
    



    

