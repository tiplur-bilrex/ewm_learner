# ewm_learner

Demo video: https://x.com/tiplur_bilrex/status/1872500868417003929

This is a program that interacts with an LLM to iteratively update an 'Explanatory World Model' (EWM). The EWM is defined in a Python file; updating can begin with a seed file that already contains a defined world model or a seed prompt requesting the LLM to initialize an EWM. 

When the EWM is updated it is evaluated by composing all composable definitions within it (type-compatable objects), computing a list of predictions (defined by tuples of inputs, functions, and outputs), then evaluating each prediction against the knowledge within an LLM. Each prediction receives a boolean 'consistent observation' or 'inconsistent observation' evaluation from the LLM. 

This observation data is then used to guide further updates to the EWM. Updates are only accepted if the total number of consistent observations increases. Updates may also be permitted if the 'organization' of the EWM improves whithout decreasing the number of consistent observations. Organization is evaluated by a custom metric.



