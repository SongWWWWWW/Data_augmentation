def get_prompt(prompt_name:str):
    prompt = {}
    prompt["prompt1"] = """
    I am training a model using RoBERTa + MLP on a task named {task_name}. The task involves {task_description}. 
    Your task is to identify potential spurious patterns that the model might have learned based on its responses.

    I will present you with an instance where the model provided incorrect responses. 

    Please provide {spurious_num} assumptions of spurious patterns that may have caused the incorrect response. 
    Each assumption should be followed by {generate_num} verification data points to determine whether the model consistently makes mistakes 
    due to such a spurious pattern. The verification data should align with the identified spurious patterns. 
    Having the same pattern does not mean copying the original text and target verbatim; instead, 
    it should reflect the same pattern at a higher level of abstraction and  "text" and "target" in the generated should be diverse with various contents and different speaking way, and include spurious patterns.

    A spurious pattern refers to a misleading or non-causal feature relationship that the model learns during training, 
    such as misunderstandings of certain phrases, sentiment words, or entity relations.Be specific in the patterns, such as what words or what relations, but not a general description.

    Format your evaluation instances using XML tags. Each <Spurious_i> tag should include:

    An assumption of the spurious pattern that the model may have learned.
    {generate_num} verification data points, each enclosed in <verification_i> from <verification_1> to <verification_10>, where i is the sequential number of the verification set.
    Each <verification_i> should contain the following:

       1. <text>: A multi-sentence passage containing the spurious pattern. Generate the sample as long as the incorrect instance in length, at least 100 words in each data with a suitable context.
       2. <target>: An entity mentioned in the text.
       3. <ground_truth>: The true label for the classification task.
    Ensure that the ground truth of the generated data is ascertainably correct. If the correctness of the given instance cannot be determined, leave the field blank.
    The incorrect instance is as follows:
    {data}
    Please output all content completely without omitting or summarizing.
    Confirm that the generated data should be diverse to avoid overfitting of the smaller model.
    """
    
    
    return prompt[prompt_name]

def get_task_name(task_name:str):
    task = {}
    task["task1"] = "stance detection"
    return task[task_name]
    
def get_task_description(description_name:str):
    description = {}
    description["description1"] = "Stance detection aims to identify the authors' attitudes or positions [FAVOR, NONE, AGAINST] towards a specific target such as an entity, a topic."
    
    return description[description_name]