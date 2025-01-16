from response import get_batch_response
import json
import re
import os
from prompt import get_task_description, get_task_name 
model = "meta.llama3-1-8b-instruct-v1:0"
def get_strategy(strategy_name):
    dt = {}
    dt["strategy1"] = strategy_1
    
    return dt[strategy_name]

def strategy_1(args):
    def transform_label(d):
        transform_dict = {
        0: 'FAVOR',
        1: 'NONE',
        2: 'AGAINST'
        }
        d_ = {
            "input_text": d["input_text"],
            "target": d["target"],
            "predicted_label": transform_dict[d["predicted_label"]],
            "true_label": transform_dict[d["true_label"]]
        }
        return d_
    def format_prompts(data,task_name,task_description,spurious_num):
        prompt = """
        I am training a model using RoBERTa + MLP on a task named {task_name}. The task involves {task_description}. 
        Your task is to identify potential spurious patterns that the model might have learned based on its responses.

        I will present you with an instance where the model provided incorrect responses.

        Please provide {spurious_num} assumptions of spurious patterns that may have caused the incorrect response. For each spurious pattern, also provide a detailed strategy for generating corresponding training data to test or mitigate the identified spurious pattern.
        
        The “generate strategy” should induce the model to increase the diversity of the generated data as much as possible.

        A spurious pattern refers to a misleading or non-causal feature relationship that the model learns during training, such as misunderstandings of certain phrases, sentiment words, or entity relations. Be specific in the patterns, such as what words or what relations, and avoid general descriptions.

        The incorrect instance is as follows:
        {data}
        Please ensure the output is formatted as follows:

        ```xml
        <SpuriousPatterns>
            <Spurious_1>
                <Pattern>Description of spurious pattern 1 (specific and detailed)</Pattern>
                <GenerateStrategy>Detailed strategy to generate training data for spurious pattern 1</GenerateStrategy>
            </Spurious_1>
            <Spurious_2>
                <Pattern>Description of spurious pattern 2 (specific and detailed)</Pattern>
                <GenerateStrategy>Detailed strategy to generate training data for spurious pattern 2</GenerateStrategy>
            </Spurious_2>
            ...
        </SpuriousPatterns>

        """
    
        # task_name = "stance detection"
        # task_description = "Stance detection aims to identify the authors' attitudes or positions [FAVOR, NONE, AGAINST] towards a specific target such as an entity, a topic."
        # spurious_num = 3
        data = transform_label(data)
        return prompt.format(task_name=task_name,task_description=task_description,data=data,spurious_num=spurious_num)
    def parse(s:str):
        pattern_regex = re.compile(r"<Pattern>(.*?)</Pattern>\s*<GenerateStrategy>(.*?)</GenerateStrategy>",re.DOTALL)

        # 匹配内容
        matches = pattern_regex.findall(s)
        ans = []
        # 提取结果
        for i, (pattern, generate_strategy) in enumerate(matches, 1):
            ans.append({
                "Pattern":pattern.strip(),
                "GenerateStrategy": generate_strategy.strip()
            })
        return ans
    def parse2(s:str):
        pattern = r"<text>(.*?)</text>\s*<target>(.*?)</target>\s*<ground_truth>(.*?)</ground_truth>"
        matches = re.findall(pattern, s)
        ans = []
        for match in matches:
            ans.append({
                "text":match[0],
                "target":match[1],
                "ground_truth":match[2]
            })
        return ans
    def format_output_prompt(spurious,generate_strategy,task_name,task_description,generate_num):
        prompt = """
        
        I am training a model using RoBERTa + MLP on a task named {task_name}. The task involves {task_description}. 
        Your task is to generate diverse and contextually appropriate training data based on the provided spurious pattern and generation strategy.

        The spurious pattern and corresponding generation strategy are as follows:
        <Spurious>
            <Pattern>{spurious}</Pattern>
            <GenerateStrategy>{generate_strategy}</GenerateStrategy>
        </Spurious>

        Based on the provided spurious pattern and generation strategy, generate {generate_num} verification data points. 
        Each generated data point should align with the spurious pattern and adhere to the specified generation strategy.

        ### Output Format:
        Each verification data point should be structured as follows:
        - <verification_i> for each data point, where `i` is the sequential number of the verification set.

        Each <verification_i> should contain the following:
        1. <text>: A multi-sentence passage (at least 100 words) containing the spurious pattern within a suitable context. 
        The passage should demonstrate the identified spurious pattern while maintaining coherence and diversity.
        2. <target>: An entity or phrase from the text that is the focus of the classification task.
        3. <ground_truth>: The true label for the classification task, ensuring logical consistency with the provided text, you can just use one of ["FAVOR","AGAINST","NONE"].

        Ensure that the generated data points are diverse, use various speaking styles, and include different entities and contexts to avoid overfitting during model fine-tuning.

        ### Example Output:
        <verification_1>
            <text>"She always goes the extra mile to assist her colleagues and solve problems effectively."</text>
            <target>"colleagues"</target>
            <ground_truth>FAVOR</ground_truth>
        </verification_1>
        <verification_2>
            <text>"The weather always changes unpredictably in this region, making planning difficult."</text>
            <target>"weather"</target>
            <ground_truth>NONE</ground_truth>
        </verification_2>
        <verification_3>
            <text>"He always delays submitting his reports, which causes unnecessary delays in the project."</text>
            <target>"reports"</target>
            <ground_truth>AGAINST</ground_truth>
        </verification_3>

        ### Now, generate the output based on the following inputs:
        <Spurious>
            <Pattern>{spurious}</Pattern>
            <GenerateStrategy>{generate_strategy}</GenerateStrategy>
        </Spurious>

        """
        # task_name = "stance detection"
        # task_description = "Stance detection aims to identify the authors' attitudes or positions [FAVOR, NONE, AGAINST] towards a specific target such as an entity, a topic."
        # generate_num = 10
        return prompt.format(task_name=task_name,task_description=task_description,generate_num=generate_num,spurious=spurious,generate_strategy=generate_strategy)

    with open(args.dev_wrong_path,"r") as f:
        data = json.load(f)
    raw_response_path = args.raw_answer_save_path
    raw_response_spurious_patterns = raw_response_path.rstrip(".json") + "_spurious_patterns.json"
    raw_response_generate_instance = raw_response_path.rstrip(".json") + "_generate_instances.json"
    task_name = get_task_name(args.task)
    task_description = get_task_description(args.task_description)
    if not os.path.exists(raw_response_spurious_patterns):
        prompts = []
        for i in range(len(data)):
            prompts.append(format_prompts(data=data[i],task_name=task_name,task_description=task_description,spurious_num=args.spurious_num))            
        answer = get_batch_response(model,prompts)
        print(f"Generate spurious patterns completely, length = {len(answer)}")
    else:
        with open(raw_response_spurious_patterns,"r") as f:
            answer = json.load(f)
        print(f"Spurious pattern is existed. Path: {raw_response_spurious_patterns}")

    patterns = []
    for i in answer:
        patterns.extend(parse(i))

    if not os.path.exists(raw_response_generate_instance):
        prompt2 = []
        for i in range(len(patterns)):
            prompt2.append(format_output_prompt(patterns[i]["Pattern"],patterns[i]["GenerateStrategy"],task_name=task_name,task_description=task_description,generate_num=args.generate_num))
            
        answer2 = get_batch_response(model,prompt2)
        print(f"Generate instances completely, length = {len(answer2)}")
        
    else:
        with open(raw_response_generate_instance,"r") as f:
            answer2 = json.load(f)
        print(f"Instances are existed. Path: {raw_response_generate_instance}")

    
    with open(raw_response_spurious_patterns,"w") as f:
        json.dump(answer,f,indent=4)
        
    with open(raw_response_generate_instance,"w") as f:
        json.dump(answer2,f,indent=4)
        
    parse_output = []
    for i in answer2:
        parse_output.extend(parse2(i))
    # print(parse_output)
    return parse_output