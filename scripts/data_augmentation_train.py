from response import get_batch_response
import json
import re
model = "meta.llama3-1-8b-instruct-v1:0"
def format_prompt(k,data):
    prompt = """
    You are given the following JSON data:

    {data}

    Your task is to generate {k} new XML examples that follow the same structure as the provided JSON. Each new example must have the same elements as the original JSON data . The new examples should reflect the underlying patterns of the original data while ensuring diversity. 

    For each new example:
    1. The structure of the XML should match the original JSON, with the same elements.
    2. The values should be different, but the format and types (e.g., numbers, strings, lists) should remain consistent with the original data.
    3. The new XML data should be valid and realistic for the context of the task.
    4. The ground_truth should be one of [FAVOR,NONE,AGAINST].

    Please output the generated data in the following XML format, where each element corresponds to a key in the original JSON:

    ```xml
    <data>
        <example_1>
            <text>Regulation of corporations has been subverted by corporations. States that incorporate corporations are not equipped to regulate corporations that are rich enough to influence elections, are rich enough to muster a legal team that can bankrupt the state. Money from corporations and their principals cannot be permitted in the political process if democracy is to survive.</text>
            <target>company</target>
            <ground_truth>AGAINST</ground_truth>
        </example_1>
        <example_2>
            <text>The whole media mess surrounding the royals is a consequence of the promotional fervor with which royal households (aka, public relations experts) developed stage-set performances for the public to devour. Prior to the Victorian era, those elaborate and lethally expensive weddings, coronations, and funerals - and the fairy tales that went along with them - just didn't exist.</text>
            <target>flag burning</target>
            <ground_truth>NONE</ground_truth>
        </example_2>
        <example_3>
            <text>Two of the main reasons people switch to cannabis from pharmaceuticals and other drugs such as alcohol: less side-effects and less withdrawal: ""Over 41% state that they use cannabis as a substitute for alcohol, 36.1% use cannabis as a substitute for illicit substances, and 67.8% use cannabis as a substitute for prescription drugs. The three main reasons cited for cannabis-related substitution are 'less withdrawal' (67.7%), 'fewer side-effects' (60.4%), and 'better symptom management' suggesting that many patients may have already identified cannabis as an effective and potentially safer adjunct or alternative to their prescription drug regimen."" [Lucas et al. Cannabis as a substitute for alcohol and other drugs: A dispensary-based survey of substitution effect in Canadian medical cannabis patients. Addiction Research & Theory. 2013]</text>
            <target>marijuana</target>
            <ground_truth>FAVOR</ground_truth>
        </example_3>
        ...
    </data>

    """
    return prompt.format(k=k,data=data)
def parse2(s:str):
    pattern = r"<text>(.*?)</text>\s*<target>(.*?)</target>\s*<ground_truth>(.*?)</ground_truth>"
    matches = re.findall(pattern, s)
    ans = []
    for match in matches:
        if match[2] in ["FAVOR","NONE","AGAINST"]:
            ans.append({
                "text":match[0],
                "target":match[1],
                "ground_truth":match[2]
            })
    return ans
def augmentation(data,k,model):
    print("Start to generate data the same as the data in train set")
    prompts = []
    for i in data:
        prompts.append(format_prompt(k,i))
    response = get_batch_response(model=model,prompts=prompts)
    parse_data = []
    for i in response:
        parse_data.extend(parse2(i))
    return parse_data
        
        
# path = "/home/ubuntu/wcc/now-task/data/train/strategy1_3.1_7_3_0.0_generate1/iter1_raw_response_parse.json"
# with open(path,"r") as f:
#     data = json.load(f)
# # for d in data:
# #     del d["target"]
# k = 5
# print(data[0])

# prompts = prompts[:10]

# # for i in response:
# #     print(i)

# parse_data = []
# for i in response:
#     parse_data.extend(parse2(i))
# print(len(parse_data))
# for i in parse_data:
#     print(i)