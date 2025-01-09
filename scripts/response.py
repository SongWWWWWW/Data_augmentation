from utils import get_model_batch_response
import math

def get_batch_response(model:str, prompts:list, batch = 30,temperature=0, max_token=8192):
    if "llama" in model:
        # meta.llama3-1-8b-instruct-v1:0
        response = []
        for i in range(math.ceil(len(prompts)/batch)):
            # print(i)
            temp_prompt = prompts[i*batch:(i+1)*batch]
            try: 
                r = get_model_batch_response(prompts=temp_prompt,model=model,temperature=temperature,max_new_tokens=max_token)
                response = response + r
            except Exception as e:
                print(e)
        
        return response
            
            
if __name__ == "__main__":
    
    prompts = []
    for i in range(31):
        prompts.append("hello")
    print(len(prompts))
    response = get_batch_response("meta.llama3-1-8b-instruct-v1:0", prompts)
    print(len(response))

    print(response)