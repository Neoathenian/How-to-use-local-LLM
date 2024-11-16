import os
import torch
import gc
import platform

def Call_LLM_linux(messages,model,tokenizer,max_new_tokens=1280):
    """This is the base function to call the LLM model. It takes a list of messages, the model and tokenizer and returns the answer (as a string).
        The list of messages has to be in the format [{"from": "human", "value": question}, {"from": "assistant", "value": answer}, ...]
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    #text_streamer = TextStreamer(tokenizer)
    #out = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=1280, use_cache=True)
    #I´m avoiding the text streamer for now, as it is verbose (it prints the generated text)
    with torch.no_grad():
        out = model.generate(input_ids=inputs, 
                            max_new_tokens=max_new_tokens,
                            use_cache=True,
                            attention_mask=(inputs != tokenizer.pad_token_id).long().to("cuda"))
    #torch.cuda.empty_cache()  # Clear GPU memory after inference
    result = tokenizer.batch_decode(out)[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()

    return result

    #return tokenizer.batch_decode(out)[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()


#from transformers import TextStreamer
def Get_answer_linux(question,model,tokenizer,system_prompt,max_new_tokens=1280):
    """This function takes a question and returns the answer from the model. Input str question, output str."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return Call_LLM_linux(messages,model,tokenizer,max_new_tokens=max_new_tokens)

def Get_answer_windows(question,model,system_prompt,max_new_tokens=1280):
    """This function takes a question and returns the answer from the model. Input str question, output str."""
    LLM_connection.model.pipeline._forward_params["max_new_tokens"]=max_new_tokens

    return model.invoke(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>
                                <|start_header_id|>user<|end_header_id|>{question}<|eot_id|>
                                <|start_header_id|>assistant<|end_header_id|>\n\n""")

    
def Get_answer_OpenAI(question,system_prompt,client,model="gpt-4o-mini",max_new_tokens=1280):
    return client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
      ],
      stream=False,
      temperature=0.00000000001,
      max_completion_tokens=max_new_tokens
    ).choices[0].message.content



def process_batch_windows(questions, model, system_prompt,max_new_tokens=1280):
    LLM_connection.model.pipeline._forward_params["max_new_tokens"]=max_new_tokens
    """Efficiently process a batch of questions."""
    prompts = [
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>{question}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        for question in questions
    ]
    output=model.pipeline(prompts)
    return [o[0]['generated_text'] for o in output]

class LLM_Connection():
    """This class is to abstract away the api connection allowing for local models.
        The idea is that the user modifies the LLM_conection.Get_answer function to connect to the model of choice (api or local model)."""
    def __init__(self,model=None,tokenizer=None,api=None):
        self.model=model
        self.tokenizer=tokenizer
        self.os_name=platform.system()
        self.api=api
        self.system_prompt="You are an assistant that has to try to answer questions the questions as faithfully as possible."

    def Get_answer(self,prompt,max_new_tokens=1280):
        """Returns the answer from the model given a prompt. Input str prompt, output str."""
        if self.model is not None:
            if self.os_name=="Windows":
                return Get_answer_windows(prompt,self.model,self.system_prompt,max_new_tokens=max_new_tokens)
            elif self.os_name=="Linux":
                return Get_answer_linux(prompt,self.model,self.tokenizer,self.system_prompt,max_new_tokens=max_new_tokens)
            else:
                print("OS not supported")
                return None
        else:
            #we use an api
            if self.api.lower() == "openai":
                return Get_answer_OpenAI(prompt,self.system_prompt,self.client,max_new_tokens=max_new_tokens)
    
    def Get_answer_batch(self,questions):
        """This function takes a list of questions and returns a list of answers. Input list of str questions, output list of str answers."""
        if self.model is not None:
            if self.os_name=="Windows":
                return process_batch_windows(questions, self.model, self.system_prompt)
            elif self.os_name=="Linux":
                return [Get_answer_linux(question,self.model,self.tokenizer,self.system_prompt) for question in questions]
            else:
                print("OS not supported")
                return None
        else:
            #we use an api
            if self.api.lower() == "openai":
                return [Get_answer_OpenAI(question,self.system_prompt,self.client) for question in questions]

    def set_model(self,model,tokenizer=None):
        self.model=model
        self.tokenizer=tokenizer
    
    def load_model_from_path(self,path):
        """This function loads a model from a path. Input str path, output None.
            We do the imports here because some libraries are os dependent. -.-
            
            https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct Download it from here
            Put the files in a folder called Llama-3.1-8B-Instruct, and put path="Llama-3.1-8B-Instruct" to load it
            """
        if self.os_name=="Windows":
            from transformers import AutoTokenizer
            from langchain_huggingface import HuggingFacePipeline
            import transformers
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.tokenizer.pad_token_id=50256 #I don´t know why this is necessary, but it is

            pipeline = transformers.pipeline(
                    "text-generation",
                    model=path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    return_full_text = False,
                    # max_length=max_seq_length,
                    truncation=True,
                    # top_p=0.9,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            self.model = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0.1})
            
        elif self.os_name=="Linux":
            from unsloth import FastLanguageModel
            from unsloth.chat_templates import get_chat_template

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,  # Ensure this is the correct path
                load_in_8bit=False,
                dtype=None,
                local_files_only=True,
                trust_remote_code=False,
                force_download=False
            )
            self.tokenizer = get_chat_template(
                self.tokenizer,
                mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
                chat_template="chatml",
            )
            self.model = FastLanguageModel.for_inference(self.model)
            
            torch.cuda.empty_cache() #I´m having issues with memory, like it´s storing stuff, idk, but this helps
    
    
    def unload_model(self):
        """Unload the model and free up resources."""
        # Delete model and tokenizer references
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache if on GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Force garbage collection
        gc.collect()

        # Clear api
        self.client = None
        self.api = None
        os.environ.pop('OPENAI_API_KEY', None)


    def load_api(self,api="OpenAI",dot_env_path=".env"):
        """Load the api connection."""
        if api.lower()=="openai":
            from openai import OpenAI
            from dotenv import load_dotenv
            #You need to put your key in a .env file, putting OPENAI_API_KEY={your key}
            load_dotenv(dot_env_path,override=True)

            self.client = OpenAI()
            self.api=api
        else:
            print("API not supported")


#This is the global LLM class that has to be modified to connect to the model of choice
LLM_connection=LLM_Connection()