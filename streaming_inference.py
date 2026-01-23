from eagle.model.ea_model import EaModel
import torch
from transformers import AutoTokenizer


def main():
    # Load model
    model = EaModel.from_pretrained(
        base_model_path="ibm-granite/granite-3.1-1b-a400m-instruct",
        ea_model_path="wantsleep/granite-3.1-1b-a400m-EAGLE3-test24",
        use_eagle3=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=-1
    )

    model.eval()

    # Load tokenizer directly to use chat_template
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-1b-a400m-instruct")

    your_message = "Explain what is KOREA"
    
    # Build prompt: 사용자 질문만 보내도록 구성
    messages = [
        {"role": "user", "content": your_message},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    print(f"Prompt:\n{prompt}\n")
    print("-" * 50)
    print("Output (streaming):\n")

    input_ids = tokenizer([prompt], return_tensors="pt").input_ids
    input_ids = input_ids.cuda()

    # Streaming mode: use ea_generate which yields tokens incrementally
    input_len = input_ids.shape[1]
    prev_len = input_len
    full_output = ""
    
    for output_ids in model.ea_generate(
        input_ids,
        temperature=0.0,
        max_new_tokens=512,
        top_k=40,
        top_p=0.9,
    ):
        # Decode only the newly generated tokens since last iteration
        current_len = output_ids.shape[1]
        if current_len > prev_len:
            new_tokens = output_ids[0, prev_len:current_len]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            # Print only the newly generated part
            if new_text:
                print(new_text, end="", flush=True)
                full_output += new_text
            prev_len = current_len
    
    print("\n" + "-" * 50)
    print(f"\nFull output:\n{full_output}")


if __name__ == "__main__":
    main()
