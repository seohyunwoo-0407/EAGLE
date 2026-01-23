from eagle.model.ea_model import EaModel
import torch
from transformers import AutoTokenizer


def main():
    # Load model
    model = EaModel.from_pretrained(
        base_model_path="ibm-granite/granite-3.1-1b-a400m-instruct",
        ea_model_path="wantsleep/granite-3.1-1b-a400m-EAGLE3",
        use_eagle3=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=-1
    )

    model.eval()

    # Load tokenizer directly to use chat_template
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-1b-a400m-instruct")

    your_message = "What is semiconductor?"
    
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

    input_ids = tokenizer([prompt], return_tensors="pt").input_ids
    input_ids = input_ids.cuda()

    output_ids = model.eagenerate(
        input_ids,
        temperature=0.0,
        max_new_tokens=512,
        top_k=40,
        top_p=0.9,
    )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Output:\n{output}")


if __name__ == "__main__":
    main()
