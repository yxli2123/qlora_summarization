import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
import utils
import argparse
import os
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402

HF_TOKEN = "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"


def main(reduced_rank, num_iter, num_bits):
    accelerator = Accelerator()
    hf_token = "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"
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                 device_map='auto',
                                                 torch_dtype=torch.float,
                                                 trust_remote_code=True)

    # Quantize
    allow_name = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h',
                  'q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
                  'out_proj', 'fc1', 'fc2']
    block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']
    utils.substitute_layer_weights_iter_quant(model,
                                              allow_name=allow_name,
                                              block_name=block_name,
                                              reduced_rank=reduced_rank,
                                              num_bits=num_bits,
                                              num_iter=num_iter,
                                              load=False,
                                              enable_lora=True)

    save_dir = os.path.join(args.model_zoo_dir, args.model_name.split('/')[-1], f"bit{num_bits}", f"iter{num_iter}", f"rank{reduced_rank}")

    model.save_pretrained(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_zoo_dir', type=str, default='/home/yli3551/yixiao_model_zoo')
    parser.add_argument('--model_name', type=str, default='facebook/bart-large',
                        help='tiiuae/falcon-7b, meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-7b-chat-hf, facebook/bart-large')
    parser.add_argument('--num_bits', type=int, default=2)
    parser.add_argument('--reduced_rank', type=int, default=8)
    parser.add_argument('--num_iter', type=int, default=5)

    args = parser.parse_args()
    main(args.reduced_rank, args.num_iter, args.num_bits)


