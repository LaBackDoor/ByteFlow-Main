import os
import random

import numpy as np
import torch
from datasets import load_dataset, IterableDataset, interleave_datasets
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from config import ByT5Config
from model import ByT5ForConditionalGeneration
import torch.nn as nn

from src.tokenizer.hybrid_tokenizer import HybridByT5PCAPTokenizer
from src.tokenizer.pcap_tokenizer import PCAPTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# Constants
MAX_SEQ_LENGTH = 1024
NOISE_DENSITY = 0.15
MEAN_NOISE_SPAN_LENGTH = 20.0


def load_c4_dataset():
    c4_en = load_dataset("allenai/c4", "en", streaming=True, trust_remote_code=True)
    return c4_en["train"]

def process_c4_example(examples, tokenizer, max_length):
    texts = examples["text"]
    tokenized_output_dict = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )

    batched_input_ids_list = tokenized_output_dict["input_ids"]
    batched_attention_mask_list = tokenized_output_dict["attention_mask"]
    input_ids_tensor_for_pos = torch.tensor(batched_input_ids_list, dtype=torch.long)

    position_indices_tensor = tokenizer.get_2d_position_indices(input_ids_tensor_for_pos)
    batched_position_indices_list = position_indices_tensor.tolist()  # Convert tensor to List[List[List[int]]]

    return {
        "input_ids": batched_input_ids_list,
        "attention_mask": batched_attention_mask_list,
        "position_indices": batched_position_indices_list
    }


def create_pcap_dataset(pcap_dir, tokenizer, max_length):
    def pcap_iterator():
        if not os.path.exists(pcap_dir):
            print(f"Warning: PCAP directory {pcap_dir} does not exist.")
            return

        pcap_files = [os.path.join(pcap_dir, f) for f in os.listdir(pcap_dir) if f.endswith(".pcap")]
        if not pcap_files:
            print(f"Warning: No PCAP files found in {pcap_dir}")
            return

        random.shuffle(pcap_files)

        for i, pcap_path in enumerate(pcap_files):
            try:
                if hasattr(tokenizer, 'tokenize_text_with_pcap'):
                    input_ids = tokenizer.tokenize_text_with_pcap("", pcap_path)
                else:
                    pcap_tokenizer = PCAPTokenizer(vocab_size=280)
                    tokenized_flows = pcap_tokenizer.tokenize_pcap(pcap_path)

                    if tokenized_flows:
                        first_flow_id = list(tokenized_flows.keys())[0]
                        input_ids = tokenized_flows[first_flow_id]
                    else:
                        print(f"Warning: No flows found in {pcap_path}")
                        continue

                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]

                attention_mask = [1] * len(input_ids)

                padding_token = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
                padding_length = max_length - len(input_ids)
                if padding_length > 0:
                    input_ids = input_ids + [padding_token] * padding_length
                    attention_mask = attention_mask + [0] * padding_length

                input_ids_tensor_for_pos = torch.tensor([input_ids],
                                                        dtype=torch.long)
                position_indices_tensor = tokenizer.get_2d_position_indices(input_ids_tensor_for_pos)
                position_indices_list = position_indices_tensor.squeeze(0).tolist()

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_indices": position_indices_list
                }

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} PCAP files")

            except Exception as e:
                print(f"Error processing {pcap_path}: {e}")
                continue

    return IterableDataset.from_generator(pcap_iterator)


def create_span_corruption_from_tokens(input_ids, attention_mask, original_position_indices, mean_span_length,
                                       noise_density, tokenizer):
    valid_indices = [i for i, mask_val in enumerate(attention_mask) if mask_val == 1]
    if len(valid_indices) < 10:
        return None, None, None, None

    valid_input_ids = [input_ids[i] for i in valid_indices]
    valid_original_positions = [original_position_indices[i] for i in valid_indices]

    num_tokens_to_mask = max(1, int(len(valid_input_ids) * noise_density))

    num_spans = max(1, int(num_tokens_to_mask / mean_span_length))

    span_lengths = np.random.poisson(lam=mean_span_length, size=num_spans)
    span_lengths = [max(1, length) for length in span_lengths]

    if not span_lengths:
        return None, None, None, None

    total_masked = sum(span_lengths)
    if total_masked > num_tokens_to_mask:

        while total_masked > num_tokens_to_mask and len(span_lengths) > 1:
            span_lengths.pop()
            total_masked = sum(span_lengths)

    if min(span_lengths) > len(valid_input_ids):
        return None, None, None, None

    if not valid_input_ids or (min(span_lengths) > len(valid_input_ids)):
        return None, None, None, None

    min_span_len_val = 0 if not span_lengths else min(span_lengths)
    available_positions = list(range(len(valid_input_ids) - min_span_len_val + 1))
    if not available_positions or len(span_lengths) > len(available_positions):
        return None, None, None, None

    if len(available_positions) < len(span_lengths):
        if not available_positions:
            return None, None, None, None
        span_starts_indices = np.random.choice(len(available_positions), len(span_lengths), replace=True)
    else:
        span_starts_indices = np.random.choice(len(available_positions), len(span_lengths), replace=False)

    span_starts = sorted([available_positions[i] for i in span_starts_indices])

    spans = []
    current_pos_in_valid = 0
    for i, start_idx_in_valid in enumerate(span_starts):
        length = span_lengths[i]
        if start_idx_in_valid < current_pos_in_valid or start_idx_in_valid + length > len(valid_input_ids):
            continue
        spans.append((start_idx_in_valid, length))
        current_pos_in_valid = start_idx_in_valid + length

    sentinel_ids = list(range(tokenizer.vocab_size - 100, tokenizer.vocab_size))

    corrupted_input_ids = []
    target_ids = []
    corrupted_position_indices = []

    pos = 0
    sentinel_idx = 0

    for start, length in spans:
        corrupted_input_ids.extend(valid_input_ids[pos:start])
        corrupted_position_indices.extend(valid_original_positions[pos:start])

        if sentinel_idx < len(sentinel_ids):
            corrupted_input_ids.append(sentinel_ids[sentinel_idx])
            corrupted_position_indices.append(
                valid_original_positions[start])

            target_ids.append(sentinel_ids[sentinel_idx])
            target_ids.extend(valid_input_ids[start:start + length])

        pos = start + length
        sentinel_idx += 1
        if sentinel_idx >= len(sentinel_ids):
            break

    corrupted_input_ids.extend(valid_input_ids[pos:])
    corrupted_position_indices.extend(valid_original_positions[pos:])

    if not corrupted_input_ids:
        return None, None, None, None

    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 1
    if corrupted_input_ids[-1] != eos_token_id:
        corrupted_input_ids.append(eos_token_id)

        last_row_pos = corrupted_position_indices[-1][0] if corrupted_position_indices else -1
        last_col_pos = corrupted_position_indices[-1][1] if corrupted_position_indices else -1
        corrupted_position_indices.append([last_row_pos + 1, last_col_pos + 1])

    if not target_ids or target_ids[-1] != eos_token_id:
        target_ids.append(eos_token_id)

    new_attention_mask = [1] * len(corrupted_input_ids)

    return corrupted_input_ids, target_ids, new_attention_mask, corrupted_position_indices


class SpanCorruptionDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, mean_span_length=20, noise_density=0.15, seed=42):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.mean_span_length = mean_span_length
        self.noise_density = noise_density
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        current_seed = self.seed
        if worker_info is not None:
            current_seed += worker_info.id

        random.seed(current_seed)
        np.random.seed(current_seed % (2 ** 32))

        for example in self.dataset:
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            position_indices = example["position_indices"]

            if isinstance(input_ids, torch.Tensor): input_ids = input_ids.tolist()
            if isinstance(attention_mask, torch.Tensor): attention_mask = attention_mask.tolist()
            if isinstance(position_indices, torch.Tensor): position_indices = position_indices.tolist()


            corrupted_input_ids, target_ids, new_attention_mask, corrupted_position_indices = \
                create_span_corruption_from_tokens(
                    input_ids,
                    attention_mask,
                    position_indices,
                    self.mean_span_length,
                    self.noise_density,
                    self.tokenizer
                )

            if corrupted_input_ids is not None:
                yield {
                    "input_ids": corrupted_input_ids,
                    "target_ids": target_ids,
                    "attention_mask": new_attention_mask,
                    "position_indices": corrupted_position_indices
                }


def collate_fn(batch):
    batch = [b for b in batch if b is not None and b.get("input_ids") is not None]
    if not batch:
        return None

    max_input_len = max(len(x["input_ids"]) for x in batch)
    max_target_len = max(len(x["target_ids"]) for x in batch)

    input_ids_batch = []
    attention_mask_batch = []
    target_ids_batch = []
    position_indices_batch = []

    pos_padding_value = [0, 0]

    for example in batch:
        padded_input = example["input_ids"] + [0] * (max_input_len - len(example["input_ids"]))
        input_ids_batch.append(padded_input)

        padded_mask = example["attention_mask"] + [0] * (max_input_len - len(example["attention_mask"]))
        attention_mask_batch.append(padded_mask)

        padded_target = example["target_ids"] + [0] * (max_target_len - len(example["target_ids"]))
        target_ids_batch.append(padded_target)

        num_pos_to_pad = max_input_len - len(example["position_indices"])
        padded_pos = example["position_indices"] + [pos_padding_value] * num_pos_to_pad
        position_indices_batch.append(padded_pos)


    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        "target_ids": torch.tensor(target_ids_batch, dtype=torch.long),
        "position_indices": torch.tensor(position_indices_batch, dtype=torch.long)
    }


def get_interleaved_dataloader(tokenizer, batch_size=2, pcap_dir=None, text_prob=0.8, pcap_prob=0.2):
    print("Creating C4 text dataset (streaming)...")
    c4_dataset = load_c4_dataset()
    tokenized_c4 = c4_dataset.map(
        lambda x: process_c4_example(x, tokenizer, MAX_SEQ_LENGTH),
        batched=True
    )

    pcap_files_exist = os.path.exists(pcap_dir) and any(f.endswith(".pcap") for f in os.listdir(pcap_dir))

    if pcap_files_exist:
        print("Creating PCAP dataset (streaming)...")
        pcap_dataset = create_pcap_dataset(pcap_dir, tokenizer, MAX_SEQ_LENGTH)

        print("Interleaving C4 and PCAP datasets with probabilities...")

        interleaved_dataset = interleave_datasets(
            [tokenized_c4, pcap_dataset],
            probabilities=[text_prob, pcap_prob],
            seed=42,
            stopping_strategy="all_exhausted"
        )
    else:
        print("No PCAP files found or PCAP directory does not exist. Using only C4 text dataset.")
        interleaved_dataset = tokenized_c4

    print("Applying span corruption (streaming)...")

    span_corruption_dataset = SpanCorruptionDataset(
        dataset=interleaved_dataset,
        tokenizer=tokenizer,
        mean_span_length=MEAN_NOISE_SPAN_LENGTH,
        noise_density=NOISE_DENSITY,
        seed=42
    )

    print("Creating DataLoader for streaming...")

    dataloader = DataLoader(
        span_corruption_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )

    return dataloader


def train_byt5_model(model, tokenizer, batch_size=2, learning_rate=5e-5,
                     save_path="byt5_model", max_steps=1_000_000,
                     pcap_dir="../data/flows", lambda_entropy_loss=0.01):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss(ignore_index=0)

    print("Creating interleaved dataloader...")
    dataloader = get_interleaved_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        pcap_dir=pcap_dir,
        text_prob=0.8,
        pcap_prob=0.2
    )

    model.train()
    step = 0
    os.makedirs(save_path, exist_ok=True)

    with tqdm(total=max_steps) as pbar:
        epoch = 0
        last_loss_for_epoch_checkpoint = None

        while step < max_steps:
            epoch += 1
            print(f"\nStarting epoch {epoch}")

            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    print(f"Skipping None batch at epoch {epoch}, batch index {batch_idx}")
                    continue

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_ids = batch["target_ids"].to(device)
                encoder_position_indices = batch["position_indices"].to(device)

                decoder_input_ids = torch.zeros_like(target_ids)
                decoder_input_ids[:, 1:] = target_ids[:, :-1].clone()

                eos_token_id_val = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 1
                decoder_input_ids[:, 0] = eos_token_id_val

                current_batch_size, dec_seq_len = decoder_input_ids.shape
                dec_pos_row = torch.arange(dec_seq_len, device=device).unsqueeze(0).repeat(current_batch_size, 1)
                dec_pos_col = dec_pos_row.clone()
                decoder_position_indices = torch.stack((dec_pos_row, dec_pos_col), dim=-1)

                logits, total_entropy_loss = model(
                    encoder_input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    encoder_attention_mask=attention_mask,
                    decoder_attention_mask=(target_ids != 0).long(),
                    encoder_position_indices=encoder_position_indices,
                    decoder_position_indices=decoder_position_indices
                )

                reshaped_logits = logits.view(-1, logits.size(-1))
                reshaped_targets = target_ids.view(-1)

                main_loss = loss_fn(reshaped_logits, reshaped_targets)
                combined_loss = main_loss + lambda_entropy_loss * total_entropy_loss

                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

                last_loss_for_epoch_checkpoint = combined_loss

                step += 1
                pbar.update(1)
                pbar.set_description(
                    f"Epoch {epoch}, Step {step}/{max_steps}, Loss: {combined_loss.item():.4f} (Main: {main_loss.item():.4f}, Entropy: {total_entropy_loss.item():.4f})")

                if step % 1000 == 0:
                    test_text_input = "Hello, how are you?"
                    test_tokens_dict = tokenizer(test_text_input, return_tensors="pt", padding=True, truncation=True,
                                                 max_length=MAX_SEQ_LENGTH)
                    test_input_ids_gen = test_tokens_dict.input_ids.to(device)
                    test_attention_mask_gen = test_tokens_dict.attention_mask.to(device)
                    test_pos_indices_gen = tokenizer.get_2d_position_indices(test_input_ids_gen).to(device)

                    with torch.no_grad():
                        output_ids = model.generate(
                            test_input_ids_gen,
                            attention_mask=test_attention_mask_gen,
                            position_indices=test_pos_indices_gen,
                            max_length=50
                        )
                        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        print(f"\nText test generation: {output_text}")

                    if os.path.exists(pcap_dir):
                        try:
                            pcap_files = [f for f in os.listdir(pcap_dir) if f.endswith(".pcap")]
                            if pcap_files:
                                test_pcap_path = os.path.join(pcap_dir, random.choice(pcap_files))
                                pcap_input_ids_list = tokenizer.tokenize_text_with_pcap("", test_pcap_path)
                                if len(pcap_input_ids_list) > MAX_SEQ_LENGTH:
                                    pcap_input_ids_list = pcap_input_ids_list[:MAX_SEQ_LENGTH]
                                else:
                                    pcap_input_ids_list += [tokenizer.pad_token_id if hasattr(tokenizer,
                                                                                              'pad_token_id') else 0] * (
                                                                       MAX_SEQ_LENGTH - len(pcap_input_ids_list))

                                pcap_input_ids_tensor = torch.tensor([pcap_input_ids_list], dtype=torch.long).to(device)
                                pcap_attention_mask_tensor = (pcap_input_ids_tensor != (
                                    tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0)).long().to(
                                    device)
                                pcap_pos_indices_tensor = tokenizer.get_2d_position_indices(pcap_input_ids_tensor).to(
                                    device)

                                with torch.no_grad():
                                    output_ids = model.generate(
                                        pcap_input_ids_tensor,
                                        attention_mask=pcap_attention_mask_tensor,
                                        position_indices=pcap_pos_indices_tensor,
                                        max_length=50
                                    )
                                    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                                    print(f"\nPCAP test generation: {output_text}")
                        except Exception as e:
                            print(f"Error during PCAP testing: {e}")

                if step >= max_steps: break
            if step >= max_steps: break

            checkpoint_path = f"{save_path}/checkpoint_epoch_{epoch}.pt"
            print(f"\nSaving epoch checkpoint to {checkpoint_path}")
            torch.save({
                'step': step,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': last_loss_for_epoch_checkpoint.item()
            }, checkpoint_path)

    print("\nTraining completed! Saving final model.")
    torch.save({
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{save_path}/final_model.pt")

    return model


def create_model_with_vocab_size(vocab_size):

    config = ByT5Config()
    config.vocab_size = vocab_size

    model = ByT5ForConditionalGeneration(config)

    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    model.apply(_init_weights)

    return model


def main():
    pcap_dir = "../../data/flows"

    print("Initializing HybridByT5PCAPTokenizer...")
    tokenizer = HybridByT5PCAPTokenizer(pcap_vocab_size=280)

    tokenizer_vocab_size = len(tokenizer)
    print(f"Tokenizer vocabulary size: {tokenizer_vocab_size}")

    print("Creating ByT5-Small model with adjusted vocabulary size...")
    model = create_model_with_vocab_size(tokenizer_vocab_size)

    print("Starting training...")
    train_byt5_model(
        model,
        tokenizer,
        batch_size=2,
        learning_rate=1e-4,
        save_path="byt5_hybrid_model_checkpoints",
        pcap_dir=pcap_dir,
        lambda_entropy_loss=0.01

    )
    print("Training complete!")


if __name__ == "__main__":
    main()