import os
import sys

# flake8: noqa: E402
# ruff: noqa: E402
# pylint: disable=wrong-import-position

import datetime
import click
import logging

logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
import transformers

transformers.logging.set_verbosity_error()

import pandas as pd
from training import (
    parse_verbose_results,
    lookbacks_first_counterfactual_template,
    ppkn_simpler_counterfactual_template,
    ppkn_simpler_counterfactual_template_keep_payload_change_key,
    ppkn_simpler_counterfactual_template_split_key_loc_maybe,
    my_ppkn_counterfactual_template_made_up_key,
)
from training import (
    ppkn_simpler_counterfactual_template_split_key_loc_new_payload,
    ppkn_simpler_counterfactual_template_split_key_loc_change_up_prev_index,
)
from training import (
    sample_answerable_question_template,
    train_experiment,
    evaluate_featurizer,
    get_counterfactual_datasets,
    sample_pkn_question_template,
    ppkn_simpler_counterfactual_template_split_key_loc,
    my_ppkn_counterfactual_template,
)
from grammar.task_to_causal_model import (
    multi_order_multi_schema_task_to_lookbacks_generic_causal_model,
    multi_order_multi_schema_task_to_lookbacks_keyload_causal_model,
)
from grammar.schemas import (
    SCHEMA_FILLING_LIQUIDS,
    SCHEMA_COLORED_SHAPES,
    SCHEMA_PEOPLE_AND_OBJECTS,
    SCHEMA_PROGRAMMING_PEOPLE_DICT,
    SCHEMA_GEOMETRY,
    SCHEMA_MUSIC_PERFORMANCE,
    SCHEMA_NUMBERED_CONTAINERS,
    SCHEMA_ANIMAL_MOVEMENTS,
    SCHEMA_LAB_EXPERIMENTS,
    SCHEMA_CHEMISTRY_EXPERIMENTS,
    SCHEMA_TRANSPORTATION,
    SCHEMA_SPORTS_EVENTS,
    SCHEMA_SPACE_OBSERVATIONS,
    SCHEMA_BOXES,
)
from neural.pipeline import LMPipeline
from experiments.residual_stream_experiment import PatchResidualStream
from experiments.aggregate_experiments import residual_stream_baselines
from causal.causal_model import CounterfactualDataset

# import inflect
from experiments.filter_experiment import FilterExperiment
from neural.LM_units import TokenPosition, get_last_token_index

import re
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm as _tqdm
from typing import DefaultDict


from datetime import datetime
import torch

# HF_TOKEN = "..." # HF
# from huggingface_hub import login
# login(token=HF_TOKEN)

POSITIONAL_SIGNAL = "positional"
KEYLOAD_SIGNAL = "keyload"
SIGNALS = [POSITIONAL_SIGNAL, KEYLOAD_SIGNAL]

schemas = [
    SCHEMA_FILLING_LIQUIDS,
    SCHEMA_MUSIC_PERFORMANCE,
    SCHEMA_PEOPLE_AND_OBJECTS,
    SCHEMA_NUMBERED_CONTAINERS,
    SCHEMA_COLORED_SHAPES,
    SCHEMA_PROGRAMMING_PEOPLE_DICT,
    SCHEMA_GEOMETRY,
    SCHEMA_ANIMAL_MOVEMENTS,
    SCHEMA_LAB_EXPERIMENTS,
    SCHEMA_CHEMISTRY_EXPERIMENTS,
    SCHEMA_TRANSPORTATION,
    SCHEMA_SPORTS_EVENTS,
    SCHEMA_SPACE_OBSERVATIONS,
    SCHEMA_BOXES,
]


def get_end_str(model_id):
    end_str = (
        "Answer:\nmodel\n"
        if "gemma" in model_id
        else (
            "Answer:assistant\n\n"
            if "llama" in model_id.lower()
            else "Answer:\nassistant\n" if "qwen" in model_id.lower() else "Oh no!"
        )
    )
    assert end_str != "Oh no!", f"Model {model_id} not supported"
    return end_str


def try_schema_checker(neural, causal, schema):
    try:
        return schema.checker(neural, causal)
    except:
        return False


def format_prompt(tokenizer, prompt, dont=False) -> str:
    if dont:
        return prompt

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )[5:]


from contextlib import contextmanager
from functools import partial
import torch

# ---- tiny helpers ----


def _get_layer_module(model, layer_idx: int):
    # LLaMA / Mistral / Gemma style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    # GPT-NeoX style
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers[layer_idx]
    # GPT-2 / OPT style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise ValueError("Unsupported model architecture: can't find layers container.")


def _num_layers(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return len(model.gpt_neox.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError("Unsupported model architecture: can't count layers.")


def _to_abs_positions(seq_len: int, positions):
    out = []
    for p in positions:
        out.append(seq_len + p if p < 0 else p)
    return out


# ---- main context manager ----


@contextmanager
def run_with_cf_hf(
    model,
    tokenizer,
    normal_str,
    cf_str,
    layer_idx=18,
    patch_pos="resid_pre",  # kept for parity; only 'resid_pre' is supported
    token_positions=None,
    device=None,
    alpha=1.0,
):
    """
    Patch the residual stream entering block `layer_idx` at `token_positions`
    with counterfactual activations taken from `cf_str`.

    Equivalent to TL's: patch 'blocks.{layer_idx}.hook_resid_pre'.
    """
    assert patch_pos == "resid_pre", "Only resid_pre is supported in this HF version."
    if token_positions is None:
        token_positions = [-1]

    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    # --- tokenize both prompts ---
    enc_cf = tokenizer(cf_str, return_tensors="pt").to(device)
    enc_norm = tokenizer(normal_str, return_tensors="pt").to(device)

    # --- forward on CF to cache resid_pre for the chosen layer ---
    # In HF, outputs.hidden_states is a tuple of length num_layers+1:
    # hidden_states[0] is embeddings output (resid_pre of layer 0),
    # hidden_states[i] is resid_pre of layer i (input to block i).
    with torch.no_grad():
        out_cf = model(**enc_cf, output_hidden_states=True, return_dict=True)
    hs_cf_all = out_cf.hidden_states
    nl = _num_layers(model)
    if not (0 <= layer_idx < nl):
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {nl-1}]")

    resid_pre_cf = hs_cf_all[layer_idx]  # shape: [B=1, T_cf, D]
    T_cf = resid_pre_cf.shape[1]

    # compute absolute indices for CF and Normal separately (support negatives)
    pos_cf = _to_abs_positions(T_cf, token_positions)
    # we will slice CF once and then reuse in the hook (shape: [len(P), D])
    patchey = resid_pre_cf[0, pos_cf, :].detach() * alpha  # [P, D]

    # Prepare normal prompt indexing
    T_norm = enc_norm["input_ids"].shape[1]
    pos_norm = _to_abs_positions(T_norm, token_positions)

    # --- hook: forward_pre on the chosen layer to modify its input hidden_states ---
    target_layer = _get_layer_module(model, layer_idx)

    def pre_hook(module, inputs):
        """
        For LLaMA/Mistral/GPT2/NeoX forward signatures, the first positional arg
        is the hidden_states entering the block (= resid_pre).
        """
        hidden_states = inputs[0]  # [B, T, D]
        if hidden_states.dim() != 3:
            return  # safety
        if hidden_states.size(1) < max(pos_norm) + 1:
            return  # sequence shorter than indices; do nothing

        # Only patch batch 0 to mirror the TL snippet
        # (extend as needed if you batch)
        hs = hidden_states.clone()
        # hs[0, pos_norm, :] = patchey.to(hs.dtype)  # broadcast [P, D] into those positions
        hs[0, pos_norm, :] = patchey.to(device=hs.device, dtype=hs.dtype)
        # rebuild args tuple
        new_inputs = (hs,) + tuple(inputs[1:])
        return new_inputs

    handle = target_layer.register_forward_pre_hook(pre_hook, with_kwargs=False)

    try:
        yield  # inside the with-block, call model on 'normal_str' or more inputs
    finally:
        handle.remove()
        if was_training:
            model.train()


def to_str_tokens(tokenizer, prompt):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    return [x.replace("▁", " ").replace("Ġ", " ") for x in tokenizer.convert_ids_to_tokens(tokens)]


def to_single_token(tokenizer, text) -> int | None:
    tokens = tokenizer.tokenize(text)
    assert len(tokens) == 1, f"Expected 1 token, got {len(tokens)}.\nText: {text}"
    return tokenizer.convert_tokens_to_ids(tokens[0])


def get_dist(
    model,
    tokenizer,
    model_name,
    train_ds,
    schema,
    num_instances,
    num_samples,
    layer,
    cat_to_query,
    messiness,
    generate=True,
    block_attn=False,
    num_fillers=0,
):
    results = {
        "normal": [],
        "cf": [],
        "source_pos": [],
        "positional_index": [],
        "keyload_index": [],
        "payload_index": [],
        "layer": [],
        "prediction": [],
        "positional_prediction": [],
        "payload_prediction": [],
        "keyload_prediction": [],
        "no_effect_prediction": [],
        "patch_effect": [],
        "dist": [],
        "distance": [],
        "generated": [],
    }
    train = train_ds[schema.name][schema.name]

    token_positions = [-1]
    end_str = get_end_str(model_name)

    for cur_index in _tqdm(range(num_samples)):
        prompt = format_prompt(tokenizer, train[cur_index]["input"]["raw_input"])
        cf_prompt = format_prompt(tokenizer, train[cur_index]["counterfactual_inputs"][0]["raw_input"])
        prompt_str_tokenized = to_str_tokens(tokenizer, prompt)
        metadata = train[cur_index]["input"]["metadata"]

        answer_indices = []
        keyload_index = None
        payload_index = None
        for i, token in enumerate(prompt_str_tokenized):
            if "qwen" in model_name.lower() and i < 10:
                continue

            if schema.matchers[cat_to_query](token):
                answer_indices.append(i)

                if prompt_str_tokenized[i].lower().strip() in metadata["keyload"].lower().strip():
                    keyload_index = len(answer_indices) - 1

                if prompt_str_tokenized[i].lower().strip() in metadata["payload"].lower().strip():
                    payload_index = len(answer_indices) - 1

        assert (
            len(answer_indices) == num_instances
        ), f"Expected {num_instances} answer indices, got {len(answer_indices)}.\nPrompt_str_tokenized: {prompt_str_tokenized}.\n{[prompt_str_tokenized[i] for i in answer_indices]}."
        assert (
            keyload_index is not None
        ), f"Keyload [{metadata['keyload']}] index is None. Prompt_str_tokenized: {prompt_str_tokenized}.\n{[prompt_str_tokenized[i] for i in answer_indices]}."

        if messiness != 4:
            assert (
                payload_index is not None
            ), f"Payload [{metadata['payload']}] index is None. Prompt_str_tokenized: {prompt_str_tokenized}.\n{[prompt_str_tokenized[i] for i in answer_indices]}."
        else:
            payload_index = -1

        pos_index = metadata["dst_index"]

        with run_with_cf_hf(
            model, tokenizer, prompt, cf_prompt, layer_idx=layer, token_positions=token_positions, alpha=1
        ):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            logits = model(input_ids).logits
            values = logits[0, -1, [to_single_token(tokenizer, prompt_str_tokenized[i]) for i in answer_indices]]

            pos_pred = values.argmax().item()

            if generate:
                pred_ids = model.generate(input_ids, max_new_tokens=schema.max_new_tokens, do_sample=False)
                pred = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                pred = pred[pred.find(end_str) + len(end_str) :]
            else:
                pred = prompt_str_tokenized[answer_indices[pos_pred]]

            if try_schema_checker(pred, metadata["positional"], schema):
                patch_effect = "positional"
            elif try_schema_checker(pred, metadata["keyload"], schema):
                patch_effect = "keyload"
            elif try_schema_checker(pred, metadata["payload"], schema):
                patch_effect = "payload"
            elif try_schema_checker(pred, metadata["no_effect"], schema):
                patch_effect = "no_effect"
            else:
                patch_effect = "unknown"

            results["normal"].append(prompt)
            results["cf"].append(cf_prompt)
            results["source_pos"].append(metadata["src_positional_index"])
            results["positional_index"].append(pos_index)
            results["keyload_index"].append(keyload_index)
            results["payload_index"].append(payload_index)
            results["layer"].append(layer)
            results["positional_prediction"].append(metadata["positional"])
            results["payload_prediction"].append(metadata["payload"])
            results["keyload_prediction"].append(metadata["keyload"])
            results["no_effect_prediction"].append(metadata["no_effect"])
            results["patch_effect"].append(patch_effect)
            results["prediction"].append(pred)
            results["dist"].append(values.tolist())
            results["distance"].append(pos_index - pos_pred)
            results["generated"].append(generate)

    print("[+] Finished getting dist")
    df = pd.DataFrame(results)
    df.to_csv(
        f"binding_results/dists/{model_name.replace('/', '_')}_{num_instances}_{num_samples}_{layer}_{cat_to_query}_{schema.name}_mess{messiness}_fill{num_fillers}_{datetime.now().strftime('%Y%m%d')}.csv",
        index=False,
    )
    return df


def log(txt):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {txt}", flush=True)


@click.command()
@click.option("--model-id", type=str, default="google/gemma-2-2b-it")
@click.option("--schema-name", type=str, default="SCHEMA_BOXES")
@click.option("--num-instances", type=int, default=20)
@click.option("--num-samples", type=int, default=1000)
@click.option("--layer", type=int, default=17)
@click.option(
    "--cat-indices-to-query",
    type=str,
    default="[0]",
    help="List of ints, e.g. [0,1]",
    callback=lambda ctx, param, value: [int(x) for x in eval(value)] if isinstance(value, str) else value,
)
@click.option("--cat-to-query", type=int, default=1)
@click.option("--generate", is_flag=True, default=False)
@click.option("--messiness", type=int, help="Messiness level: can be 0, 1, or 2")
@click.option("--num-fillers", type=int, help="Number of fillers per item", default=0)
@click.option("--write-baseline-rate", is_flag=True, default=False)
@click.option("--eval-batch-size", type=int, default=100)
@click.option(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional checkpoint path to load model weights from",
)
@click.option("--do-filter", is_flag=True, default=False)
def main(
    model_id,
    schema_name,
    num_instances,
    num_samples,
    layer,
    cat_indices_to_query,
    cat_to_query,
    messiness,
    generate=False,
    num_fillers=0,
    write_baseline_rate=False,
    eval_batch_size=100,
    checkpoint=None,
    do_filter=False,
):
    log(
        f"[+] Getting positional separability for {model_id} on {schema_name} with {num_instances} instances, {num_samples} samples, {layer} layer, {cat_indices_to_query} cat indices to query, {cat_to_query} cat to query, {messiness} messiness, {generate} generate"
    )

    log("[+] Loading model")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if checkpoint is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            resume_download=True,
            state_dict=None,
            local_files_only=False,
            # The checkpoint argument is passed as the 'revision' parameter in HuggingFace
            revision=checkpoint,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

    schema = None
    for _schema in schemas:
        if _schema.name == schema_name:
            schema = _schema
            break
    else:
        raise ValueError(f"Schema {schema_name} not found")

    causal_model = multi_order_multi_schema_task_to_lookbacks_generic_causal_model(
        [schema], num_instances, num_fillers_per_item=num_fillers, fillers=True if num_fillers > 0 else False
    )
    causal_models = {schema.name: causal_model}

    if messiness == -1:
        counterfactual_template = ppkn_simpler_counterfactual_template
    elif messiness == 0:
        counterfactual_template = ppkn_simpler_counterfactual_template_split_key_loc
    elif messiness == 1:
        counterfactual_template = ppkn_simpler_counterfactual_template_split_key_loc_change_up_prev_index
    elif messiness == 2:
        counterfactual_template = my_ppkn_counterfactual_template
    elif messiness == 3:
        counterfactual_template = lookbacks_first_counterfactual_template
    elif messiness == 4:
        counterfactual_template = ppkn_simpler_counterfactual_template_split_key_loc_new_payload
    elif messiness == 5:
        counterfactual_template = ppkn_simpler_counterfactual_template_split_key_loc_maybe
    elif messiness == 6:
        counterfactual_template = ppkn_simpler_counterfactual_template_keep_payload_change_key
    elif messiness == 8:
        counterfactual_template = my_ppkn_counterfactual_template_made_up_key
    else:
        raise ValueError(f"Invalid messiness level: {messiness}")

    log("[+] Getting counterfactual datasets")
    train_ds, test_ds, fps = get_counterfactual_datasets(
        None,
        [schema],
        num_samples=num_samples,
        num_instances=num_instances,
        cat_indices_to_query=cat_indices_to_query,
        answer_cat_id=cat_to_query,
        do_assert=True,
        do_filter=do_filter,
        counterfactual_template=counterfactual_template,
        causal_models=causal_models,
        sample_an_answerable_question=sample_answerable_question_template,
    )

    get_dist(
        model,
        tokenizer,
        model_id,
        train_ds,
        schema,
        num_instances,
        num_samples,
        layer,
        cat_to_query,
        messiness,
        generate=generate,
        num_fillers=num_fillers,
    )

    if not write_baseline_rate:
        return

    print("[+] Writing baseline rate")
    pipeline = LMPipeline(
        model_id, max_new_tokens=max(schema.max_new_tokens for schema in schemas), device="cuda", dtype=torch.float16
    )
    _, _, fps = get_counterfactual_datasets(
        pipeline,
        [schema],
        num_samples=num_samples,
        num_instances=num_instances,
        cat_indices_to_query=cat_indices_to_query,
        answer_cat_id=cat_to_query,
        do_assert=False,
        do_filter=True,
        counterfactual_template=counterfactual_template,
        causal_models=causal_models,
        sample_an_answerable_question=sample_answerable_question_template,
        batch_size=eval_batch_size,
        num_test_samples=100,
    )

    br = fps[0][0][0]
    print(f"Baseline rate: {br}")
    with open(
        f"binding_results/dists/{model_id.replace('/', '_')}_{num_instances}_{num_samples}_{layer}_{cat_to_query}_{schema.name}_mess{messiness}_fill{num_fillers}_{datetime.now().strftime('%Y%m%d')}_br.txt",
        "w",
    ) as f:
        f.write(f"{br}")


if __name__ == "__main__":
    main()
