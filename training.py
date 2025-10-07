import os
import ast
import random
import pandas as pd
from functools import partial

from tqdm import tqdm as _tqdm
from typing import Callable
from tabulate import tabulate
from datetime import datetime
from grammar.grammar import Schema
from causal.causal_model import CausalModel
from causal.causal_model import CounterfactualDataset
from experiments.filter_experiment import FilterExperiment
from neural.LM_units import TokenPosition, get_last_token_index
from experiments.residual_stream_experiment import PatchResidualStream
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from grammar.task_to_causal_model import (
    multi_schema_task_to_lookbacks_generic_causal_model,
    multi_order_multi_schema_task_to_lookbacks_generic_causal_model,
)


def _get_indices_for_querying(
    schema: Schema, cat_indices_to_query: list[int] | None = None, answer_cat_id: int | None = None
):
    """
    Get the default indices for querying and answering. By default we return all but the last category for querying, and the last category for the answer.
    """
    if cat_indices_to_query is None:
        cat_indices_to_query = list(range(len(schema.categories) - 1))

    if answer_cat_id is None:
        answer_cat_id = max([i for i in range(len(schema.categories)) if i not in cat_indices_to_query])

    return cat_indices_to_query, answer_cat_id


def sample_answerable_question_template(
    schema: Schema,
    num_instances: int,
    cat_indices_to_query: list[int] | None = None,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling an answerable question from a schema. It samples a random instance for each category, and then sets the query to a random instance.

    Args:
        schema: The schema to sample an answerable question from.
        num_instances: The number of instances (or tuples) to sample.
        cat_indices_to_query: The indices of the categories used for querying. If None, all but the last category are used for querying.
        answer_cat_id: The index of the answer category, which we query. If None, the last category not in cat_indices_to_query is used.
    """
    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    input = {}
    for cat_id in range(len(schema.categories)):
        vals = random.sample(schema.items[schema.categories[cat_id]], num_instances)
        for i, val in enumerate(vals):
            input[f"Object.{cat_id}.{i}"] = val

    for i in range(num_instances):
        input[f"Object.0.Ordinal.{i}"] = i

    if query_index_vals is None:
        query_index = random.randint(0, num_instances - 1)
    else:
        query_index = random.choice(query_index_vals)

    for cat_id in range(len(schema.categories)):
        if cat_id in cat_indices_to_query:
            input[f"Object.{cat_id}.Query"] = input[f"Object.{cat_id}.{query_index}"]
        else:
            input[f"Object.{cat_id}.Query"] = None

    input["answerCategory"] = schema.categories[answer_cat_id]
    input["schemaName"] = schema.name

    return input


def lookbacks_first_counterfactual_template(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    answer_category = schema.categories[answer_cat_id]

    existing_values = {input[f"Object.{answer_cat_id}.{instance_id}"] for instance_id in range(num_instances)}

    new_final_cat = random.sample(list(set(schema.items[answer_category]) - existing_values), 2)

    if query_index_vals is None:
        swap_index = random.choice(list(set(range(num_instances)) - {query_index}))
    else:
        swap_index = random.choice(list(set(query_index_vals) - {query_index}))

    for cat_id in range(len(schema.categories)):
        counterfactual[f"Object.{cat_id}.{swap_index}"] = input[f"Object.{cat_id}.{query_index}"]
        counterfactual[f"Object.{cat_id}.{query_index}"] = input[f"Object.{cat_id}.{swap_index}"]

    counterfactual[f"Object.{answer_cat_id}.{swap_index}"] = new_final_cat[0]
    counterfactual[f"Object.{answer_cat_id}.{query_index}"] = new_final_cat[1]

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{swap_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{swap_index}"],
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "keyload": input[f"Object.{answer_cat_id}.{query_index}"],
        "src_positional_index": query_index,
        "dst_index": swap_index,
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def train_experiment(
    train_schemas: list[Schema],
    training_datasets: dict,
    pipeline,
    token_positions,
    method="DBM+SVD",
    n_features=32,
    epochs=5,
    init_lr=1e-2,
    layer: list[int] | int = 17,
    regularization=1e-4,
    num_instances=2,
    batch_size=10,
    ordering: list[int] | None = None,
    layers_at_once=False,
    target_variable="answerPointer",
    causal_model: CausalModel | None = None,
):
    """
    Train a residual stream experiment on a set of schemas and datasets.

    Args:
        train_schemas: The schemas to train on.
        training_datasets: A dictionary of datasets to train on, keyed by schema name.
        pipeline: The pipeline to use for training.
        token_positions: The token positions to use for training.
        method: The method to use for training (DBM+SVD or DAS).
        n_features: The number of features to use for training for DAS.
        epochs: The number of training epochs.
        init_lr: The initial learning rate when training the featurizer.
        layer: The layer to use for training. If a list, we train on all layers in the list.
        regularization: The regularization coefficient for DBM+SVD.
    """
    assert method in {"DBM+SVD", "DAS", "full_vector"}
    if not isinstance(layer, list):
        layer = [layer]

    config = {
        "batch_size": batch_size,
        "training_epoch": epochs,
        "n_features": n_features,
        "init_lr": init_lr,
        "regularization_coefficient": regularization,
    }
    config["method_name"] = method

    if causal_model is None:
        if ordering is None:
            causal_model = multi_schema_task_to_lookbacks_generic_causal_model(train_schemas, num_instances)
        else:
            causal_model = multi_order_multi_schema_task_to_lookbacks_generic_causal_model(
                train_schemas, num_instances, ordering
            )

    # TODO: for now we use the checker of the first schema, but this obviously isn't ideal
    experiment = PatchResidualStream(
        pipeline, causal_model, layer, token_positions, train_schemas[0].checker, config=config
    )

    if layers_at_once:
        # If we want to train on all layers at once, we need to flatten the model units lists
        experiment.model_units_lists = [[[x[0][0] for x in experiment.model_units_lists]]]

    if method == "DBM+SVD":
        experiment.build_SVD_feature_interventions(training_datasets, verbose=False)

    base_path = f"variable_binding/{'_'.join(schema.name for schema in train_schemas)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_path = f"{base_path}_model"

    target_variables = [target_variable] if isinstance(target_variable, str) else target_variable

    method_model_dir = os.path.join(
        model_path, f"{method}_{pipeline.model.__class__.__name__}_{'-'.join(target_variables)}"
    )

    if method != "full_vector":
        experiment.train_interventions(
            training_datasets,
            target_variables,
            method="DAS" if method == "DAS" else "DBM" if method == "DBM+SVD" else "full_vector",
            verbose=False,
        )
    else:
        experiment.save_featurizers(None, method_model_dir)

    return experiment, method_model_dir


def evaluate_featurizer(
    test_schemas: list[Schema],
    test_datasets,
    pipeline,
    token_positions,
    featurizer_path: str,
    method="DBM+SVD",
    n_features=32,
    epochs=5,
    init_lr=1e-2,
    layer: list[int] | int = 17,
    num_instances=2,
    batch_size=100,
    ordering: list[int] | None = None,
    layers_at_once=False,
    causal_model: CausalModel | None = None,
    target_variable="answerPointer",
    verbose=True,
):
    config = {"batch_size": batch_size, "training_epoch": epochs, "n_features": n_features, "init_lr": init_lr}
    config["method_name"] = method

    if not isinstance(layer, list):
        layer = [layer]

    if causal_model is None:
        if ordering is None:
            causal_model = multi_schema_task_to_lookbacks_generic_causal_model(test_schemas, num_instances)
        else:
            causal_model = multi_order_multi_schema_task_to_lookbacks_generic_causal_model(
                test_schemas, num_instances, ordering
            )

    # TODO: for now we use the checker of the first schema, but this obviously isn't ideal
    experiment = PatchResidualStream(
        pipeline, causal_model, layer, token_positions, test_schemas[0].checker, config=config
    )

    experiment.load_featurizers(featurizer_path)

    if layers_at_once:
        # If we want to train on all layers at once, we need to flatten the model units lists
        experiment.model_units_lists = [[[x[0][0] for x in experiment.model_units_lists]]]

    target_variables = [target_variable] if isinstance(target_variable, str) else target_variable
    target_variable_str = "-".join(target_variables)

    base_path = f"variable_binding/{'_'.join(schema.name for schema in test_schemas)}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    results_path = f"{base_path}_results"

    raw_results, verbose = experiment.perform_interventions(
        test_datasets, verbose=verbose, target_variables_list=[target_variables], save_dir=results_path
    )

    output = []
    for layer_res in list(list(raw_results["dataset"].values())[0]["model_unit"].values()):
        output.append(
            {"layer": layer_res["metadata"]["layer"], "average_score": layer_res[target_variable_str]["average_score"]}
        )

    if len(output) == 1:
        return output[0]["average_score"], verbose

    return output, verbose


def get_counterfactual_datasets(
    pipeline,
    schemas: list[Schema],
    num_samples: int = 100,
    num_instances: int = 2,
    minimum_filter_success_rate: float = 0.8,
    cat_indices_to_query: list[int] | None = None,
    answer_cat_id: int | None = None,
    do_assert=True,
    ordering: list[int] | None = None,
    do_filter=True,
    query_index_vals: list[int] | None = None,
    counterfactual_template: Callable = lookbacks_first_counterfactual_template,
    causal_models: CausalModel | None = None,
    sample_an_answerable_question: Callable = sample_answerable_question_template,
    num_test_samples: int | None = None,
    batch_size=100,
) -> tuple[dict, dict, tuple[list, dict, dict] | None]:
    """
    Get counterfactual datasets for a set of schemas, using the lookbacks first counterfactual template.
    Args:
        pipeline: The pipeline to use for filtering.
        schemas: The schemas to get counterfactual datasets for.
        num_samples: The number of samples to generate for each schema.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        minimum_filter_success_rate: The minimum success rate for the filter, otherwise we fail.
        cat_indices_to_query: The indices of the categories used for querying. If None, all but the last category are used for querying.
        answer_cat_id: The index of the answer category, which we query. If None, the last category not in cat_indices_to_query is used.

    Returns:
        train_counterfactual_datasets: A dictionary of train counterfactual datasets, keyed by schema name.
        test_counterfactual_datasets: A dictionary of all counterfactual datasets, keyed by schema name.
        filter_percentages: A list of 2-tuples with filter percentages per schema
    """
    if num_test_samples is None:
        num_test_samples = num_samples

    if causal_models is None:
        print("Initting causal models!")
        if ordering is None:
            causal_models = {
                schema.name: multi_schema_task_to_lookbacks_generic_causal_model([schema], num_instances)
                for schema in schemas
            }
        else:
            causal_models = {
                schema.name: multi_order_multi_schema_task_to_lookbacks_generic_causal_model(
                    [schema], num_instances, ordering
                )
                for schema in schemas
            }

    sample_answerable_question = {
        schema.name: partial(
            sample_an_answerable_question,
            schema=schema,
            num_instances=num_instances,
            cat_indices_to_query=cat_indices_to_query,
            answer_cat_id=answer_cat_id,
            query_index_vals=query_index_vals,
        )
        for schema in schemas
    }

    lookbacks_first_counterfactual = {
        schema.name: partial(
            counterfactual_template,
            model=causal_models[schema.name],
            schema=schema,
            num_instances=num_instances,
            sample_answerable_question=sample_answerable_question[schema.name],
            answer_cat_id=answer_cat_id,
            query_index_vals=query_index_vals,
            cat_indices_to_query=cat_indices_to_query,
        )
        for schema in schemas
    }

    if not do_filter:
        return (
            {
                schema.name: {
                    schema.name: CounterfactualDataset.from_sampler(
                        num_samples, lookbacks_first_counterfactual[schema.name]
                    )
                }
                for schema in schemas
            },
            {
                schema.name: {
                    schema.name: CounterfactualDataset.from_sampler(
                        num_test_samples, lookbacks_first_counterfactual[schema.name]
                    )
                }
                for schema in schemas
            },
            None,
        )

    filter_percentages = []
    counterfactual_datasets = {}
    all_counterfactual_datasets = {}
    for schema in _tqdm(schemas):
        exp = FilterExperiment(pipeline, causal_models[schema.name], schema.checker)

        key = f"lookbacks_first_counterfactual_{schema.name}"
        ds1 = {key: CounterfactualDataset.from_sampler(num_samples, lookbacks_first_counterfactual[schema.name])}
        ds2 = {key: CounterfactualDataset.from_sampler(num_test_samples, lookbacks_first_counterfactual[schema.name])}
        fds1, failed_data1 = exp.filter(ds1, verbose=True, batch_size=batch_size)
        fds2, failed_data2 = exp.filter(ds2, verbose=True, batch_size=batch_size)

        try:
            f1 = len(fds1[key]) / num_samples
            f2 = len(fds2[key]) / num_test_samples
        except:
            f1, f2 = 0.0, 0.0

        if do_assert:
            assert f1 >= minimum_filter_success_rate, f"Got filter success rate of {f1} for {schema.name}"
            assert f2 >= minimum_filter_success_rate, f"Got filter success rate of {f2} for {schema.name}"

        filter_percentages.append((f1, f2))

        counterfactual_datasets[schema.name] = fds1
        all_counterfactual_datasets[schema.name] = fds2

    return counterfactual_datasets, all_counterfactual_datasets, (filter_percentages, failed_data1, failed_data2)


def get_datasets_subset_dict(counterfactual_datasets: dict, schema_indices: list[int]):
    """
    Get a subset of the counterfactual datasets, by schema index.

    Args:
        counterfactual_datasets: The counterfactual datasets to get a subset of.
        schema_indices: The indices of the schemas to get a subset of.

    Returns:
        A dictionary of the subset of counterfactual datasets, which can be passed for training.
    """
    all_datasets = list(counterfactual_datasets.values())
    filtered_datasets = [all_datasets[i] for i in schema_indices]

    joined = filtered_datasets[0].copy()
    for dataset in filtered_datasets:
        joined.update(dataset.copy())

    return joined


def get_featurizer_from_path(
    schemas: list[Schema],
    path: str,
    token_positions,
    pipeline,
    layer: list[int] | int = 17,
    ordering: list[int] | None = None,
):
    config = {"batch_size": 10}

    if not isinstance(layer, list):
        layer = [layer]

    if ordering is None:
        causal_model = multi_schema_task_to_lookbacks_generic_causal_model(schemas, 2)
    else:
        causal_model = multi_order_multi_schema_task_to_lookbacks_generic_causal_model(schemas, 2, ordering)

    # TODO: for now we use the checker of the first schema, but this obviously isn't ideal
    experiment = PatchResidualStream(pipeline, causal_model, layer, token_positions, schemas[0].checker, config=config)

    experiment.load_featurizers(path)

    return experiment.model_units_lists[0][0][0]


def get_num_of_features(schemas: list[Schema], path: str, token_positions, pipeline, layer: list[int] | int = 17):
    unit = get_featurizer_from_path(schemas, path, token_positions, pipeline, layer)
    return len(unit.get_feature_indices())


def get_featurizer_mat(schemas: list[Schema], path: str, token_positions, pipeline, layer: list[int] | int = 17):
    unit = get_featurizer_from_path(schemas, path, token_positions, pipeline, layer)
    return unit.featurizer.featurizer.rotate.weight


def read_latest_log_file(
    index=-1, path="/home/morg/students/yoavgurarieh/logs/"
) -> list[tuple[None, str, str, str, str]]:
    """
    Read the latest log file from the given path.
    """
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    sorted_files = sorted(files, key=os.path.getmtime)
    latest_file = sorted_files[index]

    # Load the event file
    event_acc = EventAccumulator(latest_file)
    event_acc.Reload()

    return list(
        zip(
            [None for _ in range(len(event_acc.Tensors("base_inputs/text_summary")))],
            [None for _ in range(len(event_acc.Tensors("base_inputs/text_summary")))],
            [
                event_acc.Tensors("base_inputs/text_summary")[i].tensor_proto.string_val[0].decode()
                for i in range(len(event_acc.Tensors("base_inputs/text_summary")))
            ],
            [
                ast.literal_eval(
                    event_acc.Tensors("counterfactual_inputs/text_summary")[i].tensor_proto.string_val[0].decode()
                )[0]
                for i in range(len(event_acc.Tensors("counterfactual_inputs/text_summary")))
            ],
            [
                ast.literal_eval(event_acc.Tensors("labels/text_summary")[i].tensor_proto.string_val[0].decode())[0]
                for i in range(len(event_acc.Tensors("labels/text_summary")))
            ],
            [
                ast.literal_eval(event_acc.Tensors("preds/text_summary")[i].tensor_proto.string_val[0].decode())[0]
                for i in range(len(event_acc.Tensors("preds/text_summary")))
            ],
        )
    )


def parse_verbose_results(verbose: list, pipeline, batches=1):
    labels = verbose[0][batches]

    inputs = []
    for i in range(batches):
        inputs.extend(verbose[0][i][0]["input"])
    raw_inputs = [x["raw_input"] for x in inputs]

    cf_inputs = []
    for i in range(batches):
        for j in range(len(verbose[0][i][0]["counterfactual_inputs"])):
            cf_inputs.append(verbose[0][i][0]["counterfactual_inputs"][j][0])

    cfs = [x["raw_input"] for x in cf_inputs]

    preds = []
    for i in range(batches):
        preds.extend(pipeline.dump(verbose[0][i][3]))

    return list(zip(inputs, cf_inputs, raw_inputs, cfs, labels, preds))


def display_verbose_results(schema: Schema, verbose: list):
    out = {"Base Input": [], "Counterfactual Input": [], "dst_index": [], "Label": [], "Prediction": [], "Correct": []}

    full_verbose = []
    for idict, cdict, inp, cf, label, pred in verbose:
        full_verbose.append((inp, cf, label, pred, schema.checker(pred, label)))
        out["Base Input"].append(inp)
        out["Counterfactual Input"].append(cf)
        out["Label"].append(label)
        out["Prediction"].append(pred)
        out["Correct"].append(schema.checker(pred, label))
        out["dst_index"].append(idict["metadata"]["dst_index"])

    # print(tabulate(full_verbose, headers=headers, tablefmt="grid", maxcolwidths=[50, 50, 15, 15]))
    return pd.DataFrame(out)


def my_ppkn_counterfactual_template(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    cat_indices_to_query: list[int] | None = None,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer = forward_res["answer"]
    if isinstance(answer, dict):
        answer = max(answer, key=lambda k: answer[k])

    for i in range(num_instances):
        if input[f"Object.{answer_cat_id}.{i}"] == answer:
            query_index = i
            break
    else:
        raise ValueError("Failed to find the query index")

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    ##########################################################################

    # Shuffle keys
    indices = list(range(num_instances))
    deranged = indices.copy()
    for x in range(1000):
        random.shuffle(deranged)
        if all(i != j for i, j in zip(indices, deranged)):
            break
    else:
        raise ValueError("Failed to find a derangement")

    index_map = dict(zip(indices, deranged))
    reverse_index_map = {v: k for k, v in index_map.items()}

    for cat_index_to_query in cat_indices_to_query + [answer_cat_id]:
        for i in range(num_instances):
            counterfactual[f"Object.{cat_index_to_query}.{i}"] = input[f"Object.{cat_index_to_query}.{index_map[i]}"]

    # Generate cf query index that's different from the normal query index and also that it's not with the same key
    for x in range(1000):
        cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))
        if (
            input[f"Object.{cat_indices_to_query[0]}.{query_index}"]
            != counterfactual[f"Object.{cat_indices_to_query[0]}.{cf_query_index}"]
        ):
            break
    else:
        raise ValueError(
            "Failed to find a cf query index that's different from the normal query index and also that it's not with the same key"
        )

    keyload = counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"]

    # Set the value of the cf query index to be a different value, but don't swap with the normal query index
    cf_swapper_index = random.choice(
        list(
            set(range(num_instances))
            - {cf_query_index, reverse_index_map[query_index], reverse_index_map[cf_query_index]}
        )
    )
    backup = counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"]
    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = counterfactual[
        f"Object.{answer_cat_id}.{cf_swapper_index}"
    ]
    counterfactual[f"Object.{answer_cat_id}.{cf_swapper_index}"] = backup

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "keyload": keyload,
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "src_positional_index": query_index,
        "src_keyload_index": index_map[cf_query_index],
        "src_payload_index": index_map[cf_swapper_index],
        "dst_index": cf_query_index,
    }

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def sample_pkn_question_template(
    schema: Schema,
    num_instances: int,
    cat_indices_to_query: list[int] | None = None,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling an answerable question from a schema. It samples a random instance for each category, and then sets the query to a random instance.

    Args:
        schema: The schema to sample an answerable question from.
        num_instances: The number of instances (or tuples) to sample.
        cat_indices_to_query: The indices of the categories used for querying. If None, all but the last category are used for querying.
        answer_cat_id: The index of the answer category, which we query. If None, the last category not in cat_indices_to_query is used.
    """
    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    input = {}
    for cat_id in range(len(schema.categories)):
        vals = random.sample(schema.items[schema.categories[cat_id]], num_instances)
        for i, val in enumerate(vals):
            input[f"Object.{cat_id}.{i}"] = val

    if query_index_vals is None:
        query_index = random.randint(0, num_instances - 1)
    else:
        query_index = random.choice(query_index_vals)

    for cat_id in range(len(schema.categories)):
        if cat_id in cat_indices_to_query:
            input[f"Object.{cat_id}.Query"] = input[f"Object.{cat_id}.{query_index}"]
        else:
            input[f"Object.{cat_id}.Query"] = None

    input["answerCategory"] = schema.categories[answer_cat_id]
    input["schemaName"] = schema.name

    return input


def my_ppkn_counterfactual_template_made_up_key(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    cat_indices_to_query: list[int] | None = None,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer = forward_res["answer"]
    for i in range(num_instances):
        if input[f"Object.{answer_cat_id}.{i}"] == answer:
            query_index = i
            break
    else:
        raise ValueError("Failed to find the query index")

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    ##########################################################################

    # Shuffle keys
    indices = list(range(num_instances))
    deranged = indices.copy()
    for x in range(1000):
        random.shuffle(deranged)
        if all(i != j for i, j in zip(indices, deranged)):
            break
    else:
        raise ValueError("Failed to find a derangement")

    index_map = dict(zip(indices, deranged))
    reverse_index_map = {v: k for k, v in index_map.items()}

    for cat_index_to_query in cat_indices_to_query + [answer_cat_id]:
        for i in range(num_instances):
            counterfactual[f"Object.{cat_index_to_query}.{i}"] = input[f"Object.{cat_index_to_query}.{index_map[i]}"]

    # Generate cf query index that's different from the normal query index and also that it's not with the same key
    for x in range(1000):
        cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))
        if (
            input[f"Object.{cat_indices_to_query[0]}.{query_index}"]
            != counterfactual[f"Object.{cat_indices_to_query[0]}.{cf_query_index}"]
        ):
            break
    else:
        raise ValueError(
            "Failed to find a cf query index that's different from the normal query index and also that it's not with the same key"
        )

    keyload = counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"]

    # Set the value of the cf query index to be a different value, but don't swap with the normal query index
    cf_swapper_index = random.choice(
        list(
            set(range(num_instances))
            - {cf_query_index, reverse_index_map[query_index], reverse_index_map[cf_query_index]}
        )
    )
    backup = counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"]
    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = counterfactual[
        f"Object.{answer_cat_id}.{cf_swapper_index}"
    ]
    counterfactual[f"Object.{answer_cat_id}.{cf_swapper_index}"] = backup

    # Set the value of the key in the cf query index to be a different *new* value, that doesn't exist in the original input
    key_cat_index = cat_indices_to_query[0]
    existing_keys = set(input[f"Object.{key_cat_index}.{i}"] for i in range(num_instances))
    new_key = random.choice(list(set(schema.items[schema.categories[key_cat_index]]) - existing_keys))
    counterfactual[f"Object.{key_cat_index}.{cf_query_index}"] = new_key

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "keyload": keyload,
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "src_positional_index": query_index,
        "src_keyload_index": index_map[cf_query_index],
        "src_payload_index": index_map[cf_swapper_index],
        "dst_index": cf_query_index,
    }

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def my_ppkn_counterfactual_template_joint_pos_key(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    cat_indices_to_query: list[int] | None = None,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer = forward_res["answer"]
    for i in range(num_instances):
        if input[f"Object.{answer_cat_id}.{i}"] == answer:
            query_index = i
            break
    else:
        raise ValueError("Failed to find the query index")

    counterfactual = input.copy()

    ##########################################################################

    # Shuffle keys
    indices = list(range(num_instances))
    deranged = indices.copy()
    for x in range(1000):
        random.shuffle(deranged)
        if all(i != j for i, j in zip(indices, deranged)):
            break
    else:
        raise ValueError("Failed to find a derangement")

    index_map = dict(zip(indices, deranged))
    reverse_index_map = {v: k for k, v in index_map.items()}

    for cat_index_to_query in cat_indices_to_query + [answer_cat_id]:
        for i in range(num_instances):
            counterfactual[f"Object.{cat_index_to_query}.{i}"] = input[f"Object.{cat_index_to_query}.{index_map[i]}"]

    # Generate cf query index that's different from the normal query index and also that it's not with the same key
    for x in range(1000):
        cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))
        if (
            input[f"Object.{cat_indices_to_query[0]}.{query_index}"]
            != counterfactual[f"Object.{cat_indices_to_query[0]}.{cf_query_index}"]
        ):
            break
    else:
        raise ValueError(
            "Failed to find a cf query index that's different from the normal query index and also that it's not with the same key"
        )

    keyload = counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"]

    # Set the value of the cf query index to be a different value, but don't swap with the normal query index
    cf_swapper_index = random.choice(
        list(
            set(range(num_instances))
            - {cf_query_index, reverse_index_map[query_index], reverse_index_map[cf_query_index]}
        )
    )
    backup = counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"]
    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = counterfactual[
        f"Object.{answer_cat_id}.{cf_swapper_index}"
    ]
    counterfactual[f"Object.{answer_cat_id}.{cf_swapper_index}"] = backup

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    # Change the key of the cf_query_index in input to be the same as the key of the cf_query_index in counterfactual
    backup = input[f"Object.{cat_indices_to_query[0]}.{cf_query_index}"]
    input[f"Object.{cat_indices_to_query[0]}.{cf_query_index}"] = counterfactual[
        f"Object.{cat_indices_to_query[0]}.{cf_query_index}"
    ]
    input[f"Object.{cat_indices_to_query[0]}.{index_map[cf_query_index]}"] = backup

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "keyload": keyload,
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "src_positional_index": query_index,
        "src_keyload_index": index_map[cf_query_index],
        "src_payload_index": index_map[cf_swapper_index],
        "dst_index": cf_query_index,
    }

    forward_res = model.run_forward(input)
    input["raw_input"] = forward_res["raw_input"]
    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def ppkn_simpler_counterfactual_template(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    if query_index_vals is None:
        swap_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    else:
        swap_index = random.choice(list(set(query_index_vals) - {query_index, cf_query_index}))

    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = input[f"Object.{answer_cat_id}.{swap_index}"]
    counterfactual[f"Object.{answer_cat_id}.{swap_index}"] = input[f"Object.{answer_cat_id}.{cf_query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "keyload": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "src_positional_index": query_index,
        # "src_keyload_index": index_map[cf_query_index],
        # "src_payload_index": index_map[cf_swapper_index],
        "dst_index": cf_query_index,
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def ppkn_simpler_counterfactual_template_keep_payload_change_key(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    if query_index_vals is None:
        swap_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    else:
        swap_index = random.choice(list(set(query_index_vals) - {query_index, cf_query_index}))

    for cat_index_to_query in cat_indices_to_query:
        counterfactual[f"Object.{cat_index_to_query}.{cf_query_index}"] = input[
            f"Object.{cat_index_to_query}.{swap_index}"
        ]
        counterfactual[f"Object.{cat_index_to_query}.{swap_index}"] = input[
            f"Object.{cat_index_to_query}.{cf_query_index}"
        ]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "keyload": input[f"Object.{answer_cat_id}.{swap_index}"],
        "src_positional_index": query_index,
        # "src_keyload_index": index_map[cf_query_index],
        # "src_payload_index": index_map[cf_swapper_index],
        "dst_index": cf_query_index,
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def ppkn_simpler_counterfactual_template_keep_payload_change_key_new_payload(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    if query_index_vals is None:
        swap_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    else:
        swap_index = random.choice(list(set(query_index_vals) - {query_index, cf_query_index}))

    for cat_index_to_query in cat_indices_to_query:
        counterfactual[f"Object.{cat_index_to_query}.{cf_query_index}"] = input[
            f"Object.{cat_index_to_query}.{swap_index}"
        ]
        counterfactual[f"Object.{cat_index_to_query}.{swap_index}"] = input[
            f"Object.{cat_index_to_query}.{cf_query_index}"
        ]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    answer_category = schema.categories[answer_cat_id]
    existing_values = {input[f"Object.{answer_cat_id}.{instance_id}"] for instance_id in range(num_instances)}
    new_final_cat = random.choice(list(set(schema.items[answer_category]) - existing_values))
    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = new_final_cat

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "keyload": None,
        "src_positional_index": query_index,
        # "src_keyload_index": index_map[cf_query_index],
        # "src_payload_index": index_map[cf_swapper_index],
        "dst_index": cf_query_index,
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def ppkn_simpler_counterfactual_template_made_up_key(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    if query_index_vals is None:
        swap_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    else:
        swap_index = random.choice(list(set(query_index_vals) - {query_index, cf_query_index}))

    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = input[f"Object.{answer_cat_id}.{swap_index}"]
    counterfactual[f"Object.{answer_cat_id}.{swap_index}"] = input[f"Object.{answer_cat_id}.{cf_query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    # Set the value of the key in the cf query index to be a different *new* value, that doesn't exist in the original input
    for key_cat_index in cat_indices_to_query:
        existing_keys = set(input[f"Object.{key_cat_index}.{i}"] for i in range(num_instances))
        new_key = random.choice(list(set(schema.items[schema.categories[key_cat_index]]) - existing_keys))
        counterfactual[f"Object.{key_cat_index}.{cf_query_index}"] = new_key

        for i in range(len(cat_indices_to_query)):
            counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
                f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
            ]

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "keyload": None,
        "src_positional_index": query_index,
        # "src_keyload_index": index_map[cf_query_index],
        # "src_payload_index": index_map[cf_swapper_index],
        "dst_index": cf_query_index,
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def ppkn_simpler_counterfactual_template_split_key_loc(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    if query_index_vals is None:
        swap_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    else:
        swap_index = random.choice(list(set(query_index_vals) - {query_index, cf_query_index}))

    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = input[f"Object.{answer_cat_id}.{swap_index}"]
    counterfactual[f"Object.{answer_cat_id}.{swap_index}"] = input[f"Object.{answer_cat_id}.{cf_query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    # Set the value of the key in the cf query index to be a different *existing* value, that does exist in the original input
    swappy_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    for key_cat_index in cat_indices_to_query:
        counterfactual[f"Object.{key_cat_index}.{cf_query_index}"] = input[f"Object.{key_cat_index}.{swappy_index}"]
        counterfactual[f"Object.{key_cat_index}.{swappy_index}"] = input[f"Object.{key_cat_index}.{cf_query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "keyload": input[f"Object.{answer_cat_id}.{swappy_index}"],
        "src_positional_index": query_index,
        # "src_keyload_index": index_map[cf_query_index],
        # "src_payload_index": index_map[cf_swapper_index],
        "dst_index": cf_query_index,
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def ppkn_simpler_counterfactual_template_split_key_loc_new_payload(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    answer = forward_res["answer"]
    for i in range(num_instances):
        if input[f"Object.{answer_cat_id}.{i}"] == answer:
            query_index = i
            break
    else:
        raise ValueError("Failed to find the query index")
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    if query_index_vals is None:
        swap_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    else:
        swap_index = random.choice(list(set(query_index_vals) - {query_index, cf_query_index}))

    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = input[f"Object.{answer_cat_id}.{swap_index}"]
    counterfactual[f"Object.{answer_cat_id}.{swap_index}"] = input[f"Object.{answer_cat_id}.{cf_query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    # Set the value of the key in the cf query index to be a different *existing* value, that does exist in the original input
    swappy_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    for key_cat_index in cat_indices_to_query:
        counterfactual[f"Object.{key_cat_index}.{cf_query_index}"] = input[f"Object.{key_cat_index}.{swappy_index}"]
        counterfactual[f"Object.{key_cat_index}.{swappy_index}"] = input[f"Object.{key_cat_index}.{cf_query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    answer_category = schema.categories[answer_cat_id]
    existing_values = {input[f"Object.{answer_cat_id}.{instance_id}"] for instance_id in range(num_instances)}
    new_final_cat = random.choice(list(set(schema.items[answer_category]) - existing_values))
    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = new_final_cat

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "keyload": input[f"Object.{answer_cat_id}.{swappy_index}"],
        "keyload_key": counterfactual[f"Object.{cat_indices_to_query[0]}.{cf_query_index}"],
        "src_positional_index": query_index,
        "dst_index": cf_query_index,
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


# Change payload and keyload to be new as well
def counterfactual_template_just_change_question_index(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    # New value for keyload
    for key_cat_index in cat_indices_to_query:
        existing_keys = set(input[f"Object.{key_cat_index}.{i}"] for i in range(num_instances))
        new_key = random.choice(list(set(schema.items[schema.categories[key_cat_index]]) - existing_keys))
        counterfactual[f"Object.{key_cat_index}.{cf_query_index}"] = new_key

        for i in range(len(cat_indices_to_query)):
            counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
                f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
            ]

    # New value for payload
    answer_category = schema.categories[answer_cat_id]
    existing_values = {input[f"Object.{answer_cat_id}.{instance_id}"] for instance_id in range(num_instances)}
    new_final_cat = random.choice(list(set(schema.items[answer_category]) - existing_values))
    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = new_final_cat

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]
    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "keyload": counterfactual[f"Object.{cat_indices_to_query[0]}.{cf_query_index}"],
        "src_positional_index": query_index,
        "dst_index": cf_query_index,
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def ppkn_simpler_counterfactual_template_split_key_loc_change_up_prev_index(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    if query_index_vals is None:
        swap_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    else:
        swap_index = random.choice(list(set(query_index_vals) - {query_index, cf_query_index}))

    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = input[f"Object.{answer_cat_id}.{swap_index}"]
    counterfactual[f"Object.{answer_cat_id}.{swap_index}"] = input[f"Object.{answer_cat_id}.{cf_query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    # Set the value of the key in the cf query index to be a different *existing* value, that does exist in the original input
    swappy_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    for key_cat_index in cat_indices_to_query:
        counterfactual[f"Object.{key_cat_index}.{cf_query_index}"] = input[f"Object.{key_cat_index}.{swappy_index}"]
        counterfactual[f"Object.{key_cat_index}.{swappy_index}"] = input[f"Object.{key_cat_index}.{cf_query_index}"]

    # Set the value of the key in the cf query index - 1 to be a different *existing* value, that does exist in the original input
    if cf_query_index != 0:
        swappy_index_for_minus_one = random.choice(
            list(set(range(num_instances)) - {query_index, cf_query_index, swappy_index})
        )
        for key_cat_index in cat_indices_to_query + [answer_cat_id]:
            og_m1 = counterfactual[f"Object.{key_cat_index}.{cf_query_index-1}"]
            counterfactual[f"Object.{key_cat_index}.{cf_query_index-1}"] = counterfactual[
                f"Object.{key_cat_index}.{swappy_index_for_minus_one}"
            ]
            counterfactual[f"Object.{key_cat_index}.{swappy_index_for_minus_one}"] = og_m1

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "keyload": input[f"Object.{answer_cat_id}.{swappy_index}"],
        "src_positional_index": query_index,
        # "src_keyload_index": index_map[cf_query_index],
        # "src_payload_index": index_map[cf_swapper_index],
        "dst_index": cf_query_index,
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def tuple_binding_counterfactual_template_for_key(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    # Set the value of the key in the cf query index to be a different *existing* value, that does exist in the original input
    for key_cat_index in cat_indices_to_query:
        counterfactual[f"Object.{key_cat_index}.{query_index}"] = input[f"Object.{key_cat_index}.{cf_query_index}"]
        counterfactual[f"Object.{key_cat_index}.{cf_query_index}"] = input[f"Object.{key_cat_index}.{query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "query_index": query_index,
        "query_index_value": input[f"Object.{answer_cat_id}.{query_index}"],
        "swapped_index": cf_query_index,
        "swapped_index_value": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "is_cf": False,
    }
    counterfactual["metadata"] = input["metadata"].copy()
    counterfactual["metadata"]["is_cf"] = True

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def tuple_binding_counterfactual_template_for_pos(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    # Set the value of the key in the cf query index to be a different *existing* value, that does exist in the original input
    for key_cat_index in cat_indices_to_query + [answer_cat_id]:
        counterfactual[f"Object.{key_cat_index}.{query_index}"] = input[f"Object.{key_cat_index}.{cf_query_index}"]
        counterfactual[f"Object.{key_cat_index}.{cf_query_index}"] = input[f"Object.{key_cat_index}.{query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{query_index}"
        ]

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "query_index": query_index,
        "query_index_value": input[f"Object.{answer_cat_id}.{query_index}"],
        "swapped_index": cf_query_index,
        "swapped_index_value": input[f"Object.{answer_cat_id}.{cf_query_index}"],
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def tuple_binding_counterfactual_template_for_key_new_key(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    # cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    for cat_id in cat_indices_to_query + [answer_cat_id]:
        cat_category = schema.categories[cat_id]
        existing_values = {input[f"Object.{cat_id}.{instance_id}"] for instance_id in range(num_instances)}
        new_cat_val = random.choice(list(set(schema.items[cat_category]) - existing_values))
        counterfactual[f"Object.{cat_id}.{query_index}"] = new_cat_val

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{query_index}"
        ]

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "query_index": query_index,
        "query_index_og_key": input[f"Object.{cat_indices_to_query[0]}.{query_index}"],
        "query_index_new_key": counterfactual[f"Object.{cat_indices_to_query[0]}.{query_index}"],
        "expected_value": input[f"Object.{answer_cat_id}.{query_index}"],
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


# def train_experiment_complete(train_schemas: list[Schema], training_datasets: dict, pipeline, token_positions, method="DBM+SVD", n_features=32, epochs=5, init_lr=1e-2, layer: list[int] | int = 17, regularization=1e-4, num_instances=2, batch_size=10, ordering: list[int] | None = None, layers_at_once=False, target_variable="answerPointer", causal_model: CausalModel | None = None):
#     """
#     Train a residual stream experiment on a set of schemas and datasets.

#     Args:
#         train_schemas: The schemas to train on.
#         training_datasets: A dictionary of datasets to train on, keyed by schema name.
#         pipeline: The pipeline to use for training.
#         token_positions: The token positions to use for training.
#         method: The method to use for training (DBM+SVD or DAS).
#         n_features: The number of features to use for training for DAS.
#         epochs: The number of training epochs.
#         init_lr: The initial learning rate when training the featurizer.
#         layer: The layer to use for training. If a list, we train on all layers in the list.
#         regularization: The regularization coefficient for DBM+SVD.
#     """
#     assert method in {"DBM+SVD", "DAS", "full_vector"}
#     if not isinstance(layer, list):
#         layer = [layer]

#     config = {"batch_size": batch_size, "training_epoch": epochs, "n_features": n_features, "init_lr": init_lr, "regularization_coefficient": regularization}
#     config["method_name"] = method

#     if causal_model is None:
#         if ordering is None:
#             causal_model = multi_schema_task_to_lookbacks_generic_causal_model(train_schemas, num_instances)
#         else:
#             causal_model = multi_order_multi_schema_task_to_lookbacks_generic_causal_model(train_schemas, num_instances, ordering)

#     # TODO: for now we use the checker of the first schema, but this obviously isn't ideal
#     experiment = PatchResidualStream(pipeline, causal_model, layer, token_positions, train_schemas[0].checker, config=config)

#     if layers_at_once:
#         # If we want to train on all layers at once, we need to flatten the model units lists
#         experiment.model_units_lists = [[[x[0][0] for x in experiment.model_units_lists]]]

#     if method == "DBM+SVD":
#         experiment.build_SVD_feature_interventions(training_datasets, verbose=False)

#     base_path = f"variable_binding/{'_'.join(schema.name for schema in train_schemas)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
#     model_path = f"{base_path}_model"

#     target_variables = ["answerPointer", "lexicalPointer"]

#     method_model_dir = os.path.join(model_path, f"{method}_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")

#     if method != "full_vector":
#         experiment.train_interventions(training_datasets, target_variables, method="DAS" if method == "DAS" else "DBM" if method == "DBM+SVD" else "full_vector", verbose=False)
#     else:
#         experiment.save_featurizers(None, method_model_dir)

#     return experiment, method_model_dir


def ppkn_simpler_counterfactual_template_split_key_loc_change_up_prev_index_fit_for_complete(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    if query_index_vals is None:
        swap_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    else:
        swap_index = random.choice(list(set(query_index_vals) - {query_index, cf_query_index}))

    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = input[f"Object.{answer_cat_id}.{swap_index}"]
    counterfactual[f"Object.{answer_cat_id}.{swap_index}"] = input[f"Object.{answer_cat_id}.{cf_query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    # Set the value of the key in the cf query index to be a different *existing* value, that does exist in the original input
    swappy_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    for key_cat_index in cat_indices_to_query:
        counterfactual[f"Object.{key_cat_index}.{cf_query_index}"] = input[f"Object.{key_cat_index}.{swappy_index}"]
        counterfactual[f"Object.{key_cat_index}.{swappy_index}"] = input[f"Object.{key_cat_index}.{cf_query_index}"]

    # Set the value of the key in the cf query index - 1 to be a different *existing* value, that does exist in the original input
    if cf_query_index != 0:
        swappy_index_for_minus_one = random.choice(
            list(set(range(num_instances)) - {query_index, cf_query_index, swappy_index})
        )
        for key_cat_index in cat_indices_to_query + [answer_cat_id]:
            og_m1 = counterfactual[f"Object.{key_cat_index}.{cf_query_index-1}"]
            counterfactual[f"Object.{key_cat_index}.{cf_query_index-1}"] = counterfactual[
                f"Object.{key_cat_index}.{swappy_index_for_minus_one}"
            ]
            counterfactual[f"Object.{key_cat_index}.{swappy_index_for_minus_one}"] = og_m1

    cf_forward_res = model.run_forward(counterfactual)
    lexical_query = cf_forward_res["lexicalQuery"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "keyload": input[f"Object.{answer_cat_id}.{swappy_index}"],
        "src_positional_index": query_index,
        # "src_keyload_index": index_map[cf_query_index],
        # "src_payload_index": index_map[cf_swapper_index],
        "dst_index": cf_query_index,
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


def ppkn_simpler_counterfactual_template_split_key_loc_new_keys_and_payloads(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    if query_index_vals is None:
        swap_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    else:
        swap_index = random.choice(list(set(query_index_vals) - {query_index, cf_query_index}))

    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = input[f"Object.{answer_cat_id}.{swap_index}"]
    counterfactual[f"Object.{answer_cat_id}.{swap_index}"] = input[f"Object.{answer_cat_id}.{cf_query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    # Set the value of the key in the cf query index to be a different *existing* value, that does exist in the original input
    swappy_index = random.choice(list(set(range(num_instances)) - {query_index, cf_query_index}))
    for key_cat_index in cat_indices_to_query:
        counterfactual[f"Object.{key_cat_index}.{cf_query_index}"] = input[f"Object.{key_cat_index}.{swappy_index}"]
        counterfactual[f"Object.{key_cat_index}.{swappy_index}"] = input[f"Object.{key_cat_index}.{cf_query_index}"]

    # Go over all categories and set a new value for the key and payload, except for the one we query (in the cf)
    for key_cat_index in cat_indices_to_query + [answer_cat_id]:
        existing_values = {input[f"Object.{key_cat_index}.{instance_id}"] for instance_id in range(num_instances)}
        # Choose num_instances-1 new values
        new_values = random.sample(
            list(set(schema.items[schema.categories[key_cat_index]]) - existing_values), num_instances
        )

        for i in range(num_instances):
            # if i == cf_query_index:# and key_cat_index == answer_cat_id:
            #     continue

            counterfactual[f"Object.{key_cat_index}.{i}"] = new_values[i]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "keyload": input[f"Object.{answer_cat_id}.{swappy_index}"],
        "src_positional_index": query_index,
        # "src_keyload_index": index_map[cf_query_index],
        # "src_payload_index": index_map[cf_swapper_index],
        "dst_index": cf_query_index,
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}


import random
from typing import List, Optional, Dict, Any, Tuple, Union

CatSpec = Optional[List[Union[int, str]]]
AnsSpec = Optional[Union[int, str]]
PerSchemaSpec = Dict[str, Tuple[CatSpec, AnsSpec]]


def _safe_get_indices_for_querying(
    schema: Schema,
    cat_indices_to_query: CatSpec = None,
    answer_cat_id: AnsSpec = None,
) -> Tuple[List[int], int]:
    """
    Resolve (cat_indices_to_query, answer_cat_id) for *this* schema.
    Accepts ints or names; drops OOB/unknown; ensures a nonempty query set that
    does not include the answer category.
    """
    cats = list(schema.categories)
    C = len(cats)
    assert C > 0

    # normalize query set -> indices
    if cat_indices_to_query is None:
        qry_idx: List[int] = []
    elif all(isinstance(x, int) for x in cat_indices_to_query):
        qry_idx = [int(x) for x in cat_indices_to_query if 0 <= int(x) < C]
    else:
        name_set = {str(x) for x in cat_indices_to_query}
        qry_idx = [i for i, n in enumerate(cats) if n in name_set]

    # normalize answer -> index
    if answer_cat_id is None:
        ans_idx: Optional[int] = None
    elif isinstance(answer_cat_id, int):
        ans_idx = answer_cat_id if 0 <= answer_cat_id < C else None
    else:
        try:
            ans_idx = cats.index(str(answer_cat_id))
        except ValueError:
            ans_idx = None

    if ans_idx is None:
        # prefer last category not in qry; fallback to last
        candidates = [i for i in range(C) if i not in set(qry_idx)]
        ans_idx = candidates[-1] if candidates else C - 1

    # ensure queries are valid and exclude answer
    qry_idx = [i for i in qry_idx if i != ans_idx]
    if (len(qry_idx) == 0) or (len(set(qry_idx)) >= C):
        qry_idx = [i for i in range(C) if i != ans_idx]

    return qry_idx, ans_idx


def _resolve_query_spec_for_schema(
    schema: Schema,
    per_schema_spec: Optional[PerSchemaSpec],
    global_cat_indices_to_query: CatSpec,
    global_answer_cat_id: AnsSpec,
) -> Tuple[List[int], int]:
    """
    If per_schema_spec contains schema.name, use that entry (allowing None inside);
    otherwise fall back to global args. Always returns schema-safe (indices, idx).
    """
    if per_schema_spec and schema.name in per_schema_spec:
        cat_q, ans_id = per_schema_spec[schema.name]
        # If entry fields are None, fall back to globals for that field
        if cat_q is None:
            cat_q_use = global_cat_indices_to_query
        else:
            cat_q_use = cat_q
        if ans_id is None:
            ans_id_use = global_answer_cat_id
        else:
            ans_id_use = ans_id
        return _safe_get_indices_for_querying(schema, cat_q_use, ans_id_use)

    # No per-schema override -> use globals
    return _safe_get_indices_for_querying(schema, global_cat_indices_to_query, global_answer_cat_id)


def sample_answerable_question_template_mixed(
    schema: Schema,
    num_instances: int,
    cat_indices_to_query: Optional[List[Union[int, str]]] = None,
    answer_cat_id: Optional[Union[int, str]] = None,
    query_index_vals: Optional[List[int]] = None,
    per_schema_spec: Optional[PerSchemaSpec] = None,
) -> Dict[str, Any]:
    """
    Sample an answerable question from a schema.

    Supports per-schema overrides via `per_schema_spec`, mapping:
        { schema.name: (cat_indices_to_query, answer_cat_id) }
    where elements can be indices or names (or None to defer to globals).
    """
    # Resolve (queries, answer) for THIS schema
    cat_indices_to_query_resolved, answer_cat_idx = _resolve_query_spec_for_schema(
        schema,
        per_schema_spec,
        cat_indices_to_query,
        answer_cat_id,
    )

    input: Dict[str, Any] = {}

    # Sample unique values per category
    for cat_id, cat_name in enumerate(schema.categories):
        vals = random.sample(schema.items[cat_name], num_instances)
        for i, val in enumerate(vals):
            input[f"Object.{cat_id}.{i}"] = val

    # Ordinals for first category (others copy via CM)
    for i in range(num_instances):
        input[f"Object.0.Ordinal.{i}"] = i

    # Choose a shared query index
    if query_index_vals is None:
        query_index = random.randint(0, num_instances - 1)
    else:
        query_index = random.choice(query_index_vals)
        if not (0 <= query_index < num_instances):
            raise ValueError(f"query_index {query_index} out of range [0, {num_instances-1}]")

    # Set queries
    for cat_id in range(len(schema.categories)):
        if cat_id in cat_indices_to_query_resolved:
            input[f"Object.{cat_id}.Query"] = input[f"Object.{cat_id}.{query_index}"]
        else:
            input[f"Object.{cat_id}.Query"] = None

    input["answerCategory"] = schema.categories[answer_cat_idx]
    input["schemaName"] = schema.name
    return input


from functools import partial
from typing import Callable


def get_counterfactual_datasets_mixed(
    pipeline,
    schemas: list[Schema],
    num_samples: int = 100,
    num_instances: int = 2,
    minimum_filter_success_rate: float = 0.8,
    cat_indices_to_query: list[int] | None = None,  # global fallback
    answer_cat_id: int | None = None,  # global fallback
    do_assert: bool = True,
    ordering: list[int] | None = None,
    do_filter: bool = True,
    query_index_vals: list[int] | None = None,
    counterfactual_template: Callable = lookbacks_first_counterfactual_template,
    causal_models: CausalModel | None = None,
    sample_an_answerable_question: Callable = sample_answerable_question_template,
    # NEW:
    per_schema_spec: dict[str, tuple[list[int] | list[str] | None, int | str | None]] | None = None,
) -> tuple[dict, dict, tuple[list, dict, dict] | None]:

    # ---------- helpers ----------
    def _normalize_query_and_answer(schema: Schema, cat_q_spec, ans_spec) -> tuple[list[int], int]:
        """
        Resolve (cat_indices_to_query, answer_cat_id) for THIS schema.
        Accepts indices or names; drops OOB/unknown; ensures non-empty query set
        that excludes the answer category.
        """
        cats = list(schema.categories)
        C = len(cats)

        # normalize query list -> indices
        if cat_q_spec is None:
            qry_idx = []
        elif all(isinstance(x, int) for x in cat_q_spec):
            qry_idx = [int(x) for x in cat_q_spec if 0 <= int(x) < C]
        else:
            name_set = {str(x) for x in cat_q_spec}
            qry_idx = [i for i, n in enumerate(cats) if n in name_set]

        # normalize answer -> index
        if ans_spec is None:
            ans_idx = None
        elif isinstance(ans_spec, int):
            ans_idx = ans_spec if 0 <= ans_spec < C else None
        else:
            try:
                ans_idx = cats.index(str(ans_spec))
            except ValueError:
                ans_idx = None

        if ans_idx is None:
            # pick last category not in qry (fallback to last)
            candidates = [i for i in range(C) if i not in set(qry_idx)]
            ans_idx = candidates[-1] if candidates else C - 1

        # ensure queries exclude answer and are non-empty but not “all”
        qry_idx = [i for i in qry_idx if i != ans_idx]
        if (len(qry_idx) == 0) or (len(set(qry_idx)) >= C):
            qry_idx = [i for i in range(C) if i != ans_idx]

        return qry_idx, ans_idx

    def _resolve_for_schema(schema: Schema):
        """Pick per-schema override if present; else use globals; then normalize."""
        if per_schema_spec and schema.name in per_schema_spec:
            ps_q, ps_a = per_schema_spec[schema.name]
        else:
            ps_q, ps_a = cat_indices_to_query, answer_cat_id
        return _normalize_query_and_answer(schema, ps_q, ps_a)

    # ---------- causal models ----------
    if causal_models is None:
        print("Initting causal models!")
        if ordering is None:
            causal_models = {
                schema.name: multi_schema_task_to_lookbacks_generic_causal_model([schema], num_instances)
                for schema in schemas
            }
        else:
            causal_models = {
                schema.name: multi_order_multi_schema_task_to_lookbacks_generic_causal_model(
                    [schema], num_instances, ordering
                )
                for schema in schemas
            }

    # ---------- per-schema samplers (pass per_schema_spec, still fine) ----------
    sample_answerable_question_per_schema = {}
    for schema in schemas:
        # We still pass per_schema_spec so the sampler can resolve names/indices too.
        sample_answerable_question_per_schema[schema.name] = partial(
            sample_an_answerable_question,
            schema=schema,
            num_instances=num_instances,
            # keep global fallbacks; sampler will override with per_schema_spec if present
            cat_indices_to_query=cat_indices_to_query,
            answer_cat_id=answer_cat_id,
            query_index_vals=query_index_vals,
            per_schema_spec=per_schema_spec,
        )

    # ---------- counterfactual template partials (USE RESOLVED INTS HERE) ----------
    lookbacks_first_counterfactual = {}
    for schema in schemas:
        resolved_q, resolved_ans = _resolve_for_schema(schema)
        lookbacks_first_counterfactual[schema.name] = partial(
            counterfactual_template,
            model=causal_models[schema.name],
            schema=schema,
            num_instances=num_instances,
            sample_answerable_question=sample_answerable_question_per_schema[schema.name],
            # CRITICAL: pass resolved values to the template (integers for this schema)
            answer_cat_id=resolved_ans,
            query_index_vals=query_index_vals,
            cat_indices_to_query=resolved_q,
        )

    # ---------- no-filter branch ----------
    if not do_filter:
        return (
            {
                schema.name: {
                    schema.name: CounterfactualDataset.from_sampler(
                        num_samples, lookbacks_first_counterfactual[schema.name]
                    )
                }
                for schema in schemas
            },
            {
                schema.name: {
                    schema.name: CounterfactualDataset.from_sampler(
                        num_samples, lookbacks_first_counterfactual[schema.name]
                    )
                }
                for schema in schemas
            },
            None,
        )

    # ---------- filtering ----------
    filter_percentages = []
    counterfactual_datasets = {}
    all_counterfactual_datasets = {}
    for schema in _tqdm(schemas):
        exp = FilterExperiment(pipeline, causal_models[schema.name], schema.checker)

        key = f"lookbacks_first_counterfactual_{schema.name}"
        ds1 = {key: CounterfactualDataset.from_sampler(num_samples, lookbacks_first_counterfactual[schema.name])}
        ds2 = {key: CounterfactualDataset.from_sampler(num_samples, lookbacks_first_counterfactual[schema.name])}
        fds1, failed_data1 = exp.filter(ds1, verbose=True, batch_size=100)
        fds2, failed_data2 = exp.filter(ds2, verbose=True, batch_size=100)

        f1 = len(fds1[key]) / num_samples
        f2 = len(fds2[key]) / num_samples

        if do_assert:
            assert f1 >= minimum_filter_success_rate, f"Got filter success rate of {f1} for {schema.name}"
            assert f2 >= minimum_filter_success_rate, f"Got filter success rate of {f2} for {schema.name}"

        filter_percentages.append((f1, f2))
        counterfactual_datasets[schema.name] = fds1
        all_counterfactual_datasets[schema.name] = fds2

    return counterfactual_datasets, all_counterfactual_datasets, (filter_percentages, failed_data1, failed_data2)


def ppkn_simpler_counterfactual_template_split_key_loc_maybe(
    model: CausalModel,
    schema: Schema,
    num_instances: int,
    sample_answerable_question: Callable,
    answer_cat_id: int | None = None,
    query_index_vals: list[int] | None = None,
    cat_indices_to_query: list[int] | None = None,
):
    """
    A template (meant to be used with partial) for sampling a question and its counterfactual, in the vein of the first counterfactual from the lookbacks paper. Used for generating a counterfactual dataset.

    Args:
        model: The causal model to use for generating the counterfactual.
        schema: The schema to generate a counterfactual for.
        num_instances: The number of instances to sample for each category (i.e. the number of tuples).
        sample_answerable_question: A callable that samples an answerable question from the schema - should be a partial
        of sample_answerable_question_template.
        answer_cat_id: The index of the answer category, which we query.
    """
    input = sample_answerable_question()
    forward_res = model.run_forward(input)
    query_index = forward_res["answerPointer"]
    cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    cat_indices_to_query, answer_cat_id = _get_indices_for_querying(schema, cat_indices_to_query, answer_cat_id)

    answer_cat_id = _get_indices_for_querying(schema, None, answer_cat_id)[1]

    counterfactual = input.copy()
    input["raw_input"] = forward_res["raw_input"]

    if query_index_vals is None:
        swap_index = random.choice(list(set(range(num_instances))))
    else:
        swap_index = random.choice(list(set(query_index_vals)))

    counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"] = input[f"Object.{answer_cat_id}.{swap_index}"]
    counterfactual[f"Object.{answer_cat_id}.{swap_index}"] = input[f"Object.{answer_cat_id}.{cf_query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    # Set the value of the key in the cf query index to be a different *existing* value, that does exist in the original input
    swappy_index = random.choice(list(set(range(num_instances))))
    for key_cat_index in cat_indices_to_query:
        counterfactual[f"Object.{key_cat_index}.{cf_query_index}"] = input[f"Object.{key_cat_index}.{swappy_index}"]
        counterfactual[f"Object.{key_cat_index}.{swappy_index}"] = input[f"Object.{key_cat_index}.{cf_query_index}"]

    for i in range(len(cat_indices_to_query)):
        counterfactual[f"Object.{cat_indices_to_query[i]}.Query"] = counterfactual[
            f"Object.{cat_indices_to_query[i]}.{cf_query_index}"
        ]

    counterfactual["raw_input"] = model.run_forward(counterfactual)["raw_input"]

    input["metadata"] = {
        "positional": input[f"Object.{answer_cat_id}.{cf_query_index}"],
        "payload": counterfactual[f"Object.{answer_cat_id}.{cf_query_index}"],
        "no_effect": input[f"Object.{answer_cat_id}.{query_index}"],
        "keyload": input[f"Object.{answer_cat_id}.{swappy_index}"],
        "src_positional_index": query_index,
        # "src_keyload_index": index_map[cf_query_index],
        # "src_payload_index": index_map[cf_swapper_index],
        "dst_index": cf_query_index,
    }

    return {"input": input, "counterfactual_inputs": [counterfactual]}
