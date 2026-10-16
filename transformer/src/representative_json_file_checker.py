"""Read and validate representative-forward JSON for SAIF pass2."""
import json
import re


def load_representative_forwards(
    json_path, *, task_name, target_layers, num_hidden_layers,
    dataset_size, batch_size, max_seq_length, pad_to_max_length,
):
    """Return {(layer, block): original 0-based forward index}.

    Accept either the multi-task selection file or one task's pass1 convergence
    report. Runtime settings must match pass1, including model, dataset order,
    batch size, tokenization and quantization. Only settings recorded in JSON
    can be checked here.
    """
    with open(json_path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Representative JSON must be an object.")
    if type(batch_size) is not int or batch_size < 1 or dataset_size < 1:
        raise ValueError("Representative pass2 requires a positive batch size and a nonempty evaluation dataset.")
    if not target_layers or any(type(layer) is not int or not 0 <= layer < num_hidden_layers for layer in target_layers):
        raise ValueError("saif_pass2_layers must contain valid model layer indices.")

    if "tasks" in data:
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported representative JSON schema_version; expected 1.")
        tasks = data["tasks"]
        if not isinstance(tasks, dict) or task_name not in tasks:
            raise ValueError(f"Task {task_name!r} is missing from representative JSON.")
        for name, actual in (("batch_size", batch_size), ("max_seq_length", max_seq_length),
                             ("pad_to_max_length", pad_to_max_length)):
            if name in data and (type(data[name]) is not type(actual) or data[name] != actual):
                raise ValueError(f"Representative JSON {name}={data[name]!r} does not match runtime {actual!r}.")
        task = tasks[task_name]
        if not isinstance(task, dict) or not isinstance(task.get("positions"), dict):
            raise ValueError(f"Task {task_name!r} must contain a positions object.")
        positions = task["positions"]
        expected_counts = [task["num_forwards"]] if "num_forwards" in task else []
    else:
        # Direct use of an existing {task}_convergence.json: same selection,
        # but batch size and tokenization metadata are unavailable in that file.
        positions = data
        expected_counts = [value["num_forwards"] for value in positions.values()
                           if isinstance(value, dict) and "num_forwards" in value]

    total_forwards = (dataset_size + batch_size - 1) // batch_size
    if any(type(count) is not int or count != total_forwards for count in expected_counts):
        raise ValueError(f"Pass1 num_forwards does not match runtime evaluation ({total_forwards} batches). Check dataset and batch size.")
    selected = {}
    for tag, info in positions.items():
        match = re.fullmatch(r"layer(0|[1-9][0-9]*)_(atten|ffn)", tag)
        if match is None:
            raise ValueError(f"Invalid representative position: {tag!r}.")
        layer, block = int(match[1]), match[2]
        if not 0 <= layer < num_hidden_layers:
            raise ValueError(f"Representative position outside model: {tag}.")
        index = info.get("medoid_forward_idx") if isinstance(info, dict) else info
        if type(index) is not int or not 0 <= index < total_forwards:
            raise ValueError(f"{tag}: invalid representative forward index {index!r}; expected 0..{total_forwards - 1}.")
        if layer in target_layers:
            selected[(layer, block)] = index
    required = {(layer, block) for layer in target_layers for block in ("atten", "ffn")}
    missing = required - set(selected)
    if missing:
        raise ValueError("Representative JSON is missing positions: " + ", ".join(
            f"layer{layer}_{block}" for layer, block in sorted(missing)))
    return selected
