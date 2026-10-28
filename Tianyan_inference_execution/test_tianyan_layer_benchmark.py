import pennylane as qml

import tianyan_layer_benchmark as benchmark


def test_benchmark_constants_are_locked_to_requested_run():
    assert benchmark.BACKEND == "tianyan-287"
    assert benchmark.SHOTS == 10
    assert benchmark.REPEATS == 3
    assert benchmark.TOKEN_QUBITS == 4
    assert benchmark.TOKEN_COUNT == 3
    assert benchmark.ATTENTION_HEADS == 4


def test_segmented_critical_paths_cover_both_layers():
    segments = benchmark.build_segments()
    attention = [segment for segment in segments if segment.layer == "attention"]
    ffn = [segment for segment in segments if segment.layer == "ffn"]

    assert [segment.name for segment in attention] == [
        "attention_head_0",
        "attention_head_1",
        "attention_head_2",
        "attention_head_3",
        "attention_mixing_round_0",
        "attention_mixing_round_1",
        "attention_softmax_part_1",
        "attention_softmax_part_2",
    ]
    assert [segment.name for segment in ffn] == [
        "ffn_gw1",
        "ffn_sigmoid_1_part_1",
        "ffn_sigmoid_1_part_2",
        "ffn_gw2",
        "ffn_sigmoid_2_part_1",
        "ffn_sigmoid_2_part_2",
    ]
    assert [segment.order for segment in attention] == list(range(len(attention)))
    assert [segment.order for segment in ffn] == list(range(len(ffn)))


def test_every_segment_is_a_terminal_measurement_free_one_shot_tape():
    for segment in benchmark.build_circuits():
        tape = segment.build_tape()
        assert tape.shots.total_shots == 10
        assert len(tape.measurements) == 0
        assert len(tape.wires) <= segment.wire_count
        assert not any(isinstance(op, qml.measurements.MidMeasureMP) for op in tape.operations)


def test_whole_ffn_is_submitted_first_and_comparisons_map_to_segments():
    circuits = benchmark.build_circuits()
    assert circuits[0].name == "ffn_whole_layer"
    assert circuits[0].role == "comparison_composite"
    assert circuits[0].wire_count == 8
    assert len(circuits[0].component_names) == 6

    critical_names = {
        circuit.name
        for circuit in circuits
        if circuit.role == "critical_path_segment"
    }
    for circuit in benchmark.build_comparison_circuits():
        assert set(circuit.component_names) <= critical_names


def test_comparison_jobs_use_the_same_ordered_gate_components_as_segments():
    circuits = {circuit.name: circuit for circuit in benchmark.build_circuits()}
    for composite in benchmark.build_comparison_circuits():
        whole_names = [operation.name for operation in composite.build_tape().operations]
        segmented_names = [
            operation.name
            for component_name in composite.component_names
            for operation in circuits[component_name].build_tape().operations
        ]
        assert whole_names == segmented_names


def test_only_unambiguous_api_fields_are_accepted_as_qpu_duration():
    fields = {
        "queueTime": 123,
        "nested.executionDuration": 4.5,
    }
    assert benchmark._explicit_qpu_duration(fields) == {
        "path": "nested.executionDuration",
        "value": 4.5,
        "unit": "ms",
    }
    assert benchmark._explicit_qpu_duration({"queueTime": 123}) is None
