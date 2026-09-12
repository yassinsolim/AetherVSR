"""Synthetic apparatus checks; these tests never create real M8 evidence."""

import copy
import json
import sys

import pytest

import m8_experiment as experiment


@pytest.fixture
def plan():
    return copy.deepcopy(experiment.read_json(experiment.ROOT / experiment.PLAN))


@pytest.mark.parametrize("key,value", [
    ("arms", ["R0", "R1"]), ("pairedSeeds", [8101, 8102, 8201]),
    ("runOrder", list(reversed(experiment.RUN_ORDER))), ("smokeSeed", 8101),
])
def test_plan_rejects_changed_design(plan, key, value):
    plan[key] = value
    with pytest.raises(experiment.GuardError, match="configuration"):
        experiment.validate_plan(plan)


@pytest.mark.parametrize("key,value", [
    ("optimizerUpdates", 16200), ("batch", 16), ("device", "cpu"),
    ("diagnosticPrefixBatches", 7), ("validationDraws", 61),
    ("initializationAbsTolerance", 1e-4), ("onlineCrop", True),
])
def test_plan_rejects_changed_training(plan, key, value):
    plan["training"][key] = value
    with pytest.raises(experiment.GuardError, match="training"):
        experiment.validate_plan(plan)


def test_registered_plan_and_no_overwrite(plan, tmp_path):
    experiment.validate_plan(plan)
    path = tmp_path / "evidence.json"
    experiment.write_new(path, {"synthetic": True})
    with pytest.raises(FileExistsError):
        experiment.write_new(path, {"synthetic": False})
    assert json.loads(path.read_text()) == {"synthetic": True}


@pytest.fixture
def registered(tmp_path, monkeypatch, plan):
    commit = "c955743" + "a" * 33
    head = "b" * 40
    contents = {experiment.PLAN: json.dumps(plan).encode(),
                experiment.REGISTRATION: b"synthetic preregistration"}
    for name, payload in contents.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    calls = []

    def fake_git(root, *args):
        assert root == tmp_path
        calls.append(args)
        if args[0] == "log":
            return f"{commit}\n{head}\n".encode()
        if args[0] == "rev-parse":
            return head.encode()
        if args[0] == "merge-base":
            return b""
        if args[0] == "show":
            return contents[args[1].split(":", 1)[1]]
        raise AssertionError(args)

    monkeypatch.setattr(experiment, "git", fake_git)
    return tmp_path, commit, head, calls


def test_registration_oldest_addition_and_ancestry(registered):
    root, commit, head, calls = registered
    evidence = experiment.registration(root, root / experiment.PLAN)
    assert evidence["registrationCommit"] == commit
    assert ("merge-base", "--is-ancestor", commit, head) in calls
    assert all("--reverse" in call and "--diff-filter=A" in call
               for call in calls if call[0] == "log")


def test_registration_changed_bytes_blocked(registered):
    root, *_ = registered
    (root / experiment.REGISTRATION).write_text("changed")
    with pytest.raises(experiment.GuardError, match="bytes changed"):
        experiment.registration(root, root / experiment.PLAN)


def test_registration_nonancestor_blocked(registered, monkeypatch):
    root, *_ = registered
    original = experiment.git

    def fake_git(root, *args):
        if args[0] == "merge-base":
            raise experiment.GuardError("not an ancestor")
        return original(root, *args)

    monkeypatch.setattr(experiment, "git", fake_git)
    with pytest.raises(experiment.GuardError, match="ancestor"):
        experiment.registration(root, root / experiment.PLAN)


def synthetic_index():
    return {"trainClips": ["train"], "valClips": ["val"], "sequences": 2,
            "trainPatches": 256, "valPatches": 16,
            "index": [{"clip": "val", "tag": "val__0", "frames": 4, "patches": 16},
                      {"clip": "train", "tag": "train__0", "frames": 8, "patches": 256}]}


def test_mapping_filters_validation_and_keeps_original_position():
    mapping = experiment.sequence_mapping(synthetic_index(), 256, 16)
    assert mapping["train"][0]["indexPosition"] == 1
    assert mapping["train"][0]["startRow"] == 0
    assert mapping["validationIndexAvailable"] is True


@pytest.mark.parametrize("mutation", ["count", "split", "membership", "duplicate"])
def test_mapping_rejects_unproven_patch_identity(mutation):
    meta = synthetic_index()
    if mutation == "count":
        meta["index"][1]["patches"] = 128
    elif mutation == "split":
        meta["index"][1]["split"] = "val"
    elif mutation == "membership":
        meta["index"][1]["clip"] = "unknown"
    else:
        meta["index"].append(copy.deepcopy(meta["index"][1]))
    with pytest.raises(experiment.GuardError):
        experiment.sequence_mapping(meta, 256, 16)


@pytest.fixture
def cache(tmp_path):
    from PIL import Image

    clips = []
    counts = {"daylight": 1, "faces": 2, "lowlight": 2, "motion": 2,
              "nature": 2, "text": 3, "texture": 2, "urban": 2}
    templates = {}
    for prefix, size in (("hr", (2560, 1440)), ("lr", (1280, 720))):
        template = tmp_path / f"{prefix}.png"
        Image.new("RGB", size).save(template)
        templates[prefix] = template
    for category, count in counts.items():
        for number in range(count):
            clip = {"id": f"{category}-{number}", "category": category}
            clips.append(clip)
            for folder in ("hr", "crf18", "crf26", "crf34"):
                directory = tmp_path / "cache" / clip["id"] / folder
                directory.mkdir(parents=True)
                prefix = "hr" if folder == "hr" else "lr"
                for frame in range(1, 9):
                    (directory / f"{prefix}_{frame:04d}.png").hardlink_to(templates[prefix])
    return tmp_path, clips


def test_cache_exact_inventory(cache):
    root, clips = cache
    files, cells = experiment.cache_inventory(root, "cache", clips)
    assert len(cells) == 48
    assert len(files) == 16 * 8 * 4
    assert all(len(cell["frames"]) == 8 for cell in cells)
    assert all(record["sha256"] for record in files.values())


@pytest.mark.parametrize("defect", ["missing", "mismatch", "dimensions", "category", "clip"])
def test_cache_rejects_missing_or_changed_cells(cache, defect):
    from PIL import Image

    root, clips = cache
    frame = root / "cache/daylight-0/crf18/lr_0001.png"
    if defect == "missing":
        frame.unlink()
    elif defect == "mismatch":
        frame.rename(frame.with_name("lr_0009.png"))
    elif defect == "dimensions":
        frame.unlink()
        Image.new("RGB", (64, 64)).save(frame)
    elif defect == "category":
        clips[0]["category"] = "faces"
    else:
        clips[0]["id"] = "unknown"
    with pytest.raises(experiment.GuardError):
        experiment.cache_inventory(root, "cache", clips)


@pytest.fixture(scope="module", autouse=True)
def torch_threads():
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_initial_equivalence_cpu_preserves_rng():
    import torch

    before = torch.get_rng_state().clone()
    evidence = experiment.initial_equivalence(8099, devices=("cpu",))
    assert torch.equal(before, torch.get_rng_state())
    assert evidence["passed"] and evidence["bordersIncluded"]
    assert all(error <= 1e-5 for error in evidence["checks"]["cpu"].values())


def synthetic_report(plan, smoke=True, rung="R0"):
    from train import update_lr, validation_updates

    budget = 8 if smoke else 81180
    seed = 8099 if smoke else 8101
    updates = validation_updates(budget, 1 if smoke else 1353)
    prefix = experiment.expected_prefix(seed, 43488)
    snapshots = [budget] if smoke else plan["training"]["snapshotUpdates"]
    return {"schema": "aethervsr.m8-training-run/1", "seed": seed, "rung": rung,
            "smokeOnly": smoke, "stepBudget": budget, "optimizerSteps": budget,
            "registrationCommit": "c955743" + "a" * 33, "device": "mps",
            "corpusDigest": plan["trainerCorpusDigest"], "validationUpdates": updates,
            "validationPsnr": [20.0] * len(updates), "bestUpdate": updates[0],
            "history": [{"update": update, "psnr": 20.0, "loss": 0.1,
                         "lr": update_lr(0.002, update - 1, budget)} for update in updates],
            "diagnosticPrefix": prefix, "streamSha256": experiment.stream_hash(prefix),
            "lrAtIndices": {str(index): update_lr(0.002, index, budget)
                            for index in (0, budget // 4, budget // 2, 3 * budget // 4, budget)},
            "snapshotPaths": {str(update): f"snapshot{update}.json" for update in snapshots},
            "snapshotSha256": {str(update): "a" * 64 for update in snapshots},
            "modelPaths": {"fixedFinal": "final.json", "bestValidation": "best.json"},
            "modelSha256": {"fixedFinal": "a" * 64, "bestValidation": "a" * 64},
            "fusionMaxAbs": {str(update): 0.0 for update in snapshots},
            "trainingSecondsIncludingValidationAndSnapshots": 1.0,
            "processSecondsThroughExport": 2.0,
            "updatesPerSecondIncludingValidationAndSnapshots": float(budget)}


@pytest.mark.parametrize("smoke", [True, False])
def test_budget_lists_without_optimizer_updates(plan, smoke):
    report = synthetic_report(plan, smoke)
    experiment.validate_report(report, plan, report, report["seed"], "R0", smoke=smoke)
    assert len(report["validationUpdates"]) == (8 if smoke else 60)
    assert report["lrAtIndices"][str(report["stepBudget"])] == 0.0
    if not smoke:
        assert 16200 not in report["validationUpdates"]
        assert "16200" in report["snapshotPaths"]


@pytest.mark.parametrize("defect", ["steps", "draw", "tie", "nan", "lr", "prefix", "stream"])
def test_report_rejects_invalid_budget_or_smoke(plan, defect):
    report = synthetic_report(plan)
    if defect == "steps":
        report["optimizerSteps"] = 7
    elif defect == "draw":
        report["validationUpdates"].append(9)
    elif defect == "tie":
        report["bestUpdate"] = 8
    elif defect == "nan":
        report["history"][0]["loss"] = float("nan")
    elif defect == "lr":
        report["lrAtIndices"]["8"] = 0.001
    elif defect == "prefix":
        report["diagnosticPrefix"].pop()
    else:
        report["streamSha256"] = "b" * 64
    with pytest.raises(experiment.GuardError):
        experiment.validate_report(report, plan, report, 8099, "R0", smoke=True)


def test_paired_reports_reject_stream_or_lr_difference(plan):
    left, right = synthetic_report(plan), synthetic_report(plan, rung="R3")
    experiment.paired_reports(left, right)
    right["streamSha256"] = "b" * 64
    with pytest.raises(experiment.GuardError, match="streamSha256"):
        experiment.paired_reports(left, right)
    right["streamSha256"] = left["streamSha256"]
    right["history"][0]["lr"] = 1.0
    with pytest.raises(experiment.GuardError, match="LR"):
        experiment.paired_reports(left, right)


def test_prefix_proof_binds_bytes_and_original_index(tmp_path):
    import torch

    experiment.write_new(tmp_path / "pairs.json", synthetic_index())
    tensors = {"lr": torch.zeros((256, 3, 64, 64), dtype=torch.uint8),
               "hr": torch.zeros((256, 3, 128, 128), dtype=torch.uint8)}
    torch.save(tensors, tmp_path / "train.pt")
    prefix = experiment.expected_prefix(8099, 256)
    proof = experiment.prefix_proof(tmp_path, prefix)
    first = proof["batches"][0]["patches"][0]
    assert first["indexPosition"] == 1
    assert first["cachedBounds"] == {"lr": [0, 0, 64, 64], "hr": [0, 0, 128, 128]}
    assert first["originalCropCoordinates"] is None
    assert "unavailable" in first["originalCropCoordinatesAvailability"]
    tensors["lr"][first["row"], 0, 0, 0] = 1
    torch.save(tensors, tmp_path / "train.pt")
    assert experiment.prefix_proof(tmp_path, prefix)["digest"] != proof["digest"]


@pytest.mark.parametrize("rung", ["R0", "R3"])
def test_reload_checks_respective_state_snapshots(tmp_path, plan, rung):
    import torch
    from aethersr import export_weights
    from train import build_model, write_model

    report = synthetic_report(plan, rung=rung)
    model = build_model(rung, 16, 2, 8099)
    initial = copy.deepcopy(model.state_dict())
    best = copy.deepcopy(initial)
    with torch.no_grad():
        model.head.bias.add_(0.01)
    final = copy.deepcopy(model.state_dict())
    report["statePath"] = "states.pt"
    optimizer = {"param_groups": [{"params": [0]}], "state": {0: {"step": torch.tensor(8.0)}}}
    torch.save({"initial": initial, "best": best, "fixedFinal": final,
                "snapshots": {8: final}, "optimizer": optimizer}, tmp_path / "states.pt")
    for key, state, update, kind in (("bestValidation", best, 1, "best-validation"),
                                      ("fixedFinal", final, 8, "fixed-final")):
        model.load_state_dict(state)
        fused = model.fuse() if hasattr(model, "fuse") else model
        training = {name: report[name] for name in ("optimizerSteps", "stepBudget", "seed", "rung",
                                                   "smokeOnly", "corpusDigest", "registrationCommit")}
        training.update(checkpointUpdate=update, checkpointKind=kind, batch=32, trainPatches=43488,
                patchesSeen=256, matchedEffectiveInitialization=True)
        payload = {"parameters": 6291, "features": 16, "depth": 2, "scale": 2,
                   "weights": export_weights(fused), "training": training}
        report["modelSha256"][key] = write_model(str(tmp_path / report["modelPaths"][key]), payload)
    report["snapshotPaths"]["8"] = report["modelPaths"]["fixedFinal"]
    report["snapshotSha256"]["8"] = report["modelSha256"]["fixedFinal"]
    evidence = experiment.verify_exports(tmp_path, report, devices=("cpu",))
    assert evidence["passed"] and evidence["reloadMaxAbs"]
    torch.save({"initial": initial, "best": final, "fixedFinal": final,
                "snapshots": {8: final}, "optimizer": optimizer}, tmp_path / "states.pt")
    with pytest.raises(experiment.GuardError, match="reloaded weight"):
        experiment.verify_exports(tmp_path, report, devices=("cpu",))
    optimizer["state"][0]["step"] = torch.tensor(7.0)
    torch.save({"initial": initial, "best": best, "fixedFinal": final,
                "snapshots": {8: final}, "optimizer": optimizer}, tmp_path / "states.pt")
    with pytest.raises(experiment.GuardError, match="update budget"):
        experiment.verify_exports(tmp_path, report, devices=("cpu",))


def test_dirty_tracked_worktree_blocks_training(monkeypatch, tmp_path):
    calls = []

    def fake_git(root, *args):
        calls.append(args)
        return b" M tools/train.py\n"

    monkeypatch.setattr(experiment, "git", fake_git)
    with pytest.raises(experiment.GuardError, match="clean tracked worktree"):
        experiment.clean_training_sources(tmp_path, {"tools/train.py": "synthetic"})
    assert len(calls) == 1


def test_untracked_implementation_is_not_in_head(monkeypatch, tmp_path):
    path = tmp_path / "tools/m8_experiment.py"
    path.parent.mkdir()
    path.write_text("synthetic uncommitted implementation")

    def fake_git(root, *args):
        if args[0] == "status":
            return b""
        raise experiment.GuardError("not tracked in HEAD")

    monkeypatch.setattr(experiment, "git", fake_git)
    with pytest.raises(experiment.GuardError, match="not tracked"):
        experiment.clean_training_sources(tmp_path, {"tools/m8_experiment.py": "synthetic"})


def test_concurrent_process_guard_excludes_only_self_ancestors():
    listing = "1 0 launchd\n10 1 shell\n20 10 python tools/m8_experiment.py --train\n"
    assert experiment.no_concurrent_training(listing, own_pid=20)["competingProcesses"] == 0
    for script in ("train.py", "m8_experiment.py", "evaluate.py", "m6-validate.py"):
        with pytest.raises(experiment.GuardError, match="concurrent"):
            experiment.no_concurrent_training(listing + f"30 10 python tools/{script}\n", own_pid=20)
    with pytest.raises(experiment.GuardError, match="absent"):
        experiment.no_concurrent_training("1 0 launchd", own_pid=20)


def test_process_inspection_fails_closed(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(experiment.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="", stderr="denied"))
    with pytest.raises(experiment.GuardError, match="cannot inspect"):
        experiment.no_concurrent_training()


def test_execution_lock_and_output_ownership(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "git", lambda *args: b"")
    with experiment.execution_lock(tmp_path):
        with pytest.raises(experiment.GuardError, match="lock exists"):
            with experiment.execution_lock(tmp_path):
                pytest.fail("lock should not be acquired")
    session = {"synthetic": True}
    experiment.run_directory(tmp_path, session, smoke=False, run_index=0)
    assert experiment.run_directory(tmp_path, session, smoke=False, run_index=1).is_dir()
    with pytest.raises(experiment.GuardError, match="refusing overwrite"):
        experiment.run_directory(tmp_path, session, smoke=False, run_index=None)
    with pytest.raises(experiment.GuardError, match="not owned"):
        experiment.run_directory(tmp_path, {"synthetic": False}, smoke=False, run_index=1)
    experiment.run_directory(tmp_path, session, smoke=True, run_index=None)
    with pytest.raises(experiment.GuardError, match="refusing overwrite"):
        experiment.run_directory(tmp_path, session, smoke=True, run_index=None)


@pytest.mark.parametrize("argv", [[], ["--prepare", "--smoke"], ["--train", "--smoke"],
                                  ["--prepare", "--run-index", "0"], ["--train", "--run-index", "6"]])
def test_cli_action_and_index_restrictions(argv, tmp_path):
    with pytest.raises(SystemExit) as error:
        experiment.main(argv, root=tmp_path)
    assert error.value.code == 2


def test_registered_command_exact_budget_and_smoke(plan, tmp_path):
    registered = {"registrationCommit": "c955743" + "a" * 33}
    for smoke, seed, rung, budget, epochs in ((False, 8102, "R3", "81180", "60"),
                                             (True, 8099, "R0", "8", "1")):
        command = experiment.run_command(plan, registered, seed, rung, smoke=smoke, root=tmp_path)
        for option, value in {"--max-steps": budget, "--epochs": epochs, "--batch": "32",
                              "--lr": "0.002", "--patch": "128", "--device": "mps",
                              "--pairs": plan["pairsDirectory"], "--seed": str(seed),
                              "--m8-registration": registered["registrationCommit"]}.items():
            assert command[command.index(option) + 1] == value
        assert ("--m8-smoke" in command) == smoke
    with pytest.raises(experiment.GuardError, match="unregistered"):
        experiment.run_command(plan, registered, 8201, "R3", smoke=False, root=tmp_path)


def test_logs_are_visible_and_preserved_without_training(tmp_path, capsys):
    command = [sys.executable, "-c", "import sys; print('synthetic stdout'); print('synthetic stderr', file=sys.stderr)"]
    out, err = tmp_path / "stdout.log", tmp_path / "stderr.log"
    code, duration = experiment.run_logged(command, tmp_path, out, err)
    assert code == 0 and duration > 0
    captured = capsys.readouterr()
    assert "synthetic stdout" in captured.out and "synthetic stderr" in captured.err
    assert out.read_text() == "synthetic stdout\n" and err.read_text() == "synthetic stderr\n"
    with pytest.raises(FileExistsError):
        experiment.run_logged(command, tmp_path, out, err)


@pytest.fixture
def execution_context(tmp_path, monkeypatch, plan):
    registered = {"registrationCommit": "c955743" + "a" * 33, "executionCommit": "b" * 40}
    inputs = {"files": {}, "digest": "d" * 64}
    code = {"synthetic-test-code": "e" * 64}
    (tmp_path / "models/m8-smoke").mkdir(parents=True)
    monkeypatch.setattr(experiment, "code_hashes", lambda root: code)
    monkeypatch.setattr(experiment, "git", lambda root, *args: registered["executionCommit"].encode())
    monkeypatch.setattr(experiment, "no_concurrent_training", lambda: {"synthetic": True})
    return tmp_path, plan, registered, inputs, code, {"synthetic": True}


@pytest.mark.parametrize("failure", ["exit", "interrupted", "verification"])
def test_failure_retains_start_logs_and_incomplete_record(execution_context, monkeypatch, failure):
    root, plan, registered, inputs, code, machine = execution_context
    attempts = []

    def fake_run(command, root, stdout, stderr):
        attempts.append(command)
        stdout.write_text("synthetic partial progress\n")
        stderr.write_text("synthetic failure\n")
        if failure == "interrupted":
            raise KeyboardInterrupt()
        if failure == "verification":
            experiment.write_new(root / "models/m8-smoke/R0-seed8099-run.json", {})
            return 0, 0.125
        return 17, 0.125

    monkeypatch.setattr(experiment, "run_logged", fake_run)
    with pytest.raises((experiment.GuardError, KeyboardInterrupt)):
        experiment.execute_run(root, plan, registered, inputs, code, machine, 8099, "R0", smoke=True, run_index=None)
    start = root / "models/m8-smoke/R0-seed8099-start.json"
    finish = root / "models/m8-smoke/R0-seed8099-execution.json"
    assert start.is_file() and finish.is_file()
    assert experiment.read_json(start)["completed"] is False
    evidence = experiment.read_json(finish)
    assert evidence["completed"] is False and evidence["error"]
    assert evidence["executionWallSeconds"] > 0
    with pytest.raises(experiment.GuardError, match="overwrite/restart"):
        experiment.execute_run(root, plan, registered, inputs, code, machine, 8099, "R0", smoke=True, run_index=None)
    assert len(attempts) == 1


def test_changed_frozen_bytes_blocked(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"synthetic frozen input")
    inputs = {"files": {"input.bin": experiment.file_record(tmp_path, "input.bin")}}
    experiment.check_frozen_files(tmp_path, inputs)
    path.write_bytes(b"changed")
    with pytest.raises(experiment.GuardError, match="bytes changed"):
        experiment.check_frozen_files(tmp_path, inputs)


def test_uncommitted_freeze_or_smoke_is_blocked(tmp_path, monkeypatch, plan):
    def uncommitted(*args):
        raise experiment.GuardError("evidence missing from HEAD")

    monkeypatch.setattr(experiment, "committed_bytes", uncommitted)
    with pytest.raises(experiment.GuardError, match="HEAD"):
        experiment.prepared_freeze(tmp_path, {}, {}, committed=True)
    with pytest.raises(experiment.GuardError, match="HEAD"):
        experiment.passing_smoke(tmp_path, plan, {}, {}, {})


def test_smoke_must_be_passing_and_code_identical(tmp_path, monkeypatch, plan):
    monkeypatch.setattr(experiment, "committed_bytes", lambda *args: None)
    experiment.write_new(tmp_path / experiment.SMOKE_REPORT,
                         {"schema": "aethervsr.m8-smoke/1", "passed": False, "smokeOnly": True})
    with pytest.raises(experiment.GuardError, match="passing smoke"):
        experiment.passing_smoke(tmp_path, plan, {}, {}, {})
    (tmp_path / experiment.INPUT_FREEZE).write_text("synthetic freeze")
    smoke = {"schema": "aethervsr.m8-smoke/1", "passed": True, "smokeOnly": True,
             "registrationCommit": "synthetic", "inputDigest": "synthetic",
             "inputFreezeSha256": experiment.sha256(tmp_path / experiment.INPUT_FREEZE),
             "codeSha256": {"old": "synthetic"}}
    (tmp_path / experiment.SMOKE_REPORT).write_text(json.dumps(smoke))
    with pytest.raises(experiment.GuardError, match="implementation changed"):
        experiment.passing_smoke(tmp_path, plan, {"digest": "synthetic"}, smoke, {"new": "synthetic"})


@pytest.fixture
def cli_context(tmp_path, monkeypatch, plan):
    registered = {"registrationCommit": "c955743" + "a" * 33, "executionCommit": "b" * 40}
    inputs = {"files": {}, "digest": "d" * 64}
    code = {"synthetic": "test-only"}
    experiment.write_new(tmp_path / experiment.PLAN, plan)
    monkeypatch.setattr(experiment, "registration", lambda *args: registered)
    monkeypatch.setattr(experiment, "code_hashes", lambda *args: code)
    monkeypatch.setattr(experiment, "freeze_inputs", lambda *args: inputs)
    monkeypatch.setattr(experiment, "machine_toolchain", lambda: {"mpsAvailable": True, "synthetic": True})

    def fake_git(root, *args):
        if args[0] == "show":
            return (root / args[1].split(":", 1)[1]).read_bytes()
        return b""

    monkeypatch.setattr(experiment, "git", fake_git)
    monkeypatch.setattr(experiment, "clean_training_sources", lambda *args: None)
    monkeypatch.setattr(experiment, "passing_smoke", lambda *args: {"synthetic": True})
    monkeypatch.setattr(experiment, "no_concurrent_training", lambda: {"synthetic": True})
    return tmp_path, registered, inputs, code


def test_prepare_allows_uncommitted_and_refuses_overwrite(cli_context, monkeypatch):
    root, *_ = cli_context
    monkeypatch.setattr(experiment, "clean_training_sources", lambda *args: pytest.fail("prepare must allow dirty code"))
    monkeypatch.setattr(experiment, "execute_run", lambda *args, **kwargs: pytest.fail("prepare must never train"))
    assert experiment.main(["--prepare"], root=root) == 0
    evidence = experiment.read_json(root / experiment.INPUT_FREEZE)
    assert evidence["schema"] == "aethervsr.m8-input-freeze/1" and evidence["toolchain"]["synthetic"]
    assert experiment.main(["--prepare"], root=root) == 1


def test_cli_train_dirty_guard_precedes_input_reads(cli_context, monkeypatch):
    root, *_ = cli_context

    def dirty(*args):
        raise experiment.GuardError("dirty tracked worktree")

    monkeypatch.setattr(experiment, "clean_training_sources", dirty)
    monkeypatch.setattr(experiment, "freeze_inputs", lambda *args: pytest.fail("must reject dirty code first"))
    assert experiment.main(["--train"], root=root) == 1
    assert not (root / "models").exists()


@pytest.fixture
def serial_cli(cli_context, monkeypatch, plan):
    root, registered, inputs, code = cli_context
    assert experiment.main(["--prepare"], root=root) == 0
    experiment.write_new(root / experiment.SMOKE_REPORT, {"synthetic": True})
    events, stored = [], {}

    def execute(root, plan, registered, inputs, code, machine, seed, rung, *, smoke, run_index):
        assert not smoke
        events.append((run_index, seed, rung))
        report = synthetic_report(plan, smoke=False, rung=rung)
        report["seed"] = seed
        execution = {"startedAt": f"2026-09-11T12:{run_index:02d}:00+00:00",
                     "finishedAt": f"2026-09-11T12:{run_index:02d}:01+00:00", "subprocessWallSeconds": 1.0}
        stored[run_index] = (report, execution)
        return report, execution

    def load(root, plan, registered, inputs, code, count):
        experiment.require(all(index in stored for index in range(count)), "missing preceding run")
        return ([stored[index][0] for index in range(count)], [stored[index][1] for index in range(count)])

    monkeypatch.setattr(experiment, "execute_run", execute)
    monkeypatch.setattr(experiment, "load_runs", load)
    return root, events, stored


def test_cli_six_runs_are_serial_and_merge_trainer_reports(serial_cli):
    root, events, _ = serial_cli
    assert experiment.main(["--train"], root=root) == 0
    assert events == [(index, seed, rung) for index, (seed, rung) in enumerate(experiment.RUN_ORDER)]
    evidence = experiment.read_json(root / experiment.RUNS_REPORT)
    assert evidence["schema"] == "aethervsr.m8-runs/1"
    assert len(evidence["runs"]) == len(evidence["chronology"]) == 6
    assert all(run["optimizerSteps"] == 81180 and "status" not in run for run in evidence["runs"])
    assert experiment.main(["--train"], root=root) == 1
    assert len(events) == 6


def test_cli_controlled_run_index_cannot_skip_or_restart(serial_cli):
    root, events, _ = serial_cli
    assert experiment.main(["--train", "--run-index", "1"], root=root) == 1
    assert experiment.main(["--train", "--run-index", "0"], root=root) == 0
    assert not (root / experiment.RUNS_REPORT).exists()
    assert experiment.main(["--train", "--run-index", "1"], root=root) == 0
    assert experiment.main(["--train", "--run-index", "3"], root=root) == 1
    assert experiment.main(["--train", "--run-index", "0"], root=root) == 1
    assert events == [(0, 8101, "R0"), (1, 8101, "R3")]


@pytest.mark.parametrize("failure", [False, True])
def test_smoke_initial_gate_precedes_both_launches(cli_context, monkeypatch, plan, failure):
    root, *_ = cli_context
    assert experiment.main(["--prepare"], root=root) == 0
    events = []

    def initial(seed):
        events.append(("initial", seed))
        experiment.require(not failure, "synthetic initialization failure")
        return {"synthetic": True, "passed": True}

    def execute(root, plan, registered, inputs, code, machine, seed, rung, *, smoke, run_index):
        events.append((rung, seed))
        assert smoke is True and run_index is None and seed == 8099
        command = experiment.run_command(plan, registered, seed, rung, smoke=True, root=root)
        assert command[command.index("--max-steps") + 1] == "8"
        assert command[command.index("--batch") + 1] == "32"
        order = 0 if rung == "R0" else 1
        return synthetic_report(plan, rung=rung), {
            "startedAt": f"2026-09-11T12:0{order}:00+00:00",
            "finishedAt": f"2026-09-11T12:0{order}:01+00:00", "subprocessWallSeconds": 1.0}

    monkeypatch.setattr(experiment, "initial_equivalence", initial)
    monkeypatch.setattr(experiment, "execute_run", execute)
    monkeypatch.setattr(experiment, "prefix_proof", lambda *args: {"synthetic": True})
    monkeypatch.setattr(experiment, "clean_training_sources", lambda *args: pytest.fail("smoke allows uncommitted code"))
    assert experiment.main(["--smoke"], root=root) == (1 if failure else 0)
    if failure:
        assert events == [("initial", 8099)]
        assert not (root / experiment.SMOKE_REPORT).exists()
    else:
        assert events == [("initial", 8099), ("R0", 8099), ("R3", 8099)]
        smoke = experiment.read_json(root / experiment.SMOKE_REPORT)
        assert smoke["schema"] == "aethervsr.m8-smoke/1" and smoke["passed"] and smoke["smokeOnly"]
        assert len(smoke["runs"]) == 2 and smoke["prefixProof"]["synthetic"]
        assert experiment.main(["--smoke"], root=root) == 1


@pytest.mark.parametrize("defect", [None, "dimensions", "dtype", "count"])
def test_corpus_geometry_and_counts_use_mmap_before_launch(tmp_path, monkeypatch, plan, defect):
    import torch

    meta = {"hrSize": [2560, 1440], "lrSize": [1280, 720], "fps": 24, "sequenceFrames": 24,
            "seed": 20260906, "structure": "gop", "preset": "medium",
            "crfDistribution": "uniform [18,36] per sequence",
            "structureSpec": {"x264params": "keyint=48:min-keyint=24:scenecut=40:bframes=3:ref=3"},
            "trainClips": [f"train{index}" for index in range(151)],
            "valClips": [f"val{index}" for index in range(16)],
            "trainPatches": 43488, "valPatches": 3072, "sequences": 453,
            "index": [{"clip": f"train{index}", "tag": f"train{index}__{sequence}",
                       "frames": 24, "patches": 96, "crf": 26}
                      for index in range(151) for sequence in range(3)]}
    calls = []

    def load(path, *, mmap, weights_only, map_location):
        assert mmap and weights_only and map_location == "cpu"
        calls.append(path.name)
        count = 43488 if path.name == "train.pt" else 3072
        if defect == "count":
            count -= 1
        return {"lr": torch.empty((count, 3, 64, 64), device="meta", dtype=torch.uint8),
                "hr": torch.empty((count, 3, 128, 64 if defect == "dimensions" else 128), device="meta",
                                  dtype=torch.float32 if defect == "dtype" else torch.uint8)}

    monkeypatch.setattr(experiment, "read_json", lambda *args: meta)
    monkeypatch.setattr(torch, "load", load)
    if defect:
        with pytest.raises(experiment.GuardError, match="expected uint8"):
            experiment.inspect_corpus(tmp_path, plan)
    else:
        _, observed = experiment.inspect_corpus(tmp_path, plan)
        assert calls == ["train.pt", "val.pt"]
        assert observed["trainSequences"] == 453
        assert observed["validationIndexAvailable"] is False


@pytest.mark.parametrize("defect", [None, "pairs-hash", "manifest-hash", "combined-digest"])
def test_input_freeze_hashes_actual_bytes_and_trainer_composition(tmp_path, monkeypatch, plan, defect):
    import hashlib

    meta = {"index": [{"clip": "synthetic", "patches": 1}], "trainClips": ["train"], "valClips": ["val"]}
    monkeypatch.setattr(experiment, "committed_bytes", lambda *args: None)
    monkeypatch.setattr(experiment, "inspect_corpus", lambda *args: (meta, {"synthetic": True}))
    monkeypatch.setattr(experiment, "cache_inventory", lambda *args: ({}, [{"synthetic": True}]))
    for key, payload in (("trainingManifest", {"clips": [{"id": "train"}]}),
                         ("validationManifest", {"clips": [{"id": "val"}]}),
                         ("productionModel", {"synthetic": True})):
        experiment.write_new(tmp_path / plan[key], payload)
        plan[f"{key}Sha256"] = experiment.sha256(tmp_path / plan[key])
    pairs = tmp_path / plan["pairsDirectory"]
    pairs.mkdir(parents=True)
    experiment.write_new(pairs / "pairs.json", meta)
    (pairs / "train.pt").write_bytes(b"synthetic train bytes, not a tensor")
    (pairs / "val.pt").write_bytes(b"synthetic val bytes, not a tensor")
    digest = hashlib.sha256(json.dumps(meta["index"], sort_keys=True).encode()
                            + (pairs / "train.pt").read_bytes() + (pairs / "val.pt").read_bytes()).hexdigest()
    plan["trainerCorpusDigest"] = digest
    frozen = {"pairsDirectory": plan["pairsDirectory"], "trainerCorpusDigest": digest,
              "trainClips": 151, "validationClips": 16, "trainSequences": 453,
              "validationSequences": 32, "trainPatches": 43488, "validationPatches": 3072,
              "files": {name: experiment.file_record(tmp_path, f"{plan['pairsDirectory']}/{name}")
                        for name in ("pairs.json", "train.pt", "val.pt")}}
    experiment.write_new(tmp_path / plan["pairsFreeze"], frozen)
    experiment.write_new(tmp_path / plan["validationFreeze"], {
        "manifestSha256": plan["validationManifestSha256"],
        "trainingManifestSha256": plan["trainingManifestSha256"],
        "audit": {"overlaps": [], "schemaProblems": []}})
    if defect == "pairs-hash":
        (pairs / "train.pt").write_bytes(b"changed")
    elif defect == "manifest-hash":
        (tmp_path / plan["trainingManifest"]).write_bytes(b"changed")
    elif defect == "combined-digest":
        plan["trainerCorpusDigest"] = "0" * 64
    if defect:
        with pytest.raises(experiment.GuardError, match="mismatch"):
            experiment.freeze_inputs(tmp_path, plan, {"registrationCommit": "synthetic"})
    else:
        evidence = experiment.freeze_inputs(tmp_path, plan, {"registrationCommit": "synthetic"})
        assert evidence["corpusDigest"] == digest
        assert evidence["digest"] == experiment.aggregate({key: value for key, value in evidence.items() if key != "digest"})


def test_registration_requires_same_addition_commit(registered, monkeypatch):
    root, _, head, _ = registered
    original = experiment.git

    def fake_git(root, *args):
        if args[0] == "log" and args[-1] == experiment.PLAN:
            return head.encode()
        return original(root, *args)

    monkeypatch.setattr(experiment, "git", fake_git)
    with pytest.raises(experiment.GuardError, match="not added together"):
        experiment.registration(root, root / experiment.PLAN)


def test_machine_inventory_does_not_request_private_identity(monkeypatch):
    from types import SimpleNamespace

    calls = []
    outputs = {("sysctl", "-n", "machdep.cpu.brand_string"): "Apple synthetic test CPU",
               ("sysctl", "-n", "hw.memsize"): "123456",
               ("sw_vers",): "ProductName: synthetic macOS\nBuildVersion: synthetic"}

    def run(args, **kwargs):
        calls.append(tuple(args))
        return SimpleNamespace(returncode=0, stdout=outputs[tuple(args)])

    monkeypatch.setattr(experiment.subprocess, "run", run)
    evidence = experiment.machine_toolchain()
    assert set(calls) == set(outputs)
    assert evidence["memoryBytes"] == 123456 and evidence["cpu"] == "Apple synthetic test CPU"
    assert not {"username", "serial", "hostname"} & evidence.keys()