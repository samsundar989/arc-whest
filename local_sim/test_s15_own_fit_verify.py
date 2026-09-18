"""Synthetic only: no P12 truth or active own-fit output is opened."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import competition_score as cs
import lifecycle_eval as le
import s14_own_fit_controls as gen
import s15_own_fit_driver as driver
import s15_own_fit_verify as verify


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(directory):
    directory.mkdir()
    result = driver.plan()
    result.update({
        "truth_sha256": driver.TRUTH_SHA,
        "base_source_sha256": sha(gen.BASE),
        "driver_sha256": sha(driver.__file__),
        "generator_sha256": sha(gen.__file__),
        "dependency_sha256": {name: sha(verify.ROOT / "local_sim" / name)
                              for name in ("competition_score.py", "lifecycle_eval.py",
                                           "mint_provenance.py", "official_config.py",
                                           "truth_reads.py")},
        "status": "complete", "evaluations": [], "evaluations_used": 19,
        "local_seconds_caps": "disabled for P12 DESIGN; not release evidence",
        "numerical_environment": {
            "python":"synthetic Python 3.10", "numpy":"synthetic NumPy 2.2",
            "platform":"synthetic Windows", "executable":"synthetic python path",
            "threads": {key:None for key in
                        ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                         "MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")}},
    })
    spec = driver.design_spec()
    split_truth = {"fit": hashlib.sha256(b"synthetic first twelve truth rows").hexdigest(),
                   "validation": hashlib.sha256(b"synthetic last twelve truth rows").hexdigest()}

    def evaluate(split, x, index):
        name = f"eval_{index:02d}_{split}"
        source = directory / (name + ".py")
        source_sha = gen.emit(source, **dict(zip(driver.NAMES,x)))
        ids = driver.SEEDS[:12] if split == "fit" else driver.SEEDS[12:]
        rows = [le.EvidenceRow(index=i, input_seed=seed, estimator_seed=seed+100,
                candidate=str(source), candidate_sha256=source_sha,
                truth_sha256=split_truth[split], width=1024, depth=16,
                budget=2**41, flops_used=(.2 + .005*x[0]) * 2**41,
                n_ops=1234, mse=(2e-8 + 1e-9*sum((v-d)**2 for v,d in zip(x,driver.X0))
                              + i*1e-10), zero_prediction_mse=.1,
                gt_noise=2e-9+i*1e-11, residual_wall_time_s=.01,
                wall_time_s=1.) for i,seed in enumerate(ids)]
        row_path = directory / (name + ".jsonl")
        row_path.write_text("".join(json.dumps(dataclasses.asdict(r),sort_keys=True)+"\n"
                                    for r in rows), encoding="utf-8")
        obs=cs.score_suite([r.to_row() for r in rows],lam=0.).objective
        official_rows=[dataclasses.replace(r.to_row(),mse=r.mse-r.gt_noise+
                       r.gt_noise*verify.LOCAL_SAMPLES/verify.OFFICIAL_SAMPLES)
                       for r in rows]
        official=cs.score_suite(official_rows,lam=0.).objective
        rec={"index":index,"split":split,"parameters":dict(zip(driver.NAMES,x)),
             "candidate":source.name,"sha256":source_sha,"rows":row_path.name,
             "rows_hash":le.rows_hash(rows),"rows_file_sha256":sha(row_path),
             "provenance":{"candidate_sha256":source_sha,
                           "truth_sha256":split_truth[split],"n_mlps":12,
                           "spec":dataclasses.asdict(spec),
                           "round_tag":spec.round_tag,"lifecycle":"grader",
                           "weights_mode":le.DEFAULT_WEIGHTS_MODE,
                           "setup_called":True,"rows_hash":le.rows_hash(rows)},
             "n":12,"n_failed":0,
             "observed_score":obs,"official_precision_score":official,
             "mean_flops":sum(r.flops_used for r in rows)/12,
             "gt_noise_mean":sum(r.gt_noise for r in rows)/12}
        if index>0 and split=="fit":
            rec["penalized_fit_objective"]=driver.objective(
                official,result["frozen_default_fit_score"],x)
        result["evaluations"].append(rec)
        if index==0: result["frozen_default_fit_score"]=official
        return official

    selected,base,held_base,held,_=driver.coordinate_fit(evaluate)
    result.update(selected_parameters=dict(zip(driver.NAMES,selected)),
                  heldout_default_official_score=held_base,
                  heldout_selected_official_score=held)
    save(directory,result)


def save(directory, result):
    for name in ("result.json","manifest.json"):
        (directory/name).write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")


def refuses(directory, mutation, needle):
    copied=directory.parent/(directory.name+"_"+needle.replace(" ","_")+"_"+
                             str(len(list(directory.parent.iterdir()))))
    shutil.copytree(directory,copied)
    result=json.loads((copied/"result.json").read_text(encoding="utf-8"))
    mutation(copied,result)
    save(copied,result)
    try: verify.verify(copied)
    except ValueError as exc:
        assert needle in str(exc), str(exc)
    else: raise AssertionError(f"did not refuse {needle}")


def main():
    with tempfile.TemporaryDirectory(prefix="own_fit_verify_test_") as td:
        directory=Path(td)/"good"
        fixture(directory)
        summary=verify.verify(directory)
        assert summary["evaluations"]==19 and summary["promotion_decision"] is None
        assert summary["heldout"]["ordered_seeds"]==list(driver.SEEDS[12:])
        assert math.isfinite(summary["heldout"]["official_precision"]["difference"])
        assert summary["numerical_environment"]["platform"]=="synthetic Windows"
        refuses(directory,lambda p,r:r.update(local_seconds_caps="official"),
                "local seconds cap declaration mismatch")
        refuses(directory,lambda p,r:r["numerical_environment"].pop("numpy"),
                "missing numerical environment identity")
        refuses(directory,lambda p,r:r["evaluations"].pop(),"missing or extra evaluations")
        refuses(directory,lambda p,r:r["evaluations"][1].update(index=0),"evaluation indices")
        refuses(directory,lambda p,r:r["evaluations"][17].update(split="fit"),"split leakage")
        refuses(directory,lambda p,r:r["evaluations"][0].update(rows_hash="0"*64),"rows_hash mismatch")
        refuses(directory,lambda p,r:r["evaluations"][0].update(sha256="0"*64),"source or row byte SHA")
        refuses(directory,lambda p,r:r["evaluations"][0]["parameters"].pop("s_lam"),
                "parameter names/order mismatch")
        refuses(directory,lambda p,r:r["evaluations"][0]["parameters"].update(extra=1.),
                "parameter names/order mismatch")
        refuses(directory,lambda p,r:r["evaluations"][0]["parameters"].update(s_lam=1.15),
                "candidate differs from maintained generator")
        refuses(directory,lambda p,r:r["evaluations"][0].update(official_precision_score=1.),"official mismatch")
        refuses(directory,lambda p,r:r.update(selected_parameters=dict.fromkeys(driver.NAMES,1.3)),"selected parameters differ")
        refuses(directory,lambda p,r:r["evaluations"][17]["provenance"].update(
            truth_sha256=hashlib.sha256(b"wrong split truth").hexdigest()),
            "row provenance mismatch")
        refuses(directory,lambda p,r:r["evaluations"][0]["provenance"].update(
            spec={"width":1024}),"lifecycle spec mismatch")
        def forged_row(p,r):
            rec=r["evaluations"][0]; path=p/rec["rows"]
            lines=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            lines[0]["input_seed"]+=100
            path.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in lines),encoding="utf-8")
            rec["rows_file_sha256"]=sha(path)
            rec["rows_hash"]=le.rows_hash([le.EvidenceRow(**x) for x in lines])
            rec["provenance"]["rows_hash"]=rec["rows_hash"]
        refuses(directory,forged_row,"ordered network identities mismatch")
        def forged_failure(p,r):
            rec=r["evaluations"][0]; path=p/rec["rows"]
            lines=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            lines[0]["error_code"]="TEST_FAILURE"
            path.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in lines),encoding="utf-8")
            rec["rows_file_sha256"]=sha(path)
            rec["rows_hash"]=le.rows_hash([le.EvidenceRow(**x) for x in lines])
            rec["provenance"]["rows_hash"]=rec["rows_hash"]
        refuses(directory,forged_failure,"noncanonical failure fallback")
        for field, replacement in (("estimator_seed", 99999999),
                                   ("gt_noise", 9e-9),
                                   ("zero_prediction_mse", .2)):
            def mismatched_network(p,r,field=field,replacement=replacement):
                rec=r["evaluations"][1]; path=p/rec["rows"]
                lines=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
                lines[0][field]=replacement
                path.write_text("".join(json.dumps(x,sort_keys=True)+"\n"
                                        for x in lines),encoding="utf-8")
                rec["rows_file_sha256"]=sha(path)
                rec["rows_hash"]=le.rows_hash([le.EvidenceRow(**x) for x in lines])
                rec["provenance"]["rows_hash"]=rec["rows_hash"]
            refuses(directory,mismatched_network,"cross-evaluation network identity mismatch")
        def forged_source_pair(p,r):
            rec=r["evaluations"][0]
            source=p/rec["candidate"]
            source.write_text(source.read_text(encoding="utf-8")+"\n# forged\n",encoding="utf-8")
            rec["sha256"]=sha(source)
            rec["provenance"]["candidate_sha256"]=rec["sha256"]
            path=p/rec["rows"]
            lines=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            for row in lines: row["candidate_sha256"]=rec["sha256"]
            path.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in lines),encoding="utf-8")
            rec["rows_file_sha256"]=sha(path)
            rec["rows_hash"]=le.rows_hash([le.EvidenceRow(**x) for x in lines])
            rec["provenance"]["rows_hash"]=rec["rows_hash"]
        refuses(directory,forged_source_pair,"candidate differs from maintained generator")
    print("own-fit verifier synthetic regressions passed: valid fixture + 20 refusals")


if __name__=="__main__":
    main()
