"""Calibration records, review sampling, and human-approved rubric versions."""
import threading
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field, model_validator

from studio import storage as db
from studio.authoring import AuthorError
from studio.calibration import calibration, correction_evidence, evaluation_task, propose_rubric, review_queue
from studio.judges import judge_connection, judge_connections
from studio.schemas import JudgeSpec, StrictModel

router = APIRouter(prefix="/api")
proposal_lock = threading.Lock()


class OutcomeLabel(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    status: Literal["pass", "fail", "unknown"]
    note: str = Field(default="", max_length=2000)


class Labels(StrictModel):
    outcomes: list[OutcomeLabel] = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def distinct(self):
        if len({outcome.id for outcome in self.outcomes}) != len(self.outcomes):
            raise ValueError("Label each outcome once per submission")
        return self


class AcceptProposal(StrictModel):
    rubric: str | None = Field(default=None, min_length=10, max_length=16000)


@router.get("/judge-lab/status")
def lab_status():
    return {"providers": judge_connections(), "codex_model": "gpt-6-astra"}


@router.get("/judges/{id}/calibration")
def judge_calibration(id: str):
    return calibration(id)


@router.get("/judges/{id}/review-queue")
def judge_review_queue(id: str, limit: int = Query(default=12, ge=1, le=40)):
    return review_queue(id, limit)


@router.post("/evaluations/{id}/labels", status_code=201)
def label_evaluation(id: str, spec: Labels):
    evaluation = db.get("evaluation", id)
    task = evaluation_task(evaluation)
    identifiers = {item["id"] for item in task["requirements"]}
    if any(item.id not in identifiers for item in spec.outcomes):
        raise HTTPException(422, "Labels must refer to outcomes in this evaluation's task snapshot.")
    judge = evaluation["judge"]
    # Append corrections; never rewrite the original judge verdict or an older label.
    return db.save("outcome_label", {"evaluation_id": id, "episode_id": evaluation["episode_id"],
        "judge_id": judge["id"], "judge_version": judge["version"], "judge": judge,
        "task": task, **spec.model_dump()})


@router.get("/judges/{id}/rubric-proposals")
def rubric_proposals(id: str):
    db.get("judge", id)
    return [item for item in db.all_records("rubric_proposal") if item["judge_id"] == id]


@router.post("/judges/{id}/rubric-proposals", status_code=201)
def create_proposal(id: str):
    judge = db.get("judge", id)
    if judge["kind"] != "vlm":
        raise HTTPException(409, "The state baseline does not interpret rubric prose. Choose a VLM judge to propose an inspection rubric.")
    connection = judge_connection(judge)
    if not connection["configured"]:
        raise HTTPException(409, connection["message"])
    evidence = correction_evidence(id)
    if not any(item["label"]["status"] != item["prediction"]["status"] for item in evidence):
        raise HTTPException(409, "Add at least one resolved human correction before proposing a rubric.")
    snapshot = calibration(id)
    try:
        proposal = propose_rubric(judge, evidence)
    except AuthorError as exc:
        raise HTTPException(502, str(exc)) from None
    return db.save("rubric_proposal", {"judge_id": id, "judge": judge, "status": "draft", **proposal,
        "calibration": snapshot, "evidence": evidence,
        "note": "Proposed from calibration labels. Improvement has not been measured. Accepting creates a new version; task requirements remain unchanged."})


@router.post("/rubric-proposals/{id}/accept", status_code=201)
def accept_proposal(id: str, spec: AcceptProposal):
    with proposal_lock:
        proposal = db.get("rubric_proposal", id)
        if proposal.get("accepted_judge_id"):
            return db.get("judge", proposal["accepted_judge_id"])
        parent = db.get("judge", proposal["judge_id"])
        model = JudgeSpec.model_validate({key: parent[key] for key in JudgeSpec.model_fields if key in parent})
        model.rubric = spec.rubric or proposal["rubric"]
        family = parent.get("root_id", parent["id"])
        versions = [judge["version"] for judge in db.all_records("judge") if judge.get("root_id", judge["id"]) == family]
        judge = db.save("judge", {**model.model_dump(), "version": max(versions) + 1,
            "parent_id": parent["id"], "root_id": family, "rubric_proposal_id": id,
            "development_calibration": proposal["calibration"]})
        db.update("rubric_proposal", id, status="accepted", accepted_judge_id=judge["id"])
        return judge
