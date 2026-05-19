from unittest.mock import MagicMock, patch

from app.core.security import create_access_token, hash_password
from app.deps.auth import verify_gpu_api_key
from app.main import app
from app.models.run import Run
from app.models.user import User


# ── helpers ───────────────────────────────────────────────────────────────────

def make_user(db, email: str = "runner@example.com") -> User:
    user = User(
        email=email,
        hashed_password=hash_password("password123"),
        is_active=True,
        auth_provider="local",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def make_run(
    db,
    user_id: int,
    *,
    video_path: str = "1/video.mp4",
    thumbnail_path: str | None = None,
) -> Run:
    run = Run(user_id=user_id, video_path=video_path, thumbnail_path=thumbnail_path, status="queued")
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def auth_header(user_id: int) -> dict:
    token = create_access_token(sub=str(user_id))
    return {"Authorization": f"Bearer {token}"}


# ── GET /runs/all ─────────────────────────────────────────────────────────────

def test_get_all_runs_empty_for_new_user(run_client, run_db):
    user = make_user(run_db)
    resp = run_client.get("/runs/all", headers=auth_header(user.id))
    assert resp.status_code == 200
    assert resp.json()["runs"] == {}


def test_get_all_runs_returns_own_runs(run_client, run_db):
    user = make_user(run_db)
    make_run(run_db, user.id, video_path="1/a.mp4")
    make_run(run_db, user.id, video_path="1/b.mp4")
    resp = run_client.get("/runs/all", headers=auth_header(user.id))
    assert resp.status_code == 200
    assert len(resp.json()["runs"]) == 2


def test_get_all_runs_excludes_other_users_runs(run_client, run_db):
    user1 = make_user(run_db, email="user1@example.com")
    user2 = make_user(run_db, email="user2@example.com")
    make_run(run_db, user2.id, video_path="2/vid.mp4")
    resp = run_client.get("/runs/all", headers=auth_header(user1.id))
    assert resp.status_code == 200
    assert resp.json()["runs"] == {}


def test_get_all_runs_requires_auth(run_client, run_db):
    resp = run_client.get("/runs/all")
    assert resp.status_code == 401


def test_get_all_runs_response_includes_thumbnail_path(run_client, run_db):
    user = make_user(run_db)
    make_run(run_db, user.id, video_path="1/c.mp4", thumbnail_path="1/c.mp4")
    resp = run_client.get("/runs/all", headers=auth_header(user.id))
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    run_data = next(iter(runs.values()))
    assert run_data["thumbnail_path"] == "1/c.mp4"


# ── GET /runs/get ─────────────────────────────────────────────────────────────

def test_get_run_returns_run(run_client, run_db):
    user = make_user(run_db)
    run = make_run(run_db, user.id)
    resp = run_client.get(f"/runs/get?run_id={run.id}", headers=auth_header(user.id))
    assert resp.status_code == 200
    assert resp.json()["id"] == run.id
    assert resp.json()["video_path"] == run.video_path


def test_get_run_returns_404_for_missing(run_client, run_db):
    user = make_user(run_db)
    resp = run_client.get("/runs/get?run_id=99999", headers=auth_header(user.id))
    assert resp.status_code == 404


def test_get_run_returns_403_for_wrong_owner(run_client, run_db):
    user1 = make_user(run_db, email="owner@example.com")
    user2 = make_user(run_db, email="thief@example.com")
    run = make_run(run_db, user2.id, video_path="2/secret.mp4")
    resp = run_client.get(f"/runs/get?run_id={run.id}", headers=auth_header(user1.id))
    assert resp.status_code == 403


def test_get_run_requires_auth(run_client, run_db):
    user = make_user(run_db)
    run = make_run(run_db, user.id)
    resp = run_client.get(f"/runs/get?run_id={run.id}")
    assert resp.status_code == 401


# ── POST /runs/create-record ──────────────────────────────────────────────────

@patch("app.services.run.service.send_message_to_queue")
def test_create_run_record_succeeds(mock_queue, run_client, run_db):
    user = make_user(run_db)
    resp = run_client.post(
        "/runs/create-record",
        json={"video_path": "1/new-video.mp4"},
        headers=auth_header(user.id),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["video_path"] == "1/new-video.mp4"
    assert body["thumbnail_path"] == "1/new-video.mp4"
    assert body["user_id"] == user.id
    mock_queue.assert_called_once()


@patch("app.services.run.service.send_message_to_queue")
def test_create_run_record_with_title(mock_queue, run_client, run_db):
    user = make_user(run_db)
    resp = run_client.post(
        "/runs/create-record",
        json={"video_path": "1/titled.mp4", "title": "My Morning Run"},
        headers=auth_header(user.id),
    )
    assert resp.status_code == 201
    assert resp.json()["title"] == "My Morning Run"


@patch("app.services.run.service.send_message_to_queue")
def test_create_run_enqueues_message(mock_queue, run_client, run_db):
    user = make_user(run_db)
    run_client.post(
        "/runs/create-record",
        json={"video_path": "1/queued.mp4"},
        headers=auth_header(user.id),
    )
    called_with = mock_queue.call_args.kwargs["message_body"]
    assert "run_id" in called_with
    assert called_with["video_path"] == "1/queued.mp4"


def test_create_run_record_requires_auth(run_client, run_db):
    resp = run_client.post("/runs/create-record", json={"video_path": "1/nope.mp4"})
    assert resp.status_code == 401


# ── POST /runs/upload-url ─────────────────────────────────────────────────────

def test_upload_url_returns_nested_structure(run_client, run_db):
    user = make_user(run_db)
    mock_video = {"signed_url": "https://storage.example.com/video-upload", "path": "1/uuid.mp4"}
    mock_thumb = {"signed_url": "https://storage.example.com/thumb-upload", "path": "1/uuid.mp4"}

    mock_storage = MagicMock()
    mock_storage.from_.return_value.create_signed_upload_url.side_effect = [mock_video, mock_thumb]

    with patch("app.services.run.service.supabase_client") as mock_sb:
        mock_sb.storage = mock_storage
        resp = run_client.post("/runs/upload-url", headers=auth_header(user.id))

    assert resp.status_code == 200
    body = resp.json()
    assert "video" in body and "thumbnail" in body
    assert body["video"]["upload_url"] == "https://storage.example.com/video-upload"
    assert body["video"]["path"] == "1/uuid.mp4"
    assert body["thumbnail"]["upload_url"] == "https://storage.example.com/thumb-upload"


def test_upload_url_returns_500_on_supabase_error(run_client, run_db):
    user = make_user(run_db)
    with patch("app.services.run.service.supabase_client") as mock_sb:
        mock_sb.storage.from_.return_value.create_signed_upload_url.side_effect = Exception("storage down")
        resp = run_client.post("/runs/upload-url", headers=auth_header(user.id))
    assert resp.status_code == 500


def test_upload_url_requires_auth(run_client, run_db):
    resp = run_client.post("/runs/upload-url")
    assert resp.status_code == 401


# ── GET /runs/thumbnail-url ───────────────────────────────────────────────────

def test_thumbnail_url_returns_404_when_no_thumbnail(run_client, run_db):
    user = make_user(run_db)
    run = make_run(run_db, user.id, thumbnail_path=None)
    resp = run_client.get(f"/runs/thumbnail-url?run_id={run.id}", headers=auth_header(user.id))
    assert resp.status_code == 404


def test_thumbnail_url_returns_signed_url(run_client, run_db):
    user = make_user(run_db)
    run = make_run(run_db, user.id, thumbnail_path="1/thumb.jpg")
    signed_url = "https://storage.example.com/signed-thumb"

    mock_storage = MagicMock()
    mock_storage.from_.return_value.create_signed_url.return_value = {"signedURL": signed_url}

    with patch("app.services.run.service.supabase_client") as mock_sb:
        mock_sb.storage = mock_storage
        resp = run_client.get(
            f"/runs/thumbnail-url?run_id={run.id}", headers=auth_header(user.id)
        )

    assert resp.status_code == 200
    assert resp.json()["url"] == signed_url


def test_thumbnail_url_returns_403_for_wrong_owner(run_client, run_db):
    user1 = make_user(run_db, email="u1@example.com")
    user2 = make_user(run_db, email="u2@example.com")
    run = make_run(run_db, user2.id, video_path="2/vid.mp4", thumbnail_path="2/thumb.jpg")
    resp = run_client.get(
        f"/runs/thumbnail-url?run_id={run.id}", headers=auth_header(user1.id)
    )
    assert resp.status_code == 403


def test_thumbnail_url_returns_500_on_supabase_error(run_client, run_db):
    user = make_user(run_db)
    run = make_run(run_db, user.id, thumbnail_path="1/thumb.jpg")
    with patch("app.services.run.service.supabase_client") as mock_sb:
        mock_sb.storage.from_.return_value.create_signed_url.side_effect = Exception("storage error")
        resp = run_client.get(
            f"/runs/thumbnail-url?run_id={run.id}", headers=auth_header(user.id)
        )
    assert resp.status_code == 500


# ── PATCH /runs/update-status ─────────────────────────────────────────────────

def test_update_status_succeeds_with_valid_key(run_client, run_db):
    user = make_user(run_db)
    run = make_run(run_db, user.id)
    app.dependency_overrides[verify_gpu_api_key] = lambda: None
    try:
        resp = run_client.patch(
            f"/runs/update-status?run_id={run.id}&new_status=completed",
            headers={"X-GPU-API-Key": "test-key"},
        )
    finally:
        del app.dependency_overrides[verify_gpu_api_key]
    assert resp.status_code == 200
    assert resp.json()["id"] == run.id


def test_update_status_rejects_invalid_key(run_client, run_db):
    user = make_user(run_db)
    run = make_run(run_db, user.id)
    resp = run_client.patch(
        f"/runs/update-status?run_id={run.id}&new_status=completed",
        headers={"X-GPU-API-Key": "wrong-key-xyz"},
    )
    assert resp.status_code == 401


def test_update_status_returns_404_for_missing_run(run_client, run_db):
    app.dependency_overrides[verify_gpu_api_key] = lambda: None
    try:
        resp = run_client.patch(
            "/runs/update-status?run_id=99999&new_status=completed",
            headers={"X-GPU-API-Key": "test-key"},
        )
    finally:
        del app.dependency_overrides[verify_gpu_api_key]
    assert resp.status_code == 404
