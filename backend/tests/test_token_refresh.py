from datetime import timedelta

from app.core.security import create_access_token, generate_user_tokens, hash_password
from app.models.user import User


def make_user(db) -> User:
    user = User(
        email="test@example.com",
        hashed_password=hash_password("password123"),
        is_active=True,
        auth_provider="local",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def expired_token(user_id: int) -> str:
    return create_access_token(sub=str(user_id), expires_delta=timedelta(minutes=-1))


# ── /auth/refresh endpoint ────────────────────────────────────────────────────

def test_refresh_endpoint_issues_new_tokens(client, db):
    user = make_user(db)
    tokens = generate_user_tokens(db, user.id)
    db.commit()

    resp = client.post("/auth/refresh", json={"refresh_token": tokens.refresh_token})

    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert "refresh_token" in body
    assert body["refresh_token"] != tokens.refresh_token  # token was rotated


def test_refresh_endpoint_rejects_invalid_token(client, db):
    resp = client.post("/auth/refresh", json={"refresh_token": "not-a-real-token"})
    assert resp.status_code == 401


def test_refresh_endpoint_rejects_reused_token(client, db):
    user = make_user(db)
    tokens = generate_user_tokens(db, user.id)
    db.commit()

    client.post("/auth/refresh", json={"refresh_token": tokens.refresh_token})
    resp = client.post("/auth/refresh", json={"refresh_token": tokens.refresh_token})

    assert resp.status_code == 401


# ── auto-refresh in protected endpoints ──────────────────────────────────────
# Uses /users/me — only needs the `users` table which exists in the test DB.

def test_valid_token_no_refresh_headers_in_response(client, db):
    user = make_user(db)
    tokens = generate_user_tokens(db, user.id)
    db.commit()

    resp = client.get("/users/me", headers={
        "Authorization": f"Bearer {tokens.access_token}",
    })

    assert resp.status_code == 200
    assert "x-new-access-token" not in resp.headers
    assert "x-new-refresh-token" not in resp.headers


def test_expired_token_with_refresh_token_succeeds(client, db):
    user = make_user(db)
    tokens = generate_user_tokens(db, user.id)
    db.commit()

    resp = client.get("/users/me", headers={
        "Authorization": f"Bearer {expired_token(user.id)}",
        "X-Refresh-Token": tokens.refresh_token,
    })

    assert resp.status_code == 200
    assert "x-new-access-token" in resp.headers
    assert "x-new-refresh-token" in resp.headers


def test_expired_token_without_refresh_token_returns_401(client, db):
    user = make_user(db)
    db.commit()

    resp = client.get("/users/me", headers={
        "Authorization": f"Bearer {expired_token(user.id)}",
    })

    assert resp.status_code == 401


def test_invalid_token_returns_401(client, db):
    resp = client.get("/users/me", headers={
        "Authorization": "Bearer totally.invalid.token",
    })
    assert resp.status_code == 401


def test_refresh_token_is_rotated_after_auto_refresh(client, db):
    user = make_user(db)
    tokens = generate_user_tokens(db, user.id)
    db.commit()

    old_refresh = tokens.refresh_token

    # First request — triggers auto-refresh, old token gets revoked
    client.get("/users/me", headers={
        "Authorization": f"Bearer {expired_token(user.id)}",
        "X-Refresh-Token": old_refresh,
    })

    # Second request with the same (now revoked) refresh token — must fail
    resp = client.get("/users/me", headers={
        "Authorization": f"Bearer {expired_token(user.id)}",
        "X-Refresh-Token": old_refresh,
    })

    assert resp.status_code == 401


def test_new_tokens_from_auto_refresh_are_usable(client, db):
    user = make_user(db)
    tokens = generate_user_tokens(db, user.id)
    db.commit()

    # Trigger auto-refresh
    resp = client.get("/users/me", headers={
        "Authorization": f"Bearer {expired_token(user.id)}",
        "X-Refresh-Token": tokens.refresh_token,
    })

    assert resp.status_code == 200
    new_access = resp.headers["x-new-access-token"]
    new_refresh = resp.headers["x-new-refresh-token"]

    # New tokens should work on the next request
    resp2 = client.get("/users/me", headers={
        "Authorization": f"Bearer {new_access}",
        "X-Refresh-Token": new_refresh,
    })

    assert resp2.status_code == 200
