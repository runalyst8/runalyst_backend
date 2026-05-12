"""add google and apple auth combined

Revision ID: 1f08647b4f5b
Revises: ccc17c366a3f
Create Date: 2026-03-31 19:44:03.117308

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1f08647b4f5b'
down_revision: Union[str, Sequence[str], None] = 'ccc17c366a3f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [c["name"] for c in inspector.get_columns("users")]
    constraints = [c["name"] for c in inspector.get_unique_constraints("users")]

    if "auth_provider" not in columns:
        op.add_column("users", sa.Column("auth_provider", sa.String(length=32), server_default="local", nullable=False))
    if "google_sub" not in columns:
        op.add_column("users", sa.Column("google_sub", sa.String(length=64), nullable=True))
    if "uq_users_google_sub" not in constraints:
        op.create_unique_constraint("uq_users_google_sub", "users", ["google_sub"])

    op.alter_column("users", "hashed_password",
                    existing_type=sa.String(length=255),
                    nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("uq_users_google_sub", "users", type_="unique")
    op.drop_column("users", "google_sub")
    op.drop_column("users", "auth_provider")
    op.alter_column("users", "hashed_password",
                    existing_type=sa.String(length=255),
                    nullable=False)
