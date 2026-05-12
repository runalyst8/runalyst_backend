"""add apple_sub to users

Revision ID: a1b2c3d4e5f6
Revises: 976136ebd876
Create Date: 2026-03-31 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '976136ebd876'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column("users", sa.Column("apple_sub", sa.String(length=255), nullable=True))
    op.create_unique_constraint("uq_users_apple_sub", "users", ["apple_sub"])


def downgrade():
    op.drop_constraint("uq_users_apple_sub", "users", type_="unique")
    op.drop_column("users", "apple_sub")
