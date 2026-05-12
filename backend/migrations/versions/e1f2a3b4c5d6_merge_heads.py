"""merge heads

Revision ID: e1f2a3b4c5d6
Revises: 7716f75da919, d1e2f3a4b5c6
Create Date: 2026-05-12

"""
from typing import Sequence, Union
from alembic import op

revision: str = 'e1f2a3b4c5d6'
down_revision: Union[str, Sequence[str], None] = ('7716f75da919', 'd1e2f3a4b5c6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
