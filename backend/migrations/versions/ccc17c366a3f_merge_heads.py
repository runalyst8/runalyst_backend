"""merge heads

Revision ID: ccc17c366a3f
Revises: 6aece6003cf0, a1b2c3d4e5f6
Create Date: 2026-03-31 18:08:57.151809

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ccc17c366a3f'
down_revision: Union[str, Sequence[str], None] = ('6aece6003cf0', 'a1b2c3d4e5f6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
