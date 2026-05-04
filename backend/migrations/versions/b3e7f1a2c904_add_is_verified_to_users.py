"""add is_verified to users

Revision ID: b3e7f1a2c904
Revises: 976136ebd876
Create Date: 2026-05-04 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = 'b3e7f1a2c904'
down_revision = '976136ebd876'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default='false')
    )
    # Backfill: existing users are considered already verified so they aren't locked out
    op.execute("UPDATE users SET is_verified = true")


def downgrade() -> None:
    op.drop_column('users', 'is_verified')
