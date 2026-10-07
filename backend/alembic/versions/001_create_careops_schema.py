from __future__ import annotations

from alembic import op

from backend.app.database import Base
from backend.app import models as _models


revision = "001_create_careops_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind(), checkfirst=True)
