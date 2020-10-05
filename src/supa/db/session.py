#  Copyright 2020 SURF.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Setup the DB, configure SQLAlchemy and define the schema.

Due to SuPA's modest DB requirements we have chosen to use `SQLite <https://sqlite.org/index.html>`_.
It is very easy to use,
does not require a full client/server setup,
and it is wickedly fast.
There are some limitations with regards to concurrency,
but those will not affect SuPA with its low DB WRITE needs;
especially when configured with the :attr:`~supa.JournalMode.WAL` journal mode.

Usage
=====

.. warning:: Due to how SuPA initializes itself,
             whatever needs to be imported from this module,
             needs to be imported locally!

When anything from this module is imported at the top of other modules
some things within this module might not yet have been correctly initialized.
Most notable the location of the SQLite database file.
Once the initialization has been completed,
these already imported objects will still refer to the old uninitialized ones.
Don't worry too much about it;
you will get an informative error.
But just to be on the safe side,
always import anything from this module locally!
"""
import sqlite3
from contextlib import closing, contextmanager
from typing import Any, Iterator

import structlog
from sqlalchemy import event, orm
from sqlalchemy.engine import Engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.pool import _ConnectionRecord

from supa import settings

logger = structlog.get_logger(__name__)


@event.listens_for(Engine, "connect")  # type: ignore
def set_sqlite_pragma(dbapi_connection: sqlite3.Connection, connection_record: _ConnectionRecord) -> None:
    """Configure certain SQLite settings.

    These settings,
    issued as SQLite `pragmas <https://sqlite.org/pragma.html>`_,
    need to be configured on each connection.
    Hence the usage of SQLAlchemy's engine's connect event.
    """
    with closing(dbapi_connection.cursor()) as cursor:
        # https://sqlite.org/pragma.html#pragma_foreign_keys
        foreign_keys = "ON"
        log = logger.bind(foreign_keys=foreign_keys)
        cursor.execute(f"PRAGMA foreign_keys={foreign_keys}")

        # https://sqlite.org/pragma.html#pragma_auto_vacuum
        auto_vacuum = "INCREMENTAL"
        log = log.bind(auto_vacuum=auto_vacuum)
        cursor.execute(f"PRAGMA auto_vacuum={auto_vacuum}")

        # https://sqlite.org/pragma.html#pragma_journal_mode
        log = log.bind(journal_mode=settings.database_journal_mode.value)
        cursor.execute(f"PRAGMA journal_mode={settings.database_journal_mode.value}")
        log.debug("Set default options for SQLite database.")


class UnconfiguredSession(scoped_session):
    """Fail safe fake session class to guard against premature SQLAlchemy session usage.

    SQLAlchemy's engine,
    and hence session,
    can only be initialized after all settings have been resolved.
    This means that we cannot initialize it at the module level.
    After all,
    at the time this module is being executed the default settings
    and those of the env file have been processed,
    those specified on the commend line might not have been processed.
    At the same time we want to keep easy access to the :data:`Session` as a module level attribute.
    So we configure the :data:`Session` with this safe guard class
    only to overwrite it with the real thing after command line options have been processed.
    """

    def __init__(self) -> None:  # noqa: D107
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> orm.Session:
        """Trap premature ``Session()`` calls and raise an exception."""
        raise Exception(
            """DB has not yet been initialized. Call `main.init_app` first. Only then (locally) import `Session` or `db_session`.

IMPORTANT
==========
Make sure you have processed all the different ways of dealing with application
configuration before you call `main.init_app`.  The env file (`supa.env`) and
the environment are handled automatically by the `supa.settings` instance.
However anything specified on the command line generally needs to be processed
explicitly in the module `supa.main`.
"""
        )


Session = UnconfiguredSession()
"""SQLAlchemy Session for accessing the database.

:data:`Session` can only be used after a call to :func:`main.init_app`.
That,
in turn,
can only be called after all application configuration has been resolved,
eg. after command line processing.
:func:`main.init_app` will replace :data:`Session` with a proper SQLAlchemy
(scoped) ``Session``.
"""


@contextmanager
def db_session() -> Iterator[scoped_session]:
    """Context manager for using an SQLAlchemy session.

    It will automatically commit the session upon leaving the context manager.
    It will rollback the session if an exception occurred while in the context manager.
    re-raising the exception afterwards.

    Example::

        my_model = MyModel(fu="fu", bar="bar")
        with db_session() as session:
            session.add(my_model)

    Raises:
        Whatever exception that was raised while the context manager was active.

    """
    session = None
    try:
        session = Session()
        yield session
        session.commit()
    except BaseException:
        logger.info("An exception occurred while doing DB work. Rolling back.")
        if session is not None:
            session.rollback()
        raise
    finally:
        if session is not None:
            session.close()
