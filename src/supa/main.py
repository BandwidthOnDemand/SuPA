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
"""
SuPA main entry point.

SuPA has a single entry point defined in this module,
namely :func:`cli`.
That is what is executed when the ``supa`` command is issued from the command-line.

The other ``@cli.command`` annotated functions in this modules implement the various sub-commands.
"""
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import UUID

import click
import grpc
import sqlalchemy
import structlog
from click import Context, Option
from tabulate import tabulate

from supa import init_app, settings
from supa.grpc_nsi import connection_provider_pb2_grpc
from supa.util.vlan import VlanRanges

logger = structlog.get_logger(__name__)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "show_default": True}


# Note: The `type: ignore`'s is the file are their to circumenvent Click's lack of type annotations.


@dataclass
class CommonOptionsState:
    """Class to capture common options shared between Click callables/sub commands."""

    database_file: Optional[Path] = None


# Trick from: https://github.com/pallets/click/issues/108
pass_common_options_state = click.make_pass_decorator(CommonOptionsState, ensure=True)
"""Define custom decorator to pass in an CommonOptionState instance as first argument.

When decorating a Click callable/sub command with the :func`common_options` decorator
the Click options defined in that decorator will become part of the sub command
as if they where defined directly on the sub command.
These common options however will not be pass in the argument list to the sub command.
Reason being that we don't know beforehand how many extra common options :func:`common_options` defines,
or if that number later changes possibly breaking existing code.
Instead we want a single state capturing object to be passed in.

Usage::

    @cli.command(context_settings=CONTEXT_SETTINGS)
    @click.option("--fu", ...)
    @click.option("--bar",...)
    @common_options              # <--- usage
    @pass_common_options_state   # <--- usage
    def my_sub_command(common_options: CommonOptionsState, fu: str, bar: str) -> None:
        # explicitly update ``settings`' attributes if they match command line options
        settings.fu = fu
        settings.bar = bar
        ...
        # with all settings resolved, we can now initialize the application properly.
        init_app()

        # actual sub command stuff
        ...
"""


def database_file_option(f):  # type: ignore
    """Define common option for specifying database file location."""

    def callback(ctx: Context, param: Option, value: Optional[str]) -> Optional[str]:
        """Update the Settings instance when the database-file option is used."""
        cos: CommonOptionsState = ctx.ensure_object(CommonOptionsState)
        if value is not None:
            cos.database_file = Path(value)

            # Update the `settings` instance so that it available application wide.
            settings.database_file = cos.database_file
        return value

    return click.option(
        "--database-file",
        type=click.Path(readable=False),
        expose_value=False,  # Don't add to sub command arg list. We have `@pass_common_options_state` for that.
        help="Location of the SQLlite database file",
        callback=callback,
    )(f)


def common_options(f):  # type: ignore
    """Provide the means to declare common options to Click callables/sub command."""
    f = database_file_option(f)
    return f


@click.group(context_settings=CONTEXT_SETTINGS)
def cli() -> None:
    """Manage the SURF ultimate Provider Agent from the command line.

    Configuration variables can be set using (in order of precedence):

    \b
    - command line options
    - environment variables
    - entries in `supa.env`

    For more information see `supa.env`.
    """
    pass


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--grpc-server-max-workers",
    default=settings.grpc_server_max_workers,
    type=int,
    help="Maximum number of workers to serve gRPC requests.",
)
@click.option(
    "--grpc-server-insecure-address-port",
    default=settings.grpc_server_insecure_address_port,
    help="Address and port to listen on.",
)
@click.option(
    "--scheduler-max-workers",
    default=settings.scheduler_max_workers,
    type=int,
    help="Maximum number of workers to execute scheduler jobs.",
)
@click.option("--domain", default=settings.domain, type=str, help="Name of the domain SuPA is responsible for.")
@click.option(
    "--network-type", default=settings.network_type, type=str, help="Name of the network SuPA is responsible for."
)
@click.option(
    "--grpc-client-insecure-address-port",
    default=settings.grpc_client_insecure_address_port,
    help="Address and port of PolyNSI.",
)
@click.option("--nsa-id", default=settings.nsa_id, type=str, help="NSA ID of SuPA.")
@common_options  # type: ignore
def serve(
    grpc_server_max_workers: int,
    grpc_server_insecure_address_port: str,
    scheduler_max_workers: int,
    domain: str,
    network_type: str,
    grpc_client_insecure_address_port: str,
    nsa_id: str,
) -> None:
    """Start the gRPC server and listen for incoming requests."""
    # Command-line options take precedence.
    settings.grpc_server_max_workers = grpc_server_max_workers
    settings.grpc_server_insecure_address_port = grpc_server_insecure_address_port
    settings.scheduler_max_workers = scheduler_max_workers
    settings.domain = domain
    settings.network_type = network_type
    settings.grpc_client_insecure_address_port = grpc_client_insecure_address_port
    settings.nsa_id = nsa_id

    init_app()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=settings.grpc_server_max_workers))
    log = logger.bind(grpc_server_max_workers=settings.grpc_server_max_workers)

    # Safe to import, now that `init_app()` has been called
    from supa.connection.provider.server import ConnectionProviderService

    connection_provider_pb2_grpc.add_ConnectionProviderServicer_to_server(ConnectionProviderService(), server)
    server.add_insecure_port(settings.grpc_server_insecure_address_port)
    log = log.bind(grpc_server_insecure_address_port=settings.grpc_server_insecure_address_port)

    server.start()
    log.info("Started Connection Provider gRPC Service.")

    server.wait_for_termination()


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option("--port-id", required=True, type=click.UUID, help="Orchestrator subscription_id on the port.")
@click.option("--name", required=True, type=str, help="Name of the Port.")
@click.option("--vlans", required=True, type=str, help="Available VLANs on the port.")
@click.option("--remote-stp", type=str, help="Remote STP (for Service Demarcation Points).")
@click.option("--bandwidth", required=True, type=int, help="In Mbps.")
@click.option("--enabled/--disabled", default=True)
@common_options  # type: ignore
def add(port_id: UUID, name: str, vlans: str, remote_stp: Optional[str], bandwidth: int, enabled: bool) -> None:
    """Add Orchestrator port to SuPA."""
    init_app(with_scheduler=False)

    # Safe to import, now that `init_app()` has been called
    from supa.db.model import Port
    from supa.db.session import db_session

    port = Port(
        port_id=port_id,
        name=name,
        vlans=str(VlanRanges(vlans)),
        remote_stp=remote_stp,
        bandwidth=bandwidth,
        enabled=enabled,
    )

    with db_session() as session:
        session.add(port)


@cli.command(name="list", context_settings=CONTEXT_SETTINGS)
@click.option("--only", type=click.Choice(("enabled", "disabled")), help="Limit list of ports [default: list all]")
@common_options  # type: ignore
def list_cmd(only: Optional[str]) -> None:
    """List Orchestrator ports made available to SuPA."""
    init_app(with_scheduler=False)
    from supa.db.model import Port
    from supa.db.session import db_session

    with db_session() as session:
        ports = session.query(Port)
        if only == "enabled":
            ports = ports.filter(Port.enabled.is_(True))
        elif only == "disabled":
            ports = ports.filter(Port.enabled.is_(False))
        ports = ports.values(Port.port_id, Port.name, Port.vlans, Port.bandwidth, Port.remote_stp, Port.enabled)
        click.echo(
            tabulate(
                tuple(ports),
                headers=("port_id", "name", "vlans", "bandwidth", "remote_stp", "enabled"),
                tablefmt="psql",
            )
        )


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option("--port-id", type=click.UUID, help="Orchestrator subscription_id on the port.")
@click.option("--name", type=str, help="Name of the Port.")
@common_options  # type: ignore
def delete(port_id: Optional[UUID], name: Optional[str]) -> None:
    """Delete Orchestrator port if not in use (or previously used).

    A port can only be deleted if it was never used in a reservation.
    Once used  a port cannot be deleted again.
    A port can be disabled though!
    This will take it out of the pool of ports
    reservations (and hence connections) are made against.
    See the `disable` command.
    """
    init_app(with_scheduler=False)
    from supa.db.model import Port
    from supa.db.session import db_session

    if port_id is None and name is None:
        click.echo("Please specify either --port-id or --name.", err=True)
    try:
        with db_session() as session:
            port = session.query(Port)
            if port_id is not None:
                port = port.get(port_id)
            else:
                port = port.filter(Port.name == name).one()
            session.delete(port)
    except sqlalchemy.exc.IntegrityError:
        click.echo(
            "Port is in use. Could not delete it. (You could disable it instead to prevent further use).", err=True
        )
    except sqlalchemy.orm.exc.NoResultFound:
        click.echo("Port could not be found.", err=True)


def _set_enable(port_id: Optional[UUID], name: Optional[str], enabled: bool) -> None:
    """Enable or disable a specific port."""
    init_app(with_scheduler=False)
    from supa.db.model import Port
    from supa.db.session import db_session

    if port_id is None and name is None:
        click.echo("Please specify either --port-id or --name.", err=True)
    try:
        with db_session() as session:
            port = session.query(Port)
            if port_id is not None:
                port = port.get(port_id)
            else:
                port = port.filter(Port.name == name).one()
            port.enabled = enabled
            click.echo(f"Port '{port.name}' has been {'enabled' if enabled else 'disabled'}.")
    except sqlalchemy.orm.exc.NoResultFound:
        click.echo("Port could not be found.", err=True)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option("--port-id", type=click.UUID, help="Orchestrator subscription_id on the port.")
@click.option("--name", type=str, help="Name of the Port.")
def enable(port_id: Optional[UUID], name: Optional[str]) -> None:
    """Enable a specific port.

    Enabling a port makes it available for reservation requests.
    """
    _set_enable(port_id, name, enabled=True)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option("--port-id", type=click.UUID, help="Orchestrator subscription_id on the port.")
@click.option("--name", type=str, help="Name of the Port.")
def disable(port_id: Optional[UUID], name: Optional[str]) -> None:
    """Disable a specific port.

    Disabling a port makes it unavailable for reservation requests.
    """
    _set_enable(port_id, name, enabled=False)
