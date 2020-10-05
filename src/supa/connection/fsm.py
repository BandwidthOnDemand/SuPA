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
"""Define the three NSI Connection Service state machines.

The NSI Connection Service defines three state machines that,
together with the message processing functions (=coordinator functions in NSI parlance),
model the behaviour of the protocol.

They are:

- :class:`ReservationStateMachine` (RSM)
- :class:`ProvisioningStateMachine` (PSM)
- :class:`LifecycleStateMachine` (LSM)

The state machines explicitly regulate the sequence in which messages are processed.
The CS messages are each assigned to one of the three state machines:
RSM, PSM and LSM.
When the first reserve request for a new Connection is received,
the function processing the reserve requests MUST coordinate the creation of the
RSM, PSM and LSM
state machines for that specific connection.

The RSM and LSM MUST be instantiated as soon as the first Connection request is received.

The PSM MUST be instantiated as soon as the first version of the reservation is committed.

"""
from statemachine import State, StateMachine


class ReservationStateMachine(StateMachine):
    """Reservation State Machine.

    .. image:: /images/ReservationStateMachine.png
    """

    ReserveStart = State("RESERVE_START", initial=True)
    ReserveChecking = State("RESERVE_CHECKING")
    ReserveHeld = State("RESERVE_HELD")
    ReserveCommitting = State("RESERVE_COMMITTING")
    ReserveFailed = State("RESERVE_FAILED")
    ReserveTimeout = State("RESERVE_TIMEOUT")
    ReserveAborting = State("RESERVE_ABORTING")

    reserve_request = ReserveStart.to(ReserveChecking)
    reserve_confirmed = ReserveChecking.to(ReserveHeld)
    reserve_failed = ReserveChecking.to(ReserveFailed)
    reserve_abort_request = ReserveFailed.to(ReserveAborting) | ReserveHeld.to(ReserveAborting)
    reserve_abort_confirmed = ReserveAborting.to(ReserveStart)
    reserve_timeout_notification = ReserveHeld.to(ReserveTimeout)
    reserve_commit_request = ReserveHeld.to(ReserveCommitting) | ReserveTimeout.to(ReserveCommitting)
    reserve_commit_confirmed = ReserveCommitting.to(ReserveStart)
    reserve_commit_failed = ReserveCommitting.to(ReserveStart)


class ProvisioningStateMachine(StateMachine):
    """Provisioning State Machine.

    .. image:: /images/ProvisioningStateMachine.png
    """

    Released = State("RELEASED", initial=True)
    Provisioning = State("PROVISIONING")
    Provisioned = State("PROVISIONED")
    Releasing = State("RELEASING")

    provision_request = Released.to(Provisioning)
    provision_confirmed = Provisioning.to(Provisioned)
    release_request = Provisioned.to(Releasing)
    release_confirmed = Releasing.to(Released)


class LifecycleStateMachine(StateMachine):
    """Lifecycle State Machine.

    .. image:: /images/LifecycleStateMachine.png
    """

    Created = State("CREATED", initial=True)
    Failed = State("FAILED")
    Terminating = State("TERMINATING")
    PassedEndTime = State("PASSED_END_TIME")
    Terminated = State("TERMINATED")

    forced_end_notification = Created.to(Failed)
    terminate_request = Created.to(Terminating) | PassedEndTime.to(Terminating) | Failed.to(Terminating)
    endtime_event = Created.to(PassedEndTime)
    terminate_confirmed = Terminating.to(Terminated)


if __name__ == "__main__":
    # If you have Graphviz and the corresponding Python package ``graphviz`` installed,
    # generating a graphical representation of the state machines is as easy as running this module.
    # The generated graphs are not the most beautiful
    # or best layed out ones,
    # but do provide the means to visually inspect the state machine definitions.
    # In addition we now have images that we can include in our documentation.
    # That is,
    # by the way,
    # the reason the images are generated in the docs/images folder of this project
    from graphviz import Digraph

    from supa import get_project_root

    output_path = get_project_root() / "docs" / "images"

    def plot_fsm(fsm: StateMachine, name: str) -> None:
        """Generate image that visualizes a state machine."""
        dg = Digraph(name=name, comment=name)
        for s in fsm.states:
            for t in s.transitions:
                dg.edge(t.source.name, t.destinations[0].name, label=t.identifier)
        dg.render(filename=name, directory=output_path, cleanup=True, format="png")

    plot_fsm(ReservationStateMachine(), "ReservationStateMachine")
    plot_fsm(ProvisioningStateMachine(), "ProvisioningStateMachine")
    plot_fsm(LifecycleStateMachine(), "LifecycleStateMachine")
