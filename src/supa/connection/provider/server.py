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
"""Implementation of the gRPC based Connection Provider Service."""
from uuid import UUID

import structlog
from grpc import ServicerContext

from supa.connection.fsm import ReservationStateMachine
from supa.db import model
from supa.grpc_nsi import connection_provider_pb2_grpc, services_pb2
from supa.grpc_nsi.connection_common_pb2 import Header, Schedule
from supa.grpc_nsi.connection_provider_pb2 import ReservationRequestCriteria, ReserveRequest, ReserveResponse
from supa.grpc_nsi.policy_pb2 import PathTrace
from supa.grpc_nsi.services_pb2 import PointToPointService
from supa.util.nsi import parse_stp
from supa.util.timestamp import as_utc_timestamp, current_timestamp, is_specified

logger = structlog.get_logger(__name__)


class ConnectionProviderService(connection_provider_pb2_grpc.ConnectionProviderServicer):
    """Implementation of the gRPC Connection Provider Service.

    Each of the methods in this class corresponds to gRPC defined ``rpc`` calls in the ``connection.provider``
    gPRC package.
    """

    def Reserve(self, pb_reserve_request: ReserveRequest, context: ServicerContext) -> ReserveResponse:
        """Request new reservation, or modify existing reservation.

        The reserve message is sent from an RA to a PA when a new reservation is being requested, or
        a modification to an existing reservation is required. The :class:`ReserveResponse` indicates that the PA
        has accepted the reservation request for processing and has assigned it the returned
        connectionId. The original ``connection_id`` will be returned for the :class:`ReserveResponse`` of a
        modification. A :class:`ReserveConfirmed` or :class:`ReserveFailed` message will be sent asynchronously to the
        RA when reserve operation has completed processing.

        Args:
            pb_reserve_request: All the details about the requested reservation.
            context: gRPC server context object.

        Returns:
            A :class:`ReserveResponse` message containing the PA assigned ``connection_id`` for this reservation
            request. This value will be unique within the context of the PA.

        """
        log = logger.bind(method="Reserve")
        logger.info("Message received.", request_message=pb_reserve_request)

        if not pb_reserve_request.connection_id:  # new reservation
            pb_header: Header = pb_reserve_request.header
            pb_criteria: ReservationRequestCriteria = pb_reserve_request.criteria
            pb_schedule: Schedule = pb_criteria.schedule
            start_time = as_utc_timestamp(pb_schedule.start_time)
            end_time = as_utc_timestamp(pb_schedule.end_time)
            pb_ptps: PointToPointService = pb_criteria.ptps
            pb_path_trace: PathTrace = pb_header.path_trace

            reservation = model.Reservation(
                correlation_id=UUID(pb_header.correlation_id),
                requester_nsa=pb_header.requester_nsa,
                provider_nsa=pb_header.provider_nsa,
                reply_to=pb_header.reply_to if pb_header.reply_to else None,
                session_security_attributes=pb_header.session_security_attributes
                if pb_header.session_security_attributes
                else None,
                global_reservation_id=pb_reserve_request.global_reservation_id,
                description=pb_reserve_request.description if pb_reserve_request.description else None,
                version=pb_criteria.version,
            )
            if is_specified(end_time):
                if end_time <= start_time:
                    raise Exception(f"End time cannot come before the start time. {start_time=!s}, {end_time=!s}")
                elif end_time <= current_timestamp():
                    raise Exception(f"End time lies in the past. {end_time=!s}")
            if is_specified(start_time):
                reservation.start_time = start_time
            if is_specified(end_time):
                reservation.end_time = end_time
            reservation.bandwidth = pb_ptps.capacity
            if pb_ptps.directionality == services_pb2.Directionality.BI_DIRECTIONAL:
                reservation.directionality = model.Directionality.Bidirectional
            else:
                reservation.directionality = model.Directionality.Unidirectional
            reservation.symmetric = pb_ptps.symmetric_path

            src_stp = parse_stp(pb_ptps.source_stp)
            reservation.src_domain = src_stp.domain
            reservation.src_network_type = src_stp.network_type
            reservation.src_port = src_stp.port
            reservation.src_vlans = str(src_stp.vlan_ranges)

            dst_stp = parse_stp(pb_ptps.dest_stp)
            reservation.dst_domain = dst_stp.domain
            reservation.dst_network_type = dst_stp.network_type
            reservation.dst_port = dst_stp.port
            reservation.dst_vlans = str(dst_stp.vlan_ranges)

            for k, v in pb_ptps.parameters.items():
                reservation.parameters.append(model.Parameter(key=k, value=v))

            if pb_path_trace.id:
                path_trace = model.PathTrace(
                    path_trace_id=pb_path_trace.id, ag_connection_id=pb_path_trace.connection_id
                )
                for pb_path in pb_path_trace.paths:
                    path = model.Path()
                    for pb_segment in pb_path.segments:
                        segment = model.Segment(segment_id=pb_segment.id, upa_connection_id=pb_segment.connection_id)
                        for pb_stp in pb_segment.stps:
                            segment.stps.append(model.Stp(stp_id=pb_stp))
                        path.segments.append(segment)
                    path_trace.paths.append(path)
                reservation.path_trace = path_trace

            rsm = ReservationStateMachine(reservation, state_field="reservation_state")
            rsm.reserve_request()

            from supa.db.session import db_session

            with db_session() as session:
                session.add(reservation)
                session.flush()  # Let DB (actually SQLAlchemy) generate the connection_id for us.
                connection_id = reservation.connection_id  # Can't reference it outside of the session, hence new var.

        from supa import scheduler
        from supa.job import ReserveJob

        scheduler.add_job(ReserveJob(connection_id))
        reserve_response = ReserveResponse(header=pb_reserve_request.header, connection_id=str(connection_id))

        log.info("Sending response.", response_message=reserve_response)
        return reserve_response
