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
"""Module with function for communication with the PolyNSI connection requester."""
import grpc

from supa import settings
from supa.grpc_nsi.connection_requester_pb2_grpc import ConnectionRequesterStub


def get_stub() -> ConnectionRequesterStub:
    """Get the connection requester stub."""
    channel = grpc.insecure_channel(settings.grpc_client_insecure_address_port)
    stub = ConnectionRequesterStub(channel)
    return stub
