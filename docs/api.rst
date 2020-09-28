.. vim:noswapfile:nobackup:nowritebackup:

API
===

supa
----

.. automodule:: supa
   :members:
   :undoc-members:

supa.main
---------

.. automodule:: supa.main
   :members:

.. click:: supa.main:cli
   :prog: supa
   :show-nested:

supa.db
-------

.. automodule:: supa.db
   :members:

supa.job
--------

.. automodule:: supa.job
   :members:

supa.connection.fsm
-------------------

.. automodule:: supa.connection.fsm
   :members:

supa.connection.error
---------------------

.. automodule:: supa.connection.error
   :members:

supa.connection.provider.server
-------------------------------

.. automodule:: supa.connection.provider.server
   :members:

supa.util.timestamp
-------------------

.. automodule:: supa.util.timestamp
   :members:

supa.util.nsi
-------------------

.. automodule:: supa.util.nsi
   :members:

supa.util.vlan
--------------

..
    Explicitely list special members to show that they are present,
    even if they are not yet properly documented.

.. automodule:: supa.util.vlan
   :members:
   :undoc-members:
   :special-members: __contains__, __iter__, __len__, __str__, __repr__, __eq__, __hash__,
                     __sub__, __and__, __or__, __xor__

supa.util.functional
--------------------

.. automodule:: supa.util.functional
   :members:
