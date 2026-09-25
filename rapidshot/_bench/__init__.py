"""The benchmark harness behind ``rapidshot benchmark``.

Private: the command line is the interface. It lived in ``benchmarks/`` until
rapidshot 2.6.1, which ships it so that anyone can reproduce the published
tables without a checkout; ``benchmarks/`` keeps thin stand-ins for every module
that moved, so the documented commands still work there.
"""
