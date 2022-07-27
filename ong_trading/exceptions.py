class InstrumentBaseException(Exception):
    """Base exception for all exceptions related to instruments"""
    pass


class PositionBelowMinimumException(InstrumentBaseException):
    """Exception risen when a deal is tried to be created below minimum position"""
    pass


class NoCashException(InstrumentBaseException):
    """Exception risen when a deal is tried to be created without cash"""
    pass
