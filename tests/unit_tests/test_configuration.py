from shared.configuration import BaseConfiguration


def test_configuration_empty() -> None:
    BaseConfiguration.from_runnable_config({})
