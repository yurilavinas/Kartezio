import unittest

from kartezio.core.components import Components, Endpoint, register
from kartezio.types import Scalar


@register(Endpoint)
class EndpointTest(Endpoint):
    def call(self, x) -> list:
        return [x[0] + self.value]

    def __init__(self, value):
        super().__init__([Scalar])
        self.value = value

    def __to_dict__(self) -> dict:
        return {"name": self.name, "args": {"value": self.value}}


class MyTestCase(unittest.TestCase):
    def test_something(self):
        endpoint = EndpointTest(42)
        self.assertEqual(endpoint.call([42])[0], 84)
        endpoint_2 = Components.instantiate("Endpoint", "EndpointTest", 21)
        self.assertEqual(endpoint_2.call([21])[0], 42)
        endpoint_3 = Endpoint.__from_dict__(endpoint.__to_dict__())
        self.assertEqual(endpoint_3.call([42])[0], 84)


if __name__ == "__main__":
    unittest.main()
