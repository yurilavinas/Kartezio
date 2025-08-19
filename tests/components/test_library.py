import unittest

import numpy as np

from kartezio.core.components import Library, Primitive, register
from kartezio.types import Matrix, Scalar


@register(Primitive)
class PrimitiveTest(Primitive):
    def __init__(self):
        super().__init__([Scalar, Scalar], Scalar, 0)

    def call(self, x: list[np.ndarray], args: list[int]):
        return x[0] + x[1]


@register(Library, "test_library")
class LibraryTest(Library):
    pass


class MyTestCase(unittest.TestCase):
    def test_something(self):
        library = LibraryTest(Matrix)
        self.assertEqual(library.size, 0)
        library.display()
        library.add_primitive(PrimitiveTest())
        self.assertEqual(library.size, 1)
        library.display()
        self.assertEqual(
            library.execute(0, [np.array([21]), np.array([21])], []), 42
        )
        library_2 = Library.__from_dict__(library.__to_dict__())
        self.assertEqual(
            library_2.execute(0, [np.array([12]), np.array([30])], []), 42
        )


if __name__ == "__main__":
    unittest.main()
