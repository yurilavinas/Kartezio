from typing import NewType

Score = NewType("Score", float)
ScoreList = list[Score]

DataType = NewType("DataType", str)
Scalar = DataType("scalar")  # shape: (1,)
Vector = DataType("vector")  # shape: (l, 1)
Matrix = DataType("matrix")  # shape: (h, w)
Tensor = DataType("tensor")  # shape: (c, h, w)

DataBatch = list[DataType]
