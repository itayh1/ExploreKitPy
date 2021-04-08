

class AttributeInfo:

    def __init__(self,attName: str, attType, attValue, numOfValues: int):
        self.attributeName = attName
        self.attributeType = attType
        self.value = attValue
        self.numOfDiscreteValues = numOfValues