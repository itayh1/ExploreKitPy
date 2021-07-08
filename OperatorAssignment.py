from Operators.Operator import Operator
from Operators.UnaryOperator import UnaryOperator


class OperatorAssignment:

    def __init__(self, sourceColumns: list, targetColumns: list, operator: Operator, secondaryOperator: UnaryOperator):
        self.sourceColumns = sourceColumns
        self.targetColumns = targetColumns
        self.operator = operator
        # a discretizer/normalizer that will be applied on the product of the previous operator
        # this operator is to be applied AFTER the main operator is complete (serves as a discretizer/normalizer)
        self.secondaryOperator = secondaryOperator

        self.filterEvaluatorScore: float = 0
        self.wrapperEvaluatorScore: float
        self.preRankerEvaluatorScore: float

    def getName(self) -> str:
        sb = ''
        sb += '{Sources:['
        for sCI in self.sourceColumns:
            sb += sCI.getName()
            sb += ','
        sb += '];'
        sb += 'Targets:['
        if self.targetColumns != None:
            for tCI in self.targetColumns:
                sb += tCI.getName()
                sb += ','

        sb += '];'
        sb += self.operator.getName()
        if self.secondaryOperator != None:
            sb += ','
            sb += self.secondaryOperator.getName()

        sb += '}'
        return sb.toString()

