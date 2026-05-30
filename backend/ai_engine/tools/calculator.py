"""
Calculator Tool - Safe mathematical expression evaluator.

Supports basic arithmetic, percentages, square roots, and exponents.
Uses safe parsing with regex and the operator module - NO raw eval().
"""

import ast
import math
import operator
import re
from typing import Dict, Optional

from .tool_registry import BaseTool


class CalculatorTool(BaseTool):
    """
    Calculator tool that safely evaluates mathematical expressions.

    Supports:
        - Basic arithmetic: +, -, *, /
        - Exponents: 2^10, 2**10
        - Square roots: sqrt 144, square root of 144
        - Percentages: 15% of 200, 20 percent of 50
        - Parentheses for grouping: (2 + 3) * 4

    Uses AST-based safe evaluation - never calls eval() on raw input.
    """

    name = "calculator"
    description = "Performs mathematical calculations safely"
    keywords = [
        "calculate", "math", "compute", "plus", "minus", "times",
        "divided", "sqrt", "square root", "percent", "add", "subtract",
        "multiply", "divide", "sum", "difference", "product",
        "+", "-", "*", "/", "^", "**"
    ]
    patterns = [
        r"what\s+is\s+[\d\.\s\+\-\*\/\^\(\)%]+",
        r"calculate\s+.+",
        r"compute\s+.+",
        r"\d+\s*[\+\-\*\/\^]\s*\d+",
        r"sqrt\s+\d+",
        r"square\s+root\s+(of\s+)?\d+",
        r"\d+\s*%\s*of\s*\d+",
        r"\d+\s+percent\s+of\s+\d+",
        r"\d+\s*\^\s*\d+",
    ]

    # Allowed operators for safe evaluation
    _operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def execute(self, message: str, user_id: Optional[str] = None, args: Optional[Dict] = None) -> Dict:
        """
        Execute a mathematical calculation from the user's message.

        Args:
            message: The user's input containing a math expression.
            user_id: Optional user identifier (not used for calculations).
            args: Optional additional arguments.

        Returns:
            Dictionary with result string, tool_used, and numeric answer.
        """
        try:
            # Extract and evaluate the expression
            expression, result = self._parse_and_calculate(message)

            # Format the result nicely (remove trailing zeros for floats)
            if isinstance(result, float) and result == int(result):
                result = int(result)

            return {
                "result": f"{expression} = {result}",
                "tool_used": "calculator",
                "answer": result
            }
        except ZeroDivisionError:
            return {
                "result": "Error: Division by zero is not allowed.",
                "tool_used": "calculator",
                "answer": None
            }
        except (ValueError, TypeError, SyntaxError) as e:
            return {
                "result": f"Error: Could not evaluate the expression. {str(e)}",
                "tool_used": "calculator",
                "answer": None
            }

    def _parse_and_calculate(self, message: str):
        """
        Parse the user's message and extract a mathematical expression.

        Handles special cases like percentages and square roots before
        falling back to general expression evaluation.

        Args:
            message: The user's input message.

        Returns:
            A tuple of (expression_string, numeric_result).
        """
        message_lower = message.lower().strip()

        # Handle percentage: "15% of 200" or "15 percent of 200"
        percent_match = re.search(r"(\d+\.?\d*)\s*(%|percent)\s*of\s*(\d+\.?\d*)", message_lower)
        if percent_match:
            percent = float(percent_match.group(1))
            number = float(percent_match.group(3))
            result = (percent / 100) * number
            return f"{percent}% of {number}", result

        # Handle square root: "sqrt 144" or "square root of 144"
        sqrt_match = re.search(r"(?:sqrt|square\s*root)\s*(?:of\s*)?(\d+\.?\d*)", message_lower)
        if sqrt_match:
            number = float(sqrt_match.group(1))
            result = math.sqrt(number)
            return f"sqrt({number})", result

        # Extract the mathematical expression from the message
        expression = self._extract_expression(message)

        # Replace ^ with ** for exponentiation
        expression = expression.replace("^", "**")

        # Safely evaluate the expression using AST
        result = self._safe_eval(expression)

        # Show the original expression (with ^ for display)
        display_expr = expression.replace("**", "^")
        return display_expr, result

    def _extract_expression(self, message: str) -> str:
        """
        Extract a mathematical expression from a natural language message.

        Args:
            message: The user's input message.

        Returns:
            A string containing the mathematical expression.
        """
        # Remove common prefixes
        cleaned = re.sub(
            r"^(what\s+is|calculate|compute|eval|solve)\s+",
            "",
            message.lower().strip()
        )

        # Remove trailing question marks and extra whitespace
        cleaned = cleaned.rstrip("?").strip()

        # Extract just the math part (numbers, operators, parentheses, dots)
        math_match = re.search(r"[\d\.\s\+\-\*\/\^\(\)]+", cleaned)
        if math_match:
            return math_match.group(0).strip()

        return cleaned

    def _safe_eval(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression using Python's AST.

        Only allows numeric literals and basic arithmetic operators.
        This is safe because it never executes arbitrary code.

        Args:
            expression: A string containing a math expression.

        Returns:
            The numeric result of the expression.

        Raises:
            ValueError: If the expression contains unsupported operations.
        """
        # Parse the expression into an AST
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            raise ValueError(f"Invalid expression: {expression}")

        # Evaluate the AST safely
        return self._eval_node(tree.body)

    def _eval_node(self, node) -> float:
        """
        Recursively evaluate an AST node.

        Only supports:
            - Numbers (int and float literals)
            - Binary operations (+, -, *, /, **)
            - Unary operations (negation, positive)

        Args:
            node: An AST node to evaluate.

        Returns:
            The numeric result.

        Raises:
            ValueError: If an unsupported node type is encountered.
        """
        # Handle numeric constants
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        # Handle binary operations (e.g., 2 + 3)
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._operators:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._operators[op_type](left, right)

        # Handle unary operations (e.g., -5)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._operators:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return self._operators[op_type](operand)

        else:
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")
