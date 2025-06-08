import pytest
from query_data import query_rag
from langchain_community.llms.ollama import Ollama
from typing import List, Tuple

EVAL_PROMPT = """
You are a test evaluator for a board game rules assistant. Compare the expected and actual responses:

Expected Response: {expected_response}
Actual Response: {actual_response}

Instructions:
1. Consider the responses equivalent if they contain the same key information
2. Ignore minor formatting differences
3. Consider numerical values equivalent if they represent the same amount
4. Answer with 'true' or 'false' only

Does the actual response match the expected response? Answer with 'true' or 'false'.
"""

class TestBoardGameRules:
    @pytest.mark.monopoly
    def test_monopoly_starting_money(self):
        assert self.query_and_validate(
            question="How much total money does a player start with in Monopoly? (Answer with the number only)",
            expected_response="$1500",
        )

    @pytest.mark.monopoly
    def test_monopoly_rent(self):
        assert self.query_and_validate(
            question="What happens if a player cannot pay rent in Monopoly?",
            expected_response="If a player cannot pay rent, they must mortgage properties or declare bankruptcy",
        )

    @pytest.mark.ticket_to_ride
    def test_ticket_to_ride_longest_train(self):
        assert self.query_and_validate(
            question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
            expected_response="10 points",
        )

    @pytest.mark.ticket_to_ride
    def test_ticket_to_ride_cards(self):
        assert self.query_and_validate(
            question="How many train cards can a player draw on their turn in Ticket to Ride?",
            expected_response="A player can draw up to 2 train cards on their turn",
        )

    @pytest.mark.general
    def test_nonexistent_rule(self):
        assert self.query_and_validate(
            question="What is the rule about flying cars in Monopoly?",
            expected_response="I cannot find that information in the rulebooks",
        )

    def query_and_validate(self, question: str, expected_response: str) -> bool:
        """Query the RAG system and validate the response."""
        try:
            response_text, sources = query_rag(question)
            prompt = EVAL_PROMPT.format(
                expected_response=expected_response, actual_response=response_text
            )

            model = Ollama(model="mistral")
            evaluation_results_str = model.invoke(prompt)
            evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

            print(f"\nTest Question: {question}")
            print(f"Expected: {expected_response}")
            print(f"Actual: {response_text}")
            print(f"Sources: {sources}")

            if "true" in evaluation_results_str_cleaned:
                print("\033[92m" + "✅ Test Passed" + "\033[0m")
                return True
            elif "false" in evaluation_results_str_cleaned:
                print("\033[91m" + "❌ Test Failed" + "\033[0m")
                return False
            else:
                raise ValueError(
                    f"Invalid evaluation result: {evaluation_results_str_cleaned}"
                )

        except Exception as e:
            print(f"\033[91m" + f"❌ Test Error: {str(e)}" + "\033[0m")
            return False

if __name__ == "__main__":
    pytest.main(["-v", "test_rag.py"])
