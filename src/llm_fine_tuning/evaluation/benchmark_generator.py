import json
import os
import random
from typing import Any, Dict, List

from ..config.settings import BENCHMARK_CONFIG, TARGET_DOMAIN
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BenchmarkGenerator:
    """Generate domain-specific benchmark datasets for EV charging stations."""

    def __init__(self):
        self.categories = BENCHMARK_CONFIG["categories"]
        self.difficulty_levels = BENCHMARK_CONFIG["difficulty_levels"]
        self.num_questions = BENCHMARK_CONFIG["num_questions"]

    def generate_benchmark_dataset(self) -> List[Dict[str, Any]]:
        """Generate a comprehensive benchmark dataset for EV charging stations."""
        logger.info("Generating benchmark dataset for %s", TARGET_DOMAIN)

        benchmark_data = []

        # Generate questions for each category and difficulty level
        question_id = 0
        for category in self.categories:
            for difficulty in self.difficulty_levels:
                questions_per_category = self.num_questions // (
                    len(self.categories) * len(self.difficulty_levels)
                )

                for _ in range(questions_per_category):
                    question_data = self._generate_question(
                        category, difficulty, question_id
                    )
                    benchmark_data.append(question_data)
                    question_id += 1

        # Save benchmark dataset
        os.makedirs("data/evaluation", exist_ok=True)
        with open("data/evaluation/benchmark_dataset.json", "w") as f:
            json.dump(benchmark_data, f, indent=2)

        logger.info("Generated %d benchmark questions", len(benchmark_data))
        return benchmark_data

    def _generate_question(
        self, category: str, difficulty: str, question_id: int
    ) -> Dict[str, Any]:
        """Generate a single question for the given category and difficulty."""

        question_templates = {
            "charging_speed": {
                "easy": [
                    "What is the typical charging speed of a Level 2 charger?",
                    "How fast can a DC fast charger charge an EV?",
                    "What is the difference between Level 1 and Level 2 charging?",
                ],
                "medium": [
                    "What factors affect the actual charging speed of an EV?",
                    "How does battery temperature impact charging speed?",
                    "What is the relationship between charger power and charging time?",
                ],
                "hard": [
                    "How do different charging protocols (CCS, CHAdeMO, Tesla) affect charging speed?",
                    "What is the impact of battery state of charge on charging curve?",
                    "How do charging networks optimize for maximum throughput?",
                ],
            },
            "connector_types": {
                "easy": [
                    "What are the main types of EV charging connectors?",
                    "What connector does Tesla use for Supercharging?",
                    "What is a Type 2 connector?",
                ],
                "medium": [
                    "What is the difference between CCS and CHAdeMO connectors?",
                    "How do connector types vary by region?",
                    "What are the power ratings for different connector types?",
                ],
                "hard": [
                    "How do connector standards evolve and what drives adoption?",
                    "What are the technical specifications for ultra-fast charging connectors?",
                    "How do connector types affect charging network interoperability?",
                ],
            },
            "installation": {
                "easy": [
                    "What is required to install a home EV charger?",
                    "How much does it cost to install a Level 2 charger?",
                    "What electrical requirements are needed for EV charging?",
                ],
                "medium": [
                    "What are the permitting requirements for commercial EV charging installation?",
                    "How do you calculate the electrical load for multiple chargers?",
                    "What are the safety considerations for EV charger installation?",
                ],
                "hard": [
                    "How do you design a scalable charging infrastructure for a fleet?",
                    "What are the grid integration challenges for large-scale charging?",
                    "How do you optimize charger placement for maximum utilization?",
                ],
            },
            "pricing": {
                "easy": [
                    "How much does it cost to charge an EV at home?",
                    "What are typical public charging station prices?",
                    "How do charging costs compare to gasoline?",
                ],
                "medium": [
                    "What are the different pricing models for public charging?",
                    "How do time-of-use rates affect charging costs?",
                    "What are the economics of commercial charging stations?",
                ],
                "hard": [
                    "How do charging networks optimize pricing for profitability?",
                    "What are the regulatory considerations for charging pricing?",
                    "How do demand charges affect commercial charging economics?",
                ],
            },
            "availability": {
                "easy": [
                    "How many public charging stations are there in the US?",
                    "Where can I find public EV charging stations?",
                    "What apps help locate charging stations?",
                ],
                "medium": [
                    "How do charging networks ensure station availability?",
                    "What are the challenges of charging station maintenance?",
                    "How do you plan charging stops for long trips?",
                ],
                "hard": [
                    "How do charging networks predict and manage demand?",
                    "What are the logistics of charging station deployment?",
                    "How do you optimize charging infrastructure for urban areas?",
                ],
            },
        }

        # Select a random question template
        templates = question_templates[category][difficulty]
        question = random.choice(templates)

        # Generate expected answer based on category and difficulty
        answer = self._generate_answer(category, difficulty, question)

        return {
            "id": f"{category}_{difficulty}_{question_id}",
            "category": category,
            "difficulty": difficulty,
            "question": question,
            "expected_answer": answer,
            "domain": TARGET_DOMAIN,
            "metadata": {"generated": True, "source": "benchmark_generator"},
        }

    def _generate_answer(self, category: str, difficulty: str, question: str) -> str:
        """Generate an expected answer for the given question."""

        # This is a simplified answer generator
        # In a real implementation, you might use an LLM API to generate more sophisticated answers

        answer_templates = {
            "charging_speed": {
                "easy": "Level 2 chargers typically provide 3-19 kW of power, adding 10-60 miles of range per hour.",
                "medium": (
                    "Charging speed depends on charger power, battery capacity, "
                    "temperature, and vehicle acceptance rate."
                ),
                "hard": (
                    "Charging protocols, battery chemistry, thermal management, "
                    "and grid capacity all influence charging speed."
                ),
            },
            "connector_types": {
                "easy": (
                    "Main types include Type 1 (J1772), Type 2 (Mennekes), CCS, "
                    "CHAdeMO, and Tesla's proprietary connector."
                ),
                "medium": (
                    "CCS combines AC and DC charging in one connector, while "
                    "CHAdeMO is DC-only. Regional standards vary."
                ),
                "hard": (
                    "Connector evolution is driven by power requirements, "
                    "safety standards, and industry collaboration."
                ),
            },
            "installation": {
                "easy": (
                    "Home installation requires a 240V circuit, proper wiring, "
                    "and often a permit. Costs range from $500-2000."
                ),
                "medium": (
                    "Commercial installations require permits, load calculations, "
                    "and compliance with electrical codes."
                ),
                "hard": (
                    "Fleet installations require load management, grid capacity "
                    "analysis, and scalable infrastructure design."
                ),
            },
            "pricing": {
                "easy": "Home charging costs $0.10-0.15/kWh, while public charging varies from $0.20-0.60/kWh.",
                "medium": "Pricing models include per-kWh, per-minute, subscription, and demand-based pricing.",
                "hard": "Network economics balance utilization, demand charges, and competitive positioning.",
            },
            "availability": {
                "easy": "There are over 50,000 public charging stations in the US, with apps like PlugShare and ChargePoint.",
                "medium": "Networks use monitoring, predictive maintenance, and real-time status updates.",
                "hard": "Demand prediction uses historical data, weather, events, and machine learning algorithms.",
            },
        }

        return answer_templates[category][difficulty]


def create_benchmark_dataset():
    """Create and save a benchmark dataset."""
    generator = BenchmarkGenerator()
    return generator.generate_benchmark_dataset()


if __name__ == "__main__":
    create_benchmark_dataset()
