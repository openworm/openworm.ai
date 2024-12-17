import modelspec
from modelspec import field, instance_of
from modelspec.base_types import Base
from typing import List


@modelspec.define
class Answer(Base):
    ref: str = field(validator=instance_of(str))
    ans: str = field(validator=instance_of(str))
    correct: bool = field(validator=instance_of(bool))


@modelspec.define
class Question(Base):
    question: str = field(validator=instance_of(str))
    # correct_answer: str = field(validator=instance_of(str))
    answers: List[Answer] = field(factory=list)


@modelspec.define
class MultipleChoiceQuiz(Base):
    title: str = field(validator=instance_of(str))
    source: str = field(validator=instance_of(str))

    questions: List[Question] = field(factory=list)


if __name__ == "__main__":
    print("Running tests")

    quiz = MultipleChoiceQuiz(title="Simple quiz", source="PG")

    q1 = Question(question="What is the capital of France?")
    q1.answers.append(Answer("A", "Madrid", False))
    q1.answers.append(Answer("B", "Paris", True))
    quiz.questions.append(q1)

    print(quiz.to_yaml())
