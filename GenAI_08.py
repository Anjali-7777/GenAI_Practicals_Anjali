"""
Practical 8: NLP Using Hugging Face Transformers
Name: Anjali Patalbansi | Program: MCA (AI & ML)

Install requirements:
pip install transformers torch

Internet access is required to download pretrained models.
"""
import torch
from transformers import (
    pipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)


def text_classification():
    print("\nEXPERIMENT 1: TEXT CLASSIFICATION")
    classifier = pipeline("sentiment-analysis")
    samples = [
        "The workshop was informative and I learned many useful skills.",
        "The service was disappointing and the delivery was very late.",
    ]
    for text, result in zip(samples, classifier(samples)):
        print("Text:", text)
        print("Prediction:", result["label"])
        print("Confidence score:", round(result["score"], 4), "\n")


def text_summarization():
    print("\nEXPERIMENT 2: TEXT SUMMARIZATION")
    text = (
        "Sarah enjoys weekends because she has time for her hobbies. "
        "She usually sleeps a little later on Saturdays and likes to read "
        "in the morning. She does not enjoy cooking every day, so she "
        "sometimes orders food from a nearby restaurant. In the afternoon, "
        "she meets her friends and plans activities for the coming week."
    )
    name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    inputs = tokenizer(
        "summarize: " + text, return_tensors="pt",
        max_length=512, truncation=True
    )
    summary_ids = model.generate(
        inputs["input_ids"], max_length=64, min_length=10, num_beams=4
    )
    print("Original text:", text)
    print("Generated summary:",
          tokenizer.decode(summary_ids[0], skip_special_tokens=True))


def question_answering():
    print("\nEXPERIMENT 3: QUESTION ANSWERING")
    context = (
        "Mira is a computer science student. She studies Python and machine "
        "learning. On weekends, she visits the public library to read "
        "technical books."
    )
    question = "What does Mira study?"
    name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForQuestionAnswering.from_pretrained(name)
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    start = int(torch.argmax(outputs.start_logits))
    end = int(torch.argmax(outputs.end_logits)) + 1
    answer_tokens = inputs["input_ids"][0][start:end]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    print("Context:", context)
    print("Question:", question)
    print("Answer:", answer)


def main():
    print("PRACTICAL 8: NLP USING HUGGING FACE TRANSFORMERS")
    text_classification()
    text_summarization()
    question_answering()
    print("\nResult: Classification, summarization, and question answering completed.")


if __name__ == "__main__":
    main()
