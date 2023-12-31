# Kaggle_LLM_Science_Exam
### Description
Goal of the Competition
Inspired by the OpenBookQA dataset, this competition challenges participants to answer difficult science-based questions written by a Large Language Model.

Your work will help researchers better understand the ability of LLMs to test themselves, and the potential of LLMs that can be run in resource-constrained environments.

### Context
As the scope of large language model capabilities expands, a growing area of research is using LLMs to characterize themselves. Because many preexisting NLP benchmarks have been shown to be trivial for state-of-the-art models, there has also been interesting work showing that LLMs can be used to create more challenging tasks to test ever more powerful models.

At the same time methods like quantization and knowledge distillation are being used to effectively shrink language models and run them on more modest hardware. The Kaggle environment provides a unique lens to study this as submissions are subject to both GPU and time limits.

The dataset for this challenge was generated by giving gpt3.5 snippets of text on a range of scientific topics pulled from wikipedia, and asking it to write a multiple choice question (with a known answer), then filtering out easy questions.

Right now we estimate that the largest models run on Kaggle are around 10 billion parameters, whereas gpt3.5 clocks in at 175 billion parameters. If a question-answering model can ace a test written by a question-writing model more than 10 times its size, this would be a genuinely interesting result; on the other hand if a larger model can effectively stump a smaller one, this has compelling implications on the ability of LLMs to benchmark and test themselves.
