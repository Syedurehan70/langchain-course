# 01 · Hello LangChain

The first chain: a `PromptTemplate` piped into `ChatGroq` to summarise a
biography and pull out two interesting facts.

- `main.py`: the chain
- `model_check.py`: lists the models currently available on the Groq account
  (useful when a model gets deprecated, which happened mid-course)

The course used `ChatOpenAI`; this version uses Groq's free tier
(`openai/gpt-oss-20b`).
