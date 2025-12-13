from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""You are an expert biographer and storyteller. Provide a comprehensive overview of {celebrity}.

**Context:**
- Style: {style}
- Length: {length}

**Structure your response as follows:**
1. **Early Life & Background**: Birth, family, formative years
2. **Career Rise**: Major milestones and breakthrough moments
3. **Notable Achievements**: Awards, records, cultural impact
4. **Personal Life**: Relationships and interests (if relevant)
5. **Legacy & Influence**: Long-term impact and current relevance

Ensure the tone matches the requested style and keep the explanation to the specified length.""",
    input_variables=["celebrity", "style", "length"],
    validate_template=True,
)

template.save("template.json")
