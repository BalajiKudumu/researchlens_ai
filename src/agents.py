from google import genai

class ResearchAgentTeam:
    def __init__(self, retriever):
        self.retriever = retriever

    @staticmethod
    def _generate(prompt, api_key, model):
        if not api_key:
            return "Gemini API key is missing. Add it in the sidebar to enable generated analysis."
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text or ""

    @staticmethod
    def _evidence_block(evidence):
        return "\n\n".join(
            f"[{x['citation']}] Paper: {x['paper']}; page: {x['page']}\n{x['text']}"
            for x in evidence
        )

    def answer_question(self, question, evidence, api_key, model):
        prompt = f"""Answer the research question using ONLY the evidence below.
Cite every substantive claim using [E1], [E2] etc. If evidence is insufficient, say so.
Question: {question}
Evidence:
{self._evidence_block(evidence)}
"""
        return self._generate(prompt, api_key, model)

    def multi_agent_answer(self, question, evidence, api_key, model):
        """Five-role sequential workflow using the same evidence set as the baselines."""
        context = self._evidence_block(evidence)
        roles = [
            ("Research paper analyst", "Extract only the facts relevant to the question, with evidence IDs."),
            ("Literature synthesis agent", "Synthesize the extracted facts into a direct, coherent answer. Cite each claim."),
            ("Methodology critic", "Check comparisons, methods and limitations for overgeneralization. Correct the draft."),
            ("Research reasoning agent", "Check whether the answer addresses all parts of the question and clearly state evidence gaps."),
            ("Evidence verification agent", "Audit every claim against its cited passage. Remove unsupported claims and return the final corrected answer. Preserve evidence IDs such as [E1]."),
        ]
        working_notes = ""
        for role, instruction in roles:
            prompt = f"""You are the {role}. Work only from the supplied evidence.
Question: {question}
Task: {instruction}
Prior agent output:
{working_notes[-6000:]}
Source evidence:
{context}
Do not invent findings, statistics, papers or citations.
"""
            working_notes = self._generate(prompt, api_key, model)
        return working_notes

    def run_research_workflow(self, topic, api_key, model):
        evidence = self.retriever.retrieve(topic, top_k=10, mode="hybrid")
        context = self._evidence_block(evidence)
        tasks = {
            "1. Research Paper Analyzer": "Extract the key problem statements, objectives, methods, datasets and findings from the evidence.",
            "2. Literature Review Agent": "Write a synthesized literature review. Identify themes, agreements and disagreements across papers.",
            "3. Methodology Comparison Agent": "Compare the methods, datasets, evaluation designs, strengths and limitations.",
            "4. Research Gap Discovery Agent": "Propose candidate research gaps. Distinguish explicit limitations from your own inferred gaps. Explain why each gap is supported.",
            "5. Evidence and Citation Verification Agent": "Audit the claims in the analysis. List claims with supporting evidence IDs and flag unsupported claims. Do not invent citations."
        }
        report = {}
        previous = ""
        for name, instruction in tasks.items():
            prompt = f"""You are the {name}. Topic: {topic}
Task: {instruction}
Use only the source evidence and cite claims with evidence IDs such as [E1].
Prior agent notes (if any): {previous[:7000]}
Source evidence:
{context}
"""
            result = self._generate(prompt, api_key, model)
            report[name] = result
            previous += f"\n{name}\n{result}\n"
        report["Source Evidence"] = "\n\n".join(
            f"**[{x['citation']}] {x['paper']} (page {x['page']})**\n\n{x['text']}"
            for x in evidence
        )
        return report
