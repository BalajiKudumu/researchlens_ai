from datetime import datetime

def build_markdown_report(topic, sections):
    lines = [f"# Research synthesis: {topic}", "",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    for heading, body in sections.items():
        lines.extend([f"## {heading}", "", body, ""])
    lines.append("> Research gaps are hypotheses for further investigation, not proof of novelty. Verify them through a systematic literature review.")
    return "\n".join(lines)
