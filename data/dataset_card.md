# Power Electronics Distillation Dataset

- Domain: Power electronics (Chinese)
- License: define before release/use (respect upstream sources)
- Source: synthetic prompts + cloud teacher model responses
- Intended use: Supervised fine-tuning (SFT) for small LLM distillation
- Safety: remove PII; avoid dangerous instructions; include disclaimers when needed

## Structure
- data/seed: prompts for teacher
- data/teacher_outputs: raw teacher responses
- data/sft: cleaned + split pairs
- data/eval: holdout evaluation set

## Schema (SFT)
Each JSONL line:
{
  \"id\": str,
  \"prompt\": str,
  \"response\": str,
  \"meta\": {
    \"language\": \"zh\",
    \"domain\": \"power-electronics\",
    \"topic\": str,
    \"subtopic\": str,
    \"task_type\": str,
    \"difficulty\": str,
    \"source\": \"teacher\"
  }
}

## Data quality
- Filter repetitive/degenerate text
- Ensure domain relevance and language consistency
- Balance topics and task types
- Use eval holdout for measurement

## Notes
- Teacher outputs should not include proprietary or licensed text without permission.
- For calculations, prefer structured explanations with final numeric answers.
