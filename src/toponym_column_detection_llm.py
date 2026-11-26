import json
import logging
from dataclasses import dataclass, field

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from src.toponym_column_detection import (
    column_mean_value_len,
    should_score_cell,
)

logger = logging.getLogger(__name__)


@dataclass
class ToponymColumnDetectionConfig:
    # Heuristics to filter columns (defaults from src.toponym_column_detection)
    isdigit_threshold: float = 0.5
    min_cell_len: int = 2
    min_scorable_fraction: float = 0.25
    max_mean_n_words: float = 2.0
    min_mean_alphabetic_value_len: float = 3

    # Model configuration
    model: str = "Qwen/Qwen3.8-27B"
    dtype: str = "auto"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enable_thinking: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 256
    raise_on_error: bool = False

    # Prompt
    n_cells: int = 20
    skip_empty: bool = True
    max_cell_chars: int = 40
    task_hint: str = ""
    system_prompt: str = (
        "You analyse OCR-read tables from historical Finnish church books "
        "(migration records: sisaan- and ulosmuuttaneet, i.e. people moving "
        "in and out of a parish). "
        "You are given the cell texts of a single table column. The OCR text "
        "may contain errors, merged or split cells and noise. "
        "Task: decide whether the column contains place names such as "
        "parishes, villages, farms, towns. Columns with personal names, "
        "dates, occupations, numbers or remarks are not toponym columns. "
        "A column is a toponym column when its cells are place names, even if "
        "the same name is repeated on many rows or the names are abbreviated "
        "or misspelled by the OCR. "
        "Decide from the cells alone and do not reconsider the decision. "
    )
    answer_format: str = (
        "Answer with JSON only, decision first: "
        '{"is_toponym": true|false, '
        '"reason": "<one short sentence>"}'
    )
    
    # Constraint decoding to valid JSON
    answer_schema: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "is_toponym": {"type": "boolean"},
                "reason": {"type": "string", "maxLength": 200},
            },
            "required": ["is_toponym", "reason"],
        }
    )

    def system_message(self) -> str:
        parts = [self.system_prompt, self.task_hint, self.answer_format]
        return " ".join(
            part.strip() for part in parts if part.strip()
        )


def is_candidate_column(
    column: list[str], config: ToponymColumnDetectionConfig
) -> bool:
    """Heuristic check that a column can be a toponym column."""
    non_empty = list(filter(bool, column))
    if not non_empty:
        return False

    n_scorable = sum(
        should_score_cell(c, config.isdigit_threshold, config.min_cell_len)
        for c in non_empty
    )
    if n_scorable / len(non_empty) < config.min_scorable_fraction:
        return False

    mean_n_words = sum(len(c.split()) for c in column) / len(column)
    if mean_n_words > config.max_mean_n_words:
        return False

    alphabetic = ["".join(filter(str.isalpha, c)) for c in column]
    if column_mean_value_len(alphabetic) < config.min_mean_alphabetic_value_len:
        return False

    return True


def render_column(column: list[str], config: ToponymColumnDetectionConfig) -> str:
    cells = [c.replace("\n", " ").strip() for c in column]
    if config.skip_empty:
        cells = [c for c in cells if c]
    cells = cells[: config.n_cells]
    return "\n".join(
        (c[: config.max_cell_chars] if c else "(empty)") for c in cells
    )


def create_client(config: ToponymColumnDetectionConfig) -> LLM:
    return LLM(
        model=config.model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
    )


def classify_columns(
    client: LLM, config: ToponymColumnDetectionConfig, columns: list[list[str]]
) -> list[bool]:
    conversations = [
        [
            {"role": "system", "content": config.system_message()},
            {"role": "user", "content": "Column cells:\n" + render_column(c, config)},
        ]
        for c in columns
    ]
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        structured_outputs=StructuredOutputsParams(json=config.answer_schema),
    )
    outputs = client.chat(
        conversations,
        sampling_params,
        chat_template_kwargs={"enable_thinking": config.enable_thinking},
        use_tqdm=True,
    )

    answers = []
    for output in outputs:
        text = output.outputs[0].text
        try:
            answer = json.loads(text)
            answers.append(answer["is_toponym"])
            logger.debug("llm reason: %s", answer.get("reason", ""))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            error = ValueError(f"unparseable answer: {text!r}")

            if config.raise_on_error:
                raise error from e
            logger.warning(
                "column classification failed, treated as non-toponym: %s",
                error
            )
            answers.append(False)

    return answers


def find_toponym_columns_batch(
    pages: list[list[list[str]]],
    config: ToponymColumnDetectionConfig | None = None,
    client: LLM | None = None,
) -> list[list[int]]:
    """
    Toponym column indices for each page, with all columns classified together.

    pages: for each page, its ditto-filled columns,
           e.g. [fill_values(col) for col in transpose_page_table(page_data)].
    """
    config = config or ToponymColumnDetectionConfig()
    pages = [
        [
            ["" if cell is None else str(cell) for cell in col]
            for col in columns
        ]
        for columns in pages
    ]

    candidates = [
        (page_i, col_i)
        for page_i, columns in enumerate(pages)
        for col_i, col in enumerate(columns)
        if is_candidate_column(col, config)
    ]
    result = [[] for _ in pages]
    if not candidates:
        return result

    if client is None:
        client = create_client(config)

    answers = classify_columns(
        client, config, [pages[page_i][col_i] for page_i, col_i in candidates]
    )
    for (page_i, col_i), answer in zip(candidates, answers):
        if answer:
            result[page_i].append(col_i)
    return result
