SYSTEM_PROMPT = "Please output {outnum} images according to the instruction: "

CONDITIONED_CREATIVE_PROMPT = (
    "Ref to the first {signal} image, generate image with other images."
)

SREF_PROMPT = (
    "Ref to the second image, transfer the style from image_2 to image_1 "
    "and generate a new image."
)

MULTIVIEW_PROMPT = "Please change views of given image, "


def get_prompt(outnum: int, task_type: str, instr: str) -> str:
    """
    Generate a prompt based on the number of images and task-specific instructions.

    Args:
        outnum (int): The number of images to generate.
        task_type (str): The type of task to perform.
        instr (str): Task-specific instruction.

    Returns:
        str: The formatted prompt string.
    """
    system_prompt = SYSTEM_PROMPT.format(outnum=outnum)
    instruction = instr

    if task_type in ("image_editing", "cref"):
        return system_prompt + instruction

    elif task_type == "conditioned_cref":
        signal_type = instruction
        return system_prompt + CONDITIONED_CREATIVE_PROMPT.format(signal=signal_type)

    elif task_type == "sref":
        return system_prompt + SREF_PROMPT

    elif task_type == "multiview":
        return system_prompt + MULTIVIEW_PROMPT + instruction

    elif task_type == "storyboard":
        story_style, instruction = instruction.split(": ", 1)
        return (
            system_prompt
            + f"Inspired by this character, a {story_style} unfolds: {instruction}"
        )

    else:
        raise ValueError(f"Unsupported task type: {task_type}")
