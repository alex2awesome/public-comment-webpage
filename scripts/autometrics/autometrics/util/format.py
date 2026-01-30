import textwrap
from typing import Optional


def get_default_formatter(dataset, *, truncate_text: Optional[int] = None):
    '''
        Format the summary
    '''
    input_column = dataset.get_input_column()
    output_column = dataset.get_output_column()
    reference_columns = dataset.get_reference_columns()

    if not input_column:
        raise ValueError("Input column not found in dataset.  When constructing your Dataset please provide input_column.")
    if not output_column:
        raise ValueError("Output column not found in dataset.  When constructing your Dataset please provide output_column.")

    def _maybe_truncate(value: str) -> str:
        if truncate_text is None:
            return value
        if value is None:
            return ""
        text = str(value)
        if len(text) <= truncate_text:
            return text
        return textwrap.shorten(text, width=truncate_text, placeholder=" …")

    def default_formatter(row_tuple):
        _, row = row_tuple  # Unpack the tuple from iterrows()
        
        # Handle case where reference_columns is None (reference-free datasets)
        if reference_columns is None or len(reference_columns) == 0:
            return (
                f"«Input ({input_column}): «{_maybe_truncate(row[input_column])}»\n"
                f"Output ({output_column}): «{_maybe_truncate(row[output_column])}»»"
            )
        
        references = [row[col] for col in reference_columns]
        references = [ref for ref in references if ref is not None]

        if not references or len(references) == 0:
            return (
                f"«Input ({input_column}): «{_maybe_truncate(row[input_column])}»\n"
                f"Output ({output_column}): «{_maybe_truncate(row[output_column])}»»"
            )
        
        if len(references) != len(reference_columns):
            print(f"Warning: Expected {len(reference_columns)} references, but found {len(references)} in row {row.name}.")
            print("references:", references)
            print("reference_columns:", reference_columns)

            raise ValueError(f"Expected {len(reference_columns)} references, but found {len(references)} in row {row.name}. Please check your dataset.")
        
        ref_str = "\n".join(
            [
                f"Reference {i+1} ({reference_columns[i]}): «{_maybe_truncate(ref)}»"
                for i, ref in enumerate(references)
            ]
        )
        return (
            f"«Input ({input_column}): «{_maybe_truncate(row[input_column])}»\n"
            f"{ref_str}\n"
            f"Output ({output_column}): «{_maybe_truncate(row[output_column])}»»"
        )
    
    return default_formatter
