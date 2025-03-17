from dataclasses import MISSING, Field, dataclass, fields, is_dataclass
from types import GenericAlias
from typing import Optional, TypeVar, get_args, get_origin


@dataclass
class Subsample:
    name: str


@dataclass
class Sample:
    id: int
    subsamples: list[Subsample]


DataclassInstanceT = TypeVar("DataclassInstanceT")


def create_dataclass_instance_from_dict(
    dataclass_type: type[DataclassInstanceT], dictionary: dict
) -> DataclassInstanceT:
    """Recursively constructs instance of provided dataclass from a dict

    Constructs all nested fields with a dataclass type. Supports lists of nested
    dataclasses

    **Does not support:**
    -  [self-referential type hints](
        https://johnscolaro.xyz/blog/python-self-referential-type-hints-with-pipes
    )
    -  Dicts with dataclasses as values
    -  Iterables other than a list

    Raises:
        TypeError: If `dataclass_type` is not a dataclass or the `dictionary` param has
            an invalid structure or types
    """

    if not is_dataclass(dataclass_type):
        raise TypeError("Param `dataclass` must be a dataclass type")

    def construct_nested_dataclass_instance(
        nested_dataclass_type: type,
        dictionary_representation: dict,
        parent_field: Optional[Field] = None,
    ):
        dictionary_to_pass_to_constructor: dict = {}

        for nested_field in fields(nested_dataclass_type):
            field_type = nested_dataclass_type.__annotations__[nested_field.name]
            field_type_without_generic = get_origin(field_type)

            field_type_generics_args = get_args(field_type)
            field_type_generic_type = (
                field_type_generics_args[0]
                if len(field_type_generics_args) > 0
                else None
            )

            if (
                nested_field.name not in dictionary_representation
                and nested_field.default == MISSING
            ):
                raise TypeError(
                    f"Field `{parent_field.name if parent_field else ''}"
                    + f".{nested_field.name}` must be specified"
                )
            elif (
                nested_field.name not in dictionary_representation
                and nested_field.default != MISSING
            ):
                continue

            if is_dataclass(field_type_without_generic) and isinstance(
                field_type_without_generic, type
            ):
                dictionary_to_pass_to_constructor[nested_field.name] = (
                    construct_nested_dataclass_instance(
                        field_type_without_generic,
                        dictionary_representation[nested_field.name],
                        nested_field,
                    )
                )
            elif (
                field_type_without_generic is list
                and isinstance(field_type, GenericAlias)
                and is_dataclass(field_type_generic_type)
                and isinstance(field_type_generic_type, type)
            ):
                dictionary_to_pass_to_constructor[nested_field.name] = [
                    construct_nested_dataclass_instance(
                        field_type_generic_type, dict_item, nested_field
                    )
                    for dict_item in dictionary_representation[nested_field.name]
                ]
            else:
                dictionary_to_pass_to_constructor[nested_field.name] = (
                    dictionary_representation[nested_field.name]
                )

        return nested_dataclass_type(**dictionary_to_pass_to_constructor)

    return construct_nested_dataclass_instance(dataclass_type, dictionary)
