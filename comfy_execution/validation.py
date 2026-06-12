from comfy_api.latest import IO


def format_value_not_in_list_details(input_name, invalid_vals, combo_options, force_truncate=False, max_items=20):
    """Build the ``details`` string for a ``value_not_in_list`` validation error.

    Returns a ``(details, truncated)`` tuple. ``details`` always names the
    offending input and value(s) so the error stays debuggable. The list of
    valid options is replaced with a short length summary when it has more
    than ``max_items`` entries (so errors don't embed entire folder listings)
    or when ``force_truncate`` is set; ``truncated`` is True in that case and
    callers should also omit the input config from the error, since it
    contains the same options list.
    """
    if force_truncate or len(combo_options) > max_items:
        list_info = f"(list of length {len(combo_options)})"
        truncated = True
    else:
        list_info = str(combo_options)
        truncated = False
    details = f"{input_name}: {', '.join(repr(v) for v in invalid_vals)} not in {list_info}"
    return details, truncated


def validate_node_input(
    received_type: str, input_type: str, strict: bool = False
) -> bool:
    """
    received_type and input_type are both strings of the form "T1,T2,...".

    If strict is True, the input_type must contain the received_type.
      For example, if received_type is "STRING" and input_type is "STRING,INT",
      this will return True. But if received_type is "STRING,INT" and input_type is
      "INT", this will return False.

    If strict is False, the input_type must have overlap with the received_type.
      For example, if received_type is "STRING,BOOLEAN" and input_type is "STRING,INT",
      this will return True.

    Supports pre-union type extension behaviour of ``__ne__`` overrides.
    """
    # If the types are exactly the same, we can return immediately
    # Use pre-union behaviour: inverse of `__ne__`
    # NOTE: this lets legacy '*' Any types work that override the __ne__ method of the str class.
    if not received_type != input_type:
        return True

    # If one of the types is '*', we can return True immediately; this is the 'Any' type.
    if received_type == IO.AnyType.io_type or input_type == IO.AnyType.io_type:
        return True

    # If the received type or input_type is a MatchType, we can return True immediately;
    # validation for this is handled by the frontend
    if received_type == IO.MatchType.io_type or input_type == IO.MatchType.io_type:
        return True

    # This accounts for some custom nodes that output lists of options as the type;
    # if we ever want to break them on purpose, this can be removed
    if isinstance(received_type, list) and input_type == IO.Combo.io_type:
        return True

    # Not equal, and not strings
    if not isinstance(received_type, str) or not isinstance(input_type, str):
        return False

    # Split the type strings into sets for comparison
    received_types = set(t.strip() for t in received_type.split(","))
    input_types = set(t.strip() for t in input_type.split(","))

    # If any of the types is '*', we can return True immediately; this is the 'Any' type.
    if IO.AnyType.io_type in received_types or IO.AnyType.io_type in input_types:
        return True

    if strict:
        # In strict mode, all received types must be in the input types
        return received_types.issubset(input_types)
    else:
        # In non-strict mode, there must be at least one type in common
        return len(received_types.intersection(input_types)) > 0
