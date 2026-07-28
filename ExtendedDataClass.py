# -*- python-fmt -*-

## Copyright (c) 2024  University of Washington.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## 1. Redistributions of source code must retain the above copyright notice, this
##    list of conditions and the following disclaimer.
##
## 2. Redistributions in binary form must reproduce the above copyright notice,
##    this list of conditions and the following disclaimer in the documentation
##    and/or other materials provided with the distribution.
##
## 3. Neither the name of the University of Washington nor the names of its
##    contributors may be used to endorse or promote products derived from this
##    software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS “AS
## IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE
## LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
## GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
## HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
## LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
## OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""ExtendedDataClass.py - A dataclass base with dict-like access and frozen-attribute enforcement."""

import typing
from dataclasses import dataclass

# Discussion here
# https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
# on using this approach
# @dataclass(ky_only=True)
# Blows up with the overridden default attrs in the below code.
# The inheritance seems to work as long as there are no data members defined in the baseclass


@dataclass
class ExtendedDataClass:
    """A derivative dataclass with the following features.

    - All data attributes must be assigned at object initialization (run-time enforcing of
    mypy [attr-defined] error)
    - `'field' in object` test works for data attributes
    - Allows iteration over data attributes like a dictionary
    """

    # Allow the object["field"] access to work
    def __getitem__(self, item: str) -> typing.Any:  # noqa: ANN401 -- value type depends on the field
        """Gets a dataclass field's value by name, dict-style.

        Args:
            item: Field name to look up.

        Returns:
            The field's current value.

        Raises:
            KeyError: If ``item`` isn't a declared dataclass field.
        """
        if item in self.__dataclass_fields__:
            return getattr(self, item)
        else:
            raise KeyError(item) from None

    def __setitem__(self, item: str, value: typing.Any) -> typing.Any:  # noqa: ANN401 -- value type depends on the field
        """Sets a dataclass field's value by name, dict-style.

        Args:
            item: Field name to set.
            value: New value for the field.

        Returns:
            The result of ``setattr`` (None).

        Raises:
            TypeError: If ``item`` isn't a declared dataclass field.
        """
        if item in self.__dataclass_fields__:
            return setattr(self, item, value)
        else:
            raise TypeError(f"{item} is not assignable")

    # Allow "'field' in object" to work
    def __contains__(self, item: str) -> bool:
        """Checks whether ``item`` is a declared dataclass field.

        Args:
            item: Field name to check.

        Returns:
            True if ``item`` is a declared field of this dataclass.
        """
        # return hasattr(self, item)
        return item in self.__dataclass_fields__

    # Prevents addition of attributes after init, but allows modification of
    # existing attributes.  Essentially the runtime check for they mypy [attr-defined] error

    # Needs the no_type_check attribute because mypy will use this type
    # check for all added attributes if present, instead of raising an error
    @typing.no_type_check
    def __setattr__(self, key: str, value: typing.Any) -> None:  # noqa: ANN401 -- value type depends on the field
        """Sets an attribute, raising if it would add a new attribute after initialization.

        Args:
            key: Attribute name to set.
            value: New value for the attribute.

        Raises:
            TypeError: If ``key`` is not an existing attribute or declared dataclass field
                (i.e. this would add a new attribute rather than update an existing one).
        """
        # The extra check for the key in __dataclass_fields__ covers the
        # case in the constructor for a field that a mutable - the attribute is not yet set,
        # but the member has been added to the dictionary
        if not hasattr(self, key) and key not in self.__dataclass_fields__:
            # ruff: noqa: UP031
            raise TypeError(f"{self!r} is a frozen class")
        object.__setattr__(self, key, value)

    # Allows iteration over the data fields like a dictionary
    def items(self) -> typing.Generator[tuple[typing.Any, typing.Any]]:
        """Iterates over this dataclass's fields like ``dict.items()``.

        Yields:
            ``(field_name, value)`` for each declared field.
        """
        for item in self.__dataclass_fields__:
            yield item, getattr(self, item)

    def keys(self) -> typing.Generator[typing.Any]:
        """Iterates over this dataclass's field names like ``dict.keys()``.

        Yields:
            Each declared field name.
        """
        yield from self.__dataclass_fields__
