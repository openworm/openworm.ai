from openworm_ai import print_
import modelspec
from modelspec import field, instance_of, optional
from modelspec.base_types import Base

from typing import List
from pathlib import Path

# Models of entities required in this pipeline (documents, etc.) defined in modelspec,
# to ease saving to/from json/yaml


@modelspec.define
class Paragraph(Base):
    """
    A model of a paragraph.

    Args:
        contents: Paragraph contents, which make up the :class:`Section`s.
    """

    contents: str = field(validator=instance_of(str))


@modelspec.define
class Reference(Base):
    """
    A model of a reference.

    Args:
        ref_str: Todo replace with more fields
    """

    ref_str: str = field(validator=instance_of(str))


@modelspec.define
class Section(Base):
    """
    A model of a section of the :class:`Document`.
    Will contain one :class:`Paragraph` or more, i.e the :class:`Paragraph`(s) in the section, probably related to the :code:`title` of the `Document <#document>`_.

    Args:
        id: The id of the section
        paragraphs: The paragraphs
    """

    id: str = field(validator=instance_of(str))
    paragraphs: List[Paragraph] = field(factory=list)


@modelspec.define
class Document(Base):
    """
    A model for documents.

    Args:
        id: The unique id of the document
        title: The document title
        source: The URL, etc.
        sections: The sections of the document
    """

    id: str = field(validator=instance_of(str))
    title: str = field(default=None, validator=optional(instance_of(str)))
    source: str = field(default=None, validator=optional(instance_of(str)))

    sections: List[Section] = field(factory=list)
    references: List[Reference] = field(factory=list)

    def to_markdown(self, file_name):
        fn = Path(file_name)
        with open(fn, "w", encoding="utf-8") as f:
            f.write(f"# {self.title}\n")
            f.write(f"\n_Generated from: {self.source}_\n")
            for section in self.sections:
                f.write(f"\n## {section.id}\n")
                for p in section.paragraphs:
                    f.write(f"\n{p.contents}\n")

        print_(f"Written contents of model {self.id} to {fn}")

    def to_plaintext(self, file_name):
        fn = Path(file_name)
        with open(fn, "w", encoding="utf-8") as f:
            f.write(f"{self.title}\n")
            f.write(f"\nGenerated from: {self.source}\n")
            for section in self.sections:
                f.write(f"\n    {section.id}\n")
                for p in section.paragraphs:
                    f.write(f"\n{p.contents}\n")

        print_(f"Written contents of model {self.id} to {fn}")


if __name__ == "__main__":
    print_("Running tests")

    doc = Document(id="MyTestDoc", source="openworm.org")
    doc.title = "My test document"

    a = Section(id="Abstract")
    a.paragraphs.append(Paragraph(contents="Blah blah blah"))
    a.paragraphs.append(Paragraph(contents="Blah2"))
    doc.sections.append(a)

    c1 = Section(id="Chapter 1")
    doc.sections.append(c1)
    c1.paragraphs.append(Paragraph(contents="More..."))

    print_(doc)
    print_(doc.sections[0].paragraphs[0].contents)
    print_(doc.sections[0].paragraphs[1].__getattribute__("contents"))

    doc.to_json_file("document.json")
    doc.to_yaml_file("document.yaml")
    print_(" >> Full document details in YAML format:\n")
    print_(doc.to_yaml())

    doc.to_markdown("document.md")
    doc.to_plaintext("document.txt")
