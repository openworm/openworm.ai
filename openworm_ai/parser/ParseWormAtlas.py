from openworm_ai import print_
from bs4 import BeautifulSoup, NavigableString, Comment
import csv

from pathlib import Path

from openworm_ai.parser.DocumentModels import Document, Section, Paragraph

MARKDOWN_DIR = "processed/markdown/wormatlas"
PLAINTEXT_DIR = "processed/plaintext/wormatlas"
JSON_DIR = "processed/json/wormatlas"

CORPUS_LOCATION = "openworm_ai/corpus"


class WormAtlasParser:
    markdown = None
    plaintext = None
    doc_model = None

    html = None

    def __init__(self, title, info):
        filename = info[0]

        self.title = title
        ref = title.replace(" ", "_")

        self.markdown = open(
            Path("%s/%s.md" % (MARKDOWN_DIR, ref)), "w", encoding="utf-8"
        )

        self.markdown.write("# %s\n\n" % title)

        self.plaintext = open(
            Path("%s/%s.txt" % (PLAINTEXT_DIR, ref)), "w", encoding="utf-8"
        )

        self.plaintext.write("%s\n\n" % title)

        src = "https://www.wormatlas.org%s" % info[1]

        self.doc_model = Document(id=title, title=title, source=src)

        verbose = False

        with open(Path(filename), encoding="ISO-8859-1") as f:
            print_("-- Opened: %s" % filename)
            self.html = f.read()

            soup = BeautifulSoup(self.html, "html.parser")

            count = 0
            for element in soup.body.find_all("table")[5].find_all(["p", "span"]):
                print_("%s ==================================" % count)
                element_str = str(element)

                if element is not None:
                    # print_(table.replace('\n', '%s\n'%count))

                    if verbose:
                        print_(
                            "%s -- |%s...|\n"
                            % (count, element_str.replace("\n", "")[:200])
                        )
                    if element.name == "p":
                        # print_(element.contents)
                        anchor = (
                            element.contents[0]["name"]
                            if (
                                len(element.contents) > 0
                                and type(element.contents[0]) is not NavigableString
                                and element.contents[0].has_attr("name")
                            )
                            else ""
                        )

                        if "19" in anchor or "20" in anchor:
                            self.process_reference(element)
                        else:
                            if (
                                "style10" not in element_str
                                and "style13x" not in element_str
                            ):
                                self.process_paragraph(element)

                    elif element.name == "span":
                        if element.attrs["class"][0] == "style10":
                            self.process_header(element, 1)
                        if element.attrs["class"][0] == "style13":
                            self.process_header(element, 2)

                count += 1

        self.finalise()

    def _get_plain_string(self, element):
        plain = ""
        for s in element.contents:
            # print_(" >>> %s" % s)
            if type(s) is Comment:
                pass
            elif type(s) is NavigableString:
                plain += "%s" % s.replace("  ", " ")
            elif s.has_attr("name"):
                pass
            elif s.has_attr("href"):
                plain += "(%s)" % s.attrs["href"]
            else:
                for c in s.contents:
                    plain += "%s" % c

        return plain.strip()

    def _fix_chars(self, text):
        subs = {"ï¿½": "'", "ï¿œ": '"'}
        for s in subs:
            text = text.replace(s, subs[s])
        return text

    def process_header(self, element, depth):
        print_("  - HEADER: %s" % str(element).replace("\n", ""))
        heading = self._get_plain_string(element)

        if len(heading) > 0 and "style13" not in heading:
            print_("  - HEADING: [%s]" % (heading))
            h_number = heading.split(" ", 1)[0]
            h_name = heading.split(" ", 1)[1]

            self.current_section = Section("%s) %s" % (h_number, h_name))
            self.doc_model.sections.append(self.current_section)

            self.markdown.write("%s %s) %s\n\n" % ("#" * (depth + 1), h_number, h_name))

            self.plaintext.write("%s%s) %s\n\n" % (" " * (depth + 1), h_number, h_name))

    def process_reference(self, reference):
        verbose = False
        reference.a["id"] = reference.a["name"]
        r_md = str(reference)

        if verbose:
            print_("  - REF: %s...\n" % (r_md[:80]))

        r_md = r_md.replace("\t", "  ")
        while "  " in r_md:
            r_md = r_md.replace("  ", " ")

        r_md = self._fix_chars(r_md).replace("\n", "")

        self.markdown.write("_%s_\n\n" % r_md)

        self.plaintext.write(
            "%s\n\n" % self._get_plain_string(reference).replace("\n", "")
        )

    def process_paragraph(self, paragraph):
        verbose = False

        p_md = str(paragraph)

        p_md = p_md.replace("\t", "  ")
        while "  " in p_md:
            p_md = p_md.replace("  ", " ")

        p_md = self._fix_chars(p_md).replace("\n", "")

        if len(p_md) > 7:
            self.markdown.write("%s\n\n" % p_md)

        ### Plaintext...

        for a in paragraph.find_all("a"):
            if "href" in a.attrs:
                a = a.replace_with(a.next_element)

        tags_to_plaintext = ["em", "strong"]
        tags_to_remove = ["u", "img"]
        tags_to_return = ["div", "br"]
        RETURN = "RETURN"

        for tag in tags_to_plaintext:
            for ee in paragraph.find_all(tag):
                cc = " ".join([str(c) for c in ee.strings])
                # print_("[%s]" % cc)
                ee = ee.replace_with(cc)

        for tag in tags_to_remove:
            for rr in paragraph.find_all(tag):
                rr = rr.replace_with("")

        for tag in tags_to_return:
            for rr in paragraph.find_all(tag):
                rr = rr.replace_with(RETURN)

        for ss in paragraph.find_all("span"):
            cc = " ".join([str(c) for c in ss.strings])
            # print_("[%s]" % cc)
            ss = ss.replace_with(cc)

        p = self._get_plain_string(paragraph)

        p = p.replace("\t", "  ")

        while "  " in p:
            p = p.replace("  ", " ")

        p = self._fix_chars(p).replace("\n", "")

        # p = p.replace("&lt;br/&gt;", "\n")
        p = p.replace("<br/>", "\n\n")
        p = p.replace("<br>", "\n\n")
        p = p.replace(RETURN, "\n\n")

        if verbose:
            print_(p)

        if len(p) > 0:
            self.plaintext.write("%s\n\n" % p)
            self.current_section.paragraphs.append(Paragraph(p))

    def finalise(self):
        self.plaintext.close()
        self.markdown.close()

        self.doc_model.to_json_file(
            "%s/%s.json" % (JSON_DIR, self.title.replace(" ", "_"))
        )


def read_all_cell_info_file():
    ref = "BasicCellInfo"
    title = "Basic information on C. elegans neurons from WormAtlas"

    markdown = open(Path("%s/%s.md" % (MARKDOWN_DIR, ref)), "w", encoding="utf-8")
    markdown.write("# %s\n\n" % title)

    plaintext = open(Path("%s/%s.txt" % (PLAINTEXT_DIR, ref)), "w", encoding="utf-8")
    plaintext.write("%s\n\n" % title)

    doc_model = Document(
        id=ref,
        title=title,
        source="https://wormatlas.org/neurons/Individual%20Neurons/Neuronframeset.html",
    )

    current_section = Section("Summary information")
    doc_model.sections.append(current_section)

    with open(
        Path(CORPUS_LOCATION + "/wormatlas/%s/all_cell_info.csv" % ref),
        newline="\n",
        encoding="utf-8",
    ) as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')

        for row in reader:
            cell_name = row[0]
            cell_type = row[1]
            cell_name_details = row[2]
            cell_lineage = row[3]
            classification = row[4]
            if cell_name != "Cell name":
                info = f"{cell_name} is a neuron in the worm C. elegans, of type {cell_type}."
                if "To be added" not in cell_name_details:
                    info += f" The name stands for {cell_name_details}."
                if "To be added" not in classification:
                    info += f" {cell_name} is a {classification}."
                if "uscle" not in cell_lineage:
                    info += f" The cell lineage of {cell_name} is {cell_lineage}."

                print_(info)
                plaintext.write("%s\n\n" % info)
                markdown.write("## %s\n\n%s\n\n" % (cell_name, info))

                current_section.paragraphs.append(Paragraph(info))

    plaintext.close()
    markdown.close()

    json_file = doc_model.to_json_file("%s/%s.json" % (JSON_DIR, ref.replace(" ", "_")))

    print_("Written to: %s" % json_file)


if __name__ == "__main__":
    guides = {}

    guides["Introduction"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Introduction.html",
        "/hermaphrodite/introduction/Introframeset.html",
    ]
    guides["Alimentary System"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Alimentary System Overview.html",
        "/hermaphrodite/alimentary/Alimframeset.html",
    ]

    guides["Pharynx"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Alimentary System Pharynx.html",
        "/hermaphrodite/pharynx/Phaframeset.html",
    ]
    guides["Intestine"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Alimentary System Intestine.html",
        "/hermaphrodite/intestine/Intframeset.html",
    ]
    guides["Rectum and Anus"] = [
        CORPUS_LOCATION
        + "/wormatlas/Handbook - Alimentary System Rectum and Anus.html",
        "/hermaphrodite/rectum/Rectframeset.html",
    ]
    guides["Coelomocyte System"] = [
        CORPUS_LOCATION + "/wormatlas/Hermaphrodite Coelomocyte System.html",
        "/hermaphrodite/coelomocyte/Coelomoframeset.html",
    ]
    guides["The Cuticle"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - The Cuticle.html",
        "/hermaphrodite/cuticle/Cutframeset.html",
    ]

    guides["Hypodermis"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Epithelial System Hypodermis.html",
        "/hermaphrodite/hypodermis/Hypframeset.html",
    ]
    guides["Seam cells"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Epithelial System Seam Cells.html",
        "/hermaphrodite/seam cells/Seamframeset.html",
    ]
    guides["Interfacial epithelial cells"] = [
        CORPUS_LOCATION
        + "/wormatlas/Handbook - Epithelial System Interfacial Cells.html",
        "hermaphrodite/interfacial/Interframeset.html",
    ]
    guides["Atypical epithelial cells"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Epithelial System Atypical Cells.html",
        "/hermaphrodite/atypical/Atypframeset.html",
    ]
    guides["Excretory System"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Excretory System.html",
        "/hermaphrodite/excretory/Excframeset.html",
    ]
    guides["Muscle System Introduction"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Muscle System Introduction.html",
        "/hermaphrodite/muscleintro/MusIntroframeset.html",
    ]
    guides["Somatic Muscles"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Muscle System Somatic Muscle.html",
        "/hermaphrodite/musclesomatic/MusSomaticframeset.html",
    ]
    guides["Nonstriated Muscle"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Muscle System Nonstriated Muscle.html",
        "/hermaphrodite/musclenonstriated/MusNonstriframeset.html",
    ]
    guides["GLR Cells"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Muscle System GLR Cells.html",
        "/hermaphrodite/muscleGLR/MusGLRframeset.html",
    ]
    guides["Head Mesodermal Cell"] = [
        CORPUS_LOCATION
        + "/wormatlas/Handbook - Muscle System Head Mesodermal Cell.html",
        "/hermaphrodite/muscleheadcell/Mushmcframeset.html",
    ]

    guides["Nervous System"] = [
        CORPUS_LOCATION
        + "/wormatlas/Handbook - Nervous System General Description.html",
        "/hermaphrodite/nervous/Neuroframeset.html",
    ]
    guides["Neuronal Support Cells"] = [
        CORPUS_LOCATION
        + "/wormatlas/Handbook - Nervous System Neuronal Support Cells.html",
        "/hermaphrodite/neuronalsupport/Neurosupportframeset.html",
    ]
    guides["Reproductive System"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Reproductive System Overview.html",
        "/hermaphrodite/reproductive/Reproframeset.html",
    ]
    guides["Somatic Gonad"] = [
        CORPUS_LOCATION
        + "/wormatlas/Handbook - Reproductive System Somatic Gonad.html",
        "/hermaphrodite/somatic gonad/Somframeset.html",
    ]
    guides["Germ Line"] = [
        CORPUS_LOCATION
        + "/wormatlas/Handbook - Reproductive System Part III Germ Line.html",
        "/hermaphrodite/germ line/Germframeset.html",
    ]
    guides["Egg-laying Apparatus"] = [
        CORPUS_LOCATION
        + "/wormatlas/Handbook - Reproductive System Part VI Egg-laying Apparatus.html",
        "/hermaphrodite/egglaying apparatus/Eggframeset.html",
    ]
    guides["Gap Junctions"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Gap Junctions.html",
        "/hermaphrodite/gapjunctions/Gapjunctframeset.html",
    ]

    guides["Pericellular Structures"] = [
        CORPUS_LOCATION + "/wormatlas/Handbook - Pericellular Structures.html",
        "/hermaphrodite/pericellular/Periframeset.html",
    ]

    """
    guides["XXX"] = [
        CORPUS_LOCATION + "xxx",
        "xxx",
    ]"""

    import os

    # rint(openworm_ai.__file__)
    print_(os.getcwd())
    with open(
        Path("processed/markdown/wormatlas/README.md"), "w", encoding="utf-8"
    ) as readme:
        readme.write("""
## WormAtlas Handbooks

**[Basic Info on _C. elegans_ neurons](BasicCellInfo.md)**
 
The following handbooks from [WormAtlas](https://www.wormatlas.org/handbookhome.htm) have been translated to Markdown format

""")

        # guides = {}
        for g in guides:
            wbp = WormAtlasParser(g, guides[g])
            readme.write(f"**[{g}]({g.replace(' ', '_')}.md)**\n\n")

        read_all_cell_info_file()
